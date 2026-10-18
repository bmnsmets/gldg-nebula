"""Running the Nebula binaries: finding them and their material files."""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .io.electrons import read_electrons, write_electrons
from .io.triangles import write_tri

__all__ = ["find_binary", "find_material", "gpu_available", "run_nebula"]

# The repository this package is part of, for its build folder
_REPOSITORY = Path(__file__).resolve().parents[2]


def gpu_available() -> bool:
    """Whether nvidia-smi lists a GPU."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError, subprocess.SubprocessError:
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def _find_executable(name: str) -> Path | None:
    candidates = []
    if "NEBULA_BIN_DIR" in os.environ:
        candidates.append(Path(os.environ["NEBULA_BIN_DIR"]) / name)
    found = shutil.which(name)
    if found is not None:
        candidates.append(Path(found))
    candidates.append(_REPOSITORY / "build" / "bin" / name)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_binary(gpu: bool | None = None) -> Path:
    """Find nebula_gpu or nebula_cpu_mt.

    Looks in $NEBULA_BIN_DIR, on the PATH, and in the build/bin folder of
    this repository.

    Args:
        gpu: True for nebula_gpu, False for nebula_cpu_mt, None for
            nebula_gpu if it is found and there is a GPU, else nebula_cpu_mt.

    Raises:
        FileNotFoundError: If the binary is not found.
    """
    if gpu is None:
        binary = _find_executable("nebula_gpu")
        if binary is not None and gpu_available():
            return binary
        gpu = False
    name = "nebula_gpu" if gpu else "nebula_cpu_mt"
    binary = _find_executable(name)
    if binary is None:
        raise FileNotFoundError(
            f"{name} not found in $NEBULA_BIN_DIR, on the PATH or in "
            f"{_REPOSITORY / 'build' / 'bin'}; build Nebula with CMake first"
        )
    return binary


def find_material(name: str, directories: Sequence[str | os.PathLike] = ()) -> Path:
    """Find the material file <name>.mat.

    Looks in $NEBULA_MATERIALS_DIR, the given directories, and the
    build/materials folder of this repository.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    searched = []
    if "NEBULA_MATERIALS_DIR" in os.environ:
        searched.append(Path(os.environ["NEBULA_MATERIALS_DIR"]))
    searched += [Path(d) for d in directories]
    searched.append(_REPOSITORY / "build" / "materials")
    for directory in searched:
        if (directory / f"{name}.mat").is_file():
            return directory / f"{name}.mat"
    raise FileNotFoundError(
        f"material file {name}.mat not found in {', '.join(map(str, searched))}; "
        f"generate it with cstool or with -DNEBULA_MATERIALS when building"
    )


def run_nebula(
    triangles: np.ndarray,
    primaries: np.ndarray,
    materials: Sequence[str | os.PathLike],
    *,
    seed: int,
    binary: str | os.PathLike,
    workdir: str | os.PathLike | None = None,
    progress: Callable[[float | None], None] | None = None,
) -> np.ndarray:
    """Run Nebula and return the detected electrons.

    The geometry and primary electrons are written to files in ``workdir``
    (by default a temporary directory that is removed afterwards), together
    with the detected electrons (output.det) and Nebula's log (nebula.log).

    Args:
        triangles: Geometry, with dtype `TRIANGLE_DTYPE`.
        primaries: Primary electrons, with dtype `ELECTRON_DTYPE`.
        materials: Paths of the material files, in the order of their ids.
        seed: Random seed for Nebula.
        binary: Path of nebula_gpu or nebula_cpu_mt.
        workdir: Directory to keep the files in.
        progress: Called while Nebula runs, with the fraction of primary
            electrons it has started simulating (0 to 1), or with None while
            nebula_gpu does its pre-scan. Nebula reports this once a second.

    Returns:
        The detected electrons, with dtype `ELECTRON_DTYPE`.

    Raises:
        RuntimeError: If Nebula fails; the message includes its log.
    """
    with tempfile.TemporaryDirectory(prefix="nebula-") as temporary:
        directory = Path(workdir if workdir is not None else temporary)
        directory.mkdir(parents=True, exist_ok=True)
        write_tri(directory / "geometry.tri", triangles)
        write_electrons(directory / "primaries.pri", primaries)
        command = [
            os.fspath(binary),
            f"--seed={seed}",
            os.fspath(directory / "geometry.tri"),
            os.fspath(directory / "primaries.pri"),
            *(os.fspath(material) for material in materials),
        ]
        with (
            open(directory / "output.det", "wb") as output,
            open(directory / "nebula.log", "wb") as log,
        ):
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE)
            assert process.stderr is not None
            _follow_log(process.stderr, log, progress)
            returncode = process.wait()
        if returncode != 0:
            log_text = (directory / "nebula.log").read_text(errors="replace")
            raise RuntimeError(
                f"{Path(binary).name} failed with exit code {returncode}:\n{log_text}"
            )
        return read_electrons(directory / "output.det")


_PROGRESS = re.compile(r"Progress (\d+(?:\.\d+)?)%")


def _follow_log(stream, log, progress: Callable[[float | None], None] | None) -> None:
    """Copy Nebula's stderr to the log, reporting its progress as it comes.

    Nebula rewrites one line with carriage returns: "Executing pre-scan ..."
    (nebula_gpu) and then "Progress 12.34%" once a second.
    """
    last: float | None | str = "nothing"  # the last value reported
    line = ""  # text since the last carriage return or newline
    while chunk := os.read(stream.fileno(), 65536):
        log.write(chunk)
        log.flush()
        if progress is None:
            continue
        *finished, line = re.split(r"[\r\n]", line + chunk.decode(errors="replace"))
        for text in [*finished, line]:
            if (match := _PROGRESS.search(text)) is not None:
                value: float | None = float(match.group(1)) / 100
            elif "pre-scan" in text:
                value = None
            else:
                continue
            if value != last:
                progress(value)
                last = value
