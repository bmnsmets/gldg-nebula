"""Finding the Nebula binaries and material files, and running Nebula."""

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .io.electrons import ELECTRON_DTYPE, read_electrons, write_electrons
from .io.triangles import write_tri

__all__ = [
    "find_binary",
    "find_material",
    "gpu_available",
    "run_nebula",
    "stream_nebula",
]

# The repository root: build/bin holds the binaries, materials/ the material files
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
    materials folder of this repository.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    searched = []
    if "NEBULA_MATERIALS_DIR" in os.environ:
        searched.append(Path(os.environ["NEBULA_MATERIALS_DIR"]))
    searched += [Path(d) for d in directories]
    searched.append(_REPOSITORY / "materials")
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
    """Run Nebula and return the detected electrons (`ELECTRON_DTYPE`).

    The input files, the output (output.det) and Nebula's log (nebula.log) are
    written to ``workdir``, by default a temporary directory.

    Args:
        triangles: Geometry, with dtype `TRIANGLE_DTYPE`.
        primaries: Primary electrons, with dtype `ELECTRON_DTYPE`.
        materials: Paths of the material files, in the order of their ids.
        seed: Random seed for Nebula.
        binary: Path of nebula_gpu or nebula_cpu_mt.
        workdir: Directory to keep the files in.
        progress: Called about once a second with the fraction (0 to 1) of
            primaries Nebula has started, or None during nebula_gpu's pre-scan.

    Raises:
        RuntimeError: If Nebula fails; the message includes its log.
    """
    with tempfile.TemporaryDirectory(prefix="nebula-") as temporary:
        directory = Path(workdir if workdir is not None else temporary)
        command = _write_inputs(
            directory, triangles, primaries, materials, seed, binary
        )
        with (
            open(directory / "output.det", "wb") as output,
            open(directory / "nebula.log", "wb") as log,
        ):
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE)
            assert process.stderr is not None
            _follow_log(process.stderr, log, progress)
            returncode = process.wait()
        _check_exit(binary, returncode, directory)
        return read_electrons(directory / "output.det")


def stream_nebula(
    triangles: np.ndarray,
    primaries: np.ndarray,
    materials: Sequence[str | os.PathLike],
    on_detected: Callable[[np.ndarray], None],
    *,
    seed: int,
    binary: str | os.PathLike,
    workdir: str | os.PathLike | None = None,
    progress: Callable[[float | None], None] | None = None,
) -> int:
    """Like `run_nebula`, but pass the detected electrons on in batches while it runs.

    The electrons are not kept; with ``workdir`` they are also written to
    output.det there.

    Args:
        triangles, primaries, materials, seed, binary, workdir, progress:
            See `run_nebula`.
        on_detected: Called, in order and from another thread, with each batch
            (`ELECTRON_DTYPE`), which it may keep but not change.

    Returns:
        The number of detected electrons.

    Raises:
        RuntimeError: If Nebula fails; the message includes its log.
        Exception: Whatever ``on_detected`` raised, once Nebula has finished.
    """
    with tempfile.TemporaryDirectory(prefix="nebula-") as temporary:
        directory = Path(workdir if workdir is not None else temporary)
        command = _write_inputs(
            directory, triangles, primaries, materials, seed, binary
        )
        with (
            open(directory / "nebula.log", "wb") as log,
            open(directory / "output.det", "wb")
            if workdir is not None
            else contextlib.nullcontext() as copy,
            subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as process,
        ):
            assert process.stdout is not None and process.stderr is not None
            reader = _DetectedReader(process.stdout, on_detected, copy)
            reader.start()
            try:
                _follow_log(process.stderr, log, progress)
            except BaseException:
                process.kill()
                raise
            finally:
                returncode = process.wait()
                reader.join()
        _check_exit(binary, returncode, directory)
        if reader.error is not None:
            raise reader.error
        if reader.partial:
            raise RuntimeError(
                f"{Path(binary).name} output ends in a partial record of "
                f"{reader.partial} bytes"
            )
        return reader.count


def _write_inputs(
    directory: Path,
    triangles: np.ndarray,
    primaries: np.ndarray,
    materials: Sequence[str | os.PathLike],
    seed: int,
    binary: str | os.PathLike,
) -> list[str]:
    """Write the geometry and primaries to directory and return Nebula's command line."""
    directory.mkdir(parents=True, exist_ok=True)
    write_tri(directory / "geometry.tri", triangles)
    write_electrons(directory / "primaries.pri", primaries)
    return [
        os.fspath(binary),
        f"--seed={seed}",
        os.fspath(directory / "geometry.tri"),
        os.fspath(directory / "primaries.pri"),
        *(os.fspath(material) for material in materials),
    ]


def _check_exit(binary: str | os.PathLike, returncode: int, directory: Path) -> None:
    if returncode != 0:
        log_text = (directory / "nebula.log").read_text(errors="replace")
        raise RuntimeError(
            f"{Path(binary).name} failed with exit code {returncode}:\n{log_text}"
        )


# Bytes read from Nebula's output at a time, and the pipe size asked for
_READ_SIZE = 2**20


class _DetectedReader(threading.Thread):
    """Reads detected electrons from Nebula's output and passes them on.

    It keeps reading after an error (kept in ``error``), so that Nebula never
    waits on a full pipe.
    """

    def __init__(self, stream, on_detected: Callable[[np.ndarray], None], copy) -> None:
        super().__init__(name="nebula-output", daemon=True)
        self._stream = stream
        self._on_detected = on_detected
        self._copy = copy
        self.count = 0  # electrons passed on
        self.partial = 0  # bytes of an incomplete record at the end
        self.error: Exception | None = None

    def run(self) -> None:
        fd = self._stream.fileno()
        try:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, _READ_SIZE)
        except ImportError, AttributeError, OSError:
            pass  # not Linux, or too large: keep the default pipe size
        size = ELECTRON_DTYPE.itemsize
        rest = np.empty(0, np.uint8)  # an incomplete record from the last read
        while True:
            # A new buffer each time, as on_detected may keep the records
            buffer = np.empty(len(rest) + _READ_SIZE, np.uint8)
            buffer[: len(rest)] = rest
            n = os.readv(fd, [buffer[len(rest) :]])
            if n == 0:
                break
            end = len(rest) + n
            whole = end - end % size
            rest = buffer[whole:end].copy()
            if self._copy is not None:
                try:
                    self._copy.write(buffer[end - n : end])
                except Exception as error:  # noqa: BLE001 - raised once Nebula is done
                    self._copy = None
                    self.error = self.error or error
            if whole and self.error is None:
                try:
                    self._on_detected(buffer[:whole].view(ELECTRON_DTYPE))
                    self.count += whole // size
                except Exception as error:  # noqa: BLE001 - raised once Nebula is done
                    self.error = error
        self.partial = len(rest)


_PROGRESS = re.compile(r"Progress (\d+(?:\.\d+)?)%")


def _follow_log(stream, log, progress: Callable[[float | None], None] | None) -> None:
    """Copy Nebula's stderr to the log and report its progress.

    Nebula overwrites one line: "Executing pre-scan ..." (nebula_gpu), then
    "Progress 12.34%" once a second.
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
