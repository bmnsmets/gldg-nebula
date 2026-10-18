"""Tests for nebula.runner."""

import io
import os
import stat

import pytest
from nebula import runner
from nebula.cli import _Console
from nebula.io.electrons import make_electrons
from nebula.io.triangles import make_triangles


def make_executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_binary_from_environment(tmp_path, monkeypatch):
    make_executable(tmp_path / "nebula_cpu_mt")
    monkeypatch.setenv("NEBULA_BIN_DIR", os.fspath(tmp_path))
    assert runner.find_binary(gpu=False) == tmp_path / "nebula_cpu_mt"


def test_find_binary_falls_back_to_cpu(tmp_path, monkeypatch):
    make_executable(tmp_path / "nebula_gpu")
    make_executable(tmp_path / "nebula_cpu_mt")
    monkeypatch.setenv("NEBULA_BIN_DIR", os.fspath(tmp_path))
    monkeypatch.setattr(runner, "gpu_available", lambda: False)
    assert runner.find_binary() == tmp_path / "nebula_cpu_mt"
    monkeypatch.setattr(runner, "gpu_available", lambda: True)
    assert runner.find_binary() == tmp_path / "nebula_gpu"


def test_find_binary_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBULA_BIN_DIR", raising=False)
    monkeypatch.setenv("PATH", os.fspath(tmp_path))
    monkeypatch.setattr(runner, "_REPOSITORY", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.find_binary(gpu=True)


def test_find_material(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBULA_MATERIALS_DIR", raising=False)
    monkeypatch.setattr(runner, "_REPOSITORY", tmp_path / "repository")
    (tmp_path / "silicon.mat").write_bytes(b"")
    assert runner.find_material("silicon", [tmp_path]) == tmp_path / "silicon.mat"
    with pytest.raises(FileNotFoundError):
        runner.find_material("unobtainium", [tmp_path])


FAKE_NEBULA = r"""#!/bin/sh
printf 'This is Nebula\n' >&2
printf ' \rExecuting pre-scan | running: 10' >&2
sleep 0.05
printf ' \rProgress 10.00%%' >&2
sleep 0.05
printf ' \rProgress 10.00%% \rProgress 100.00%%' >&2
printf '\n\nTiming (seconds):\n' >&2
exit ${FAKE_EXIT:-0}
"""


def run_fake(tmp_path, values):
    binary = tmp_path / "nebula_fake"
    binary.write_text(FAKE_NEBULA)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    triangles = make_triangles([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], [0, -123])
    primaries = make_electrons([0, 0, 1], [0, 0, -1], 100, [0, 0])
    return runner.run_nebula(
        triangles,
        primaries,
        [],
        seed=1,
        binary=binary,
        workdir=tmp_path / "work",
        progress=values.append,
    )


def test_run_nebula_reports_progress(tmp_path):
    values = []
    detected = run_fake(tmp_path, values)
    assert len(detected) == 0
    assert values == [None, 0.1, 1.0]
    log = (tmp_path / "work" / "nebula.log").read_text()
    assert log.startswith("This is Nebula\n") and "Timing (seconds)" in log


def test_run_nebula_failure_includes_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    with pytest.raises(RuntimeError, match="exit code 3") as error:
        run_fake(tmp_path, [])
    assert "This is Nebula" in str(error.value)


def test_console_progress_bar():
    stream = io.StringIO()
    with _Console(file=stream, disable=False) as console:
        console.report("starting")
        for value in [None, 0.5, 1.0]:
            console.progress(value)
        console.report("done")
    text = stream.getvalue()
    assert text.startswith("starting\n") and text.endswith("done\n")
    frames = text.split("\r")
    assert any(frame.startswith("pre-scan:") for frame in frames)
    assert any(frame.startswith("simulating:  50%") for frame in frames)
    assert any(frame.startswith("finishing: 100%") for frame in frames)


def test_console_quiet_and_not_a_terminal():
    stream = io.StringIO()
    with _Console(file=stream, quiet=True, disable=False) as console:
        console.report("hidden")
        console.progress(0.5)
    assert stream.getvalue() == ""
    stream = io.StringIO()  # not a terminal: tqdm leaves the bar out
    with _Console(file=stream) as console:
        console.report("shown")
        console.progress(0.5)
    assert stream.getvalue() == "shown\n"
