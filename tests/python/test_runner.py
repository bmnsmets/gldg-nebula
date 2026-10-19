"""Tests for nebula.runner."""

import io
import os
import stat

import numpy as np
import pytest
from nebula import runner
from nebula.cli import _Console
from nebula.io.electrons import ELECTRON_DTYPE, make_electrons
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


# Writes the records of $FAKE_RECORDS to stdout in pieces of odd sizes,
# which split records, with pauses so that the reader sees them separately
FAKE_STREAMING_NEBULA = r"""#!/usr/bin/env python3
import os, sys, time
data = open(os.environ["FAKE_RECORDS"], "rb").read()
sys.stderr.write("This is Nebula\n")
start = 0
for size in [1, 35, 37, 1000, 71, 5000, 100000]:
    os.write(1, data[start : start + size])
    start += size
    time.sleep(0.002)
os.write(1, data[start:])
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


def stream_fake(tmp_path, monkeypatch, records, on_detected, workdir=None):
    binary = tmp_path / "nebula_fake_streaming"
    binary.write_text(FAKE_STREAMING_NEBULA)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    path = tmp_path / "records.bin"
    path.write_bytes(records.tobytes() if isinstance(records, np.ndarray) else records)
    monkeypatch.setenv("FAKE_RECORDS", os.fspath(path))
    triangles = make_triangles([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], [0, -123])
    primaries = make_electrons([0, 0, 1], [0, 0, -1], 100, [0, 0])
    return runner.stream_nebula(
        triangles, primaries, [], on_detected, seed=1, binary=binary, workdir=workdir
    )


def random_records(n):
    rng = np.random.default_rng(3)
    records = np.zeros(n, dtype=ELECTRON_DTYPE)
    records["x"] = rng.normal(size=n)
    records["E"] = rng.uniform(0, 1000, n)
    records["px"] = rng.integers(0, 100, n)
    return records


def test_stream_nebula_passes_on_all_records(tmp_path, monkeypatch):
    records = random_records(5000)
    batches = []
    count = stream_fake(tmp_path, monkeypatch, records, batches.append)
    assert count == len(records) and len(batches) > 1
    assert all(batch.dtype == ELECTRON_DTYPE for batch in batches)
    assert np.concatenate(batches).tobytes() == records.tobytes()


def test_stream_nebula_keeps_output_in_workdir(tmp_path, monkeypatch):
    records = random_records(3000)
    work = tmp_path / "work"
    count = stream_fake(tmp_path, monkeypatch, records, lambda batch: None, work)
    assert count == len(records)
    assert (work / "output.det").read_bytes() == records.tobytes()
    assert (work / "nebula.log").read_text().startswith("This is Nebula")


def test_stream_nebula_raises_callback_error(tmp_path, monkeypatch):
    def fail(batch):
        raise KeyError("callback failed")

    work = tmp_path / "work"
    records = random_records(3000)
    with pytest.raises(KeyError, match="callback failed"):
        stream_fake(tmp_path, monkeypatch, records, fail, work)
    # Nebula's output was still read to the end
    assert (work / "output.det").read_bytes() == records.tobytes()


def test_stream_nebula_failure_and_partial_record(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="partial record of 10 bytes"):
        stream_fake(
            tmp_path, monkeypatch, random_records(2).tobytes() + bytes(10), print
        )
    monkeypatch.setenv("FAKE_EXIT", "3")
    with pytest.raises(RuntimeError, match="exit code 3") as error:
        stream_fake(tmp_path, monkeypatch, random_records(2), print)
    assert "This is Nebula" in str(error.value)
