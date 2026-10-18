"""Tests for nebula.runner."""

import os
import stat

import pytest
from nebula import runner


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
