"""Runs the known-answer test of the Philox generator that nebula_gpu uses."""

import subprocess

import pytest
from nebula import runner

BINARY = runner._find_executable("test_philox")

pytestmark = [
    pytest.mark.requires_nebula,
    pytest.mark.skipif(BINARY is None, reason="test_philox not built"),
]


def test_philox_known_answers():
    result = subprocess.run([BINARY], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
