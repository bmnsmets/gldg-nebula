"""Tests for nebula.io.electrons."""

import io
from pathlib import Path

import numpy as np
import pytest
from nebula.io import (
    DEPOSIT_DTYPE,
    ELECTRON_DTYPE,
    make_electrons,
    read_deposits,
    read_electrons,
    write_deposits,
    write_electrons,
)

SEM_PRI = Path(__file__).parents[2] / "examples" / "tutorial-sem" / "sem.pri"


def random_electrons(n, seed=0):
    rng = np.random.default_rng(seed)
    electrons = np.empty(n, dtype=ELECTRON_DTYPE)
    for name in ["x", "y", "z", "dx", "dy", "dz"]:
        electrons[name] = rng.normal(size=n)
    electrons["E"] = rng.uniform(100, 1000, size=n)
    electrons["px"] = rng.integers(-1000, 1000, size=n)
    electrons["py"] = rng.integers(-1000, 1000, size=n)
    return electrons


def test_dtype_matches_tutorial():
    tutorial_dtype = np.dtype(
        [(name, "=f") for name in ["x", "y", "z", "dx", "dy", "dz", "E"]]
        + [("px", "=i"), ("py", "=i")]
    )
    assert ELECTRON_DTYPE.itemsize == 36
    assert ELECTRON_DTYPE == tutorial_dtype


@pytest.mark.skipif(not SEM_PRI.exists(), reason="sem.pri has not been generated")
def test_read_tutorial_primaries():
    electrons = read_electrons(SEM_PRI)
    assert electrons.dtype == ELECTRON_DTYPE
    assert len(electrons) == SEM_PRI.stat().st_size // 36
    assert np.all(electrons["E"] == 500)
    assert np.all(electrons["dz"] == -1)


def test_round_trip(tmp_path):
    electrons = random_electrons(100)
    path = tmp_path / "primaries.pri"
    write_electrons(path, electrons)
    assert path.stat().st_size == 100 * 36
    np.testing.assert_array_equal(read_electrons(path), electrons)

    mapped = read_electrons(path, mmap=True)
    np.testing.assert_array_equal(mapped, electrons)
    assert not mapped.flags.writeable

    buffer = io.BytesIO()
    write_electrons(buffer, electrons)
    buffer.seek(0)
    read = read_electrons(buffer)
    np.testing.assert_array_equal(read, electrons)
    assert read.flags.writeable


def test_write_in_parts():
    electrons = random_electrons(10)
    buffer = io.BytesIO()
    write_electrons(buffer, electrons[:4])
    write_electrons(buffer, electrons[4:])
    buffer.seek(0)
    np.testing.assert_array_equal(read_electrons(buffer), electrons)


def test_write_converts_fields():
    electrons = random_electrons(5)
    names = list(reversed(ELECTRON_DTYPE.names))
    converted = np.empty(5, dtype=[(name, "f8") for name in names] + [("extra", "i8")])
    for name in ELECTRON_DTYPE.names:
        converted[name] = electrons[name]
    buffer = io.BytesIO()
    write_electrons(buffer, converted)
    buffer.seek(0)
    np.testing.assert_array_equal(read_electrons(buffer), electrons)


def test_read_empty(tmp_path):
    path = tmp_path / "empty.pri"
    path.write_bytes(b"")
    assert read_electrons(path).shape == (0,)
    assert read_electrons(path, mmap=True).shape == (0,)


def test_read_truncated(tmp_path):
    path = tmp_path / "truncated.pri"
    path.write_bytes(b"\0" * 35)
    with pytest.raises(ValueError):
        read_electrons(path)
    with pytest.raises(ValueError):
        read_electrons(io.BytesIO(b"\0" * 35))


def test_mmap_needs_path():
    with pytest.raises(TypeError):
        read_electrons(io.BytesIO(), mmap=True)


@pytest.mark.parametrize(
    "electrons",
    [
        np.zeros(3, dtype=[("x", "f4"), ("y", "f4")]),  # missing fields
        np.zeros((2, 2), dtype=ELECTRON_DTYPE),  # not one-dimensional
        np.array([(np.nan, 0, 0, 0, 0, -1, 500, 0, 0)], dtype=ELECTRON_DTYPE),  # not finite
        np.array(
            [(1e40, 0, 0, 0, 0, -1, 500, 0, 0)],
            dtype=[(n, "f8") for n in ELECTRON_DTYPE.names],
        ),  # float32 overflow
        np.array(
            [(0, 0, 0, 0, 0, -1, 500, 0.5, 0)],
            dtype=[(n, "f8") for n in ELECTRON_DTYPE.names],
        ),  # non-integer pixel
        np.array(
            [(0, 0, 0, 0, 0, -1, 500, 0, 2**31)],
            dtype=[(n, "i8") for n in ELECTRON_DTYPE.names],
        ),  # pixel out of int32 range
    ],
)
def test_write_invalid(electrons):
    with pytest.raises(ValueError):
        write_electrons(io.BytesIO(), electrons)


def test_make_electrons_broadcasts():
    position = [[0, 0, 35], [1, 2, 35]]
    electrons = make_electrons(position, [0, 0, -1], 500, [[3, 4], [5, 6]])
    assert electrons.dtype == ELECTRON_DTYPE
    assert electrons.shape == (2,)
    np.testing.assert_array_equal(electrons["x"], [0, 1])
    np.testing.assert_array_equal(electrons["y"], [0, 2])
    np.testing.assert_array_equal(electrons["dz"], [-1, -1])
    np.testing.assert_array_equal(electrons["E"], [500, 500])
    np.testing.assert_array_equal(electrons["px"], [3, 5])
    np.testing.assert_array_equal(electrons["py"], [4, 6])


def test_make_electrons_single():
    electrons = make_electrons([0, 0, 1], [0, 0, -1], 1000, [0, 0])
    assert electrons.shape == (1,)


@pytest.mark.parametrize(
    "position, direction, energy, pixel",
    [
        (np.zeros((2, 3)), [0, 0, -1], [1, 2, 3], [0, 0]),  # mismatched N
        (np.zeros((2, 2)), [0, 0, -1], 500, [0, 0]),  # bad position shape
        ([0, 0, 0], [0, 0, 0], 500, [0, 0]),  # zero direction
        ([0, 0, np.inf], [0, 0, -1], 500, [0, 0]),  # non-finite position
        ([0, 0, 0], [0, 0, -1], 0, [0, 0]),  # non-positive energy
        ([0, 0, 0], [0, 0, -1], 500, [0.5, 0]),  # non-integer pixel
        ([0, 0, 0], [0, 0, -1], 500, [2**31, 0]),  # pixel out of int32 range
    ],
)
def test_make_electrons_invalid(position, direction, energy, pixel):
    with pytest.raises(ValueError):
        make_electrons(position, direction, energy, pixel)


def random_deposits(n, seed=0):
    rng = np.random.default_rng(seed)
    deposits = np.empty(n, dtype=DEPOSIT_DTYPE)
    for name in ["x", "y", "z"]:
        deposits[name] = rng.normal(size=n)
    deposits["E"] = rng.uniform(100, 1000, size=n)
    deposits["dE"] = rng.uniform(0, 100, size=n)
    deposits["px"] = rng.integers(-1000, 1000, size=n)
    deposits["py"] = rng.integers(-1000, 1000, size=n)
    return deposits


def test_deposit_dtype_matches_nebula():
    # nebula_cpu_edep writes x, y, z, E before, energy lost (float), then px, py (int)
    assert DEPOSIT_DTYPE.itemsize == 28
    assert DEPOSIT_DTYPE.names == ("x", "y", "z", "E", "dE", "px", "py")
    assert [DEPOSIT_DTYPE[n].kind for n in DEPOSIT_DTYPE.names] == ["f"] * 5 + ["i"] * 2


def test_deposits_round_trip(tmp_path):
    deposits = random_deposits(100)
    path = tmp_path / "deposits.bin"
    write_deposits(path, deposits)
    assert path.stat().st_size == 100 * 28
    np.testing.assert_array_equal(read_deposits(path), deposits)
    np.testing.assert_array_equal(read_deposits(path, mmap=True), deposits)

    buffer = io.BytesIO()
    write_deposits(buffer, deposits[:30])
    write_deposits(buffer, deposits[30:])
    buffer.seek(0)
    np.testing.assert_array_equal(read_deposits(buffer), deposits)


def test_read_deposits_truncated():
    with pytest.raises(ValueError):
        read_deposits(io.BytesIO(b"\0" * 27))


@pytest.mark.parametrize(
    "deposits",
    [
        np.zeros(3, dtype=[("x", "f4"), ("E", "f4")]),  # missing fields
        np.zeros(3, dtype=ELECTRON_DTYPE),  # electrons have no dE
        np.array([(0, 0, 0, 500, np.nan, 0, 0)], dtype=DEPOSIT_DTYPE),  # not finite
        np.array(
            [(0, 0, 0, 500, 10, 0.5, 0)],
            dtype=[(n, "f8") for n in DEPOSIT_DTYPE.names],
        ),  # non-integer pixel
    ],
)
def test_write_deposits_invalid(deposits):
    with pytest.raises(ValueError):
        write_deposits(io.BytesIO(), deposits)


@pytest.mark.parametrize("field", ["x", "dz", "E"])
def test_write_checks_every_chunk(field):
    # Records that already have the file's dtype are written as they are,
    # but checked in chunks: a bad value after the first chunk is found too
    electrons = random_electrons(3 * 2**16 + 5)
    stream = io.BytesIO()
    write_electrons(stream, electrons)
    assert stream.getvalue() == electrons.tobytes()
    electrons[2**17 + 3][field] = np.inf
    with pytest.raises(ValueError, match=f"field {field} must be finite"):
        write_electrons(io.BytesIO(), electrons)
