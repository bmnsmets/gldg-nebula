"""Binary electron files: primaries (.pri), detected electrons (.det) and energy deposits.

The files have no header, only fixed-size records one after another.

.pri and .det files use `ELECTRON_DTYPE` (36 bytes):

- ``x``, ``y``, ``z``: position in nm; ``dx``, ``dy``, ``dz``: direction (any
  length); ``E``: kinetic energy in eV (all float32);
- ``px``, ``py``: pixel indices (int32). Nebula copies them from each primary to
  the electrons it causes, so they can tag electrons in any way.

``nebula_gpu`` and ``nebula_cpu_mt`` write the detected electrons, in no fixed
order, to standard output. ``nebula_cpu_edep`` also writes one `DEPOSIT_DTYPE`
record (28 bytes) per scattering event: position (nm), energy before the event
``E`` and energy lost ``dE`` (eV), and the primary's ``px``, ``py``.

Inside a material, energies include its ``barrier``: a 500 eV electron in
silicon (barrier 12.44 eV) has ``E`` = 512.44 eV. In vacuum they are the usual
kinetic energy.
"""

import os
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions

__all__ = [
    "DEPOSIT_DTYPE",
    "ELECTRON_DTYPE",
    "make_electrons",
    "read_deposits",
    "read_electrons",
    "write_deposits",
    "write_electrons",
]

ELECTRON_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("dx", "<f4"),
        ("dy", "<f4"),
        ("dz", "<f4"),
        ("E", "<f4"),
        ("px", "<i4"),
        ("py", "<i4"),
    ]
)
"""One electron in a .pri or .det file (36 bytes)."""

DEPOSIT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("E", "<f4"),
        ("dE", "<f4"),
        ("px", "<i4"),
        ("py", "<i4"),
    ]
)
"""One energy deposit written by ``nebula_cpu_edep`` (28 bytes)."""


def _check_size(n_bytes: int, dtype: np.dtype) -> None:
    if n_bytes % dtype.itemsize != 0:
        raise ValueError(
            f"file size {n_bytes} is not a multiple of {dtype.itemsize} bytes"
        )


def _check_pixels(pixels: np.ndarray, name: str) -> None:
    if pixels.dtype == np.int32:
        return
    if not (
        np.issubdtype(pixels.dtype, np.integer) or np.all(pixels == np.round(pixels))
    ):
        raise ValueError(f"{name} must be integers")
    info = np.iinfo(np.int32)
    if np.any(pixels < info.min) or np.any(pixels > info.max):
        raise ValueError(f"{name} must fit in an int32")


# Records checked at a time before writing (a chunk is faster than all at once)
_CHECK_CHUNK = 2**16


def _read_records(
    file: str | os.PathLike | BinaryIO, dtype: np.dtype, mmap: bool
) -> np.ndarray:
    if isinstance(file, (str, os.PathLike)):
        n_bytes = os.path.getsize(file)
        _check_size(n_bytes, dtype)
        if mmap and n_bytes > 0:
            return np.memmap(file, dtype=dtype, mode="r")
        return np.fromfile(file, dtype=dtype)

    if mmap:
        raise TypeError("mmap needs a path, not a file object")
    data = file.read()
    _check_size(len(data), dtype)
    return np.frombuffer(data, dtype=dtype).copy()


def _write_records(
    file: str | os.PathLike | BinaryIO, records: np.ndarray, dtype: np.dtype, what: str
) -> None:
    records = np.asarray(records)
    names = records.dtype.names or ()
    missing = [name for name in dtype.names or () if name not in names]
    if missing:
        raise ValueError(f"{what} are missing the fields {', '.join(missing)}")
    if records.ndim > 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {records.shape}")
    _check_pixels(records["px"], "field px")
    _check_pixels(records["py"], "field py")
    records = records.reshape(-1)
    if records.dtype != dtype:
        # Values too large for float32 become inf, which the check below reports
        with np.errstate(over="ignore"):
            records = recfunctions.require_fields(records, dtype)
    floats = [name for name in dtype.names or () if dtype[name].kind == "f"]
    for start in range(0, len(records), _CHECK_CHUNK):
        chunk = records[start : start + _CHECK_CHUNK]
        for name in floats:
            if not np.all(np.isfinite(chunk[name])):
                raise ValueError(f"field {name} must be finite (in float32)")

    if isinstance(file, (str, os.PathLike)):
        records.tofile(file)
    else:
        file.write(records.tobytes())


def read_electrons(
    file: str | os.PathLike | BinaryIO, *, mmap: bool = False
) -> np.ndarray:
    """Read a .pri or .det file as an array with dtype `ELECTRON_DTYPE`.

    Args:
        file: Path of the file, or an open binary file.
        mmap: Map the file into memory instead of reading it (for files larger
            than memory); the array is then read-only. Paths only.

    Raises:
        ValueError: If the file size is not a whole number of records.
    """
    return _read_records(file, ELECTRON_DTYPE, mmap)


def write_electrons(file: str | os.PathLike | BinaryIO, electrons: np.ndarray) -> None:
    """Write a .pri or .det file.

    To write a large file in parts, call this once per part with an open file.

    Args:
        file: Path of the file, or an open binary file.
        electrons: Array with (at least) the fields of `ELECTRON_DTYPE`; other
            fields are ignored.

    Raises:
        ValueError: If a field is missing, a value is not finite, or a pixel
            index does not fit in an int32.
    """
    _write_records(file, electrons, ELECTRON_DTYPE, "electrons")


def read_deposits(
    file: str | os.PathLike | BinaryIO, *, mmap: bool = False
) -> np.ndarray:
    """Read an energy deposit file as an array with dtype `DEPOSIT_DTYPE`.

    Arguments as for `read_electrons`.
    """
    return _read_records(file, DEPOSIT_DTYPE, mmap)


def write_deposits(file: str | os.PathLike | BinaryIO, deposits: np.ndarray) -> None:
    """Write an energy deposit file (fields of `DEPOSIT_DTYPE`).

    Arguments as for `write_electrons`.
    """
    _write_records(file, deposits, DEPOSIT_DTYPE, "deposits")


def make_electrons(
    position: npt.ArrayLike,
    direction: npt.ArrayLike,
    energy: npt.ArrayLike,
    pixel: npt.ArrayLike,
) -> np.ndarray:
    """Make an array of electrons (`ELECTRON_DTYPE`) from separate arrays.

    A value that is the same for all electrons can be given once.

    Args:
        position: Positions in nm, shape (N, 3) or (3,).
        direction: Directions (any length), shape (N, 3) or (3,).
        energy: Kinetic energies in eV, shape (N,) or a number.
        pixel: Pixel indices (px, py), shape (N, 2) or (2,).

    Raises:
        ValueError: If the shapes do not match or a value is invalid.
    """
    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    pixel = np.asarray(pixel)
    for name, array, columns in [
        ("position", position, 3),
        ("direction", direction, 3),
        ("pixel", pixel, 2),
    ]:
        if array.ndim not in (1, 2) or array.shape[-1] != columns:
            raise ValueError(
                f"{name} must have shape (N, {columns}) or ({columns},), got {array.shape}"
            )
    if energy.ndim > 1:
        raise ValueError(
            f"energy must have shape (N,) or be a scalar, got {energy.shape}"
        )
    try:
        (n,) = np.broadcast_shapes(
            position.shape[:-1],
            direction.shape[:-1],
            energy.shape,
            pixel.shape[:-1],
            (1,),
        )
    except ValueError:
        raise ValueError(
            "position, direction, energy and pixel have different numbers of electrons"
        ) from None

    if not (np.all(np.isfinite(position)) and np.all(np.isfinite(direction))):
        raise ValueError("position and direction must be finite")
    if not np.all(np.any(direction != 0, axis=-1)):
        raise ValueError("direction must not be zero")
    if not np.all((energy > 0) & np.isfinite(energy)):
        raise ValueError("energy must be positive and finite")
    _check_pixels(pixel, "pixel indices")

    electrons = np.empty(n, dtype=ELECTRON_DTYPE)
    for i, name in enumerate(["x", "y", "z"]):
        electrons[name] = position[..., i]
    for i, name in enumerate(["dx", "dy", "dz"]):
        electrons[name] = direction[..., i]
    electrons["E"] = energy
    electrons["px"] = pixel[..., 0]
    electrons["py"] = pixel[..., 1]
    return electrons
