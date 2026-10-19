"""Binary electron files: primaries (.pri), detected electrons (.det) and energy deposits.

These files have no header; they hold one fixed-size record after another.

Primary and detected electrons use the 36-byte `ELECTRON_DTYPE` record:

- ``x``, ``y``, ``z``: position in nm (float32);
- ``dx``, ``dy``, ``dz``: direction, need not be normalised (float32);
- ``E``: kinetic energy in eV (float32);
- ``px``, ``py``: pixel indices (int32). Nebula does not interpret these, it only
  copies them from a primary electron to all detected electrons it causes, so
  they can tag electrons in any way.

A .pri file holds the primary electrons that are input to Nebula, with their
starting position, direction and energy. ``nebula_gpu`` and ``nebula_cpu_mt``
write the detected electrons to standard output (normally redirected to a .det
file), with the position, direction and energy at which they hit a detector.
The order of detected electrons is not deterministic.

``nebula_cpu_edep`` writes the detected electrons to the file given with
``--detect-filename``, and energy deposits to ``--deposit-filename`` (standard
output by default). These use the 28-byte `DEPOSIT_DTYPE` record, one per
scattering event:

- ``x``, ``y``, ``z``: position of the event in nm (float32);
- ``E``: kinetic energy of the electron before the event in eV (float32);
- ``dE``: energy lost in the event in eV (float32);
- ``px``, ``py``: pixel indices of the primary electron (int32).

Inside a material, Nebula measures kinetic energy from the bottom of the
conduction band instead of from the vacuum level: an electron gains the
material's ``barrier`` property when it enters the material and loses it when it
leaves. So ``E`` in a deposit is higher than the vacuum energy by the barrier;
a 500 eV primary electron entering silicon (barrier 12.44 eV) has ``E`` =
512.44 eV. Energies in .pri and .det files are measured in the same way, so
they are vacuum energies for electrons that start in, or are detected from,
vacuum, which is the usual case.
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
        raise ValueError(f"file size {n_bytes} is not a multiple of {dtype.itemsize} bytes")


def _check_pixels(pixels: np.ndarray, name: str) -> None:
    if pixels.dtype == np.int32:
        return
    if not (np.issubdtype(pixels.dtype, np.integer) or np.all(pixels == np.round(pixels))):
        raise ValueError(f"{name} must be integers")
    info = np.iinfo(np.int32)
    if np.any(pixels < info.min) or np.any(pixels > info.max):
        raise ValueError(f"{name} must fit in an int32")


# Records per chunk when checking values before writing
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
    # In chunks that stay in the CPU cache while each field is checked
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
    """Read a .pri or .det file.

    Args:
        file: Path of the file, or an open binary file.
        mmap: Memory-map the file instead of reading it, for files too large to
            fit in memory. The returned array is then read-only. Only for paths.

    Returns:
        A one-dimensional array with dtype `ELECTRON_DTYPE`.

    Raises:
        ValueError: If the file size is not a multiple of the record size.
        TypeError: If ``mmap`` is used with a file object.
    """
    return _read_records(file, ELECTRON_DTYPE, mmap)


def write_electrons(file: str | os.PathLike | BinaryIO, electrons: np.ndarray) -> None:
    """Write a .pri or .det file.

    To write a large file in parts, open it once and call this function for
    each part with the open file.

    Args:
        file: Path of the file, or an open binary file.
        electrons: Structured array with (at least) the fields of
            `ELECTRON_DTYPE`. Fields are converted to the types in the file;
            other fields are ignored.

    Raises:
        ValueError: If fields are missing, the array is not one-dimensional,
            a value is not finite, or a pixel index is not an integer in the
            int32 range.
    """
    _write_records(file, electrons, ELECTRON_DTYPE, "electrons")


def read_deposits(
    file: str | os.PathLike | BinaryIO, *, mmap: bool = False
) -> np.ndarray:
    """Read an energy deposit file written by ``nebula_cpu_edep``.

    Args:
        file: Path of the file, or an open binary file.
        mmap: Memory-map the file instead of reading it, for files too large to
            fit in memory. The returned array is then read-only. Only for paths.

    Returns:
        A one-dimensional array with dtype `DEPOSIT_DTYPE`.

    Raises:
        ValueError: If the file size is not a multiple of the record size.
        TypeError: If ``mmap`` is used with a file object.
    """
    return _read_records(file, DEPOSIT_DTYPE, mmap)


def write_deposits(file: str | os.PathLike | BinaryIO, deposits: np.ndarray) -> None:
    """Write an energy deposit file.

    To write a large file in parts, open it once and call this function for
    each part with the open file.

    Args:
        file: Path of the file, or an open binary file.
        deposits: Structured array with (at least) the fields of
            `DEPOSIT_DTYPE`. Fields are converted to the types in the file;
            other fields are ignored.

    Raises:
        ValueError: If fields are missing, the array is not one-dimensional,
            a value is not finite, or a pixel index is not an integer in the
            int32 range.
    """
    _write_records(file, deposits, DEPOSIT_DTYPE, "deposits")


def make_electrons(
    position: npt.ArrayLike,
    direction: npt.ArrayLike,
    energy: npt.ArrayLike,
    pixel: npt.ArrayLike,
) -> np.ndarray:
    """Make an array of electrons, such as primary electrons, from separate arrays.

    The arguments are broadcast against each other, so a value that is the same
    for all electrons can be given once.

    Args:
        position: Positions in nm, with shape (N, 3) or (3,).
        direction: Directions, with shape (N, 3) or (3,). Need not be normalised.
        energy: Kinetic energies in eV, with shape (N,) or a scalar.
        pixel: Pixel indices (px, py), with shape (N, 2) or (2,).

    Returns:
        A one-dimensional array with dtype `ELECTRON_DTYPE`.

    Raises:
        ValueError: If the shapes do not match, a value is not finite, a
            direction is zero, an energy is not positive, or a pixel index is
            not an integer in the int32 range.
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
        raise ValueError(f"energy must have shape (N,) or be a scalar, got {energy.shape}")
    try:
        (n,) = np.broadcast_shapes(
            position.shape[:-1], direction.shape[:-1], energy.shape, pixel.shape[:-1], (1,)
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
