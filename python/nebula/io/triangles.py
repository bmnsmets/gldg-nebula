"""The .tri geometry format: read and write triangles with their material ids.

A .tri file is a text file with one triangle per line, as 11 whitespace-separated
columns::

    material_in material_out x0 y0 z0 x1 y1 z1 x2 y2 z2

Coordinates are in nm. The normal ``(r1 - r0) x (r2 - r0)`` points from
``material_in`` to ``material_out``. Material ids 0, 1, ... are the .mat files
given to Nebula; negative ids are `SpecialMaterial`. Blank lines and lines
starting with ``#`` are ignored.

In numpy, triangles are an array with `TRIANGLE_DTYPE`: ``material_in`` and
``material_out`` (int32), and ``vertices`` (float32, shape (3, 3) as [vertex,
xyz], in nm). float32 is the precision Nebula simulates with.
"""

import enum
import os
import warnings
from typing import IO

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions

__all__ = [
    "TRIANGLE_DTYPE",
    "SpecialMaterial",
    "make_triangles",
    "read_tri",
    "write_tri",
]

TRIANGLE_DTYPE = np.dtype(
    [
        ("material_in", "<i4"),
        ("material_out", "<i4"),
        ("vertices", "<f4", (3, 3)),
    ]
)
"""One triangle of a .tri file."""

_N_COLUMNS = 11


class SpecialMaterial(enum.IntEnum):
    """Negative material ids with a special meaning, as in Nebula's ``special_materials``."""

    NOP = -128  #: No-op
    TERMINATOR = -127  #: Destroy the electron
    DETECTOR = -126  #: Flag as detected
    DETECTOR_LT50 = -125  #: Flag as detected, if energy < 50 eV
    DETECTOR_GE50 = -124  #: Flag as detected, if energy >= 50 eV
    VACUUM = -123  #: Go to vacuum propagation
    MIRROR = -122  #: Specularly reflect


def _check_material_ids(materials: np.ndarray, name: str) -> None:
    if not (
        np.issubdtype(materials.dtype, np.integer)
        or np.all(materials == np.round(materials))
    ):
        raise ValueError(f"{name} must be integers")
    info = np.iinfo(np.int32)
    if np.any(materials < info.min) or np.any(materials > info.max):
        raise ValueError(f"{name} must fit in an int32")


def read_tri(file: str | os.PathLike | IO[str]) -> np.ndarray:
    """Read a .tri file (a path or an open text file) as an array with `TRIANGLE_DTYPE`.

    Raises:
        ValueError: If a line does not have 11 columns, or a material id is not
            an integer.
    """
    with warnings.catch_warnings():
        # loadtxt warns about files without triangles; those are valid here
        warnings.filterwarnings("ignore", message=".*input contained no data")
        data = np.loadtxt(file, comments="#", ndmin=2, dtype=np.float64)
    if data.size == 0:
        return np.empty(0, dtype=TRIANGLE_DTYPE)
    if data.shape[1] != _N_COLUMNS:
        raise ValueError(
            f"expected {_N_COLUMNS} columns per triangle, got {data.shape[1]}"
        )
    _check_material_ids(data[:, :2], "material ids")

    triangles = np.empty(len(data), dtype=TRIANGLE_DTYPE)
    triangles["material_in"] = data[:, 0]
    triangles["material_out"] = data[:, 1]
    # Rounds to float32 like Nebula does; values too large become inf there too
    with np.errstate(over="ignore"):
        triangles["vertices"] = data[:, 2:].reshape(-1, 3, 3)
    return triangles


def write_tri(file: str | os.PathLike | IO[str], triangles: np.ndarray) -> None:
    """Write a .tri file; `read_tri` gives back exactly the same float32 values.

    Args:
        file: Path of the file, or an open text file.
        triangles: Array with (at least) the fields of `TRIANGLE_DTYPE`; other
            fields are ignored.

    Raises:
        ValueError: If a field is missing, a coordinate is not finite, or a
            material id does not fit in an int32.
    """
    triangles = np.asarray(triangles)
    names = triangles.dtype.names or ()
    missing = [name for name in TRIANGLE_DTYPE.names or () if name not in names]
    if missing:
        raise ValueError(f"triangles are missing the fields {', '.join(missing)}")
    if triangles.ndim > 1:
        raise ValueError(
            f"triangles must be one-dimensional, got shape {triangles.shape}"
        )
    triangles = triangles.reshape(-1)
    if triangles["vertices"].shape[1:] != (3, 3):
        raise ValueError(
            f"field vertices must have shape (3, 3), got {triangles['vertices'].shape[1:]}"
        )
    _check_material_ids(triangles["material_in"], "field material_in")
    _check_material_ids(triangles["material_out"], "field material_out")
    # Values too large for float32 become inf, which the check below reports
    with np.errstate(over="ignore"):
        triangles = recfunctions.require_fields(triangles, TRIANGLE_DTYPE)
    vertices = triangles["vertices"].reshape(-1, 9)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("field vertices must be finite (in float32)")

    # str of a numpy float32 is its shortest form that reads back exactly
    lines = (
        f"{m_in} {m_out} " + " ".join(map(str, coords)) + "\n"
        for m_in, m_out, coords in zip(
            triangles["material_in"].tolist(),
            triangles["material_out"].tolist(),
            vertices,
        )
    )
    if isinstance(file, (str, os.PathLike)):
        with open(file, "w") as f:
            f.writelines(lines)
    else:
        file.writelines(lines)


def make_triangles(vertices: npt.ArrayLike, materials: npt.ArrayLike) -> np.ndarray:
    """Make an array of triangles (`TRIANGLE_DTYPE`) from separate arrays.

    Args:
        vertices: Vertices in nm, shape (N, 3, 3) or (N, 9).
        materials: (material_in, material_out) per triangle, shape (N, 2), or
            (2,) for the same pair for all.

    Raises:
        ValueError: If a shape is wrong or a value is invalid.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim not in (2, 3) or vertices.shape[1:] not in ((3, 3), (9,)):
        raise ValueError(
            f"vertices must have shape (N, 3, 3) or (N, 9), got {vertices.shape}"
        )
    vertices = vertices.reshape(-1, 3, 3)

    materials = np.asarray(materials)
    try:
        materials = np.broadcast_to(materials, (len(vertices), 2))
    except ValueError:
        raise ValueError(
            f"materials must have shape (2,) or ({len(vertices)}, 2), got {materials.shape}"
        ) from None
    _check_material_ids(materials, "material ids")

    triangles = np.empty(len(vertices), dtype=TRIANGLE_DTYPE)
    triangles["material_in"] = materials[:, 0]
    triangles["material_out"] = materials[:, 1]
    # Check after rounding to float32, so that values too large for it are caught
    with np.errstate(over="ignore"):
        triangles["vertices"] = vertices
    if not np.all(np.isfinite(triangles["vertices"])):
        raise ValueError("vertices must be finite (in float32)")
    return triangles
