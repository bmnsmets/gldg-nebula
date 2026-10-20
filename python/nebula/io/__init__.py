"""Readers and writers for Nebula's file formats.

Depends only on numpy (and h5py for .mat files), not on the Nebula binaries.
"""

from .electrons import (
    DEPOSIT_DTYPE,
    ELECTRON_DTYPE,
    make_electrons,
    read_deposits,
    read_electrons,
    write_deposits,
    write_electrons,
)
from .materials import Axis, Material, Quantity, Table, read_mat, write_mat
from .triangles import (
    TRIANGLE_DTYPE,
    SpecialMaterial,
    make_triangles,
    read_tri,
    write_tri,
)

__all__ = [
    "DEPOSIT_DTYPE",
    "ELECTRON_DTYPE",
    "TRIANGLE_DTYPE",
    "Axis",
    "Material",
    "Quantity",
    "SpecialMaterial",
    "Table",
    "make_electrons",
    "make_triangles",
    "read_deposits",
    "read_electrons",
    "read_mat",
    "read_tri",
    "write_deposits",
    "write_electrons",
    "write_mat",
    "write_tri",
]
