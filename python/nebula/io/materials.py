"""The .mat material format (HDF5): material properties and cross-section tables.

A .mat file is an HDF5 file, normally made by cstool, with:

- *properties*, stored as attributes of the root group. These are strings
  (``name``, ``conductor_type``, ...), numbers with a unit (``barrier``,
  ``fermi``, ``band_gap``, ``phonon_loss``, ``effective_A``), or plain numbers;
- *tables*, float64 datasets in groups such as ``full_penn`` or
  ``kieft/elastic``, each with a ``units`` attribute;
- *axes*: the coordinates along a table dimension (e.g. ``full_penn/energy``),
  stored next to its tables (as HDF5 "dimension scales") with a ``units``
  attribute. Tables can share an axis; a dimension can have none.

Unit strings such as ``"eV"`` or ``"nm^-1"`` are kept as they are, for Nebula to parse.

With Nebula's default physics, a material needs the properties ``barrier``,
``fermi``, ``phonon_loss`` and ``effective_A`` (``band_gap`` is optional), and
the tables ``full_penn/imfp``, ``full_penn/omega_icdf``, ``full_penn/q_icdf``,
``ionization/binding_icdf``, ``kieft/elastic/imfp`` and
``kieft/elastic/costheta_icdf``.

Material files are hundreds of MB, so `read_mat` reads a table only when it is used.
"""

import dataclasses
import os
from collections.abc import Iterable

import h5py
import numpy as np
import numpy.typing as npt

__all__ = ["Axis", "Material", "Quantity", "Table", "read_mat", "write_mat"]

_QUANTITY_DTYPE = np.dtype([("value", np.float64), ("unit", h5py.vlen_dtype(bytes))])


@dataclasses.dataclass(frozen=True)
class Quantity:
    """A number with a unit, such as a material property ``Quantity(12.44, "eV")``."""

    value: float
    unit: str


@dataclasses.dataclass(eq=False)
class Axis:
    """The coordinates along one dimension of a table."""

    name: str
    values: np.ndarray
    unit: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1:
            raise ValueError(f"axis {self.name} must be one-dimensional")


class Table:
    """A table of a material: its values, unit and an optional axis per dimension.

    A table from `read_mat` reads its values from the file on first use of `data`.
    """

    def __init__(
        self,
        data: npt.ArrayLike,
        unit: str = "",
        axes: Iterable[Axis | None] | None = None,
    ):
        """Make a table.

        Args:
            data: The table values.
            unit: Unit of the values.
            axes: One `Axis` or None per dimension of ``data``; by default None
                for all dimensions.
        """
        self._data: np.ndarray | None = np.asarray(data, dtype=np.float64)
        self._source: tuple[str, str] | None = None
        self._shape = self._data.shape
        self.unit = unit
        self.axes = tuple(axes) if axes is not None else (None,) * self._data.ndim

    @classmethod
    def _from_file(cls, path: str, name: str, shape, unit: str, axes) -> Table:
        table = cls.__new__(cls)
        table._data = None
        table._source = (path, name)
        table._shape = tuple(shape)
        table.unit = unit
        table.axes = tuple(axes)
        return table

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data, available without loading it."""
        return self._shape

    @property
    def data(self) -> np.ndarray:
        """The table values. Loaded from the file on first access."""
        if self._data is None:
            self._data = self._read()
        return self._data

    @data.setter
    def data(self, data: npt.ArrayLike) -> None:
        self._data = np.asarray(data, dtype=np.float64)
        self._shape = self._data.shape

    def _read(self) -> np.ndarray:
        """Return the data, reading it from the file without caching it."""
        if self._data is not None:
            return self._data
        assert self._source is not None
        path, name = self._source
        with h5py.File(path, "r") as f:
            return f[name][()]

    def __repr__(self) -> str:
        axes = ", ".join("None" if a is None else a.name for a in self.axes)
        return f"Table(shape={self.shape}, unit={self.unit!r}, axes=({axes}))"


@dataclasses.dataclass
class Material:
    """The contents of a .mat file.

    Attributes:
        properties: Material properties by name.
        tables: Tables by their path in the file, without leading slash, e.g.
            ``"full_penn/imfp"``.
    """

    properties: dict[str, Quantity | str | float] = dataclasses.field(
        default_factory=dict
    )
    tables: dict[str, Table] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        """The ``name`` property."""
        return str(self.properties["name"])


def _decode(value) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _read_property(value) -> Quantity | str | float:
    if isinstance(value, (bytes, str)):
        return _decode(value)
    value = np.asarray(value)
    if value.dtype.names == ("value", "unit"):
        return Quantity(float(value["value"]), _decode(value["unit"].item()))
    return float(value)


def read_mat(file: str | os.PathLike) -> Material:
    """Read a .mat file. Table values are read only when first used."""
    path = os.path.abspath(file)
    material = Material()
    with h5py.File(path, "r") as f:
        for key, value in f.attrs.items():
            material.properties[key] = _read_property(value)

        def visit(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset) or h5py.h5ds.is_scale(obj.id):
                return
            axes = []
            for dim in obj.dims:
                if len(dim) == 0:
                    axes.append(None)
                    continue
                scale = dim[0]
                scale_name = scale.attrs.get("NAME", scale.name.rsplit("/", 1)[-1])
                axes.append(
                    Axis(
                        _decode(scale_name),
                        scale[()],
                        _decode(scale.attrs.get("units", "")),
                    )
                )
            material.tables[name] = Table._from_file(
                path, name, obj.shape, _decode(obj.attrs.get("units", "")), axes
            )

        f.visititems(visit)
    return material


def _check_ascii(text: str, what: str) -> None:
    if not text.isascii():
        raise ValueError(f"{what} must be ASCII, got {text!r}")


def _group_of(key: str) -> tuple[str, str]:
    group, _, name = key.rpartition("/")
    return group, name


def _check_material(file: str | os.PathLike, material: Material) -> dict:
    """Check that the material can be written; return the axes per group."""
    for key, value in material.properties.items():
        _check_ascii(key, "property names")
        if isinstance(value, Quantity):
            _check_ascii(value.unit, f"unit of property {key}")
        elif isinstance(value, str):
            _check_ascii(value, f"property {key}")
        elif not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
            value, bool
        ):
            raise ValueError(f"property {key} must be a Quantity, str or number")  # noqa: TRY004

    group_axes: dict[str, dict[str, Axis]] = {}
    for key, table in material.tables.items():
        _check_ascii(key, "table names")
        _check_ascii(table.unit, f"unit of table {key}")
        if len(table.axes) != len(table.shape):
            raise ValueError(
                f"table {key} has {len(table.shape)} dimensions but {len(table.axes)} axes"
            )
        group, _ = _group_of(key)
        axes = group_axes.setdefault(group, {})
        for dim, axis in enumerate(table.axes):
            if axis is None:
                continue
            _check_ascii(axis.name, f"axis names of table {key}")
            _check_ascii(axis.unit, f"unit of axis {axis.name} of table {key}")
            if len(axis.values) != table.shape[dim]:
                raise ValueError(
                    f"axis {axis.name} of table {key} has {len(axis.values)} values, "
                    f"but dimension {dim} has size {table.shape[dim]}"
                )
            if axis.name in axes:
                other = axes[axis.name]
                if other.unit != axis.unit or not np.array_equal(
                    other.values, axis.values
                ):
                    raise ValueError(
                        f"group {group or '/'} has different axes named {axis.name}"
                    )
            axes[axis.name] = axis

    for key in material.tables:
        group, name = _group_of(key)
        if name in group_axes.get(group, {}):
            raise ValueError(f"table {key} has the same name as an axis in its group")

    if os.path.exists(file):
        for key, table in material.tables.items():
            if (
                table._data is None
                and table._source is not None
                and os.path.samefile(file, table._source[0])
            ):
                raise ValueError(
                    f"cannot write to {file}: table {key} has not been loaded from it yet"
                )
    return group_axes


def write_mat(file: str | os.PathLike, material: Material) -> None:
    """Write a .mat file in the layout cstool uses, overwriting ``file``.

    Tables not yet read are copied from their file one at a time.

    Raises:
        ValueError: If the axes do not match the tables, a text is not ASCII, a
            property has an unsupported type, or ``file`` is the file unread
            tables still come from.
    """
    group_axes = _check_material(file, material)

    with h5py.File(file, "w") as f:
        for key, value in material.properties.items():
            if isinstance(value, Quantity):
                f.attrs[key] = np.array(
                    (value.value, value.unit.encode("ascii")), dtype=_QUANTITY_DTYPE
                )
            elif isinstance(value, str):
                f.attrs[key] = value.encode("ascii")
            else:
                f.attrs[key] = float(value)

        scales = {}
        for group, axes in group_axes.items():
            parent = f.require_group(group) if group else f
            for name, axis in axes.items():
                scale = parent.create_dataset(name, data=axis.values)
                scale.attrs["units"] = axis.unit.encode("ascii")
                scale.make_scale(name)
                scales[group, name] = scale

        for key, table in material.tables.items():
            group, _ = _group_of(key)
            dataset = f.create_dataset(
                key, data=np.asarray(table._read(), dtype=np.float64)
            )
            dataset.attrs["units"] = table.unit.encode("ascii")
            for dim, axis in enumerate(table.axes):
                if axis is not None:
                    dataset.dims[dim].attach_scale(scales[group, axis.name])
