"""Tests for nebula.io.materials."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from nebula.io import Axis, Material, Quantity, Table, read_mat, write_mat

SILICON_MAT = Path(__file__).parents[2] / "build" / "materials" / "silicon.mat"

NEBULA_TABLES = {
    "full_penn/imfp",
    "full_penn/omega_icdf",
    "full_penn/q_icdf",
    "ionization/binding_icdf",
    "kieft/elastic/imfp",
    "kieft/elastic/costheta_icdf",
}


@pytest.fixture
def material():
    rng = np.random.default_rng(0)
    energy = Axis("energy", np.geomspace(1, 1000, 8), "eV")
    loss_frac = Axis("loss_frac", np.geomspace(1e-4, 1, 5), "")
    return Material(
        properties={
            "name": "test",
            "barrier": Quantity(12.44, "eV"),
            "effective_A": Quantity(4.66e-23, "g"),
            "scale": 2.5,
        },
        tables={
            "full_penn/imfp": Table(rng.uniform(size=8), "nm^-1", [energy]),
            "full_penn/omega_icdf": Table(rng.uniform(size=(8, 6)), "eV", [energy, None]),
            "ionization/binding_icdf": Table(
                rng.uniform(size=(8, 5, 4)), "eV", [energy, loss_frac, None]
            ),
            "outer_shells": Table([1.0, 2.0, 3.0], "eV"),
        },
    )


def assert_tables_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        a, e = actual[key], expected[key]
        assert a.shape == e.shape, key
        assert a.unit == e.unit, key
        np.testing.assert_array_equal(a.data, e.data)
        assert len(a.axes) == len(e.axes)
        for axis_a, axis_e in zip(a.axes, e.axes):
            if axis_e is None:
                assert axis_a is None
            else:
                assert (axis_a.name, axis_a.unit) == (axis_e.name, axis_e.unit)
                np.testing.assert_array_equal(axis_a.values, axis_e.values)


def test_round_trip(tmp_path, material):
    path = tmp_path / "test.mat"
    write_mat(path, material)
    read = read_mat(path)
    assert read.properties == material.properties
    assert read.name == "test"
    assert_tables_equal(read.tables, material.tables)


def test_shared_axis_written_once(tmp_path, material):
    path = tmp_path / "test.mat"
    write_mat(path, material)
    with h5py.File(path, "r") as f:
        assert h5py.h5ds.is_scale(f["full_penn/energy"].id)
        assert h5py.h5ds.is_scale(f["ionization/loss_frac"].id)
        assert "full_penn/energy" not in read_mat(path).tables
        imfp = f["full_penn/imfp"]
        omega = f["full_penn/omega_icdf"]
        assert imfp.dims[0][0] == omega.dims[0][0] == f["full_penn/energy"]


def test_table_data_is_lazy(tmp_path, material):
    path = tmp_path / "test.mat"
    write_mat(path, material)
    table = read_mat(path).tables["ionization/binding_icdf"]
    assert table._data is None
    assert table.shape == (8, 5, 4)
    np.testing.assert_array_equal(
        table.data, material.tables["ionization/binding_icdf"].data
    )
    assert table._data is not None


def test_write_copies_unloaded_tables(tmp_path, material):
    write_mat(tmp_path / "a.mat", material)
    read = read_mat(tmp_path / "a.mat")
    read.properties["barrier"] = Quantity(13.0, "eV")
    write_mat(tmp_path / "b.mat", read)
    assert all(table._data is None for table in read.tables.values())
    copy = read_mat(tmp_path / "b.mat")
    assert copy.properties["barrier"] == Quantity(13.0, "eV")
    assert_tables_equal(copy.tables, material.tables)


def test_cannot_overwrite_unloaded_source(tmp_path, material):
    path = tmp_path / "test.mat"
    write_mat(path, material)
    read = read_mat(path)
    with pytest.raises(ValueError):
        write_mat(path, read)
    for table in read.tables.values():
        table.data  # load everything
    write_mat(path, read)
    assert_tables_equal(read_mat(path).tables, material.tables)


def test_compatible_with_cstool(tmp_path, material):
    from cstool.common import datafile, units

    # Read a file written by cstool
    path = tmp_path / "cstool.mat"
    energy = np.geomspace(1, 1000, 8)
    imfp = np.linspace(0, 1, 8)
    with datafile(path, "w") as f:
        f.set_property("name", "cstool")
        f.set_property("fermi", 7.83 * units.eV, "eV")
        f.set_property("count", 3)
        group = f.create_group("full_penn")
        group.add_scale("energy", energy * units.eV, "eV")
        group.add_dataset("imfp", imfp * units("nm^-1"), ("energy",), "nm^-1")
    with datafile(path, "r") as f:
        assert f.get_property("name") == "cstool"
        assert f.get_property("count") == 3
    read = read_mat(path)
    assert read.properties == {
        "name": "cstool",
        "fermi": Quantity(7.83, "eV"),
        "count": 3.0,
    }
    table = read.tables["full_penn/imfp"]
    assert table.unit == "nm^-1"
    np.testing.assert_array_equal(table.data, imfp)
    assert table.axes[0].name == "energy"
    np.testing.assert_array_equal(table.axes[0].values, energy)

    # Read a file written by write_mat with cstool, and check that the
    # attributes have the same HDF5 types as cstool's
    ours = tmp_path / "ours.mat"
    write_mat(ours, material)
    with h5py.File(path, "r") as theirs_f, h5py.File(ours, "r") as ours_f:
        for obj, name in [("/", "name"), ("/", "fermi"), ("full_penn/imfp", "units")]:
            theirs_type = theirs_f[obj].attrs.get_id(name).get_type()
            ours_name = "barrier" if name == "fermi" else name
            ours_type = ours_f[obj].attrs.get_id(ours_name).get_type()
            assert theirs_type == ours_type, (obj, name)
    with datafile(ours, "r") as f:
        assert f.get_property("name") == "test"
        assert f.get_property("barrier") == 12.44 * units.eV
        imfp = f.get_group("full_penn").get_dataset("imfp")
        assert imfp.units == units("nm^-1")
        np.testing.assert_array_equal(
            imfp.magnitude, material.tables["full_penn/imfp"].data
        )


@pytest.mark.parametrize(
    "tables",
    [
        {"t": Table(np.zeros(3), "", [Axis("e", np.arange(4))])},  # axis length
        {"t": Table(np.zeros((3, 3)), "", [None])},  # number of axes
        {  # same-named axes that differ
            "g/a": Table(np.zeros(3), "", [Axis("e", np.arange(3), "eV")]),
            "g/b": Table(np.zeros(3), "", [Axis("e", np.arange(3) + 1, "eV")]),
        },
        {"t": Table(np.zeros(3), "µm")},  # non-ASCII unit
        {"g/e": Table(np.zeros(3), "", [Axis("e", np.arange(3))])},  # name clash
    ],
)
def test_write_invalid_tables(tmp_path, tables):
    with pytest.raises(ValueError):
        write_mat(tmp_path / "bad.mat", Material(tables=tables))


@pytest.mark.parametrize("value", [[1.0], True, Quantity(1.0, "Å"), "naïve"])
def test_write_invalid_properties(tmp_path, value):
    with pytest.raises(ValueError):
        write_mat(tmp_path / "bad.mat", Material(properties={"p": value}))


@pytest.mark.skipif(not SILICON_MAT.exists(), reason="silicon.mat has not been built")
def test_read_cstool_material():
    silicon = read_mat(SILICON_MAT)
    assert silicon.name == "silicon"
    assert silicon.properties["barrier"].unit == "eV"
    assert silicon.properties["barrier"].value == pytest.approx(12.44)
    assert NEBULA_TABLES <= silicon.tables.keys()
    q_icdf = silicon.tables["full_penn/q_icdf"]
    assert q_icdf.shape == (128, 512, 512)
    assert q_icdf._data is None
    assert q_icdf.axes[0].name == "energy"
    assert q_icdf.axes[1:] == (None, None)
