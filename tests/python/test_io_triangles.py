"""Tests for nebula.io.triangles."""

import io
from pathlib import Path

import numpy as np
import pytest
from nebula.io import (
    TRIANGLE_DTYPE,
    SpecialMaterial,
    make_triangles,
    read_tri,
    write_tri,
)

SEM_TRI = Path(__file__).parents[2] / "examples" / "tutorial-sem" / "sem.tri"


def random_triangles(n, seed=0):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(scale=100, size=(n, 3, 3))
    materials = rng.choice([0, 1, 2, *SpecialMaterial], size=(n, 2))
    return make_triangles(vertices, materials)


def test_read_tutorial_geometry():
    triangles = read_tri(SEM_TRI)
    assert triangles.dtype == TRIANGLE_DTYPE
    assert triangles.ndim == 1
    assert triangles["vertices"].shape == (len(triangles), 3, 3)
    assert triangles["vertices"].dtype == np.float32
    assert triangles[0]["material_in"] == 0
    assert triangles[0]["material_out"] == SpecialMaterial.VACUUM
    np.testing.assert_array_equal(
        triangles[0]["vertices"], [[-500, -500, 0], [-20, -500, 0], [-20, 500, 0]]
    )


def test_read_skips_comments_and_blank_lines():
    text = "# header\n\n   # indented comment\n1 -123 0 0 0 1 0 0 0 1 0\n\n"
    triangles = read_tri(io.StringIO(text))
    np.testing.assert_array_equal(triangles["material_in"], [1])
    np.testing.assert_array_equal(triangles["material_out"], [-123])
    np.testing.assert_array_equal(
        triangles["vertices"], [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]
    )


def test_read_rounds_to_float32():
    triangles = read_tri(io.StringIO("0 1 0.1 0 0 1 0 0 0 1 0\n"))
    assert triangles["vertices"][0, 0, 0] == np.float32(0.1)


def test_read_empty():
    triangles = read_tri(io.StringIO("# nothing\n"))
    assert triangles.shape == (0,)
    assert triangles.dtype == TRIANGLE_DTYPE


@pytest.mark.parametrize(
    "line",
    [
        "1 -123 0 0 0 1 0 0 0 1\n",  # 10 columns
        "1.5 -123 0 0 0 1 0 0 0 1 0\n",  # non-integer material
    ],
)
def test_read_invalid(line):
    with pytest.raises(ValueError):
        read_tri(io.StringIO(line))


def test_round_trip_exact(tmp_path):
    triangles = random_triangles(50)
    triangles[0]["vertices"] = [[0.1, -500, 0], [1e-9, 2.5e7, 0], [-0.0, 1 / 3, 7]]

    path = tmp_path / "geometry.tri"
    write_tri(path, triangles)
    np.testing.assert_array_equal(read_tri(path), triangles)

    buffer = io.StringIO()
    write_tri(buffer, triangles)
    buffer.seek(0)
    np.testing.assert_array_equal(read_tri(buffer), triangles)


def test_round_trip_exact_wide_range():
    rng = np.random.default_rng(1)
    n = 10_000
    vertices = rng.normal(size=(n, 3, 3)) * 10 ** rng.uniform(-6, 6, size=(n, 3, 3))
    triangles = make_triangles(vertices, [0, 1])
    buffer = io.StringIO()
    write_tri(buffer, triangles)
    buffer.seek(0)
    np.testing.assert_array_equal(read_tri(buffer), triangles)


def test_write_format():
    buffer = io.StringIO()
    triangles = make_triangles(
        [[[0, 0, 0], [0.1, 0, 0], [0, 1, 0]], [[1 / 3, 0, 0], [0, 0, 0], [0, 0, 0]]],
        [0, SpecialMaterial.VACUUM],
    )
    write_tri(buffer, triangles)
    assert buffer.getvalue().splitlines() == [
        "0 -123 0.0 0.0 0.0 0.1 0.0 0.0 0.0 1.0 0.0",
        "0 -123 0.33333334 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
    ]


def test_write_converts_fields():
    triangles = random_triangles(5)
    converted = np.empty(
        5,
        dtype=[
            ("vertices", "f8", (3, 3)),
            ("material_out", "f8"),
            ("material_in", "i8"),
            ("extra", "i8"),
        ],
    )
    for name in TRIANGLE_DTYPE.names:
        converted[name] = triangles[name]
    buffer = io.StringIO()
    write_tri(buffer, converted)
    buffer.seek(0)
    np.testing.assert_array_equal(read_tri(buffer), triangles)


FLOAT_TRIANGLE_DTYPE = np.dtype(
    [("material_in", "f8"), ("material_out", "f8"), ("vertices", "f8", (3, 3))]
)


@pytest.mark.parametrize(
    "triangles",
    [
        np.zeros(3, dtype=[("material_in", "i4"), ("material_out", "i4")]),  # missing
        np.zeros((2, 2), dtype=TRIANGLE_DTYPE),  # not one-dimensional
        np.array([(0.5, 1, np.zeros((3, 3)))], dtype=FLOAT_TRIANGLE_DTYPE),  # material
        np.array([(2**31, 1, np.zeros((3, 3)))], dtype=FLOAT_TRIANGLE_DTYPE),  # int32
        np.array([(0, 1, np.full((3, 3), 1e40))], dtype=FLOAT_TRIANGLE_DTYPE),  # float32
        np.zeros(
            1,
            dtype=[("material_in", "i4"), ("material_out", "i4"), ("vertices", "f8", 9)],
        ),  # vertices field with the wrong shape
    ],
)
def test_write_invalid(triangles):
    with pytest.raises(ValueError):
        write_tri(io.StringIO(), triangles)


def test_write_non_finite_vertices():
    triangles = make_triangles(np.zeros((1, 3, 3)), [0, 1])
    triangles["vertices"][0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        write_tri(io.StringIO(), triangles)


def test_make_triangles():
    vertices = np.arange(18.0).reshape(2, 3, 3)
    triangles = make_triangles(vertices.reshape(2, 9), [[0, 1], [2, 3]])
    assert triangles.dtype == TRIANGLE_DTYPE
    np.testing.assert_array_equal(triangles["vertices"], vertices)
    np.testing.assert_array_equal(triangles["material_in"], [0, 2])
    np.testing.assert_array_equal(triangles["material_out"], [1, 3])


def test_make_triangles_broadcasts_materials():
    triangles = make_triangles(np.zeros((3, 3, 3)), [1, SpecialMaterial.MIRROR])
    np.testing.assert_array_equal(triangles["material_in"], [1, 1, 1])
    np.testing.assert_array_equal(triangles["material_out"], [-122, -122, -122])


@pytest.mark.parametrize(
    "vertices, materials",
    [
        (np.zeros((2, 3, 2)), [0, 1]),  # bad vertices shape
        (np.zeros((2, 3, 3)), [[0, 1]] * 3),  # materials don't match vertices
        (np.full((1, 3, 3), np.nan), [0, 1]),  # non-finite coordinate
        (np.full((1, 3, 3), 1e40), [0, 1]),  # too large for float32
        (np.zeros((1, 3, 3)), [0.5, 1]),  # non-integer material
        (np.zeros((1, 3, 3)), [2**31, 1]),  # material out of int32 range
    ],
)
def test_make_triangles_invalid(vertices, materials):
    with pytest.raises(ValueError):
        make_triangles(vertices, materials)
