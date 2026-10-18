"""Tests for nebula.geometry."""

import io
from pathlib import Path

import numpy as np
import pytest
from nebula.geometry import (
    height_map,
    plot_triangles,
    triangulate_zmap,
    zmap_from_png,
)
from nebula.io import (
    TRIANGLE_DTYPE,
    SpecialMaterial,
    make_triangles,
    read_tri,
    write_tri,
)
from PIL import Image

SEM_TRI = Path(__file__).parents[2] / "examples" / "tutorial-sem" / "sem.tri"


def save_png(path, array, mode=None):
    image = Image.fromarray(np.asarray(array))
    if mode is not None and image.mode != mode:
        image = image.convert(mode)
    image.save(path, format="PNG")
    return path


def test_orientation(tmp_path):
    # 3 wide, 2 high; white pixels at the top-left and bottom-right
    pixels = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    zmap = zmap_from_png(save_png(tmp_path / "a.png", pixels, "L"), height=30)
    assert zmap.shape == (3, 2)
    expected = np.zeros((3, 2))
    expected[0, 1] = 30  # top-left: x = 0, y = top
    expected[2, 0] = 30  # bottom-right: x = right, y = 0
    np.testing.assert_array_equal(zmap, expected)
    assert zmap.flags.c_contiguous


def test_linear_mapping(tmp_path):
    pixels = np.array([[0, 51, 255]], dtype=np.uint8)
    zmap = zmap_from_png(save_png(tmp_path / "a.png", pixels, "L"), height=30, base=5)
    assert zmap.dtype == np.float32
    np.testing.assert_allclose(zmap[:, 0], 5 + np.array([0, 0.2, 1]) * 30, rtol=1e-6)


def test_16_bit(tmp_path):
    pixels = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = save_png(tmp_path / "a.png", pixels)
    assert Image.open(path).mode == "I;16"
    zmap = zmap_from_png(path, height=1)
    np.testing.assert_allclose(zmap[:, 0], [0, 32768 / 65535, 1], rtol=1e-7)
    assert zmap[1, 0] != np.float32(128 / 255)  # not reduced to 8 bits


def test_colour(tmp_path):
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
    zmap = zmap_from_png(save_png(tmp_path / "rgb.png", rgb, "RGB"), height=10)
    np.testing.assert_allclose(zmap[:, 0], [0, 10, 2.99], rtol=1e-6)

    # Transparency is ignored
    rgba = np.array([[[255, 255, 255, 0], [0, 0, 0, 255]]], dtype=np.uint8)
    zmap = zmap_from_png(save_png(tmp_path / "rgba.png", rgba, "RGBA"), height=10)
    np.testing.assert_allclose(zmap[:, 0], [10, 0])

    # Palette image
    palette = Image.fromarray(rgb).convert("P")
    palette.save(tmp_path / "p.png", format="PNG")
    zmap = zmap_from_png(tmp_path / "p.png", height=10)
    np.testing.assert_allclose(zmap[:2, 0], [0, 10])


def test_negative_height(tmp_path):
    pixels = np.array([[0, 255]], dtype=np.uint8)
    zmap = zmap_from_png(save_png(tmp_path / "a.png", pixels, "L"), height=-20, base=10)
    np.testing.assert_array_equal(zmap[:, 0], [10, -10])


def test_file_object():
    buffer = io.BytesIO()
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    np.testing.assert_array_equal(zmap_from_png(buffer, height=1)[:, 0], [0, 1])


@pytest.mark.parametrize("height, base", [(np.nan, 0), (1, np.inf)])
def test_invalid_numbers(tmp_path, height, base):
    path = save_png(tmp_path / "a.png", np.zeros((1, 1), dtype=np.uint8), "L")
    with pytest.raises(ValueError):
        zmap_from_png(path, height=height, base=base)


def test_not_a_png(tmp_path):
    path = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(path, format="JPEG")
    with pytest.raises(ValueError):
        zmap_from_png(path, height=1)


# Triangulation


def surface_of(triangles, material=0):
    return triangles[triangles["material_in"] == material]


def mesh_height(surface, xy):
    """Height of the triangulated surface at points xy (N, 2)."""
    v = surface["vertices"].astype(np.float64)
    a, b, c = v[:, 0, :2], v[:, 1, :2], v[:, 2, :2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    heights = np.full(len(xy), np.nan)
    for start in range(0, len(xy), 2000):
        p = xy[start : start + 2000, None, :]
        l1 = (
            (b[:, 0] - p[..., 0]) * (c[:, 1] - p[..., 1])
            - (b[:, 1] - p[..., 1]) * (c[:, 0] - p[..., 0])
        ) / det
        l2 = (
            (c[:, 0] - p[..., 0]) * (a[:, 1] - p[..., 1])
            - (c[:, 1] - p[..., 1]) * (a[:, 0] - p[..., 0])
        ) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        k = np.argmax(inside, axis=1)
        rows = np.arange(len(k))
        z = (
            l1[rows, k] * v[k, 0, 2]
            + l2[rows, k] * v[k, 1, 2]
            + l3[rows, k] * v[k, 2, 2]
        )
        heights[start : start + 2000] = np.where(inside.any(axis=1), z, np.nan)
    return heights


def coverage(surface, xy):
    """Number of surface triangles containing each point, seen from above."""
    v = surface["vertices"].astype(np.float64)
    a, b, c = v[:, 0, :2], v[:, 1, :2], v[:, 2, :2]

    def side(p, q, r):
        return (q[:, 0] - p[..., 0]) * (r[:, 1] - p[..., 1]) - (q[:, 1] - p[..., 1]) * (
            r[:, 0] - p[..., 0]
        )

    p = xy[:, None, :]
    inside = (side(p, a, b) > 0) & (side(p, b, c) > 0) & (side(p, c, a) > 0)
    return inside.sum(axis=1)


def sample_positions(nx, ny, pixel_size=1.0):
    x = (np.arange(nx) - (nx - 1) / 2) * pixel_size
    y = (np.arange(ny) - (ny - 1) / 2) * pixel_size
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def gaussian_bump(n=200, height=30.0, width=15.0):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return height * np.exp(-((x - n / 2) ** 2 + (y - 0.4 * n) ** 2) / (2 * width**2))


def test_flat_zmap():
    triangles = triangulate_zmap(
        np.full((10, 20), 5.0), terminator_z=None, mirrors=False, detector=None
    )
    assert len(triangles) == 2
    assert np.all(triangles["vertices"][..., 2] == 5)
    np.testing.assert_array_equal(np.unique(triangles["vertices"][..., 0]), [-5, 5])
    np.testing.assert_array_equal(np.unique(triangles["vertices"][..., 1]), [-10, 10])


def test_plane_needs_few_triangles():
    x, y = np.meshgrid(np.arange(50), np.arange(40), indexing="ij")
    zmap = 0.3 * x - 0.2 * y + 10
    triangles = triangulate_zmap(zmap, terminator_z=None, mirrors=False, detector=None)
    # Corners plus some samples next to them: the edge values are held for half
    # a pixel up to the box edges, so the surface is not exactly the plane
    assert len(triangles) <= 18
    heights = mesh_height(triangles, sample_positions(*zmap.shape))
    max_error = 0.01 * (zmap.max() - zmap.min())
    assert np.max(np.abs(heights - zmap.ravel())) <= max_error + 1e-4


def test_pixel_size():
    zmap = gaussian_bump(40, height=10, width=5)
    triangles = triangulate_zmap(zmap, pixel_size=2.5, box_size=200, mirrors=False)
    surface = surface_of(triangles)
    # The box, and the z-map's 100 nm in the middle of it
    np.testing.assert_array_equal(
        [surface["vertices"][..., 0].min(), surface["vertices"][..., 0].max()],
        [-100, 100],
    )
    max_error = 0.01 * (zmap.max() - zmap.min()) + 1e-4
    heights = mesh_height(surface, sample_positions(40, 40, pixel_size=2.5))
    np.testing.assert_allclose(heights, zmap.ravel(), atol=max_error)
    # Padded with zeros from one pixel beyond the last pixel centre (48.75
    # nm); the padding is triangulated with the rest, so also within tolerance
    np.testing.assert_allclose(
        mesh_height(surface, np.array([[51.25, 0], [90, 0]])), 0, atol=max_error
    )


@pytest.mark.parametrize("height, tolerance", [(30, 0.01), (30, 0.002), (3, 0.01)])
def test_smooth_bump_within_tolerance(height, tolerance):
    zmap = gaussian_bump(height=height) - 5  # range = height (the bump starts near 0)
    max_error = tolerance * (zmap.max() - zmap.min())
    triangles = triangulate_zmap(zmap, tolerance=tolerance)
    surface = surface_of(triangles)
    heights = mesh_height(surface, sample_positions(*zmap.shape))
    assert not np.any(np.isnan(heights))
    # Allow for rounding the vertices to float32
    assert np.max(np.abs(heights - zmap.ravel())) <= max_error + 1e-4
    assert len(surface) < 0.05 * 2 * zmap.size


def test_tolerance_is_relative():
    # Scaling the heights scales the allowed error, so the mesh stays the same
    zmap = gaussian_bump()
    low = triangulate_zmap(zmap, mirrors=False, detector=None, terminator_z=None)
    high = triangulate_zmap(10 * zmap, mirrors=False, detector=None, terminator_z=None)
    assert len(low) == len(high)
    np.testing.assert_array_equal(low["vertices"][..., :2], high["vertices"][..., :2])


def test_no_cracks_or_overlaps():
    zmap = gaussian_bump(100)
    for box in [None, 300, (300, 100)]:
        triangles = triangulate_zmap(
            zmap, box_size=box, detector=None, terminator_z=None, mirrors=False
        )
        lx, ly = (100, 100) if box is None else np.broadcast_to(box, (2,))
        rng = np.random.default_rng(0)
        xy = rng.uniform([-lx / 2, -ly / 2], [lx / 2, ly / 2], size=(5000, 2))
        np.testing.assert_array_equal(coverage(triangles, xy), 1)
        v = triangles["vertices"].astype(np.float64)
        area = 0.5 * np.abs(
            (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
            - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])
        )
        assert area.sum() == pytest.approx(lx * ly)


@pytest.mark.parametrize("box", [None, 300, (300, 100)])
def test_edges_shared(box):
    # The padded z-map is triangulated as a whole, so also with padding every
    # edge inside the box is shared by exactly two triangles
    triangles = triangulate_zmap(
        gaussian_bump(100),
        box_size=box,
        detector=None,
        terminator_z=None,
        mirrors=False,
    )
    hx, hy = (50, 50) if box is None else np.broadcast_to(box, (2,)) / 2
    edges = {}
    for tri in triangles["vertices"]:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            key = tuple(sorted([tuple(tri[i]), tuple(tri[j])]))
            edges[key] = edges.get(key, 0) + 1
    for (p, q), count in edges.items():
        on_boundary = (p[0] == q[0] and abs(p[0]) == hx) or (
            p[1] == q[1] and abs(p[1]) == hy
        )
        assert count == (1 if on_boundary else 2)


def test_padding_thinner_than_a_pixel():
    zmap = np.full((10, 10), 7.0)
    triangles = triangulate_zmap(
        zmap, box_size=10.5, detector=None, terminator_z=None, mirrors=False
    )
    # From the last pixel centre (4.5 nm) to 0 at the box edge (5.25 nm)
    points = np.array([[0, 0], [4.5, 0], [4.875, 0], [5.25, 0]])
    np.testing.assert_allclose(
        mesh_height(triangles, points), [7, 7, 3.5, 0], atol=1e-5
    )


def test_orientation_and_materials():
    triangles = triangulate_zmap(gaussian_bump(100), material=3, box_size=200)
    surface = surface_of(triangles, 3)
    assert np.all(surface["material_out"] == SpecialMaterial.VACUUM)
    v = surface["vertices"].astype(np.float64)
    normal_z = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (
        v[:, 1, 1] - v[:, 0, 1]
    ) * (v[:, 2, 0] - v[:, 0, 0])
    assert np.all(normal_z > 0)


def test_padding_places_zmap_in_middle():
    zmap = np.zeros((100, 100))
    zmap[80, 10] = 20  # a single raised pixel
    triangles = triangulate_zmap(zmap, box_size=500, tolerance=0.01)
    surface = surface_of(triangles)
    v = surface["vertices"]
    np.testing.assert_array_equal([v[..., 0].min(), v[..., 0].max()], [-250, 250])
    np.testing.assert_array_equal([v[..., 1].min(), v[..., 1].max()], [-250, 250])
    heights = mesh_height(surface, np.array([[30.5, -39.5], [200, 200], [-240, 0]]))
    np.testing.assert_allclose(heights, [20, 0, 0], atol=1e-5)


def test_padding_one_axis():
    zmap = np.full((100, 50), 7.0)
    triangles = triangulate_zmap(
        zmap, box_size=(300, 50), detector=None, terminator_z=None, mirrors=False
    )
    # From the last pixel centre (x = 49.5) to the zero sample one pixel further out
    points = np.array([[0, 0], [49.5, 0], [50, 0], [50.5, 0], [100, 0], [-100, 20]])
    np.testing.assert_allclose(
        mesh_height(triangles, points), [7, 7, 3.5, 0, 0, 0], atol=1e-5
    )


def test_default_planes():
    zmap = gaussian_bump(50, height=20)
    triangles = triangulate_zmap(zmap, box_size=100)
    detector = triangles[triangles["material_in"] == SpecialMaterial.DETECTOR]
    terminator = triangles[triangles["material_in"] == SpecialMaterial.TERMINATOR]
    mirrors = triangles[triangles["material_in"] == SpecialMaterial.MIRROR]
    assert len(detector) == 2 and len(terminator) == 2 and len(mirrors) == 8
    np.testing.assert_allclose(detector["vertices"][..., 2], zmap.max() + 10, rtol=1e-6)
    assert np.all(terminator["vertices"][..., 2] == -10_000)
    np.testing.assert_allclose(
        [mirrors["vertices"][..., 2].min(), mirrors["vertices"][..., 2].max()],
        [-10_000, zmap.max() + 10],
        rtol=1e-6,
    )
    assert set(np.abs(mirrors["vertices"][..., :2]).max(axis=(1, 2))) == {50}


def test_planes_optional():
    zmap = gaussian_bump(50, height=20)
    triangles = triangulate_zmap(
        zmap, detector=SpecialMaterial.DETECTOR_LT50, detector_z=40, terminator_z=None
    )
    detector = triangles[triangles["material_in"] == SpecialMaterial.DETECTOR_LT50]
    assert np.all(detector["vertices"][..., 2] == 40)
    assert not np.any(triangles["material_in"] == SpecialMaterial.TERMINATOR)
    mirrors = triangles[triangles["material_in"] == SpecialMaterial.MIRROR]
    # Without a terminator, the mirrors start at the lowest point of the surface
    assert mirrors["vertices"][..., 2].min() == np.float32(zmap.min())
    bare = triangulate_zmap(zmap, detector=None, terminator_z=None, mirrors=False)
    assert set(np.unique(bare["material_in"])) == {0}


@pytest.mark.parametrize(
    "zmap, kwargs",
    [
        (np.zeros((10, 10)), {"box_size": 9}),  # box smaller than the z-map
        (np.zeros((10, 10)), {"detector_z": 0}),  # detector cuts the surface
        (np.zeros((10, 10)), {"terminator_z": 0}),  # terminator cuts the surface
        (np.full((10, 10), np.nan), {}),
        (np.zeros(10), {}),
        (np.zeros((10, 10)), {"pixel_size": 0}),
        (np.zeros((10, 10)), {"tolerance": 0}),
        (
            np.zeros((10, 10)),
            {"terminator_z": None, "detector": None},
        ),  # zero-height mirrors
    ],
)
def test_triangulate_invalid(zmap, kwargs):
    with pytest.raises(ValueError):
        triangulate_zmap(zmap, **kwargs)


def test_triangulation_round_trip(tmp_path):
    triangles = triangulate_zmap(gaussian_bump(60), box_size=100)
    write_tri(tmp_path / "bump.tri", triangles)
    np.testing.assert_array_equal(read_tri(tmp_path / "bump.tri"), triangles)


def test_height_map():
    zmap = gaussian_bump(100)
    triangles = triangulate_zmap(zmap, box_size=200)
    centres = np.arange(100) - 49.5
    heights = height_map(triangles, centres, centres)
    assert heights.shape == (100, 100)
    # The same as interpolating the surface triangles, and within tolerance
    np.testing.assert_allclose(
        heights.ravel(),
        mesh_height(surface_of(triangles), sample_positions(100, 100)),
        atol=1e-4,
    )
    assert np.max(np.abs(heights - zmap)) <= 0.01 * np.ptp(zmap) + 1e-4
    # Padding is 0, outside the box is NaN; any order of coordinates
    grid = height_map(triangles, [150.0, 90.0, 0.0], [0.0])
    assert np.isnan(grid[0, 0])
    assert grid[1, 0] == pytest.approx(0, abs=0.01 * np.ptp(zmap))
    assert grid[2, 0] == pytest.approx(heights[49:51, 49:51].mean(), abs=1)


def test_height_map_ignores_special_triangles():
    # Mirrors, detector (15 nm) and terminator (-10 um) are not surfaces
    triangles = triangulate_zmap(np.full((10, 10), 5.0), box_size=20)
    np.testing.assert_allclose(height_map(triangles, [0, 9.9], [0])[:, 0], [5, 0])


def test_height_map_takes_highest():
    # The tutorial's PMMA line has vertical walls and lies on silicon: at
    # its walls and on top, the highest height (30 nm) counts
    heights = height_map(read_tri(SEM_TRI), [-100, -20, 0, 20, 100, 600], [0])
    np.testing.assert_array_equal(heights[:5, 0], [0, 30, 30, 30, 0])
    assert np.isnan(heights[5, 0])  # outside the 1 um box


def test_height_map_without_material():
    planes = read_tri(SEM_TRI)[-4:]  # mirrors and terminator only
    assert np.all(np.isnan(height_map(planes, [0, 10], [0, 10])))


# Plotting


@pytest.fixture
def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def drawn_colours(ax):
    """Face colours of the surfaces, which are drawn as one collection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    (collection,) = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    return collection.get_facecolor()


def lines(ax):
    """Lines drawn (the mesh edges), by colour: {colour: segments (N, 2, 3)}."""
    from matplotlib.colors import to_hex
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    return {
        to_hex(c.get_color()[0]): np.array(c._segments3d)
        for c in ax.collections
        if isinstance(c, Line3DCollection)
    }


def has_colour(ax, colour):
    """Whether faces of exactly this (unshaded) colour are drawn."""
    from matplotlib.colors import to_rgba

    return bool(np.any(np.all(np.isclose(drawn_colours(ax), to_rgba(colour)), axis=1)))


def pixels(ax, point, radius=0):
    """RGB colours of the rendered pixels around a 3-D point, (n, n, 3)."""
    from mpl_toolkits.mplot3d import proj3d

    ax.figure.canvas.draw()
    x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
    px, py = ax.transData.transform((x, y))
    image = np.asarray(ax.figure.canvas.buffer_rgba())[..., :3] / 255
    row, column = image.shape[0] - round(py), round(px)
    return image[row - radius : row + radius + 1, column - radius : column + radius + 1]


def test_plot_default_geometry(plt):
    from mpl_toolkits.mplot3d import Axes3D

    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    ax = plot_triangles(triangles, material_names={0: "silicon"})
    assert isinstance(ax, Axes3D)
    # The terminator 10 um down is neither drawn nor mentioned
    assert legend_labels(ax) == ["silicon | vacuum", "mirrors (cut off)", "detector"]
    assert has_colour(ax, "tab:blue") and has_colour(ax, "tab:green")
    assert not has_colour(ax, "tab:red")
    # Surface (0 to 20 nm) and detector (30 nm), plus half that range below
    np.testing.assert_allclose(ax.get_zlim(), (-15, 30), atol=1e-3)


def test_plot_z_range_shows_terminator(plt):
    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    ax = plot_triangles(triangles, z_range=(-10_000, 40))
    assert legend_labels(ax) == ["0 | vacuum", "terminator", "mirrors", "detector"]
    assert has_colour(ax, "tab:red")
    assert ax.get_zlim() == (-10_000, 40)


def test_plot_z_range_leaves_out_detector(plt):
    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    ax = plot_triangles(triangles, z_range=(-5, 25))  # detector at 30 nm
    assert legend_labels(ax) == ["0 | vacuum", "mirrors (cut off)"]
    assert not has_colour(ax, "tab:green")


def test_outline():
    from nebula.geometry import _outline

    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    groups = {
        m: triangles[triangles["material_in"] == m]["vertices"] for m in [-126, -122]
    }
    # Detector: the 4 sides of its plane, not the diagonal
    assert len(_outline(groups[-126].astype(float))) == 4
    # Mirrors: 4 vertical corners, 4 top and 4 bottom edges, no diagonals
    mirrors = _outline(groups[-122].astype(float))
    assert len(mirrors) == 12
    direction = mirrors[:, 1] - mirrors[:, 0]
    assert np.all(np.count_nonzero(direction, axis=1) == 1)  # all axis-aligned


def test_plot_outlines_hidden_by_surface(plt):
    def blue_pixels(point):
        # Blue clearly above red: the mirror outline, not the orange surface
        # or the grey background
        rgb = pixels(ax, point, radius=3)
        return int(np.sum(rgb[..., 2] - rgb[..., 0] > 0.1))

    ax = plot_triangles(read_tri(SEM_TRI))
    # The back-left mirror corner runs behind the surface below z = 0, the
    # front-right one in front of it
    assert blue_pixels((-500, 500, -10)) == 0
    assert blue_pixels((500, -500, -10)) > 0


def test_plot_into_axes_and_exaggeration(plt):
    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    ax = plt.figure().add_subplot(projection="3d")
    assert plot_triangles(triangles, ax, z_exaggeration=1) is ax
    assert ax.get_zlabel() == "z (nm)"
    # A 100 nm wide box with a 45 nm z range needs no exaggeration ...
    assert plot_triangles(triangles).get_zlabel() == "z (nm)"
    # ... but a 500 nm wide one does: 0.3 * 500 / 45
    wide = triangulate_zmap(gaussian_bump(50, height=20), box_size=500)
    assert plot_triangles(wide).get_zlabel() == "z (nm), x3.33"


def test_plot_materials(plt):
    a = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    b = [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    triangles = np.concatenate(
        [make_triangles([a], [0, 1]), make_triangles([b], [1, SpecialMaterial.VACUUM])]
    )
    ax = plot_triangles(triangles)
    assert legend_labels(ax) == ["0 | 1", "1 | vacuum"]
    # Both triangles are horizontal, so equally shaded: different colours
    assert len(np.unique(np.round(drawn_colours(ax), 6), axis=0)) == 2


def test_plot_occlusion_order(plt):
    # Matplotlib only sorts faces by depth within one collection, and by their
    # centres; the top of the tutorial's PMMA line must not be hidden by the
    # large silicon plane around it
    from matplotlib.colors import to_rgba

    ax = plot_triangles(read_tri(SEM_TRI), material_names={0: "silicon", 1: "pmma"})
    pixel = pixels(ax, (0, 0, 30))[0, 0]

    # Compare hues: the pixel's colour direction against PMMA (brown) and silicon (orange)
    def similarity(colour):
        c = np.array(to_rgba(colour)[:3])
        return pixel @ c / (np.linalg.norm(pixel) * np.linalg.norm(c))

    assert similarity("tab:brown") > similarity("tab:orange")


def test_plot_tutorial_geometry(plt):
    ax = plot_triangles(read_tri(SEM_TRI), material_names={0: "silicon", 1: "pmma"})
    assert legend_labels(ax) == [
        "silicon | vacuum",
        "silicon | pmma",
        "pmma | vacuum",
        "mirrors (cut off)",
        "detector < 50 eV",
    ]


def test_plot_nothing(plt):
    with pytest.raises(ValueError):
        plot_triangles(np.empty(0, dtype=TRIANGLE_DTYPE))


def test_plot_edges(plt):
    from matplotlib.colors import to_hex

    triangles = triangulate_zmap(gaussian_bump(50, height=20), box_size=100)
    surface = triangles[triangles["material_in"] == 0]
    assert to_hex("k") not in lines(plot_triangles(triangles))
    mesh = lines(plot_triangles(triangles, edges=True))[to_hex("k")]
    # Every edge once: Euler's formula for a triangulated rectangle (also with
    # padding, which is triangulated together with the z-map)
    vertices = len(np.unique(surface["vertices"].reshape(-1, 3), axis=0))
    assert len(mesh) == vertices + len(surface) - 1
