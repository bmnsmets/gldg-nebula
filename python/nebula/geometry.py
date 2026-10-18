"""Building Nebula geometry, starting from height maps (z-maps).

A z-map is a two-dimensional float32 array ``zmap[ix, iy]`` of surface heights
in nm. The first index runs along x (to the right), the second along y
(upwards), the same convention as Nebula's coordinates and the ``px``/``py``
pixel indices. Seen from above (+z), a z-map therefore looks like an image
displayed with ``plt.imshow(zmap.T, origin="lower")``. A z-map has no lateral
scale of its own; the size of its pixels in nm is given when it is turned into
triangles.

`triangulate_zmap` turns a z-map into Nebula triangles. It treats the z-map as
samples of a smooth surface and uses as few triangles as possible while staying
within a given height tolerance of every sample: flat areas get a few large
triangles, slopes and edges many small ones. `plot_triangles` shows such
geometry in 3-D, coloured by material.
"""

import os
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
from matplotlib import image
from scipy.spatial import Delaunay

from .io.triangles import SpecialMaterial, make_triangles

__all__ = ["plot_triangles", "triangulate_zmap", "zmap_from_png"]

# ITU-R 601 luma, as used by Pillow to convert colour images to greyscale
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def zmap_from_png(
    file: str | os.PathLike | BinaryIO,
    height: float,
    base: float = 0.0,
) -> np.ndarray:
    """Read a PNG image as a z-map.

    The intensity of each pixel is mapped linearly to a height: black becomes
    ``base`` and white becomes ``base + height``, so
    ``z = base + I / I_max * height``, with ``I_max`` = 255 for 8-bit and
    65535 for 16-bit images. Colour images are converted to greyscale with
    ``0.299 R + 0.587 G + 0.114 B``; transparency is ignored. 16-bit colour
    images are read with 8-bit precision.

    The bottom-left pixel of the image becomes ``zmap[0, 0]``, so the z-map
    seen from above looks like the image.

    Args:
        file: Path of the PNG file, or an open binary file.
        height: Height of white pixels above black ones, in nm. May be
            negative, to make white pixels lower.
        base: Height of black pixels, in nm.

    Returns:
        A float32 array ``zmap[ix, iy]`` with shape (image width, image height).

    Raises:
        ValueError: If ``height`` or ``base`` is not finite, or the file is not
            a PNG image.
    """
    if not (np.isfinite(height) and np.isfinite(base)):
        raise ValueError("height and base must be finite")

    if isinstance(file, (str, os.PathLike)):
        file = os.fspath(file)
    # For PNG files, imread returns floats from 0 to 1, scaled by the bit depth
    try:
        pixels = image.imread(file, format="png")
    except SyntaxError as error:  # what Pillow raises for files that are not PNG
        raise ValueError(f"{file} is not a PNG image") from error
    if pixels.ndim == 3:
        intensity = pixels[..., :3] @ _LUMA_WEIGHTS
    else:
        intensity = pixels.astype(np.float64)

    z = (base + intensity * height).astype(np.float32)
    # Image rows run from the top down; z-map y runs from the bottom up
    return np.ascontiguousarray(z[::-1, :].T)


def _axis_samples(
    n: int, pixel_size: float, box: float, axis: str
) -> tuple[np.ndarray, int, bool]:
    """Sample coordinates along one axis of the z-map padded to the box size.

    Returns the coordinates, the number of zero pixels added on each side, and
    whether the axis is padded at all. Besides the pixel centres, there is one
    extra sample on each box edge: it holds the edge value of the z-map if the
    z-map fills the box, or is zero if the box is larger.
    """
    width = n * pixel_size
    padded = not np.isclose(box, width)
    if padded and box < width:
        raise ValueError(
            f"box size along {axis} ({box} nm) must be at least the z-map size ({width} nm)"
        )
    # Whole zero pixels that fit on each side; a narrower rest up to the box
    # edge is covered by the edge sample
    pad = int(np.floor((box - width) / (2 * pixel_size) + 1e-9)) if padded else 0
    centres = (np.arange(-pad, n + pad) - (n - 1) / 2) * pixel_size
    return np.concatenate([[-box / 2], centres, [box / 2]]), pad, padded


def _greedy_tin(points: np.ndarray, z: np.ndarray, start: np.ndarray, max_error: float):
    """Triangulate samples with as few vertices as possible (greedy insertion).

    Starting from the samples ``start``, repeatedly adds, in every triangle
    that has samples further than ``max_error`` (in nm) from the triangulated
    surface, the sample that is furthest away, until all samples are within
    ``max_error``.

    Returns:
        Indices of the vertices into ``points``, and the triangles as indices
        into the vertices.
    """
    vertices = np.asarray(start)
    while True:
        tri = Delaunay(points[vertices])
        simplex = tri.find_simplex(points)
        if np.any(simplex < 0):
            raise RuntimeError("sample outside the triangulation")
        transform = tri.transform[simplex]
        b = np.einsum("mij,mj->mi", transform[:, :2], points - transform[:, 2])
        weights = np.column_stack([b, 1 - b.sum(axis=1)])
        corners = vertices[tri.simplices[simplex]]
        error = np.abs((weights * z[corners]).sum(axis=1) - z)
        error[vertices] = 0

        bad = np.flatnonzero(error > max_error)
        if len(bad) == 0:
            return vertices, tri.simplices
        # The worst sample in each triangle that has samples too far away
        bad = bad[np.lexsort((-error[bad], simplex[bad]))]
        _, first = np.unique(simplex[bad], return_index=True)
        vertices = np.concatenate([vertices, bad[first]])


def _rectangle(x0: float, x1: float, y0: float, y1: float, z: float) -> list:
    """Two horizontal triangles covering a rectangle, counter-clockwise from above."""
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return [[a, b, c], [a, c, d]]


def triangulate_zmap(
    zmap: npt.ArrayLike,
    pixel_size: float = 1.0,
    *,
    tolerance: float = 0.01,
    box_size: float | tuple[float, float] | None = None,
    material: int = 0,
    terminator_z: float | None = -10_000.0,
    mirrors: bool = True,
    detector: int | None = SpecialMaterial.DETECTOR,
    detector_z: float | None = None,
) -> np.ndarray:
    """Turn a z-map into Nebula geometry.

    The surface separates ``material`` below from vacuum above. It is an
    adaptive triangulation that is within ``tolerance`` times the height range
    of the z-map (its maximum minus its minimum) of every z-map sample.
    The z-map's pixels are ``pixel_size`` wide, with the samples at their
    centres, so a z-map of N pixels is N * pixel_size wide.

    The geometry is centred on the origin: x runs from -Lx/2 to Lx/2 and y
    from -Ly/2 to Ly/2, with (Lx, Ly) the box size. If the box is the size of
    the z-map, the edge values of the z-map are held up to the box edges. If
    it is larger, the z-map is placed in the middle and padded with zeros to
    the box size before it is triangulated, so the surface goes from the edge
    values to height 0 over one pixel beyond the z-map. The padding is whole
    pixels on each side, plus a narrower strip at the box edges if the box is
    not a whole number of pixels larger. The number of samples, and so the
    time to triangulate, grows with the area of the box.

    Optionally, the box gets a terminator plane at the bottom,
    mirrors on the four sides and a detector plane at the top, as in Nebula's
    SEM tutorial.

    Nebula's simulation domain is the extent of all triangles plus 1 nm.
    Electrons that leave it are lost, so without a terminator plane the
    material ends at the lowest point of the surface, and electrons that pass
    through the detector plane are lost right after it. Primary electrons must
    start between the top of the surface and the detector plane; a
    `SpecialMaterial.DETECTOR` plane detects electrons crossing it in either
    direction.

    Args:
        zmap: Heights in nm, indexed ``zmap[ix, iy]`` (see the module docstring).
        pixel_size: Width of the z-map's pixels in nm.
        tolerance: Largest allowed height difference between the
            triangulated surface and any z-map sample, as a fraction of the
            z-map's height range: 0.01 means 1 % of max - min. For a z-map
            from an 8-bit PNG, a value below 1/255 makes the triangulation
            follow the steps between grey levels, which costs many triangles.
            A flat z-map is reproduced exactly.
        box_size: Size (Lx, Ly) of the box in nm, or one number for a square
            box. Each side must be at least the z-map size. By default, the
            box is the size of the z-map.
        material: Material id below the surface. Above it is vacuum.
        terminator_z: Height in nm of the terminator plane at the bottom of
            the box, or None for none. Must be below the surface. It should be deeper
            than electrons can travel into the material: about 2 to 10 times
            the electron range at the beam energy. In silicon the range is
            roughly 10 nm at 0.5 keV, 0.5 um at 5 keV and 9 um at 30 keV, so
            for example -100 nm suits 0.5 to 1 keV, -1000 nm up to 5 to
            8 keV, and the default -10000 nm up to about 30 keV. Too shallow
            removes electrons that would have backscattered; too deep only
            makes the simulation slower (about 30 % for -10000 nm instead
            of -100 nm at 0.5 keV).
        mirrors: Whether to put mirrors on the four sides of the box, from the
            terminator (or the lowest point of the surface) to the detector
            (or the highest point of the surface).
        detector: Material id of the detector plane at the top of the box,
            such as `SpecialMaterial.DETECTOR` or
            `SpecialMaterial.DETECTOR_LT50`, or None for none.
        detector_z: Height in nm of the detector plane. Must be above the
            surface. By default, 10 nm above its highest point.

    Returns:
        An array of triangles with dtype `TRIANGLE_DTYPE`, which can be
        written with `write_tri`.

    Raises:
        ValueError: If an argument is invalid, the box is smaller than the
            z-map, the terminator or detector plane
            would cut the surface, or the mirrors would have no height (a flat
            surface without terminator and detector).
    """
    zmap = np.asarray(zmap, dtype=np.float64)
    if zmap.ndim != 2 or zmap.size == 0:
        raise ValueError(f"zmap must be a non-empty 2-D array, got shape {zmap.shape}")
    if not np.all(np.isfinite(zmap)):
        raise ValueError("zmap must be finite")
    if not (np.isfinite(pixel_size) and pixel_size > 0):
        raise ValueError("pixel_size must be positive")
    if not (np.isfinite(tolerance) and tolerance > 0):
        raise ValueError("tolerance must be positive")

    nx, ny = zmap.shape
    if box_size is None:
        box_x, box_y = nx * pixel_size, ny * pixel_size
    else:
        box_x, box_y = np.broadcast_to(np.asarray(box_size, dtype=np.float64), (2,))
    x, pad_x, padded_x = _axis_samples(nx, pixel_size, box_x, "x")
    y, pad_y, padded_y = _axis_samples(ny, pixel_size, box_y, "y")

    # Samples: the z-map padded with zeros to the box, plus samples on the box
    # edges that are zero, or hold the edge values if the z-map fills the box
    z = np.pad(zmap, [(pad_x, pad_x), (pad_y, pad_y)])
    z = np.pad(z, [(1, 1), (0, 0)], mode="constant" if padded_x else "edge")
    z = np.pad(z, [(0, 0), (1, 1)], mode="constant" if padded_y else "edge")
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()])
    z = z.ravel()
    corners = np.ravel_multi_index(
        ([0, 0, -1 % len(x), -1 % len(x)], [0, -1 % len(y), 0, -1 % len(y)]), gx.shape
    )

    # The floor keeps rounding errors from counting as errors for flat z-maps
    max_error = max(tolerance * float(zmap.max() - zmap.min()), 1e-6)
    vertices, simplices = _greedy_tin(points, z, corners, max_error)
    surface = np.concatenate([points[vertices], z[vertices, None]], axis=1)[simplices]
    # Counter-clockwise seen from above, so that the normals point up
    edge1 = surface[:, 1, :2] - surface[:, 0, :2]
    edge2 = surface[:, 2, :2] - surface[:, 0, :2]
    area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    surface = surface[area != 0]
    surface[area[area != 0] < 0] = surface[area[area != 0] < 0][:, [0, 2, 1]]

    hx, hy = box_x / 2, box_y / 2
    triangles = [make_triangles(surface, [material, SpecialMaterial.VACUUM])]

    z_low, z_high = float(z.min()), float(z.max())
    top = z_high
    if detector is not None:
        top = z_high + 10 if detector_z is None else detector_z
        if top <= z_high:
            raise ValueError(
                f"detector_z must be above the surface (highest point {z_high} nm)"
            )
        triangles.append(
            make_triangles(_rectangle(-hx, hx, -hy, hy, top), [detector] * 2)
        )
    if terminator_z is not None:
        if terminator_z >= z_low:
            raise ValueError(
                f"terminator_z must be below the surface (lowest point {z_low} nm)"
            )
        terminator = _rectangle(-hx, hx, -hy, hy, terminator_z)
        triangles.append(make_triangles(terminator, [SpecialMaterial.TERMINATOR] * 2))
    if mirrors:
        bottom = terminator_z if terminator_z is not None else z_low
        if top <= bottom:
            raise ValueError(
                "mirrors need a terminator, a detector or a surface that is not flat"
            )
        walls = []
        for (xa, ya), (xb, yb) in [
            ((-hx, -hy), (hx, -hy)),
            ((hx, -hy), (hx, hy)),
            ((hx, hy), (-hx, hy)),
            ((-hx, hy), (-hx, -hy)),
        ]:
            a, b = (xa, ya, bottom), (xb, yb, bottom)
            c, d = (xb, yb, top), (xa, ya, top)
            walls += [[a, b, c], [a, c, d]]
        triangles.append(make_triangles(walls, [SpecialMaterial.MIRROR] * 2))
    return np.concatenate(triangles)


# Plot styles of special materials, drawn as outlines: label, colour
_SPECIAL_STYLES = {
    SpecialMaterial.TERMINATOR: ("terminator", "tab:red"),
    SpecialMaterial.MIRROR: ("mirrors", "tab:blue"),
    SpecialMaterial.DETECTOR: ("detector", "tab:green"),
    SpecialMaterial.DETECTOR_LT50: ("detector < 50 eV", "limegreen"),
    SpecialMaterial.DETECTOR_GE50: ("detector >= 50 eV", "darkgreen"),
    SpecialMaterial.NOP: ("no-op", "tab:gray"),
}
# Colours of material interfaces: those of tab10 not used for special materials
_INTERFACE_COLOURS = [
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]
# Horizontal planes, left out when outside the z range; other special
# triangles (vertical walls) are cut off at the z range instead
_PLANES = {
    SpecialMaterial.TERMINATOR,
    SpecialMaterial.DETECTOR,
    SpecialMaterial.DETECTOR_LT50,
    SpecialMaterial.DETECTOR_GE50,
}


def _triangle_groups(triangles: np.ndarray) -> np.ndarray:
    """Group of each triangle: its special material id, or 0 for an interface
    between materials (including vacuum)."""
    special = np.array(list(_SPECIAL_STYLES))
    m_in, m_out = triangles["material_in"], triangles["material_out"]
    return np.where(
        np.isin(m_out, special), m_out, np.where(np.isin(m_in, special), m_in, 0)
    )


def _shading(polys: np.ndarray) -> np.ndarray:
    """Brightness from 0.4 to 1 of triangles lit from the upper left, for plots."""
    normals = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[normals[:, 2] < 0] *= -1  # light both sides alike
    light = np.array([-1.0, 1.0, 1.0]) / np.sqrt(3)
    return 0.4 + 0.6 * np.clip(normals @ light, 0, 1)


def _subdivide(polys: np.ndarray, max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Split triangles into four until no edge is longer than max_edge.

    Returns the pieces and, for each piece, the index of its triangle.
    """
    source = np.arange(len(polys))
    while True:
        edges = np.linalg.norm(polys - np.roll(polys, 1, axis=1), axis=2).max(axis=1)
        split = edges > max_edge
        if not split.any():
            return polys, source
        a, b, c = polys[split, 0], polys[split, 1], polys[split, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        quarters = np.stack(
            [
                np.stack([a, ab, ca], axis=1),
                np.stack([ab, b, bc], axis=1),
                np.stack([ca, bc, c], axis=1),
                np.stack([ab, bc, ca], axis=1),
            ]
        ).reshape(-1, 3, 3)
        polys = np.concatenate([polys[~split], quarters])
        source = np.concatenate([source[~split], np.tile(source[split], 4)])


def _strips(segments: np.ndarray, max_length: float, half_width: float) -> np.ndarray:
    """Thin strips of triangles along line segments (N, 2, 3), for plots.

    Each segment is cut into pieces of at most max_length, and each piece
    becomes two crossed strips, so that it is visible from any direction.
    """
    start, end = segments[:, 0], segments[:, 1]
    count = np.maximum(
        np.ceil(np.linalg.norm(end - start, axis=1) / max_length), 1
    ).astype(int)
    index = np.repeat(np.arange(len(segments)), count)
    step = np.concatenate([np.arange(n) for n in count])
    fraction = np.stack([step, step + 1], axis=1) / count[index, None]
    a = start[index, None] + fraction[..., None] * (end - start)[index, None]
    p, q = a[:, 0], a[:, 1]

    direction = (q - p) / np.linalg.norm(q - p, axis=1, keepdims=True)
    across = np.cross(direction, [0.0, 0.0, 1.0])
    vertical = np.linalg.norm(across, axis=1) < 1e-6
    across[vertical] = np.cross(direction[vertical], [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    other = np.cross(direction, across)
    triangles = []
    for side in [across * half_width, other * half_width]:
        corners = [p + side, q + side, q - side, p - side]
        triangles += [
            np.stack(corners[:3], axis=1),
            np.stack([corners[0], corners[2], corners[3]], axis=1),
        ]
    return np.concatenate(triangles)


def _edges(polys: np.ndarray) -> np.ndarray:
    """The three edges of each triangle, (3 N, 2, 3)."""
    return np.stack([polys, np.roll(polys, -1, axis=1)], axis=2).reshape(-1, 2, 3)


def _unique_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first occurrence of each edge (in either direction), and
    for every edge the number of the distinct edge it is."""
    ends = np.round(edges, 6).reshape(-1, 6)
    first, second = ends[:, :3], ends[:, 3:]
    # Put the lexicographically smaller end first, so both directions match
    difference = second - first
    component = np.argmax(difference != 0, axis=1)
    smaller_first = difference[np.arange(len(ends)), component] >= 0
    keys = np.where(
        smaller_first[:, None], ends, np.concatenate([second, first], axis=1)
    )
    _, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return index, inverse.ravel()


def _outline(polys: np.ndarray) -> np.ndarray:
    """Edges of triangles, without those between triangles in the same plane.

    For planes and walls made of rectangles, this leaves the outlines of the
    rectangles, without their diagonals.
    """
    edges = _edges(polys)
    normals = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    normals = np.repeat(
        normals / np.linalg.norm(normals, axis=1, keepdims=True), 3, axis=0
    )
    index, inverse = _unique_edges(edges)
    keep = np.ones(len(index), dtype=bool)
    for edge in np.flatnonzero(np.bincount(inverse) == 2):
        a, b = np.flatnonzero(inverse == edge)
        if abs(normals[a] @ normals[b]) > 1 - 1e-9:
            keep[edge] = False
    return edges[index[keep]]


def plot_triangles(
    triangles: np.ndarray,
    ax=None,
    *,
    z_range: tuple[float, float] | None = None,
    z_exaggeration: float | None = None,
    material_names: dict[int, str] | None = None,
    edges: bool = False,
):
    """Plot Nebula geometry in 3-D, coloured by material.

    Interfaces between materials (including vacuum) are drawn as surfaces,
    with one colour per pair of materials, and shaded so that the relief of a
    surface is visible. Terminators, mirrors, detectors and no-op triangles are
    drawn as outlines, in fixed colours.

    By default, the plot shows the height range of the interfaces and
    detectors, plus half that range below it. Terminator and detector planes
    outside this range (such as the 10 um deep terminator of
    `triangulate_zmap`) are left out, also from the legend, and mirror walls
    are cut off at it, so that the surface stays visible. Pass ``z_range`` to
    show more.

    Matplotlib draws 3-D triangles one by one, in order of the depth of their
    centres, so this is slow for more than a few 10^4 triangles. To make
    surfaces and outlines hide each other correctly, large triangles are split
    into smaller pieces for drawing (the triangles themselves are not
    changed), and outlines are drawn as thin strips of triangles.

    Args:
        triangles: Array with dtype `TRIANGLE_DTYPE`, for example from
            `triangulate_zmap` or `read_tri`.
        ax: 3-D axes to plot into. By default, a new figure is made.
        z_range: Lowest and highest z to show, in nm.
        z_exaggeration: Factor by which heights are stretched. By default, the
            height of the plot is at least 30 % of its width; 1 is true scale.
        material_names: Names for material ids in the legend, such as
            ``{0: "silicon"}``. By default, the ids are shown.
        edges: Whether to draw the edges of the interface triangles, to show
            the mesh. They are drawn on top, so hidden edges show too.

    Returns:
        The 3-D axes.

    Raises:
        ValueError: If there are no triangles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    triangles = np.asarray(triangles)
    if len(triangles) == 0:
        raise ValueError("there are no triangles to plot")
    vertices = triangles["vertices"].astype(np.float64)
    groups = _triangle_groups(triangles)
    is_special = groups != 0

    if z_range is None:
        reference = ~np.isin(
            groups, [SpecialMaterial.TERMINATOR, SpecialMaterial.MIRROR]
        )
        z_all = vertices[reference][..., 2] if reference.any() else vertices[..., 2]
        z_lo, z_hi = float(z_all.min()), float(z_all.max())
        z_range = (z_lo - max(0.5 * (z_hi - z_lo), 1.0), max(z_hi, z_lo + 1.0))
    z_min, z_max = z_range

    def name(material: int) -> str:
        if material_names is not None and material in material_names:
            return material_names[material]
        return "vacuum" if material == SpecialMaterial.VACUUM else str(material)

    if ax is None:
        ax = plt.figure(figsize=(9, 7)).add_subplot(projection="3d")
    # Draw in the order the collections are added: surfaces, then lines on top
    ax.computed_zorder = False
    x, y = vertices[..., 0], vertices[..., 1]
    width = max(x.max() - x.min(), y.max() - y.min())
    if z_exaggeration is None:
        z_exaggeration = max(1.0, 0.3 * width / (z_max - z_min))
    scale = np.array([1.0, 1.0, z_exaggeration])  # from nm to plotted lengths
    handles = []

    # Matplotlib sorts faces by depth only within one collection, and only by
    # their centres. So everything that should hide or be hidden goes into one
    # collection: the interfaces, split into pieces of at most 1/16 of the
    # width, and the outlines of special materials as thin strips, cut into
    # pieces of the same length.
    polys, faces, seams = [], [], []

    # Interfaces between materials, one colour per pair, shaded
    interfaces = vertices[~is_special]
    if len(interfaces):
        pairs = np.sort(
            np.column_stack([triangles["material_in"], triangles["material_out"]])[
                ~is_special
            ],
            axis=1,
        )
        unique_pairs, first, pair_index = np.unique(
            pairs, axis=0, return_index=True, return_inverse=True
        )
        colours = np.empty((len(interfaces), 4))
        shade = _shading(interfaces * scale)
        for n, i in enumerate(np.argsort(first)):
            a, b = unique_pairs[i]
            # Vacuum (the lowest special id) last: "silicon | vacuum"
            label = f"{name(b)} | {name(a)}" if a < 0 else f"{name(a)} | {name(b)}"
            colour = _INTERFACE_COLOURS[n % len(_INTERFACE_COLOURS)]
            selected = pair_index.ravel() == i
            colours[selected] = to_rgba(colour)
            colours[selected, :3] *= shade[selected, None]
            handles.append(Patch(facecolor=colour, label=label))
        pieces, source = _subdivide(interfaces * scale, width / 16)
        polys.append(pieces)
        faces.append(colours[source])
        # Edges in the pieces' own colour hide the seams between them
        seams.append(np.full(len(pieces), 0.3))

    # Special materials, as outlines; planes outside the z range are left out
    for material, (label, colour) in _SPECIAL_STYLES.items():
        special = vertices[groups == material]
        if len(special) == 0:
            continue
        z = special[..., 2]
        if material in _PLANES and (z.min() < z_min or z.max() > z_max):
            continue
        outline = _outline(special)
        if z.min() < z_min or z.max() > z_max:
            label += " (cut off)"
            # Leave out edges that lie entirely beyond the z range
            beyond = np.all(outline[..., 2] < z_min, axis=1) | np.all(
                outline[..., 2] > z_max, axis=1
            )
            outline = outline[~beyond].copy()
            outline[..., 2] = np.clip(outline[..., 2], z_min, z_max)
        strips = _strips(outline * scale, width / 16, width / 800)
        polys.append(strips)
        faces.append(np.tile(to_rgba(colour), (len(strips), 1)))
        seams.append(np.full(len(strips), 0.6))
        handles.append(Line2D([], [], color=colour, label=label))

    if polys:
        faces = np.concatenate(faces)
        ax.add_collection3d(
            Poly3DCollection(
                np.concatenate(polys) / scale,
                facecolors=faces,
                edgecolors=faces,
                linewidths=np.concatenate(seams),
                axlim_clip=True,
            )
        )
    if edges and len(interfaces):
        # Drawn on top, as there are too many to draw as strips
        all_edges = _edges(interfaces)
        mesh = all_edges[_unique_edges(all_edges)[0]]
        ax.add_collection3d(
            Line3DCollection(
                mesh, colors=[to_rgba("k", 0.5)], linewidths=0.2, axlim_clip=True
            )
        )

    # Axes
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z_min, z_max)
    height = z_max - z_min
    ax.set_box_aspect((x.max() - x.min(), y.max() - y.min(), height * z_exaggeration))
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)" if z_exaggeration == 1 else f"z (nm), x{z_exaggeration:.3g}")
    ax.view_init(elev=25, azim=-60)
    ax.legend(handles=handles, loc="upper left", fontsize="small")
    return ax
