"""Building Nebula geometry from height maps (z-maps), and plotting it.

A z-map is a float32 array ``zmap[ix, iy]`` of surface heights in nm, with x to
the right and y up (show it with ``plt.imshow(zmap.T, origin="lower")``). Its
pixel size is given when it is turned into triangles.

`triangulate_zmap` makes triangles that follow the z-map within a height
tolerance: few large ones on flat areas, many small ones on slopes and edges.
`plot_triangles` shows geometry in 3-D.
"""

import os
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
from matplotlib import image
from scipy.spatial import Delaunay

from .io.triangles import SpecialMaterial, make_triangles

__all__ = ["height_map", "plot_triangles", "triangulate_zmap", "zmap_from_png"]

# Weights of red, green and blue in greyscale, as Pillow uses
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def zmap_from_png(
    file: str | os.PathLike | BinaryIO,
    height: float,
    base: float = 0.0,
) -> np.ndarray:
    """Read a PNG image as a z-map: black is ``base``, white ``base + height``.

    Grey levels in between are linear. Colour images are turned to greyscale
    (16-bit colour with 8-bit precision); transparency is ignored. The
    bottom-left pixel becomes ``zmap[0, 0]``, so seen from above the z-map
    looks like the image.

    Args:
        file: Path of the PNG file, or an open binary file.
        height: Height of white above black, in nm; negative makes white lower.
        base: Height of black, in nm.

    Returns:
        A float32 array ``zmap[ix, iy]`` of shape (image width, image height).

    Raises:
        ValueError: If ``height`` or ``base`` is not finite, or the file is not a PNG.
    """
    if not (np.isfinite(height) and np.isfinite(base)):
        raise ValueError("height and base must be finite")

    if isinstance(file, (str, os.PathLike)):
        file = os.fspath(file)
    # imread gives PNG pixels as 0 (black) to 1 (white), whatever the bit depth
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
    """Sample positions along one axis: the pixel centres, padded to the box.

    Returns the positions, the number of zero pixels added on each side, and
    whether the box is larger than the z-map. The first and last positions
    are the box edges.
    """
    width = n * pixel_size
    padded = not np.isclose(box, width)
    if padded and box < width:
        raise ValueError(
            f"box size along {axis} ({box} nm) must be at least the z-map size ({width} nm)"
        )
    # Whole zero pixels on each side; the edge sample covers any narrower rest
    pad = int(np.floor((box - width) / (2 * pixel_size) + 1e-9)) if padded else 0
    centres = (np.arange(-pad, n + pad) - (n - 1) / 2) * pixel_size
    return np.concatenate([[-box / 2], centres, [box / 2]]), pad, padded


def _greedy_tin(points: np.ndarray, z: np.ndarray, start: np.ndarray, max_error: float):
    """Triangulate samples using few of them as vertices.

    Starting from ``start``, each round adds, per triangle, the sample furthest
    from the surface, until all samples are within ``max_error`` (nm).

    Returns:
        The vertices as indices into ``points``, and the triangles as indices
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
        # The worst sample of each triangle that has one too far away
        bad = bad[np.lexsort((-error[bad], simplex[bad]))]
        _, first = np.unique(simplex[bad], return_index=True)
        vertices = np.concatenate([vertices, bad[first]])


def _rectangle(x0: float, x1: float, y0: float, y1: float, z: float) -> list:
    """Two horizontal triangles covering a rectangle, normals up."""
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
    """Turn a z-map into Nebula geometry: ``material`` below the surface, vacuum above.

    The geometry is centred on the origin. A box larger than the z-map is
    filled with height 0 around it; otherwise the z-map's edge heights extend
    to the box edges. Larger boxes take longer to triangulate. The box can get
    a terminator plane at the bottom, mirrors on its sides and a detector plane
    at the top, as in Nebula's SEM tutorial.

    Electrons that leave the box (plus 1 nm) are lost, so without a terminator
    the material ends at the lowest point of the surface. Primary electrons
    must start between the surface and the detector plane.

    Args:
        zmap: Heights in nm, ``zmap[ix, iy]``.
        pixel_size: Width of a z-map pixel in nm.
        tolerance: Largest height error allowed, as a fraction of the z-map's
            height range (0.01 is 1 %). For an 8-bit PNG, below 1/255 follows
            every grey-level step and costs many triangles.
        box_size: Box size (Lx, Ly) in nm, or one number for a square; at least
            the z-map size, which is the default.
        material: Material id below the surface.
        terminator_z: Height in nm of the terminator plane, or None. It must be
            deeper than electrons get; in silicon about -100 nm up to 1 keV,
            -1000 nm up to 5 keV and -10000 nm up to 30 keV. Deeper only costs time.
        mirrors: Whether to put mirrors on the four sides, from the terminator
            (or the surface's lowest point) to the detector (or its highest).
        detector: Material id of the detector plane, such as
            `SpecialMaterial.DETECTOR`, or None.
        detector_z: Height in nm of the detector plane; by default 10 nm above
            the surface.

    Returns:
        Triangles with dtype `TRIANGLE_DTYPE`.

    Raises:
        ValueError: If an argument is invalid, the box is smaller than the
            z-map, a plane would cut the surface, or the mirrors would have no
            height.
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

    # The z-map padded with zeros to the box, plus samples on the box edges:
    # zero, or the edge values if the z-map fills the box
    z = np.pad(zmap, [(pad_x, pad_x), (pad_y, pad_y)])
    z = np.pad(z, [(1, 1), (0, 0)], mode="constant" if padded_x else "edge")
    z = np.pad(z, [(0, 0), (1, 1)], mode="constant" if padded_y else "edge")
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()])
    z = z.ravel()
    corners = np.ravel_multi_index(
        ([0, 0, -1 % len(x), -1 % len(x)], [0, -1 % len(y), 0, -1 % len(y)]), gx.shape
    )

    # At least 1e-6 nm, so that rounding does not split flat z-maps
    max_error = max(tolerance * float(zmap.max() - zmap.min()), 1e-6)
    vertices, simplices = _greedy_tin(points, z, corners, max_error)
    surface = np.concatenate([points[vertices], z[vertices, None]], axis=1)[simplices]
    # Drop triangles with no area seen from above; turn the rest so normals point up
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


def height_map(
    triangles: np.ndarray, xs: npt.ArrayLike, ys: npt.ArrayLike
) -> np.ndarray:
    """Height of the top surface of a geometry at each grid point, seen from above.

    Only triangles with a material on a side count (not mirrors, detectors and
    the like). Where there are several heights, as under overhangs, the highest
    is taken.

    Args:
        triangles: Array with dtype `TRIANGLE_DTYPE`.
        xs, ys: Grid coordinates in nm.

    Returns:
        Heights in nm, float64 ``[ix, iy]``; NaN where no triangle covers the point.
    """
    triangles = np.asarray(triangles)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    heights = np.full((len(xs), len(ys)), np.nan)

    has_material = (triangles["material_in"] >= 0) | (triangles["material_out"] >= 0)
    polys = triangles["vertices"][has_material].astype(np.float64)
    a, b, c = polys[:, 0], polys[:, 1], polys[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    vertical = det == 0

    # On sorted coordinates, the grid points near a triangle form one block
    x_order, y_order = np.argsort(xs), np.argsort(ys)
    x_sorted, y_sorted = xs[x_order], ys[y_order]
    sorted_heights = np.full_like(heights, np.nan)
    margin = 1e-6  # nm, so that points on edges count
    low, high = polys[..., :2].min(axis=1), polys[..., :2].max(axis=1)
    i0 = np.searchsorted(x_sorted, low[:, 0] - margin, side="left")
    i1 = np.searchsorted(x_sorted, high[:, 0] + margin, side="right")
    j0 = np.searchsorted(y_sorted, low[:, 1] - margin, side="left")
    j1 = np.searchsorted(y_sorted, high[:, 1] + margin, side="right")
    for t in np.flatnonzero(~vertical & (i1 > i0) & (j1 > j0)):
        px = x_sorted[i0[t] : i1[t], None]
        py = y_sorted[None, j0[t] : j1[t]]
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = polys[t]
        weight_a = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / det[t]
        weight_b = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / det[t]
        weight_c = 1 - weight_a - weight_b
        inside = (weight_a >= -1e-9) & (weight_b >= -1e-9) & (weight_c >= -1e-9)
        z = np.where(inside, weight_a * az + weight_b * bz + weight_c * cz, np.nan)
        block = sorted_heights[i0[t] : i1[t], j0[t] : j1[t]]
        np.fmax(block, z, out=block)

    heights[np.ix_(x_order, y_order)] = sorted_heights
    return heights


# Label and colour of special materials, drawn as outlines
_SPECIAL_STYLES = {
    SpecialMaterial.TERMINATOR: ("terminator", "tab:red"),
    SpecialMaterial.MIRROR: ("mirrors", "tab:blue"),
    SpecialMaterial.DETECTOR: ("detector", "tab:green"),
    SpecialMaterial.DETECTOR_LT50: ("detector < 50 eV", "limegreen"),
    SpecialMaterial.DETECTOR_GE50: ("detector >= 50 eV", "darkgreen"),
    SpecialMaterial.NOP: ("no-op", "tab:gray"),
}
# Colours of material interfaces: the tab10 colours not used above
_INTERFACE_COLOURS = [
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]
# Horizontal planes: left out when outside the z range (walls are cut off instead)
_PLANES = {
    SpecialMaterial.TERMINATOR,
    SpecialMaterial.DETECTOR,
    SpecialMaterial.DETECTOR_LT50,
    SpecialMaterial.DETECTOR_GE50,
}


def _triangle_groups(triangles: np.ndarray) -> np.ndarray:
    """Per triangle: its special material id, or 0 for an interface between materials."""
    special = np.array(list(_SPECIAL_STYLES))
    m_in, m_out = triangles["material_in"], triangles["material_out"]
    return np.where(
        np.isin(m_out, special), m_out, np.where(np.isin(m_in, special), m_in, 0)
    )


def _shading(polys: np.ndarray) -> np.ndarray:
    """Brightness (0.4 to 1) of triangles lit from the upper left."""
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
    """Line segments (N, 2, 3) drawn as thin triangles, to plot lines among surfaces.

    Each piece of at most max_length becomes two crossed strips, visible from
    any direction.
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
    """Find edges that are the same in either direction.

    Returns the index of the first copy of each distinct edge, and for every
    edge the number of its distinct edge.
    """
    ends = np.round(edges, 6).reshape(-1, 6)
    first, second = ends[:, :3], ends[:, 3:]
    # Put the smaller end (by x, then y, then z) first, so both directions match
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

    For rectangles this leaves the outline, without the diagonal.
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

    Interfaces between materials are shaded surfaces, one colour per pair of
    materials. Terminators, mirrors and detectors are outlines. By default the
    height range is that of the interfaces and detectors, plus half of it
    below; planes outside it (like a deep terminator) are left out and walls
    cut off. Slow beyond a few 10^4 triangles.

    Args:
        triangles: Array with dtype `TRIANGLE_DTYPE`, for example from
            `triangulate_zmap` or `read_tri`.
        ax: 3-D axes to plot into. By default, a new figure is made.
        z_range: Lowest and highest z to show, in nm.
        z_exaggeration: Stretch of heights; 1 is true scale. By default the plot
            is at least 30 % as high as it is wide.
        material_names: Names for material ids in the legend, such as
            ``{0: "silicon"}``. By default, the ids are shown.
        edges: Whether to draw the mesh of the interfaces (on top, so hidden
            edges show too).

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

    # Matplotlib sorts faces by the depth of their centres, and only within
    # one collection. So surfaces and outlines go into one collection, all in
    # pieces of at most 1/16 of the width, so that they hide each other right.
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
