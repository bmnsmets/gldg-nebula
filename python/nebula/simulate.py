"""Running simulations defined in TOML files (see `nebula.config`)."""

import dataclasses
import importlib.metadata
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .beam import scan_positions, scan_primaries
from .config import Config, TriGeometry, ZmapGeometry
from .geometry import height_map, triangulate_zmap, zmap_from_png
from .imaging import DetectorCounter
from .io.triangles import SpecialMaterial, read_tri
from .runner import find_binary, find_material, stream_nebula

__all__ = ["Result", "run", "save_results"]


@dataclasses.dataclass
class Result:
    """The outcome of a run.

    Attributes:
        images: Yield per pixel of each detector, float32 ``[ix, iy]``.
        primaries: Number of primary electrons per pixel.
        xs, ys: Pixel centres in nm.
        binary: The Nebula binary that was used.
        triangles: Number of triangles of the geometry.
        detected: Number of electrons Nebula detected.
        seconds: Time Nebula took.
        zmap: Height in nm of the highest surface of the geometry at each
            beam position, float32 ``[ix, iy]``, see `nebula.geometry.height_map`.
    """

    images: dict[str, np.ndarray]
    primaries: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    binary: Path
    triangles: int
    detected: int
    seconds: float
    zmap: np.ndarray | None = None


def run(
    config: Config,
    *,
    gpu: bool | None = None,
    workdir: str | os.PathLike | None = None,
    report: Callable[[str], None] | None = None,
    progress: Callable[[float | None], None] | None = None,
) -> Result:
    """Run a simulation: make the geometry and primaries, run Nebula, apply detectors.

    Args:
        config: The run, from `nebula.config.load_config`.
        gpu: True for nebula_gpu, False for nebula_cpu_mt, None to use
            nebula_gpu if there is a GPU.
        workdir: Directory to keep Nebula's input and output files in; by
            default they are removed.
        report: Called with a line of text about each step: the scan, the
            geometry and the binary before Nebula starts, and the detectors
            after it.
        progress: Called while Nebula runs, see `nebula.runner.run_nebula`.

    Raises:
        ValueError: If the beam does not start between the surface and the
            detector plane, or outside the box.
    """
    geometry, beam = config.geometry, config.beam
    made = _geometry(geometry)
    triangles, surface_top = made.triangles, made.surface_top

    detector_z = _detector_z(triangles)
    if not surface_top < beam.z < detector_z:
        raise ValueError(
            f"[beam] z ({beam.z} nm) must be above the surface ({surface_top} nm) "
            f"and below the detector plane ({detector_z} nm)"
        )

    # Scan: the given area (its outer edges), or the geometry's default
    x = beam.x if beam.x is not None else made.scan_x
    y = beam.y if beam.y is not None else made.scan_y
    step = beam.step if beam.step is not None else made.step
    if step is None:
        raise ValueError("[beam] needs step for a triangle file geometry")
    xs, ys = scan_positions(*x, step), scan_positions(*y, step)
    (x_min, x_max), (y_min, y_max) = made.box_x, made.box_y
    if x[0] < x_min or x[1] > x_max or y[0] < y_min or y[1] > y_max:
        raise ValueError(
            f"the scan must lie inside the box (x {x_min:g} to {x_max:g}, "
            f"y {y_min:g} to {y_max:g} nm)"
        )

    nx_scan, ny_scan = len(xs), len(ys)
    rng = np.random.default_rng(config.seed)
    primaries, counts = scan_primaries(
        xs,
        ys,
        energy=beam.energy,
        z=beam.z,
        axis=beam.axis,
        avg_electrons=beam.avg_electrons,
        poisson=beam.poisson,
        beam_std=beam.beam_std,
        rng=rng,
    )

    directory = config.path.parent if config.path is not None else Path.cwd()
    materials = [find_material(name, [directory]) for name in geometry.materials]
    binary = find_binary(gpu)
    if report is not None:
        total = int(counts.sum())
        report(
            f"{config.name}: {nx_scan} x {ny_scan} pixels of {step:g} nm, "
            f"{_count(total)} primaries at {beam.energy:g} eV "
            f"(≈{total / counts.size:.0f} per pixel), seed {config.seed}"
            f"{' (random)' if config.random_seed else ''}"
        )
        n_materials = len(geometry.materials)
        source = (
            f" from {geometry.file.name}" if isinstance(geometry, TriGeometry) else ""
        )
        report(
            f"geometry: {len(triangles)} triangles{source}, {n_materials} "
            f"material{'s' if n_materials != 1 else ''}"
        )
        report(f"running {binary.name} ({binary})")
        report(
            f"applying {len(config.detectors)} detector"
            f"{'s' if len(config.detectors) != 1 else ''} to the electrons as they come"
        )
    start = time.perf_counter()
    with DetectorCounter(config.detectors, xs, ys) as counter:
        detected = stream_nebula(
            triangles,
            primaries,
            materials,
            counter.add,
            seed=config.seed,
            binary=binary,
            workdir=workdir,
            progress=progress,
        )
        seconds = time.perf_counter() - start
        images = counter.images(counts)
    result = Result(images, counts, xs, ys, binary, len(triangles), detected, seconds)
    # The highest surface Nebula simulated, on the same grid as the images
    result.zmap = height_map(triangles, xs, ys).astype(np.float32)
    return result


class _Geometry(NamedTuple):
    """Triangles of a run, with what the scan needs to know about them (in nm)."""

    triangles: np.ndarray
    surface_top: float  # the beam must start above this
    box_x: tuple[float, float]  # the scan must lie inside the box
    box_y: tuple[float, float]
    scan_x: tuple[float, float]  # default scan
    scan_y: tuple[float, float]
    step: float | None  # default step; None for triangle files


def _geometry(geometry: ZmapGeometry | TriGeometry) -> _Geometry:
    """Make or read the triangles of a geometry.

    A triangle file's box, and default scan, is the x-y extent of its
    triangles. A z-map's box is its box size, and its default scan its
    own pixels.

    Raises:
        ValueError: If a triangle file uses more materials than are given.
    """
    if isinstance(geometry, TriGeometry):
        triangles = read_tri(geometry.file)
        used = max(triangles["material_in"].max(), triangles["material_out"].max())
        if used >= len(geometry.materials):
            raise ValueError(
                f"{geometry.file.name} uses material id {used}, but [geometry] "
                f"materials names only {len(geometry.materials)}"
            )
        has_material = (triangles["material_in"] >= 0) | (
            triangles["material_out"] >= 0
        )
        vertices = triangles["vertices"]
        surface_top = float(vertices[has_material][..., 2].max(initial=0.0))
        low, high = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
        box_x, box_y = (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))
        return _Geometry(triangles, surface_top, box_x, box_y, box_x, box_y, None)

    zmap = zmap_from_png(
        geometry.file, height=geometry.z_max - geometry.z_min, base=geometry.z_min
    )
    triangles = triangulate_zmap(
        zmap,
        geometry.pixel_size,
        tolerance=geometry.tolerance,
        box_size=geometry.box_size,
        terminator_z=geometry.z_terminator,
        mirrors=geometry.mirrors,
        detector=SpecialMaterial.DETECTOR,
        detector_z=geometry.z_detector,
    )
    nx, ny = zmap.shape
    half_x, half_y = nx * geometry.pixel_size / 2, ny * geometry.pixel_size / 2
    box = geometry.box_size or (2 * half_x, 2 * half_y)
    return _Geometry(
        triangles,
        max(float(zmap.max()), 0.0),
        (-box[0] / 2, box[0] / 2),
        (-box[1] / 2, box[1] / 2),
        (-half_x, half_x),
        (-half_y, half_y),
        geometry.pixel_size,
    )


def _detector_z(triangles: np.ndarray) -> float:
    """Height in nm of the lowest detector triangle, the plane the beam must stay below."""
    detector = (triangles["material_in"] == SpecialMaterial.DETECTOR) | (
        triangles["material_out"] == SpecialMaterial.DETECTOR
    )
    if not detector.any():
        raise ValueError("the geometry has no detector plane (material -126)")
    return float(triangles["vertices"][detector][..., 2].min())


def _count(n: int) -> str:
    """A number of electrons, short: 950, 12.3k, 4.0M."""
    if n >= 999_500:
        return f"{n / 1e6:.1f}M"
    if n >= 10_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def save_results(path: str | os.PathLike, config: Config, result: Result) -> None:
    """Save the yields of a run as a safetensors file.

    The file holds one float32 image ``[ix, iy]`` per detector, named after
    it, ``primaries`` (int32, primary electrons per pixel), and ``x`` and
    ``y`` (float32, pixel centres in nm). ``zmap`` (float32 ``[ix, iy]``)
    holds the height in nm of the highest surface of the geometry at each
    beam position. Its metadata holds the TOML text,
    the Nebula binary, the seed and the gldg-nebula version.
    """
    from safetensors.numpy import save_file

    tensors = {
        name: np.ascontiguousarray(image) for name, image in result.images.items()
    }
    tensors["primaries"] = result.primaries.astype(np.int32)
    tensors["x"] = result.xs.astype(np.float32)
    tensors["y"] = result.ys.astype(np.float32)
    if result.zmap is not None:
        tensors["zmap"] = np.ascontiguousarray(result.zmap, dtype=np.float32)
    metadata = {
        "name": config.name,
        "config": config.text,
        "binary": result.binary.name,
        "seed": str(config.seed),
        "gldg-nebula": importlib.metadata.version("gldg-nebula"),
    }
    save_file(tensors, os.fspath(path), metadata=metadata)
