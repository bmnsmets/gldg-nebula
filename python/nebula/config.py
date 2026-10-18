"""Simulation runs defined in TOML files.

A run file describes the geometry (a z-map image), the beam and its scan, and
the detectors that select detected electrons afterwards::

    name = "cell65"
    seed = 1                          # optional; random if left out

    [geometry.zmap]
    file = "cell65morphilt-znom.png"  # relative to the TOML file
    tolerance = 0.01                  # optional, fraction of the height range
    pixel_size = 1                    # nm, optional
    box_size = [500, 500]             # nm, optional (default: the z-map size)
    z_min = 0                         # nm, height of black, optional
    z_max = 50                        # nm, height of white
    z_terminator = -1000              # nm, optional (default -10000)
    z_detector = 60                   # nm, optional (default 10 nm above the top)
    mirrors = true                    # optional
    materials = ["silicon"]           # .mat files by name; one for z-maps

    [beam]
    energy = 500                      # eV
    z = 55                            # nm, between the surface and z_detector
    axis = [0, 0, -1]                 # optional
    avg_electrons = 50                # per pixel
    poisson = true                    # optional
    beam_std = 1.0                    # nm, optional
    x = [-240, 240]                   # nm, optional: edges of the scanned area
    y = [-240, 240]                   # nm, optional
    step = 2                          # nm, optional (default: the pixel size)

    [detectors.se]
    energy = [0, 50]

    [detectors.bse_q1]
    energy = [50, inf]
    azimuth = [0, 90]

``x`` and ``y`` are the outer edges of the scanned area, which is divided into
pixels of ``step`` by ``step``; the beam scans their centres. For example,
``x = [-100, 100]`` with ``step = 1`` gives 200 pixels, centred at -99.5, ...,
99.5. Without ``x``, ``y`` and ``step``, the beam scans the centres of the
z-map's pixels.

Detectors select electrons that hit Nebula's detector plane (at
``z_detector``), which itself records every electron. Each filter is a range
``[min, max]``, min included and max not; ``inf`` and ``-inf`` give open ends.
All filters of a detector must hold:

- ``energy``: kinetic energy at the detector plane, in eV;
- ``dx``, ``dy``: position where the electron hits the detector plane, minus
  the position of the beam for its pixel, in nm;
- ``r``: distance from the beam axis on the detector plane, in nm;
- ``polar``: angle between the electron's direction and +z, in degrees
  (0 is straight up);
- ``azimuth``: angle of the direction in the x-y plane, in degrees from +x
  counter-clockwise, 0 to 360. A range with min > max wraps around 0, so
  ``[315, 45]`` is centred on +x.
"""

import dataclasses
import math
import os
import secrets
import tomllib
from pathlib import Path

__all__ = ["Beam", "Config", "ConfigError", "Detector", "ZmapGeometry", "load_config"]

FILTERS = ("energy", "dx", "dy", "r", "polar", "azimuth")
# Names used for other tensors in the output file
RESERVED_NAMES = ("primaries", "x", "y", "zmap")

Range = tuple[float, float]


class ConfigError(ValueError):
    """A run file that is not valid: a value is missing, wrong or unknown."""


@dataclasses.dataclass(frozen=True)
class ZmapGeometry:
    """Geometry made from a z-map image, see `nebula.geometry.triangulate_zmap`."""

    file: Path
    z_max: float
    materials: tuple[str, ...]
    z_min: float = 0.0
    tolerance: float = 0.01
    pixel_size: float = 1.0
    box_size: tuple[float, float] | None = None
    z_terminator: float = -10_000.0
    z_detector: float | None = None
    mirrors: bool = True


@dataclasses.dataclass(frozen=True)
class Beam:
    """The primary electrons and the pixels they are scanned over."""

    energy: float
    z: float
    avg_electrons: float
    axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    poisson: bool = True
    beam_std: float = 0.0
    x: Range | None = None
    y: Range | None = None
    step: float | None = None


@dataclasses.dataclass(frozen=True)
class Detector:
    """A selection of detected electrons; None means no filter."""

    name: str
    energy: Range | None = None
    dx: Range | None = None
    dy: Range | None = None
    r: Range | None = None
    polar: Range | None = None
    azimuth: Range | None = None


@dataclasses.dataclass(frozen=True)
class Config:
    """A simulation run."""

    name: str
    geometry: ZmapGeometry
    beam: Beam
    detectors: tuple[Detector, ...]
    seed: int = 0
    path: Path | None = None
    text: str = ""
    random_seed: bool = False  # the file has no seed, so it was drawn at random


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _finite(value, where: str) -> float:
    number = _number(value, where)
    if not math.isfinite(number):
        raise ConfigError(f"{where} must be finite, got {value!r}")
    return number


def _positive(value, where: str) -> float:
    number = _finite(value, where)
    if number <= 0:
        raise ConfigError(f"{where} must be positive, got {value!r}")
    return number


def _vector(value, length: int, where: str, finite: bool = True) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{where} must be a list of {length} numbers, got {value!r}")
    convert = _finite if finite else _number
    return tuple(convert(v, where) for v in value)


def _range(value, where: str, wrap: bool = False) -> Range:
    low, high = _vector(value, 2, where, finite=False)
    if math.isnan(low) or math.isnan(high):
        raise ConfigError(f"{where} must not contain nan")
    if low >= high and not (wrap and low != high):
        raise ConfigError(f"{where} must be [min, max] with min < max, got {value!r}")
    return low, high


def _bool(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {value!r}")
    return value


def _table(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a table")
    return dict(value)


def _no_other_keys(table: dict, where: str) -> None:
    if table:
        raise ConfigError(f"unknown keys in {where}: {', '.join(sorted(table))}")


def _required(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"{where} needs {key}")
    return table.pop(key)


def _geometry(table: dict, directory: Path) -> ZmapGeometry:
    table = _table(table, "[geometry]")
    if set(table) != {"zmap"}:
        raise ConfigError(
            "[geometry] must contain exactly one geometry, [geometry.zmap]"
        )
    zmap = _table(table["zmap"], "[geometry.zmap]")
    where = "[geometry.zmap]"

    file = _required(zmap, "file", where)
    if not isinstance(file, str):
        raise ConfigError(f"{where} file must be a string")
    path = (directory / file).resolve()
    if not path.is_file():
        raise ConfigError(f"{where} file {file!r} not found (looked for {path})")

    materials = _required(zmap, "materials", where)
    if not isinstance(materials, list) or not all(
        isinstance(m, str) for m in materials
    ):
        raise ConfigError(f"{where} materials must be a list of material names")
    if len(materials) != 1:
        raise ConfigError(f"{where} z-map geometries take exactly one material")

    values: dict = {
        "file": path,
        "materials": tuple(materials),
        "z_max": _finite(_required(zmap, "z_max", where), f"{where} z_max"),
    }
    for key in ["z_min", "z_terminator", "z_detector"]:
        if key in zmap:
            values[key] = _finite(zmap.pop(key), f"{where} {key}")
    for key in ["tolerance", "pixel_size"]:
        if key in zmap:
            values[key] = _positive(zmap.pop(key), f"{where} {key}")
    if "box_size" in zmap:
        box = _vector(zmap.pop("box_size"), 2, f"{where} box_size")
        values["box_size"] = tuple(_positive(b, f"{where} box_size") for b in box)
    if "mirrors" in zmap:
        values["mirrors"] = _bool(zmap.pop("mirrors"), f"{where} mirrors")
    _no_other_keys(zmap, where)
    return ZmapGeometry(**values)


def _beam(table: dict) -> Beam:
    beam = _table(table, "[beam]")
    where = "[beam]"
    values: dict = {
        "energy": _positive(_required(beam, "energy", where), f"{where} energy"),
        "z": _finite(_required(beam, "z", where), f"{where} z"),
        "avg_electrons": _positive(
            _required(beam, "avg_electrons", where), f"{where} avg_electrons"
        ),
    }
    if "axis" in beam:
        axis = _vector(beam.pop("axis"), 3, f"{where} axis")
        if not any(axis):
            raise ConfigError(f"{where} axis must not be zero")
        values["axis"] = axis
    if "poisson" in beam:
        values["poisson"] = _bool(beam.pop("poisson"), f"{where} poisson")
    if "beam_std" in beam:
        std = _finite(beam.pop("beam_std"), f"{where} beam_std")
        if std < 0:
            raise ConfigError(f"{where} beam_std must not be negative")
        values["beam_std"] = std
    for key in ["x", "y"]:
        if key in beam:
            low, high = _vector(beam.pop(key), 2, f"{where} {key}")
            if low >= high:
                raise ConfigError(
                    f"{where} {key} must be [start, end] with start < end"
                )
            values[key] = (low, high)
    if "step" in beam:
        values["step"] = _positive(beam.pop("step"), f"{where} step")
    _no_other_keys(beam, where)
    return Beam(**values)


def _detectors(table: dict) -> tuple[Detector, ...]:
    detectors = []
    for name, filters in _table(table, "[detectors]").items():
        where = f"[detectors.{name}]"
        if not name.isidentifier() or name in RESERVED_NAMES:
            raise ConfigError(
                f"{where}: detector names must be identifiers other than "
                f"{', '.join(RESERVED_NAMES)}"
            )
        filters = _table(filters, where)
        values = {
            key: _range(filters.pop(key), f"{where} {key}", wrap=key == "azimuth")
            for key in FILTERS
            if key in filters
        }
        _no_other_keys(filters, where)
        detectors.append(Detector(name, **values))
    if not detectors:
        raise ConfigError("define at least one detector in [detectors.<name>]")
    return tuple(detectors)


def load_config(path: str | os.PathLike) -> Config:
    """Read a simulation run from a TOML file.

    Args:
        path: Path of the TOML file. Files it refers to are looked up
            relative to it.

    Returns:
        The run.

    Raises:
        ConfigError: If the file is not valid TOML, a value is missing or
            invalid, or there are unknown keys.
    """
    path = Path(path)
    text = path.read_text()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from None

    name = _required(data, "name", path.name)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{path.name}: name must be a non-empty string")
    seed = data.pop("seed", None)
    random_seed = seed is None
    if random_seed:
        seed = secrets.randbelow(2**32)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ConfigError(f"{path.name}: seed must be an integer from 0 to 2^32 - 1")
    geometry = _geometry(_required(data, "geometry", path.name), path.parent)
    beam = _beam(_required(data, "beam", path.name))
    detectors = _detectors(_required(data, "detectors", path.name))
    _no_other_keys(data, path.name)
    return Config(
        name, geometry, beam, detectors, seed, path.resolve(), text, random_seed
    )
