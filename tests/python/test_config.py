"""Tests for nebula.config."""

import math
from pathlib import Path

import numpy as np
import pytest
from nebula.config import ConfigError, TriGeometry, ZmapGeometry, load_config
from PIL import Image

EXAMPLES = Path(__file__).parents[2] / "examples"
CELL65 = EXAMPLES / "z-map" / "cell65.toml"
LINE_SPACE = EXAMPLES / "line-space" / "line-space-00.toml"

MINIMAL = """
name = "test"

[geometry.zmap]
file = "map.png"
z_max = 20
materials = ["silicon"]

[beam]
energy = 500
z = 25
avg_electrons = 10

[detectors.se]
energy = [0, 50]
"""

TRI = MINIMAL.replace(
    """[geometry.zmap]
file = "map.png"
z_max = 20
materials = ["silicon"]""",
    """[geometry]
file = "geometry.tri"
materials = ["silicon", "pmma"]""",
).replace("avg_electrons = 10", "avg_electrons = 10\nstep = 2")


@pytest.fixture
def directory(tmp_path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "map.png")
    (tmp_path / "geometry.tri").write_text("")  # only read by nebula.simulate
    return tmp_path


def write(directory, text):
    path = directory / "run.toml"
    path.write_text(text)
    return path


def test_cell65():
    config = load_config(CELL65)
    assert config.name == "cell65"
    assert config.geometry.file.is_file()
    assert len(config.geometry.materials) == 1  # z-maps take one material
    names = [detector.name for detector in config.detectors]
    assert names[:2] == ["se", "bse"]
    se, bse = config.detectors[:2]
    assert se.energy == (0, 50)
    assert bse.energy == (50, math.inf)
    assert config.text == CELL65.read_text()

    assert isinstance(config.geometry, ZmapGeometry)


def test_line_space():
    config = load_config(LINE_SPACE)
    assert isinstance(config.geometry, TriGeometry)
    assert config.geometry.file == LINE_SPACE.parent / "tri" / "geometry_000000.tri"
    assert config.geometry.materials == ("silicon", "pmma")


def test_tri(directory):
    config = load_config(write(directory, TRI))
    assert config.geometry == TriGeometry(
        directory / "geometry.tri", ("silicon", "pmma")
    )
    assert config.beam.step == 2


@pytest.mark.parametrize(
    "change",
    [
        ("step = 2", ""),  # a triangle file has no pixel size
        ('file = "geometry.tri"', 'file = "missing.tri"'),
        ('file = "geometry.tri"', ""),
        ('materials = ["silicon", "pmma"]', "materials = []"),
        ('materials = ["silicon", "pmma"]', ""),
        ("[geometry]", "[geometry]\nz_max = 20"),  # unknown key
        ("[beam]", "[geometry.zmap]\nfile = 'map.png'\n\n[beam]"),  # two geometries
    ],
)
def test_invalid_tri(directory, change):
    old, new = change
    assert old in TRI
    with pytest.raises(ConfigError):
        load_config(write(directory, TRI.replace(old, new)))


def test_random_seed(directory):
    # Without a seed, each load draws another (equal with probability 2^-32)
    path = write(directory, MINIMAL)
    assert load_config(path).seed != load_config(path).seed
    config = load_config(
        write(directory, MINIMAL.replace('name = "test"', 'name = "test"\nseed = 7'))
    )
    assert config.seed == 7 and not config.random_seed


def test_defaults(directory):
    config = load_config(write(directory, MINIMAL))
    geometry, beam = config.geometry, config.beam
    assert config.random_seed and 0 <= config.seed < 2**32
    assert geometry.file == (directory / "map.png").resolve()
    assert (geometry.z_min, geometry.tolerance, geometry.pixel_size) == (0, 0.01, 1)
    assert geometry.box_size is None and geometry.z_detector is None
    assert geometry.z_terminator == -10_000 and geometry.mirrors
    assert beam.axis == (0, 0, -1) and beam.poisson and beam.beam_std == 0
    assert beam.x is None and beam.y is None and beam.step is None


def test_filters(directory):
    text = (
        MINIMAL
        + """
[detectors.quadrant]
energy = [50, inf]
dx = [-inf, 0]
r = [0, 20]
polar = [0, 45]
azimuth = [315, 45]
"""
    )
    quadrant = load_config(write(directory, text)).detectors[1]
    assert quadrant.dx == (-math.inf, 0)
    assert quadrant.azimuth == (315, 45)  # wraps around 0
    assert quadrant.dy is None


@pytest.mark.parametrize(
    "change",
    [
        ("z_max = 20", "z_max = 20\ncolour = 1"),  # unknown key
        ("energy = [0, 50]", 'energy = "<50"'),  # not a range
        ("energy = [0, 50]", "energy = [50, 0]"),  # min > max
        ("energy = [0, 50]", "energy = [0, 50, 100]"),
        ("energy = [0, 50]", "polar = [45, 0]"),  # only azimuth wraps
        ('materials = ["silicon"]', 'materials = ["silicon", "pmma"]'),
        ('file = "map.png"', 'file = "missing.png"'),
        ("[detectors.se]", "[detectors.primaries]"),  # reserved name
        ("[detectors.se]", "[detectors.zmap]"),  # reserved name
        ("[detectors.se]\nenergy = [0, 50]\n", ""),  # no detectors
        ("avg_electrons = 10", "avg_electrons = 0"),
        ("avg_electrons = 10", "avg_electrons = 10\naxis = [0, 0, 0]"),
        ("z = 25", 'z = "25"'),
        ('name = "test"', 'name = "test"\nseed = -1'),
        ("[beam]", "[beam]\nx = [10, -10]"),
        ("[beam]", "[beam]\ny = [10, 10]"),  # no width
    ],
)
def test_invalid(directory, change):
    old, new = change
    assert old in MINIMAL
    with pytest.raises(ConfigError):
        load_config(write(directory, MINIMAL.replace(old, new)))


def test_invalid_toml(directory):
    with pytest.raises(ConfigError):
        load_config(write(directory, "name = "))
