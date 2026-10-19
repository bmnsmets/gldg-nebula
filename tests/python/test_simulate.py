"""Tests for the geometry checks of nebula.simulate, which do not need Nebula."""

import numpy as np
import pytest
from nebula.config import Beam, Config, Detector, TriGeometry
from nebula.geometry import triangulate_zmap
from nebula.io.triangles import SpecialMaterial, write_tri
from nebula.simulate import _detector_z, _geometry, run


def tri_file(tmp_path, detector=True, material=0):
    """A 20 x 10 nm block 5 nm high, with its detector plane at 15 nm."""
    triangles = triangulate_zmap(
        np.full((20, 10), 5.0),
        terminator_z=-100,
        material=material,
        detector=SpecialMaterial.DETECTOR if detector else None,
        detector_z=15,
    )
    path = tmp_path / "geometry.tri"
    write_tri(path, triangles)
    return path


def config(path, z=10.0, x=None):
    return Config(
        "test",
        TriGeometry(path, ("silicon",)),
        Beam(energy=500, z=z, avg_electrons=1, x=x, step=2),
        (Detector("se", energy=(0, 50)),),
    )


def test_tri_geometry(tmp_path):
    made = _geometry(TriGeometry(tri_file(tmp_path), ("silicon",)))
    assert made.surface_top == 5
    assert made.box_x == made.scan_x == (-10, 10)
    assert made.box_y == made.scan_y == (-5, 5)
    assert made.step is None
    assert _detector_z(made.triangles) == 15


def test_too_few_materials(tmp_path):
    with pytest.raises(ValueError, match="material id 1"):
        _geometry(TriGeometry(tri_file(tmp_path, material=1), ("silicon",)))


def test_no_detector(tmp_path):
    made = _geometry(TriGeometry(tri_file(tmp_path, detector=False), ("silicon",)))
    with pytest.raises(ValueError, match="detector"):
        _detector_z(made.triangles)


@pytest.mark.parametrize(
    ("z", "x"),
    [
        (3.0, None),  # below the surface
        (20.0, None),  # above the detector
        (10.0, (-12.0, 8.0)),  # outside the box
    ],
)
def test_invalid_beam(tmp_path, z, x):
    # These are checked before Nebula or the materials are looked for
    with pytest.raises(ValueError):
        run(config(tri_file(tmp_path), z=z, x=x))
