"""Tests for nebula.beam."""

import numpy as np
import pytest
from nebula.beam import scan_positions, scan_primaries


def test_scan_positions():
    # x = [-100, 100] with step 1: 200 pixels, centred at -99.5, ..., 99.5
    centres = scan_positions(-100, 100, 1)
    assert len(centres) == 200
    assert (centres[0], centres[-1]) == (-99.5, 99.5)
    np.testing.assert_allclose(scan_positions(-4, 4, 2), [-3, -1, 1, 3])
    np.testing.assert_allclose(scan_positions(0, 2, 2), [1])
    for start, end, step in [(0, 5, 2), (0, 0, 1), (0, 1, 2)]:
        with pytest.raises(ValueError):
            scan_positions(start, end, step)


def test_fixed_number_per_pixel():
    rng = np.random.default_rng(0)
    xs, ys = np.array([-1.0, 1.0]), np.array([0.0, 2.0, 4.0])
    electrons, counts = scan_primaries(
        xs, ys, energy=500, z=30, avg_electrons=4, poisson=False, rng=rng
    )
    np.testing.assert_array_equal(counts, np.full((2, 3), 4))
    assert len(electrons) == 24
    # Without spread, every electron starts at its pixel centre
    np.testing.assert_array_equal(electrons["x"], xs[electrons["px"]])
    np.testing.assert_array_equal(electrons["y"], ys[electrons["py"]])
    assert np.all(electrons["z"] == 30) and np.all(electrons["E"] == 500)
    assert np.all(electrons["dz"] == -1)


def test_poisson_and_spread():
    rng = np.random.default_rng(1)
    xs = ys = np.arange(20.0)
    electrons, counts = scan_primaries(
        xs,
        ys,
        energy=1000,
        z=10,
        axis=[0, 1, -1],
        avg_electrons=50,
        beam_std=2.0,
        rng=rng,
    )
    assert counts.sum() == len(electrons)
    assert counts.std() > 0 and abs(counts.mean() - 50) < 1
    np.testing.assert_array_equal(
        np.bincount(electrons["px"] * 20 + electrons["py"], minlength=400),
        counts.ravel(),
    )
    offset = electrons["x"] - xs[electrons["px"]]
    assert abs(offset.std() - 2.0) < 0.05
    assert np.all(electrons["dy"] == 1)
