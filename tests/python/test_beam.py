"""Tests for nebula.beam."""

import numpy as np
import pytest
from nebula.beam import scan_positions, scan_primaries
from nebula.io import make_electrons


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


def reference_scan_primaries(
    xs, ys, *, energy, z, axis, avg_electrons, poisson, beam_std, rng
):
    # The straightforward implementation scan_primaries replaced
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    shape = (len(xs), len(ys))
    if poisson:
        counts = rng.poisson(avg_electrons, size=shape)
    else:
        counts = np.full(shape, round(avg_electrons))
    pixel = np.repeat(np.arange(counts.size), counts.ravel())
    ix, iy = np.unravel_index(pixel, shape)
    position = np.column_stack(
        [
            xs[ix] + rng.normal(0, beam_std, len(pixel)),
            ys[iy] + rng.normal(0, beam_std, len(pixel)),
            np.full(len(pixel), z),
        ]
    )
    electrons = make_electrons(position, axis, energy, np.column_stack([ix, iy]))
    return electrons, counts


@pytest.mark.parametrize(
    ("axis", "poisson", "beam_std"),
    [((0, 0, -1), True, 1.5), ((0.3, -2, -5), False, 0.0), ((1, 0, 0), True, 0.0)],
)
def test_scan_primaries_matches_reference(axis, poisson, beam_std):
    xs, ys = np.linspace(-7.3, 11, 9), np.linspace(2, 5.5, 4)
    arguments = {
        "energy": 812.5,
        "z": 33.3,
        "axis": axis,
        "avg_electrons": 6.4,
        "poisson": poisson,
        "beam_std": beam_std,
    }
    electrons, counts = scan_primaries(
        xs, ys, rng=np.random.default_rng(5), **arguments
    )
    expected, expected_counts = reference_scan_primaries(
        xs, ys, rng=np.random.default_rng(5), **arguments
    )
    np.testing.assert_array_equal(counts, expected_counts)
    assert electrons.dtype == expected.dtype
    assert electrons.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "change",
    [
        {"axis": (0, 0, 0)},
        {"axis": (0, np.nan, -1)},
        {"energy": 0},
        {"energy": -5},
        {"z": np.inf},
        {"beam_std": np.nan},
        {"xs": [0, np.nan]},
        {"ys": [np.inf]},
    ],
)
def test_scan_primaries_invalid(change):
    arguments = {
        "xs": [0.0, 1.0],
        "ys": [0.0],
        "energy": 500,
        "z": 10,
        "avg_electrons": 2,
    }
    with pytest.raises(ValueError):
        scan_primaries(**(arguments | change), rng=np.random.default_rng(0))
