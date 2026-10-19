"""Tests for nebula.imaging."""

import math

import numpy as np
from nebula.config import Detector
from nebula.imaging import detector_mask, electron_quantities, yield_images
from nebula.io import make_electrons

XS = np.array([0.0, 10.0])
YS = np.array([0.0, 10.0, 20.0])


def electrons(position, direction, energy, pixel):
    return make_electrons(position, direction, energy, pixel)


def test_quantities():
    e = electrons(
        [[13, 9, 60], [0, 25, 60]],
        [[1, 0, 1], [0, -1, 0.0001]],
        [10, 400],
        [[1, 1], [0, 2]],
    )
    q = electron_quantities(e, XS, YS)
    np.testing.assert_allclose(q["dx"], [3, 0])
    np.testing.assert_allclose(q["dy"], [-1, 5])
    np.testing.assert_allclose(q["r"], [math.hypot(3, 1), 5])
    np.testing.assert_allclose(q["polar"], [45, 90], atol=0.01)
    np.testing.assert_allclose(q["azimuth"], [0, 270])
    np.testing.assert_allclose(q["energy"], [10, 400])


def test_ranges_include_min_exclude_max():
    e = electrons([[0, 0, 60]] * 3, [0, 0, 1], [49.5, 50, 50.5], [0, 0])
    q = electron_quantities(e, XS, YS)
    se = detector_mask(Detector("se", energy=(0, 50)), q)
    bse = detector_mask(Detector("bse", energy=(50, math.inf)), q)
    np.testing.assert_array_equal(se, [True, False, False])
    np.testing.assert_array_equal(bse, [False, True, True])


def test_azimuth_wraps():
    angles = np.radians([0, 30, 90, 200, 340])
    direction = np.column_stack([np.cos(angles), np.sin(angles), np.ones(5)])
    q = electron_quantities(electrons([0, 0, 60], direction, 100, [0, 0]), XS, YS)
    around_x = detector_mask(Detector("x", azimuth=(315, 45)), q)
    np.testing.assert_array_equal(around_x, [True, True, False, False, True])
    second = detector_mask(Detector("q2", azimuth=(90, 180)), q)
    np.testing.assert_array_equal(second, [False, False, True, False, False])


def test_combined_filters():
    e = electrons(
        [[1, 0, 60], [30, 0, 60], [1, 0, 60]], [0, 0, 1], [100, 100, 10], [[0, 0]] * 3
    )
    q = electron_quantities(e, XS, YS)
    inner_bse = Detector("inner", energy=(50, math.inf), r=(0, 20))
    np.testing.assert_array_equal(detector_mask(inner_bse, q), [True, False, False])


def test_yield_images():
    # Pixel (0, 0): 4 primaries, 2 SE and 1 BSE; pixel (1, 2): 2 primaries, 1 SE;
    # pixel (1, 0): no primaries
    e = electrons(
        [0, 0, 60],
        [0, 0, 1],
        [10, 20, 300, 5],
        [[0, 0], [0, 0], [0, 0], [1, 2]],
    )
    primaries = np.array([[4, 1, 1], [0, 1, 2]])
    detectors = [Detector("se", energy=(0, 50)), Detector("bse", energy=(50, math.inf))]
    images = yield_images(e, detectors, XS, YS, primaries)
    assert images["se"].dtype == np.float32 and images["se"].shape == (2, 3)
    np.testing.assert_allclose(images["se"], [[0.5, 0, 0], [np.nan, 0, 0.5]])
    np.testing.assert_allclose(images["bse"], [[0.25, 0, 0], [np.nan, 0, 0]])


def test_quantities_subset():
    e = electrons(
        [[13, 9, 60], [0, 25, 60]], [[1, 0, 1], [0, -1, 0.5]], 10, [[1, 1], [0, 2]]
    )
    full = electron_quantities(e, XS, YS)
    polar = electron_quantities(e, XS, YS, keys=("polar",))
    assert list(polar) == ["polar"]
    np.testing.assert_array_equal(polar["polar"], full["polar"])
    r = electron_quantities(e, XS, YS, keys=["r"])
    assert list(r) == ["r"]
    np.testing.assert_array_equal(r["r"], full["r"])


def test_yield_images_chunked_matches_reference():
    # Many small chunks on several threads give exactly the all-at-once result
    rng = np.random.default_rng(1)
    n = 5000
    position = np.column_stack(
        [rng.uniform(-5, 15, n), rng.uniform(-5, 25, n), np.full(n, 60)]
    )
    direction = rng.normal(size=(n, 3))
    pixel = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 3, n)])
    e = electrons(position, direction, rng.exponential(60, n), pixel)
    primaries = np.array([[40, 0, 30], [20, 25, 0]])
    detectors = [
        Detector("all"),
        Detector("se", energy=(0, 50)),
        Detector("bse", energy=(50, math.inf), polar=(10, 55)),
        Detector("wrap", azimuth=(300, 60.5)),
        Detector("box", dx=(-2, 3.5), dy=(-np.inf, 4)),
        Detector("ring", r=(1, 6), polar=(90, 180), azimuth=(0, 180)),
    ]
    images = yield_images(e, detectors, XS, YS, primaries, _chunk_size=7)

    q = electron_quantities(e, XS, YS)
    index = np.ravel_multi_index((e["px"], e["py"]), primaries.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        for detector in detectors:
            counts = np.bincount(index[detector_mask(detector, q)], minlength=6)
            expected = np.where(
                primaries > 0, counts.reshape(primaries.shape) / primaries, np.nan
            ).astype(np.float32)
            np.testing.assert_array_equal(images[detector.name], expected)
    np.testing.assert_array_equal(np.isnan(images["all"]), primaries == 0)
