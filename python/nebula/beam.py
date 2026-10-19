"""Generating primary electrons: a beam scanned over a grid of pixels."""

import numpy as np
import numpy.typing as npt

from .io.electrons import ELECTRON_DTYPE, make_electrons

__all__ = ["scan_positions", "scan_primaries"]


def scan_positions(start: float, end: float, step: float) -> np.ndarray:
    """Centres of the pixels of width step that cover start to end.

    For example, start = -100, end = 100 and step = 1 give 200 pixels,
    centred at -99.5, -98.5, ..., 99.5.

    Raises:
        ValueError: If end - start is not a whole number (at least 1) of steps.
    """
    pixels = (end - start) / step
    n = round(pixels)
    if n < 1 or abs(pixels - n) > 1e-6 * max(1.0, abs(pixels)):
        raise ValueError(
            f"the scan from {start} to {end} nm is not a whole number of {step} nm pixels"
        )
    return start + step * (np.arange(n) + 0.5)


def scan_primaries(
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    *,
    energy: float,
    z: float,
    axis: npt.ArrayLike = (0, 0, -1),
    avg_electrons: float,
    poisson: bool = True,
    beam_std: float = 0.0,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Primary electrons for a beam scanned over the pixels xs x ys.

    Each pixel gets ``avg_electrons`` electrons, or a Poisson-distributed
    number with that mean. They start at height ``z`` around the pixel centre,
    with a Gaussian spread ``beam_std`` in x and y, and go in the direction
    ``axis``. Their ``px`` and ``py`` are the indices of the pixel in xs and ys.

    Args:
        xs, ys: Pixel centres in nm.
        energy: Kinetic energy in eV.
        z: Starting height in nm.
        axis: Direction of the electrons.
        avg_electrons: Mean number of electrons per pixel.
        poisson: Whether to draw the number per pixel from a Poisson
            distribution. Otherwise, ``avg_electrons`` is rounded.
        beam_std: Standard deviation of the starting position in x and y, in nm.
        rng: Random number generator.

    Returns:
        The electrons, and the number of electrons per pixel (int64, shape
        (len(xs), len(ys))).
    """
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    shape = (len(xs), len(ys))
    if poisson:
        counts = rng.poisson(avg_electrons, size=shape)
    else:
        counts = np.full(shape, round(avg_electrons))
    # The values shared by all electrons are checked once, as make_electrons
    # would check them for each electron
    make_electrons([0, 0, z], axis, energy, [0, 0])
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and np.isfinite(beam_std)):
        raise ValueError("position and direction must be finite")

    ix = np.repeat(np.arange(shape[0]), counts.sum(axis=1))
    iy = np.repeat(np.tile(np.arange(shape[1]), shape[0]), counts.ravel())
    n = len(ix)
    electrons = np.empty(n, dtype=ELECTRON_DTYPE)
    electrons["x"] = xs[ix] + rng.normal(0, beam_std, n)
    electrons["y"] = ys[iy] + rng.normal(0, beam_std, n)
    electrons["z"] = z
    for name, value in zip(["dx", "dy", "dz"], np.asarray(axis, dtype=np.float64), strict=True):
        electrons[name] = value
    electrons["E"] = energy
    electrons["px"] = ix
    electrons["py"] = iy
    return electrons, counts
