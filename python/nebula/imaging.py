"""Turning detected electrons into images: detectors, yields per pixel and plots."""

import math
import os
import tomllib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import FILTERS, RESERVED_NAMES, Detector

__all__ = ["detector_mask", "electron_quantities", "plot_results", "yield_images"]


def electron_quantities(
    electrons: np.ndarray, xs: npt.ArrayLike, ys: npt.ArrayLike
) -> dict[str, np.ndarray]:
    """The quantities detectors filter on, for each detected electron.

    Args:
        electrons: Detected electrons (`ELECTRON_DTYPE`), with ``px`` and
            ``py`` the indices of their pixel in xs and ys.
        xs, ys: Pixel centres (beam positions) in nm.

    Returns:
        ``energy`` (eV), ``dx``, ``dy``, ``r`` (nm, relative to the beam
        position of the pixel), ``polar`` and ``azimuth`` (degrees), see
        `nebula.config`.
    """
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    dx = electrons["x"] - xs[electrons["px"]]
    dy = electrons["y"] - ys[electrons["py"]]
    direction = np.column_stack(
        [electrons["dx"], electrons["dy"], electrons["dz"]]
    ).astype(np.float64)
    length = np.linalg.norm(direction, axis=1)
    return {
        "energy": electrons["E"].astype(np.float64),
        "dx": dx,
        "dy": dy,
        "r": np.hypot(dx, dy),
        "polar": np.degrees(np.arccos(np.clip(direction[:, 2] / length, -1, 1))),
        "azimuth": np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) % 360,
    }


def detector_mask(detector: Detector, quantities: dict[str, np.ndarray]) -> np.ndarray:
    """Which electrons a detector selects: all its ranges [min, max) hold."""
    mask = np.ones(len(quantities["energy"]), dtype=bool)
    for key in FILTERS:
        bounds = getattr(detector, key)
        if bounds is None:
            continue
        low, high = bounds
        value = quantities[key]
        if key == "azimuth" and low > high:  # wraps around 0
            mask &= (value >= low) | (value < high)
        else:
            mask &= (value >= low) & (value < high)
    return mask


def yield_images(
    electrons: np.ndarray,
    detectors: tuple[Detector, ...] | list[Detector],
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    primaries: np.ndarray,
) -> dict[str, np.ndarray]:
    """Yield of each detector per pixel: selected electrons per primary.

    Args:
        electrons: Detected electrons, with ``px``, ``py`` their pixel.
        detectors: The detectors.
        xs, ys: Pixel centres in nm.
        primaries: Number of primary electrons per pixel, shape (len(xs), len(ys)).

    Returns:
        For each detector, a float32 image ``[ix, iy]``; NaN where a pixel had
        no primary electrons.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    shape = (len(xs), len(ys))
    quantities = electron_quantities(electrons, xs, ys)
    pixel = np.ravel_multi_index((electrons["px"], electrons["py"]), shape)
    images = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for detector in detectors:
            selected = pixel[detector_mask(detector, quantities)]
            counts = np.bincount(selected, minlength=shape[0] * shape[1]).reshape(shape)
            images[detector.name] = np.where(
                primaries > 0, counts / primaries, np.nan
            ).astype(np.float32)
    return images


_UNITS = {
    "energy": " eV",
    "dx": " nm",
    "dy": " nm",
    "r": " nm",
    "polar": "\u00b0",
    "azimuth": "\u00b0",
}


def _describe_range(key: str, bounds) -> str:
    """A filter as text, e.g. "energy \u2265 50 eV" or "azimuth 0\u201390\u00b0"."""
    low, high = (float(b) for b in bounds)
    unit = _UNITS.get(key, "")
    if math.isinf(low) and math.isinf(high):
        return f"{key} any"
    if math.isinf(high):
        return f"{key} \u2265 {low:g}{unit}"
    if math.isinf(low):
        return f"{key} < {high:g}{unit}"
    return f"{key} {low:g}\u2013{high:g}{unit}"


def plot_results(
    source: str | os.PathLike,
    *,
    detectors: Sequence[str] | None = None,
    cmap: str = "gray",
    zmap_cmap: str = "viridis",
    percentile: float = 99,
    columns: int = 3,
):
    """Plot the results of a run: the z-map and the yield map of each detector.

    Reads a safetensors file written by ``nebula RUN.toml`` (see
    `nebula.simulate.save_results`). Every image in it other than
    ``primaries``, ``x``, ``y`` and ``zmap`` is a detector, whatever its name;
    they are shown in the order of the run file stored in the file's metadata
    (or alphabetically), after the z-map if there is one.

    Yields are shown in greyscale like SEM images (black low, white high) and
    the z-map in another colour map, so that it is not mistaken for a yield.
    Each panel has its own colour bar. The yield maps' scales start at 0 and
    end at the given percentile of their values, so that a few noisy pixels
    do not set the scale; the z-map uses its full range. Pixels without
    primary electrons (NaN) are shown in red.

    Args:
        source: Path of the safetensors file.
        detectors: Names of the detectors to show, in this order. By default,
            all of them.
        cmap: Colour map of the yield maps.
        zmap_cmap: Colour map of the z-map.
        percentile: Percentile of each yield map's values at which its
            colour scale ends (it starts at 0); 100 for the maximum.
        columns: Number of panels per row.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If ``detectors`` names images that are not in the file, or
            the file has no images to show.
    """
    import matplotlib.pyplot as plt
    from safetensors import safe_open
    from safetensors.numpy import load_file

    tensors = load_file(os.fspath(source))
    with safe_open(os.fspath(source), "numpy") as f:
        metadata = f.metadata() or {}
    try:
        config = tomllib.loads(metadata["config"]) if "config" in metadata else {}
    except tomllib.TOMLDecodeError:
        config = {}
    filters = config.get("detectors", {})

    # Detector images: everything with the image shape that is not reserved
    shape = tensors["primaries"].shape if "primaries" in tensors else None
    available = [
        name
        for name, tensor in tensors.items()
        if name not in RESERVED_NAMES
        and tensor.ndim == 2
        and (shape is None or tensor.shape == shape)
    ]
    if detectors is None:
        in_config = [name for name in filters if name in available]
        names = in_config + sorted(set(available) - set(in_config))
    else:
        unknown = [name for name in detectors if name not in available]
        if unknown:
            raise ValueError(
                f"no detector images {', '.join(unknown)} in {source}; "
                f"it has {', '.join(sorted(available)) or 'none'}"
            )
        names = list(detectors)

    panels = []
    if "zmap" in tensors:
        panels.append(
            ("zmap", tensors["zmap"], "surface height", zmap_cmap, "height (nm)")
        )
    for name in names:
        image = tensors[name]
        title = (
            f"{name}: mean {np.nanmean(image):.3g}"
            if np.isfinite(image).any()
            else name
        )
        if isinstance(filters.get(name), dict) and filters[name]:
            described = ", ".join(
                _describe_range(k, v) for k, v in filters[name].items()
            )
            title += f"\n{described}"
        panels.append((name, image, title, cmap, "yield"))
    if not panels:
        raise ValueError(f"no images to plot in {source}")

    # Pixel edges in nm, from the pixel centres x and y (or pixel indices)
    def edges(key: str, n: int) -> tuple[float, float]:
        if key in tensors and len(tensors[key]) == n:
            centres = tensors[key].astype(np.float64)
            step = centres[1] - centres[0] if n > 1 else 1.0
            return centres[0] - step / 2, centres[-1] + step / 2
        return -0.5, n - 0.5

    nx, ny = panels[0][1].shape
    (x0, x1), (y0, y1) = edges("x", nx), edges("y", ny)
    unit = " (nm)" if "x" in tensors else " (pixel)"

    columns = max(1, min(columns, len(panels)))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4.4 * rows), squeeze=False
    )
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    for ax, (name, image, title, colours, label) in zip(axes.ravel(), panels):
        finite = image[np.isfinite(image)]
        # Heights are exact, so the z-map uses its full range
        if not finite.size:
            low, high = 0.0, 1.0
        elif name == "zmap":
            # Heights are exact, so the z-map uses its full range
            low, high = float(finite.min()), float(finite.max())
        else:
            # Yields start at 0, so that maps can be compared by eye
            low, high = 0.0, float(np.percentile(finite, percentile))
        if high <= low:
            low, high = low - 0.5, high + 0.5
        extend = {
            (False, False): "neither",
            (True, False): "min",
            (False, True): "max",
            (True, True): "both",
        }[
            (
                bool(finite.size and finite.min() < low),
                bool(finite.size and finite.max() > high),
            )
        ]
        colour_map = plt.get_cmap(colours).with_extremes(bad="tab:red")
        shown = ax.imshow(
            np.ma.masked_invalid(image).T,
            origin="lower",
            cmap=colour_map,
            vmin=low,
            vmax=high,
            extent=(x0, x1, y0, y1),
            interpolation="nearest",
        )
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03, extend=extend, label=label)
        ax.set_title(title, fontsize="medium")
        ax.set_xlabel("x" + unit)
        ax.set_ylabel("y" + unit)

    # Figure title from what the file says about the run
    parts = [metadata.get("name") or config.get("name") or Path(source).stem]
    beam = config.get("beam", {})
    if "energy" in beam:
        parts.append(f"{beam['energy']:g} eV")
    scan = f"{nx} x {ny} pixels"
    if "x" in tensors and nx > 1:
        scan += f" of {float(tensors['x'][1] - tensors['x'][0]):g} nm"
    parts.append(scan)
    if "primaries" in tensors:
        parts.append(f"≈{tensors['primaries'].mean():.0f} primaries per pixel")
    fig.suptitle(", ".join(parts))
    fig.tight_layout()
    return fig
