"""From detected electrons to images: detector filters, yield per pixel, plots."""

import math
import os
import threading
import tomllib
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from .config import FILTERS, RESERVED_NAMES, Detector

__all__ = [
    "DetectorCounter",
    "detector_mask",
    "electron_quantities",
    "plot_results",
    "yield_images",
]


def electron_quantities(
    electrons: np.ndarray,
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    keys: Iterable[str] = FILTERS,
) -> dict[str, np.ndarray]:
    """The values detectors filter on (see `nebula.config`), per electron.

    Args:
        electrons: Detected electrons (`ELECTRON_DTYPE`); ``px``, ``py`` index
            xs and ys.
        xs, ys: Pixel centres (beam positions) in nm.
        keys: Which values to compute; by default all.

    Returns:
        float64 arrays by name: ``energy`` (eV), ``dx``, ``dy``, ``r`` (nm,
        relative to the beam), ``polar``, ``azimuth`` (degrees).
    """
    keys = set(keys)
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    quantities = {}
    if "energy" in keys:
        quantities["energy"] = electrons["E"].astype(np.float64)
    if keys & {"dx", "r"}:
        dx = electrons["x"] - xs[electrons["px"]]
    if keys & {"dy", "r"}:
        dy = electrons["y"] - ys[electrons["py"]]
    if "dx" in keys:
        quantities["dx"] = dx
    if "dy" in keys:
        quantities["dy"] = dy
    if "r" in keys:
        quantities["r"] = np.hypot(dx, dy)
    if keys & {"polar", "azimuth"}:
        direction = np.column_stack(
            [electrons["dx"], electrons["dy"], electrons["dz"]]
        ).astype(np.float64)
    if "polar" in keys:
        length = np.linalg.norm(direction, axis=1)
        cos_polar = np.clip(direction[:, 2] / length, -1, 1)
        quantities["polar"] = np.degrees(np.arccos(cos_polar))
    if "azimuth" in keys:
        azimuth = np.arctan2(direction[:, 1], direction[:, 0])
        quantities["azimuth"] = np.degrees(azimuth) % 360
    return quantities


def detector_mask(detector: Detector, quantities: dict[str, np.ndarray]) -> np.ndarray:
    """Which electrons pass all of a detector's filters [min, max)."""
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


def _count_selected(
    electrons: np.ndarray,
    detectors: Sequence[Detector],
    xs: np.ndarray,
    ys: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Electrons each detector keeps per pixel, shape (detectors, pixels)."""
    keys = {key for d in detectors for key in FILTERS if getattr(d, key) is not None}
    quantities = electron_quantities(electrons, xs, ys, keys)
    pixel = np.ravel_multi_index((electrons["px"], electrons["py"]), shape)
    size = shape[0] * shape[1]
    counts = np.empty((len(detectors), size), dtype=np.int64)
    for i, detector in enumerate(detectors):
        if any(getattr(detector, key) is not None for key in FILTERS):
            selected = pixel[detector_mask(detector, quantities)]
        else:
            selected = pixel
        counts[i] = np.bincount(selected, minlength=size)
    return counts


class DetectorCounter:
    """Counts the electrons each detector keeps per pixel, batch by batch.

    Batches are counted on several threads; the result does not depend on how
    the electrons were split into batches. Use it in a ``with`` block, or call
    `close` when done.

    Args:
        detectors: The detectors.
        xs, ys: Pixel centres in nm.
        chunk_size: Electrons counted at a time.
        workers: Number of threads; by default up to 8.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        *,
        chunk_size: int = 2**16,
        workers: int | None = None,
    ) -> None:
        self.detectors = list(detectors)
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.shape = (len(self.xs), len(self.ys))
        if workers is None:
            # More threads do not help: counting is limited by memory speed
            workers = max(1, min(8, os.cpu_count() or 1))
        self._chunk_size = chunk_size
        self._pool = ThreadPoolExecutor(max_workers=workers)
        # Chunks waiting for a thread. Limited, so that a fast producer waits
        # instead of queueing all its electrons
        self._queued: deque[Future] = deque()
        self._max_queued = 4 * workers
        self._pending: list[np.ndarray] = []  # electrons not yet in a chunk
        self._pending_size = 0
        # Each thread adds to its own counts, which are summed at the end
        self._local = threading.local()
        self._totals: list[np.ndarray] = []
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Stop the threads, discarding chunks not yet counted."""
        self._pool.shutdown(wait=True, cancel_futures=True)

    def add(self, electrons: np.ndarray) -> None:
        """Count a batch of detected electrons (`ELECTRON_DTYPE`)."""
        if len(electrons) == 0:
            return
        self._pending.append(electrons)
        self._pending_size += len(electrons)
        if self._pending_size < self._chunk_size:
            return
        if len(self._pending) == 1:
            batch = self._pending[0]
        else:
            batch = np.concatenate(self._pending)
        full = len(batch) - len(batch) % self._chunk_size
        for start in range(0, full, self._chunk_size):
            self._submit(batch[start : start + self._chunk_size])
        self._pending = [batch[full:]] if full < len(batch) else []
        self._pending_size = len(batch) - full

    def counts(self) -> np.ndarray:
        """Finish counting; int64 counts of shape (detectors, pixels), pixels in [ix, iy] order."""
        for electrons in self._pending:
            self._submit(electrons)
        self._pending, self._pending_size = [], 0
        while self._queued:
            self._queued.popleft().result()
        total = np.zeros((len(self.detectors), self.shape[0] * self.shape[1]), np.int64)
        for counts in self._totals:
            total += counts
        return total

    def images(self, primaries: np.ndarray) -> dict[str, np.ndarray]:
        """Yield per pixel of each detector: electrons kept per primary.

        Args:
            primaries: Primary electrons per pixel, shape (len(xs), len(ys)).

        Returns:
            A float32 image ``[ix, iy]`` per detector; NaN where there were no
            primaries.
        """
        images = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for detector, counts in zip(self.detectors, self.counts(), strict=True):
                images[detector.name] = np.where(
                    primaries > 0, counts.reshape(self.shape) / primaries, np.nan
                ).astype(np.float32)
        return images

    def _submit(self, chunk: np.ndarray) -> None:
        # Raise errors of finished chunks; wait if too many are queued
        while self._queued and (
            self._queued[0].done() or len(self._queued) >= self._max_queued
        ):
            self._queued.popleft().result()
        self._queued.append(self._pool.submit(self._count, chunk))

    def _count(self, chunk: np.ndarray) -> None:
        counts = _count_selected(chunk, self.detectors, self.xs, self.ys, self.shape)
        total = getattr(self._local, "total", None)
        if total is None:
            total = self._local.total = np.zeros_like(counts)
            with self._lock:
                self._totals.append(total)
        total += counts


def yield_images(
    electrons: np.ndarray,
    detectors: tuple[Detector, ...] | list[Detector],
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    primaries: np.ndarray,
    *,
    _chunk_size: int = 2**16,
) -> dict[str, np.ndarray]:
    """Yield per pixel of each detector, for all electrons at once.

    See `DetectorCounter` for electrons that come in batches.

    Args:
        electrons: Detected electrons (`ELECTRON_DTYPE`).
        detectors: The detectors.
        xs, ys: Pixel centres in nm.
        primaries: Primary electrons per pixel, shape (len(xs), len(ys)).

    Returns:
        A float32 image ``[ix, iy]`` per detector; NaN where there were no
        primaries.
    """
    with DetectorCounter(detectors, xs, ys, chunk_size=_chunk_size) as counter:
        counter.add(electrons)
        return counter.images(primaries)


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
    """Plot a results file: the z-map, then the yield map of each detector.

    Reads a file from `nebula.simulate.save_results`. Detectors appear in the
    order of the run file. Yield maps are greyscale (like SEM images), scaled
    from 0 to the given percentile so that a few noisy pixels do not set the
    scale; the z-map is in colour over its full range. Pixels without
    primaries are red.

    Args:
        source: Path of the safetensors file.
        detectors: Names of the detectors to show, in this order. By default,
            all of them.
        cmap: Colour map of the yield maps.
        zmap_cmap: Colour map of the z-map.
        percentile: Where each yield map's colour scale ends; 100 for the maximum.
        columns: Number of panels per row.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If a detector in ``detectors`` is not in the file, or there
            is nothing to plot.
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
