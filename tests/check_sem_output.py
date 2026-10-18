"""Sanity checks on the output of the SEM tutorial (examples/tutorial-sem/).

Usage: python check_sem_output.py sem.pri output.det [output2.det ...]

For each detector file, this builds the SEM image the same way as
examples/tutorial-sem/sem-analysis.py, saves it as <output>.png, and checks that
 - electrons were detected, with a plausible secondary electron yield;
 - the edges of the PMMA line (x = -20 nm and x = +20 nm) are brighter than
   the flat silicon substrate, which is the main feature of the image.
If more than one detector file is given (e.g. from nebula_gpu and
nebula_cpu_mt), the yields must also agree with each other.

Exits with a nonzero status if any check fails.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same layout as in sem-pri.py / sem-analysis.py
electron_dtype = np.dtype(
    [
        ("x", "=f"),
        ("y", "=f"),
        ("z", "=f"),  # Position
        ("dx", "=f"),
        ("dy", "=f"),
        ("dz", "=f"),  # Direction
        ("E", "=f"),  # Energy
        ("px", "=i"),
        ("py", "=i"),
    ]
)  # Pixel index

# Pixel grid of sem-pri.py: x from -50 to 50 nm, y from -100 to 100 nm, 1 nm steps
N_PX, N_PY = 101, 201


def x_to_px(x):
    return round(x + 50)


failures = []


def check(condition, message):
    print("  [{}] {}".format("ok" if condition else "FAIL", message))
    if not condition:
        failures.append(message)


def analyse(det_file, n_primaries):
    print(f"{det_file}:")
    data = np.fromfile(det_file, dtype=electron_dtype)
    H, _, _ = np.histogram2d(
        data["px"],
        data["py"],
        bins=[np.arange(N_PX + 1) - 0.5, np.arange(N_PY + 1) - 0.5],
    )

    plt.figure()
    plt.imshow(H.T, cmap="gray", vmin=0)
    plt.colorbar()
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    png = os.path.splitext(det_file)[0] + ".png"
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"  image saved to {png}")

    SE_yield = len(data) / n_primaries
    check(len(data) > 0, f"detected {len(data)} electrons")
    check(np.all(data["E"] < 50), "all detected electrons are below 50 eV")
    check(0.05 < SE_yield < 5, f"SE yield {SE_yield:.3f} is plausible")

    # Mean signal per pixel column, normalised to the number of primaries per pixel
    column = H.mean(axis=1) / (n_primaries / (N_PX * N_PY))
    substrate = np.concatenate(
        [column[: x_to_px(-30)], column[x_to_px(30) + 1 :]]
    ).mean()
    line_top = column[x_to_px(-10) : x_to_px(10) + 1].mean()
    left_edge = column[x_to_px(-22) : x_to_px(-18) + 1].max()
    right_edge = column[x_to_px(18) : x_to_px(22) + 1].max()
    print(
        f"  column yield: substrate {substrate:.3f}, line top {line_top:.3f}, edges {left_edge:.3f} / {right_edge:.3f}"
    )
    check(
        left_edge > 1.2 * substrate and right_edge > 1.2 * substrate,
        "line edges are brighter than the substrate",
    )
    check(
        left_edge > 1.2 * line_top and right_edge > 1.2 * line_top,
        "line edges are brighter than the top of the line",
    )

    return SE_yield


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    pri_file = sys.argv[1]
    n_primaries = os.path.getsize(pri_file) // electron_dtype.itemsize
    print(f"{n_primaries} primary electrons in {pri_file}")

    yields = [analyse(det_file, n_primaries) for det_file in sys.argv[2:]]

    if len(yields) > 1:
        print("Comparing runs:")
        spread = (max(yields) - min(yields)) / np.mean(yields)
        check(spread < 0.03, f"SE yields agree within 3% (spread {spread:.2%})")

    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
