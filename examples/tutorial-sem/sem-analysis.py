import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Find out which file to open
if len(sys.argv) < 2:
    print("No output file provided")
    sys.exit()
filename = sys.argv[1]
if not os.path.exists(filename):
    print(f"File {filename} cannot be found")
    sys.exit()


# This is a numpy datatype that corresponds to output files
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

# Open the output file
data = np.fromfile(filename, dtype=electron_dtype)
print(f"Number of electrons detected: {len(data)}")


# Make a histogram of pixel indices
xmin = data["px"].min()
xmax = data["px"].max()
ymin = data["py"].min()
ymax = data["py"].max()
H, xedges, yedges = np.histogram2d(
    data["px"],
    data["py"],
    bins=[
        np.linspace(xmin - 0.5, xmax + 0.5, xmax - xmin + 2),
        np.linspace(ymin - 0.5, ymax + 0.5, ymax - ymin + 2),
    ],
)

# Make a plot
plt.imshow(H.T, cmap="gray", vmin=0)
plt.colorbar()
plt.xlabel("x pixel")
plt.ylabel("y pixel")
pngname = os.path.splitext(filename)[0] + ".png"
plt.savefig(pngname, dpi=150, bbox_inches="tight")
print(f"Image saved to {pngname}")
plt.show()
