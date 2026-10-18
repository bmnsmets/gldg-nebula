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


# Make a histogram of energies
N_bins = 4 * int(np.max(data["E"]))
spectrum, bin_edges = np.histogram(data["E"], bins=N_bins)

# Make a line plot
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.plot(bin_centers, spectrum)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity (a.u.)")
plt.show()
