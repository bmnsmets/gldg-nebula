import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Find out which file to open
if len(sys.argv) < 2:
	print("No output file provided")
	sys.exit()
filename = sys.argv[1]
if not os.path.exists(filename):
	print("File {} cannot be found".format(filename))
	sys.exit()


# This is a numpy datatype that corresponds to output files
electron_dtype = np.dtype([
    ('x',  '=f'), ('y',  '=f'), ('z',  '=f'), # Position
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'), # Direction
    ('E',  '=f'),                             # Energy
    ('px', '=i'), ('py', '=i')])              # Pixel index

# Open the output file
data = np.fromfile(filename, dtype=electron_dtype)
print("Number of electrons detected: {}".format(len(data)))


# Make a histogram of pixel indices
xmin = data['px'].min()
xmax = data['px'].max()
ymin = data['py'].min()
ymax = data['py'].max()
H, xedges, yedges = np.histogram2d(data['px'], data['py'],
	bins = [
		np.linspace(xmin-.5, xmax+.5, xmax-xmin+2),
		np.linspace(ymin-.5, ymax+.5, ymax-ymin+2)
	])

# Make a plot
plt.imshow(H.T, cmap='gray', vmin=0)
plt.colorbar()
plt.xlabel('x pixel')
plt.ylabel('y pixel')
pngname = os.path.splitext(filename)[0] + '.png'
plt.savefig(pngname, dpi=150, bbox_inches='tight')
print("Image saved to {}".format(pngname))
plt.show()
