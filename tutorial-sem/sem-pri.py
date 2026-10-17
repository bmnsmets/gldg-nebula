import numpy as np

# SEM image

# Parameters:
z = 35                            # starting z
xpx = np.linspace(-50, 50, 101)   # x pixels: between -50nm and +50nm, in steps of 1nm
ypx = np.linspace(-100, 100, 201) # y pixels: between -100nm and +100nm, in steps of 1nm
energy = 500                      # Beam energy, in eV
epx = 100                         # Number of electrons per pixel
sigma = 1                         # Standard deviation of Gaussian beam spot size
poisson = True                    # Whether to use Poisson shotnoise


# This is a numpy datatype that corresponds to pri files
electron_dtype = np.dtype([
    ('x',  '=f'), ('y',  '=f'), ('z',  '=f'), # Position
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'), # Direction
    ('E',  '=f'),                             # Energy
    ('px', '=i'), ('py', '=i')])              # Pixel index


# Open file
with open('sem.pri', 'wb') as file:
	# Iterate over pixels
	for i, xmid in enumerate(xpx):
		for j, ymid in enumerate(ypx):
			# Number of electrons in this specific pixel
			N_elec = np.random.poisson(epx) if poisson else epx

			# Allocate numpy buffer
			buffer = np.empty(N_elec, dtype=electron_dtype)

			# Fill with data
			buffer['x'] = np.random.normal(xmid, sigma, N_elec)
			buffer['y'] = np.random.normal(ymid, sigma, N_elec)
			buffer['z'] = z
			buffer['dx'] = 0
			buffer['dy'] = 0
			buffer['dz'] = -1
			buffer['E'] = energy
			buffer['px'] = i
			buffer['py'] = j

			# Write buffer to file
			buffer.tofile(file)
