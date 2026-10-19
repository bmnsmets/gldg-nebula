# Nebula (GLDG fork)

This is the Nebula Monte Carlo simulator for electron-matter interaction. 
Full instructions on how to use this software are in the
[documentation](https://nebula-simulator.github.io).

This fork is self-contained: besides the simulator, it includes
[cstool](third_party/cstool), which generates the material files Nebula needs,
and builds ELSEPA, which cstool depends on. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the origin and licences of
the bundled components.

## Requirements

- cmake 3.18 or newer
- A C++17 compiler, e.g. gcc 8 or newer
- HDF5 1.8.13 or newer, including the high-level libraries (e.g. `libhdf5-dev` on Debian/Ubuntu)
- A Fortran compiler, e.g. gfortran, to build ELSEPA
- [uv](https://docs.astral.sh/uv/), for the Python environment. It downloads
  Python 3.14 if needed.
- Optional: CUDA 11.0 or newer, for the GPU version (12.8 or newer for Blackwell GPUs)
- Internet access during the first build, to download ELSEPA, or
  [a local copy of the archive](third_party/elsepa/README.md)

## Building

```
cmake -S . -B build
cmake --build build -j
```

This builds:
- `nebula_gpu` (if a CUDA compiler is found), `nebula_cpu_mt` and `nebula_cpu_edep` in `build/bin`
- ELSEPA in `build/elsepa`
- the Python environment, with cstool, in `.venv` (see [below](#python-environment))
- a material file for each parameter file in
  [third_party/cstool/data/materials](third_party/cstool/data/materials), in
  `build/materials`. This is the slowest step: a few minutes and about 550 MB
  per material.

Options for the first `cmake` command (they are remembered for later builds):

| Option | Effect |
|---|---|
| `-DNEBULA_MATERIALS="silicon;pmma"` | Generate only these materials (default `all`; empty for none) |
| `-DCMAKE_CUDA_ARCHITECTURES="..."` | GPU architectures to compile for (see below) |
| `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` | CUDA compiler to use. Default: from `$CUDACXX`, `PATH`, `$CUDA_HOME` or the newest `/usr/local/cuda*` |
| `-DNEBULA_ENABLE_CUDA=OFF` | CPU version only |
| `-DELSEPA_ZIP=FILE` | Build ELSEPA from a local copy of the archive |
| `-DNEBULA_PYTHON_ENV=OFF` | No Python environment (and no uv needed) |

### GPU architectures

By default, `nebula_gpu` contains native code for compute capabilities 7.5,
8.0, 8.6, 8.9, 9.0, 10.0 and 12.0 (Turing to Blackwell, as far as your CUDA
compiler supports them), plus PTX for the newest, which the driver can compile
for future GPUs. To choose your own set, pass a
[CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
list:

```
# native code + PTX for each
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;90;120"          

# native code only
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80-real;120-real"   

# only the GPUs in this machine
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native               
```


## Python environment

The repository is a uv project ([pyproject.toml](pyproject.toml)); the build
runs `uv sync` to set up `.venv`, with cstool installed in editable mode. Use it through
 `source .venv/bin/activate` or `uv run`, and add packages with
`uv add <package>`.


## Running the z-map examples

[examples/z-map](examples/z-map) contains simulations of SEM images of
surfaces described by a height map (z-map): a greyscale PNG in which black is
the lowest and white the highest point. Each example is a run file (TOML) next
to its PNG:

| Run file | Geometry | Material |
|---|---|---|
| [cell65.toml](examples/z-map/cell65.toml) | [cell65-zmap.png](examples/z-map/cell65-zmap.png) | copper |
| [fins.toml](examples/z-map/fins.toml) | [fins-zmap.png](examples/z-map/fins-zmap.png) | silicon |

After [building](#building) (the material files used must be in
`build/materials`), run an example with:

```
uv run nebula examples/z-map/cell65.toml
```

This turns the PNG into a triangle mesh, makes the primary electrons for the
beam scan, runs `nebula_gpu` (or `nebula_cpu_mt` if there is no GPU; `--cpu`
forces it), applies the detectors of the run file, and saves next to it:

- `cell65.safetensors`: the yield (detected electrons per primary electron)
  per pixel of each detector, the height of the surface at each beam position
  (`zmap`), the number of primary electrons per pixel, and the run file itself;
- `cell65-plot.png`: a plot of the surface height and of each detector's
  yield map.

To make the plot again, for example of only some detectors:

```
uv run nebula plot examples/z-map/cell65.safetensors --detectors se bse
```

`uv run nebula -h` and `uv run nebula plot -h` list all options. To load the
results in Python:

```python
from safetensors.numpy import load_file

results = load_file("examples/z-map/cell65.safetensors")
se = results["se"]  # yield per pixel, indexed [ix, iy], x to the right and y up
```

### Run files

A run file has four parts; the comments in the examples explain each value:

- `[geometry.zmap]`: the PNG, its pixel size, the heights of black and white,
  the size of the simulated box, the depth of the terminator plane (electrons
  that reach it are removed; a few times the electron range at the beam energy
  is enough), the height of the detector plane, and the material, by the name
  of its file in `build/materials`. Instead, `[geometry]` can name a
  ready-made triangle file (`file = "geometry.tri"`) and its materials
  (`materials = ["silicon", "pmma"]`, for material ids 0, 1, ...), as in
  [line-space-00.toml](examples/line-space/line-space-00.toml). It must have a
  detector plane; `[beam] step` is then required, and the scan defaults to the
  whole geometry.
- `[beam]`: the energy and starting height of the primary electrons, the
  number of electrons per pixel, the size of the beam spot, and the scanned
  area (`x`, `y`: its outer edges in nm; `step`: the pixel size).
- `[detectors.<name>]`: detectors that select from the electrons reaching the
  detector plane, each giving one yield map. Every filter is a range
  `[min, max)` (`inf` for no upper limit) on the energy (eV), the angle
  between the electron's direction and the beam axis (`polar`, degrees), the
  direction around the axis (`azimuth`, degrees from +x, counter-clockwise),
  or the position on the detector plane relative to the beam (`dx`, `dy`, `r`,
  nm). For example, secondary electrons are `energy = [0, 50]`, and a BSE
  quadrant detector is `energy = [50, inf]` with an `azimuth` quarter and a
  `polar` range set by the detector's hole and outer radius.

To simulate your own surface, copy one of the examples, point `file` to your
PNG and adjust the values.


## Citing

If you publish results obtained with Nebula, please cite the papers listed in
the [documentation](https://nebula-simulator.github.io). Material files made
with cstool use ELSEPA, so also cite F. Salvat, A. Jablonski and C.J. Powell,
Comput. Phys. Commun. 165 (2005) 157, doi:10.1016/j.cpc.2004.09.006.
