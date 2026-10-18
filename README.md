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
for future GPUs. Native code also runs on GPUs with a higher minor version
(e.g. 8.6 code on 8.7). To choose your own set, pass a
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

PTX only loads if the NVIDIA driver is at least as new as the CUDA toolkit
that compiled it (PTX from CUDA 13.4 fails on a CUDA 13.2 driver); native code
has no such restriction within a CUDA major version. So when building for
other machines, list their architectures explicitly. If `nebula_gpu` has no
usable code for a GPU, it stops with a CUDA error message.

## Running the SEM tutorial

The files of the [SEM tutorial](https://nebula-simulator.github.io/tutorial-sem/)
are in [examples/tutorial-sem](examples/tutorial-sem). After building:

```
cd examples/tutorial-sem
python sem-pri.py
nebula_gpu sem.tri sem.pri ../../build/materials/silicon.mat ../../build/materials/pmma.mat > output.det     # or nebula_cpu_mt
python sem-analysis.py output.det
```

`tests/run_tutorial_sem.sh` runs these steps on both the GPU and the CPU, and
checks the result:

```
tests/run_tutorial_sem.sh --mat-dir build/materials /tmp/sem-run
```

## Python environment

The repository is a uv project ([pyproject.toml](pyproject.toml)); the build
runs `uv sync` to set up `.venv`, with cstool installed in editable mode. Use it through
`build/nebula-env.sh`, `.venv/bin/activate` or `uv run`, and add packages with
`uv add <package>`.


## Citing

If you publish results obtained with Nebula, please cite the papers listed in
the [documentation](https://nebula-simulator.github.io). Material files made
with cstool use ELSEPA, so also cite F. Salvat, A. Jablonski and C.J. Powell,
Comput. Phys. Commun. 165 (2005) 157, doi:10.1016/j.cpc.2004.09.006.
