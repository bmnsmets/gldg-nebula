# Nebula

This is the Nebula Monte Carlo simulator for electron-matter interaction. 
Full instructions on how to use this software are in the
[documentation](https://nebula-simulator.github.io).

This forked repo is self-contained. 
Besides the simulator, it includes [cstool](third_party/cstool), which generates the material files Nebula needs,
and the recipe to build ELSEPA, which cstool depends on. A single script builds
all of it. 
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the origin and licences of the bundled components.

## Requirements

- cmake 3.18 or newer
- A C++17 compiler, e.g. gcc 8 or newer
- HDF5 1.8.13 or newer, including the high-level libraries (e.g. `libhdf5-dev` on Debian/Ubuntu)
- A Fortran compiler, e.g. gfortran, to build ELSEPA
- Python 3.9 or newer with `venv`, for cstool (on Debian/Ubuntu, `python3-venv`).
  [uv](https://docs.astral.sh/uv/) is used if it is installed.
- Optional: CUDA 11.0 or newer, for the GPU version. Blackwell GPUs (sm_100, sm_120) need CUDA 12.8 or newer.
- Internet access during the first build, to download ELSEPA. You can also
  [supply the archive yourself](third_party/elsepa/README.md).

## Building

```
./build.sh
source build/nebula-env.sh
```

This builds `nebula_gpu` (if a CUDA compiler is found), `nebula_cpu_mt` and
`nebula_cpu_edep` in `build/bin`, and ELSEPA in `build/elsepa`. It also creates
a Python environment with cstool in `build/venv`. Sourcing `build/nebula-env.sh`
puts all of these on your `PATH`. Run `./build.sh --help` for all options. The
most useful ones:

| Option | Effect |
|---|---|
| `--materials "silicon pmma"` | Also generate `silicon.mat` and `pmma.mat` in `build/materials`, from the parameter files in [third_party/cstool/data/materials](third_party/cstool/data/materials). Takes a few minutes per material. |
| `--cuda-arch "LIST"` | GPU architectures to compile for (see below) |
| `--cuda-home DIR` | CUDA toolkit to use. By default `nvcc` is taken from `PATH`, `$CUDA_HOME` or the newest `/usr/local/cuda*`. |
| `--no-cuda` | CPU version only |
| `--elsepa-zip FILE` | Build ELSEPA from a local copy of the archive, without downloading |

### GPU architectures

By default, `nebula_gpu` contains native code for compute capabilities 7.5,
8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, as far as your CUDA compiler supports them.
That covers Turing, Ampere, Ada, Hopper and Blackwell. It also contains PTX for
the newest of these, which the driver can compile for future GPUs. Native code
also runs on GPUs with the same major and a higher minor compute capability
(e.g. 8.6 code runs on 8.7, and 12.0 on 12.1).

To choose your own set, pass a
[CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
list:

```
./build.sh --cuda-arch "80;90;120"            # native code + PTX for each
./build.sh --cuda-arch "80-real;120-real"     # native code only
./build.sh --cuda-arch native                 # only the GPUs in this machine
```

The PTX can only be used if the NVIDIA driver is at least as new as the CUDA
toolkit that compiled it. 
For example, PTX from CUDA 13.4 does not load with a driver that supports CUDA 13.2. 
Native code does not have this restriction within the same CUDA major version. 
So when building for other machines, list their architectures explicitly instead of relying on the PTX. 
If `nebula_gpu` has no usable code for a GPU, it stops with a CUDA error message.

Without the script, the plain CMake build works as in upstream Nebula:
`cmake -S . -B build && cmake --build build`.

## Running the SEM tutorial

The files of the [SEM tutorial](https://nebula-simulator.github.io/tutorial-sem/)
are in [tutorial-sem](tutorial-sem):

```
./build.sh --materials "silicon pmma"
source build/nebula-env.sh
mkdir sem-run && cd sem-run
cp ../tutorial-sem/* ../build/materials/*.mat .
python sem-pri.py
nebula_gpu sem.tri sem.pri silicon.mat pmma.mat > output.det     # or nebula_cpu_mt
python sem-analysis.py output.det
```

`tests/run_tutorial_sem.sh WORKDIR` runs these steps automatically, on both
the GPU and the CPU, and checks the result:

```
tests/run_tutorial_sem.sh --mat-dir build/materials /tmp/sem-run
```

## Citing

If you publish results obtained with Nebula, please cite the papers listed in
the [documentation](https://nebula-simulator.github.io). Material files made
with cstool use ELSEPA, so also cite F. Salvat, A. Jablonski and C.J. Powell,
Comput. Phys. Commun. 165 (2005) 157, doi:10.1016/j.cpc.2004.09.006.
