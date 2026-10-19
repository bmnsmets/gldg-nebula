# Agent notes for gldg-nebula

Nebula (C++/CUDA electron-matter Monte Carlo simulator), a self-contained fork:
it bundles cstool (third_party/cstool, makes material files) and builds ELSEPA.
Python tools are in python/nebula (import name `nebula`, distribution `gldg-nebula`).

## Build and run
- `cmake -S . -B build && cmake --build build -j` builds build/bin/{nebula_gpu,
  nebula_cpu_mt,nebula_cpu_edep}, the uv venv (.venv) and build/materials/*.mat
  (slow: minutes and ~550 MB per material; editing cstool sources regenerates them all).
- Python: `uv run ...` (Python 3.14, uv project in pyproject.toml).
- Simulate: `uv run nebula examples/z-map/cell65.toml` (TOML run file, see
  python/nebula/config.py) → NAME.safetensors + NAME-plot.png;
  `uv run nebula plot X.safetensors` replots.

## Checks (run before finishing)
- `uv run pytest tests/python` (no binaries needed); `uv run pytest tests/integration`
  needs the build (marker `requires_nebula`; includes build/bin/test_philox, the
  known-answer test of the GPU random number generator).
- If ruff and/or ty are available: `uvx ruff check` and `uvx ruff format` on changed files; `uvx ty check python/nebula/`.

## Layout
- python/nebula/io/: file formats — triangles.py (.tri, TRIANGLE_DTYPE),
  electrons.py (.pri/.det/deposits, ELECTRON_DTYPE), materials.py (.mat, HDF5).
- geometry.py (PNG → z-map → adaptive triangulation, height_map, plot_triangles),
  config.py (run files), beam.py, runner.py (finds binaries/materials, runs Nebula),
  imaging.py (detector filters, yields, plot_results), simulate.py, cli.py.

## Conventions
- Data are numpy structured arrays; z-maps and images are indexed [ix, iy], x right,
  y up (plot with `imshow(a.T, origin="lower")`). Lengths in nm, energies in eV.
- Vertices are float32 (Nebula is single precision). Material ids ≥ 0 index the .mat
  files; negative ids are SpecialMaterial (−123 vacuum, −126 detector, −122 mirror, …).
- Google-style docstrings in plain English. Match the surrounding code; keep functions small.
- Validate inputs with clear ValueError messages (ConfigError for run files).

## Nebula facts that matter
- The simulation domain is the triangles' bounding box + 1 nm; electrons leaving it
  are lost. Primaries always start in vacuum. The triangle normal (r1−r0)×(r2−r0)
  points from material_in to material_out.
- Detector planes detect in both directions; kinetic energy inside a material includes
  its barrier.
- To benchmark, run a binary directly on kept inputs (`nebula X.toml --keep DIR`) and
  read its "Simulation" timer. nebula_gpu is deterministic for a given seed;
  nebula_cpu_mt is not.
- nebula_gpu keeps no random-generator state: each number is Philox4x32-10 of
  (seed; particle slot, iteration, kernel, draw) (common/util/philox.h). A kernel
  that draws must handle each slot in at most one thread and pass its own kernel id.
  elastic/inelastic are limited by memory bandwidth (per-particle data is expensive);
  init (octree traversal) by dependent-load latency and warp divergence. Octree leaves
  split above 8 triangles (_split_count): 2–4 is slower, and copying triangles into
  the leaves or into an aligned layout did not help (more cache footprint).
- To profile kernels per source line: `cmake -S . -B build-prof -DCMAKE_CUDA_FLAGS=-lineinfo
  -DCMAKE_CUDA_ARCHITECTURES=120 -DNEBULA_MATERIALS="" -DNEBULA_BUILD_ELSEPA=OFF
  -DNEBULA_PYTHON_ENV=OFF` (same code as build/). ncu needs GPU counter access (sudo here).
- matplotlib 3-D sorts faces by depth only within one collection; plot_triangles
  relies on that.

## Don't
- Commit generated files (*.mat, *.pri, *.det, *.safetensors, examples/z-map/*-plot.png).
- Change vendored code in third_party/ without noting it in its CHANGES and in
  THIRD_PARTY_NOTICES.md.
