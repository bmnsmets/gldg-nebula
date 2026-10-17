# Third-party components

This repository is a fork of [Nebula](https://github.com/Nebula-simulator/Nebula)
(BSD 3-Clause, see [LICENSE](LICENSE)). The fork bundles the other pieces needed
to run Nebula, so that one script builds everything. They are listed below.

## cstool — `third_party/cstool/`

Generates the material files that Nebula reads.

- Upstream: https://github.com/Nebula-simulator/cstool, commit
  `0c739eb3fcc3fe5297e74c601ac4a9546db596cf` (version 1.0.7, 2024-08-21)
- Licence: BSD 3-Clause, Copyright (c) 2020, Nebula-simulator. The full text is
  in [third_party/cstool/LICENSE](third_party/cstool/LICENSE).
- Local modifications (also listed under `1.0.7+gldg` in
  [third_party/cstool/CHANGES](third_party/cstool/CHANGES)):
  - `setup.py` replaced by `pyproject.toml`, because `distutils` no longer
    exists in Python 3.12 and newer.
  - The ENDF data files are bundled, and are located with `importlib.resources`
    instead of the removed `pkg_resources` module. A download only happens if
    they are missing, and goes to a user cache folder.
  - The ELSEPA folder can be given with the `ELSEPA_DIR` environment variable.
  - A numpy warning about uninitialised memory is fixed. The results are
    unchanged.
  - The version is marked `1.0.7+gldg`.

## ENDF/B-VII.1 data — `third_party/cstool/cstool/data/endf_data/`

The atomic relaxation, electro-atomic and photo-atomic sublibraries of
ENDF/B-VII.1, from the National Nuclear Data Center (NNDC), Brookhaven National
Laboratory (https://www.nndc.bnl.gov/endf-b7.1/). The files are unmodified;
their checksums are in `endf_sources.json`.

Reference: M.B. Chadwick et al., Nuclear Data Sheets 112 (2011) 2887–2996,
doi:10.1016/j.nds.2011.11.002.

## ELSEPA — not included, downloaded at build time

cstool needs ELSEPA, by F. Salvat, A. Jablonski and C.J. Powell. ELSEPA is
licensed under the
[CPC Non-profit Use Licence](https://www.elsevier.com/about/policies-and-standards/open-access-licenses/elsevier-user/cpc),
which does not allow redistribution, so this repository contains only the build
recipe. The build downloads the original 2004 code and verifies its checksum.
Publications that use it must cite Comput. Phys. Commun. 165 (2005) 157,
doi:10.1016/j.cpc.2004.09.006. Details are in
[third_party/elsepa/README.md](third_party/elsepa/README.md).

## Tutorial files — `tutorial-sem/`

`sem.tri`, `sem-pri.py`, `sem-analysis.py` and `reels-analysis.py` come from
the Nebula documentation, https://nebula-simulator.github.io/ (Nebula-simulator).
