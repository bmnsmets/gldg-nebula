# ELSEPA

cstool uses ELSEPA to compute Mott cross sections for elastic scattering:

> F. Salvat, A. Jablonski and C.J. Powell, "ELSEPA — Dirac partial-wave
> calculation of elastic scattering of electrons and positrons by atoms,
> positive ions and molecules", Comput. Phys. Commun. 165 (2005) 157–190,
> doi:[10.1016/j.cpc.2004.09.006](https://doi.org/10.1016/j.cpc.2004.09.006)

Publications that use material files generated with cstool must cite this paper.

## Why the source is not in this repository

ELSEPA (version of September 27, 2004) is distributed by the Computer Physics
Communications Program Library under the
[CPC Non-profit Use Licence](https://www.elsevier.com/about/policies-and-standards/open-access-licenses/elsevier-user/cpc).
That licence limits use to academic and other non-profit purposes. It also
forbids passing the source or executables to third parties without the authors'
written permission. This repository is public, so ELSEPA is not committed to it.

Instead, the CMake build in this folder downloads the archive when you configure
the project. It verifies the archive's SHA-256 checksum and compiles `elscata`
and `elscatm` with your Fortran compiler. By building it, you obtain ELSEPA under
the terms of the CPC licence.

| | |
|---|---|
| Download URL | `https://nebula-simulator.github.io/data/elsepa.zip` (the archive linked from the Nebula/cstool installation instructions) |
| SHA-256 | `86cb42ada162498e0a2e886e5fd6c06195b583e820b52d7b8e4dff0b276b3342` |
| Original source | CPC Program Library, program ADUS_v1_0, now on Mendeley Data: doi:[10.17632/5zzrz874tt.1](https://doi.org/10.17632/5zzrz874tt.1) |

The archive also contains Windows executables (`*.exe`), which are not used.

## Building

`./build.sh` in the repository root builds ELSEPA automatically. The results go to
`<build>/elsepa/`, together with the `z_*.den`/`z_*.dfs` data files that ELSEPA
reads at run time. cstool finds that folder through the `ELSEPA_DIR` environment
variable, which `<build>/nebula-env.sh` sets.

Offline builds: download the archive on another machine and pass it with
`./build.sh --elsepa-zip /path/to/elsepa.zip` (or
`cmake -DELSEPA_ZIP=/path/to/elsepa.zip`). The checksum is verified either way.

To skip ELSEPA entirely, use `cmake -DNEBULA_BUILD_ELSEPA=OFF`. cstool will
not work without it.
