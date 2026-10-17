# ENDF/B-VII.1 data

These three files are the unmodified ENDF/B-VII.1 atomic relaxation, electro-atomic
and photo-atomic sublibraries, as distributed by the National Nuclear Data Center
(NNDC) at Brookhaven National Laboratory:

| File               | Source URL                                                            | SHA-1                                      |
|--------------------|-----------------------------------------------------------------------|--------------------------------------------|
| `atomic_relax.zip` | https://www.nndc.bnl.gov/endf-b7.1/zips/ENDF-B-VII.1-atomic_relax.zip | `8f1d2aa39b0d2705635a7d474cf05d1c79b13564` |
| `electrons.zip`    | https://www.nndc.bnl.gov/endf-b7.1/zips/ENDF-B-VII.1-electrons.zip    | `6c532bcf3f650c2a696574306d85e01fdda24b15` |
| `photoat.zip`      | https://www.nndc.bnl.gov/endf-b7.1/zips/ENDF-B-VII.1-photoat.zip      | `ae0ba033e0ee01e6c06c07fa6eb9de552ef139f7` |

They are included so that cstool runs without network access. cstool verifies the
checksums above (from `../endf_sources.json`) before use.

Reference: M.B. Chadwick et al., "ENDF/B-VII.1 Nuclear Data for Science and
Technology: Cross Sections, Covariances, Fission Product Yields and Decay Data",
Nuclear Data Sheets 112 (2011) 2887-2996, doi:10.1016/j.nds.2011.11.002.
