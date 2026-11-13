# References and verification

**verified** = a test compares against a published number.
**pending** = cited in code, but no test checks a published number yet
(placeholder tests in `pytests/` are marked `reference` and skipped).

| Source | Used in | Status |
|---|---|---|
| CAST (2007), doi:10.1088/1475-7516/2007/04/010 | solar_flux | verified (total 3.75e11, mean 4.20, peak 3.0 keV) — `tests/` |
| Raffelt & Stodolsky (1988), doi:10.1103/PhysRevD.37.1237 | conversion (uniform) | verified (9 T, 9.26 m → 1.70e-17) — `tests/` |
| IGRF-14, doi:10.5281/zenodo.14012302; NOAA/BGS pyIGRF14 | bfield | verified (pyIGRF14 values) — `tests/` |
| Davoudiasl & Huber (2006), hep-ph/0509293 | conversion (path) | pending |
| Davoudiasl & Huber, arXiv:0804.3543 | sensitivity | pending |
| Helene (1983), doi:10.1016/0167-5087(83)90709-3 | sensitivity | pending (only n = 0 analytic case checked) |
| NIST XCOM Si photoelectric (bundled table, 2–50 keV) | instrument (via xraylib) | verified (xraylib within 1%) — `pytests/` |
| Si photoabsorption below 2 keV (NIST XCOM 1–1.8 keV, Henke/CXRO) | instrument | pending |
| xraylib, Schoonjans et al. (2011), doi:10.1016/j.sab.2011.09.011 | instrument, background/transport | source of cross sections |
| NIST XrayMassCoef (dry air) | atmosphere | pending |
| NRLMSIS 2.1 (pymsis) | atmosphere | pending |
| van Bibber et al. (1989), doi:10.1103/PhysRevD.39.2089 | atmosphere (plasma mass) | pending |
| Gruber et al. (1999), astro-ph/9903492 | background (CXB) | pending |
| Churazov et al. (2008), astro-ph/0608252 | background (albedo) | pending |
| Ajello et al. (2008), arXiv:0808.3377 | background (dark Earth) | pending |
| Tawa et al. (2008), arXiv:0803.0616 | background (particle) | pending |
| Smart & Shea (2005), doi:10.1016/j.asr.2004.09.015 | bfield (cutoff) | pending |
| Angel (1979), doi:10.1086/157397 | lobster geometry | pending |
| Als-Nielsen & McMorrow (2011), ch. 3 (Fresnel, Debye–Waller) | lobster reflectance | analytic checks only (R = 1 below θc; exact value at 2θc) |
| Published MPO effective area (e.g. SVOM MXT) | lobster | pending |

When a reference value is sourced: put the number and citation in the test,
remove the skip, and change the status here to verified.
