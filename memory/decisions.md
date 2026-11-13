# Design rules

- **No hidden normalizations.** Uncalibrated inputs (dark Earth, particle
  level, solar GOES, lines) must be passed in explicitly.
- **Reject, don't extrapolate.** Fits and tables raise outside their valid
  range (e.g. CXB < 3 keV, Si table outside 2–50 keV, IGRF outside 1900–2030).
- **Weak mixing only.** Conversion assumes P ≪ 1.
- **Same g everywhere.** Production and conversion use the same coupling.
- **Effective area includes QE.** Never apply QE twice; never apply photon
  QE or area to particle background.
- **Expected counts only.** No random Poisson events are drawn.
- **Offline by default where possible.** Astropy uses bundled IERS data; MSIS
  needs `f107`, `f107a`, `ap` supplied to avoid downloading activity data.
- **Short physical keyword names** (`e`, `g`, `b`, `r`, `k`, `t`, `h`, `inc`).
- **Docstrings state units, shapes, assumptions, and a reference.**
- **Dependencies:** `requirements.txt` mirrors `pyproject.toml` dependencies
  plus pytest. Change both together and update its "last tested" line.
- **Tests:** simple sanity checks now; published-value checks are added only
  with a cited source (see `pytests/README.md`).
