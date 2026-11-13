# Project

**Goal:** forecast what a low-Earth-orbit X-ray detector would see when looking
at the Sun through Earth's magnetic field: solar axions converting to X-rays
in the geomagnetic field, plus the backgrounds.

**Pipeline:** solar axion flux → orbit and Sun geometry → IGRF field along the
line of sight → conversion probability (with air mass and absorption) →
detector response → signal and background counts → coupling limit.
`functions/orbit_counts.simulate_counts` runs the whole chain.

**Out of scope (not modeled):** aurora, fluorescence, multiple scattering,
SAA/trapped-particle bursts, activation, detector energy redistribution,
saturation, Poisson event draws. Drag and third-body forces in the orbit.

**Status:** forecasting tool, not a calibrated background prediction. Most
outputs are not yet checked against published values — see
[references.md](references.md).
