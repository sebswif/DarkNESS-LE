# Project

**Goal:** forecast what a low-Earth-orbit X-ray detector would see when looking
at the Sun through Earth's magnetic field: solar axions converting to X-rays
in the geomagnetic field, plus the backgrounds.

**Pipeline:** solar axion flux → orbit and Sun geometry → IGRF field along the
line of sight → conversion probability (with air mass and absorption) →
detector response → signal and background counts → coupling limit.
`functions/orbit_counts.simulate_counts` runs the whole chain.

**Target:** soft X-rays from solar axion/ALP conversion in Earth's field.

**Study type:** trade and feasibility study of a CubeSat payload with a
generic CCD, comparing:

1. Wide field of view, no optics (bare/collimated CCD).
2. Lobster-eye optics, sized to the bus. Long axis sets focal length.

| Bus | Layout (U) | Approx. envelope | Focal length available |
|---|---|---|---|
| 6U | 2×3×1 | 22.6 × 34 × 10 cm | ~30 cm |
| 8U | 2×4×1 | 22.6 × 45 × 10 cm | ~40 cm |
| Linear 6U | 6×1×1 | 10 × 10 × 68 cm | ~60 cm |

Linear 6U is an option because deployers such as NanoRacks accommodate it.

First sizing pass (2026-11-12, `functions/lobster.py`, ideal smooth Ir,
d = 40 µm, w = 8 µm, best L/d = 60; axion-flux-weighted 0.5–10 keV spot
area, no QE): 6U 20×9 cm → 0.8 cm²; 8U 20×9 cm → 1.4 cm²; linear 6U
9×9 cm → 2.9 cm². Area grows ~f²; at 2–3 keV only ~2–3 cm of aperture
half-width reflects, so wide apertures are mostly dead glass.

**Prior work:** the sibling project `../gecosax-cubesat` has a validated
optics trade (bare vs lobster at f ≤ 100 cm) and a geometric lobster model.

**Out of scope (not modeled):** aurora, fluorescence, multiple scattering,
SAA/trapped-particle bursts, activation, detector energy redistribution,
saturation, Poisson event draws. Drag and third-body forces in the orbit.

**Status:** forecasting tool, not a calibrated background prediction. Most
outputs are not yet checked against published values — see
[references.md](references.md).
