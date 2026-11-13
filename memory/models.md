# Models by module

Details and full assumptions are in each docstring.

- **solar_flux** — CAST (2007) fit for the solar Primakoff spectrum; flux ∝ g².
- **conversion** — weak-mixing conversion (P ≪ 1). Uniform: P = (g B L / 2)².
  Path: sampled integral with phase from (ma² − mg²) and absorption back to
  the detector. Flux × probability ∝ g⁴.
- **geometry** — circular starting orbit, RK4 with optional J2 (axis along
  GCRS z). Default 500 km, 97.4°.
- **bfield** — IGRF-14 spherical harmonics to degree 13, valid 1900–2030.
  `cutoff_rigidity` is a tilted-dipole vertical Størmer estimate, not tracing.
- **lineofsight** — straight ray toward the Sun; stops at a 60 km shell
  (modeling cutoff, not physics).
- **atmosphere** — NRLMSIS 2.1 density via pymsis; dry-air composition for
  attenuation and photon plasma mass.
- **instrument** — ideal silicon photoelectric QE (2–50 keV table, log-log
  interpolation, no extrapolation); bare and simple lobster-eye areas.
- **background/models** — CXB (Gruber 1999, ≥ 3 keV), Earth albedo (Churazov
  2008), dark Earth (Ajello 2008 shape, caller-supplied normalization),
  particle background (flat + Gaussian lines, caller-calibrated), solar
  thermal continuum normalized to GOES flux.
- **background/transport** — single-scattered sunlight (Rayleigh + Compton,
  xraylib dry air) through MSIS density.
- **sensitivity** — trapezoid counts, Helene (1983) Poisson upper limit,
  g limit = g × (limit / counts)^¼.
- **orbit_counts** — ties all of the above over an orbit; observes only in
  eclipse with Sun tangent altitude < 120 km.
- **physics** — legacy duplicate of `instrument.silicon_qe`; slated for removal.
