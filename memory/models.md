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
- **instrument** — ideal silicon photoelectric QE from xraylib `CS_Photo`
  (0.1–100 keV, exact K edge); no filter or dead layer yet. Bare area.
  `lobster_effective_area` is a superseded placeholder (blind to focal length).
- **lobster** — first-principles square-pore optic, on-axis point source:
  graze arcsin(x/2f); per-axis reflection count floor(u) or floor(u)+1 with
  u = (L/d)·tanθ; odd–odd → spot, odd–even → arms; separable over a
  rectangular aperture; Fresnel + Debye–Waller with xraylib constants.
  Spot ≈ hypot(d, 2f·slope_error). No off-axis, shear, tile gaps, or QE.
  Built independently of GECOSAX; its 1.5 cm² (6U, 30 cm) agrees with our
  0.8–1.4 cm² range (ideal Ir, 2026-11-12 demo).
- **background/models** — CXB (Gruber 1999, ≥ 3 keV), Earth albedo (Churazov
  2008), dark Earth (Ajello 2008 shape, caller-supplied normalization),
  particle background (flat + Gaussian lines, caller-calibrated), solar
  thermal continuum normalized to GOES flux (1 W/m² = 6.2415×10¹¹
  keV cm⁻² s⁻¹; was 10⁶ too low before 2026-11-12).
- **background/transport** — single-scattered sunlight (Rayleigh + Compton,
  xraylib dry air) through MSIS density.
- **sensitivity** — trapezoid counts, Helene (1983) Poisson upper limit,
  g limit = g × (limit / counts)^¼.
- **orbit_counts** — ties all of the above over an orbit; observes only in
  eclipse with Sun tangent altitude < 120 km.
- **physics** — legacy duplicate of `instrument.silicon_qe`; slated for removal.
