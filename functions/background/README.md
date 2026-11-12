# Background spectra and transport

`models.py` supplies `cxb`, `dark_earth`, `nxb`, and `solar_spectrum`.
`earth_albedo` and measured `particle_background` remain alternative calibration
utilities; the orbit pipeline does not add overlapping Earth models together.
`transport.py` supplies `ray_column` and `solar_scattered`. Ordinary package
exports replace the former file/package name collision.

`ray_column` integrates MSIS mass density along WGS84 rays. It returns atmospheric
columns in g/cm^2 and path profiles, including solid-Earth obstruction. A blocked
ray's finite column covers the visible atmosphere only: set its through-Earth
transmission to zero. The old exponential `transmission()` was removed.

`solar_scattered` integrates incoming solar illumination, differential Rayleigh
and Compton scattering, and outgoing attenuation. Both attenuation and scattering
use xraylib's NIST dry-air mass mixture. Compton recoil and its energy Jacobian
are included, but atomic momentum broadening is not. Illumination uses parallel
rays from the Sun center; the orbit loop uses a finite disk for direct sunlight.
At momentum transfers below 0.001 inverse Angstrom, the incoherent scattering
factor is continued proportional to momentum transfer squared, avoiding
xraylib's spline-domain limit and approaching zero in the forward direction.
Incoming columns are adaptively interpolated along each viewing ray with
`column_tolerance=0.001` log-column error control, checked at segment midpoints.
Set this to zero to evaluate every incoming path directly as a reference.
Every evaluated atmospheric column still uses the requested path spacing.

`background_budget(e, exposure, components)` now integrates a dictionary of
**already folded detector differential rates** [counts/s/keV]. It returns
component counts and `total`. Do not apply area, QE or solid angle again:

```python
from functions.background import background_budget
counts = background_budget([3, 10], exposure=100,
                           components={"particle": 0.01, "cxb": [0.02, 0.005]})
```

For a full orbit use `functions.orbit_counts.simulate_counts`; the main README
contains a runnable example and output descriptions.

## Physical limitations

- Fixed dry-air composition at MSIS density is an approximation above the
  well-mixed atmosphere. Activity inputs are measured values or explicit assumptions.
- Solar spectrum is a thermal continuum normalized to GOES, without solar lines.
- Dark Earth is a uniform fitted spectrum, uncalibrated below 15 keV. Its required
  normalization and the detector-particle parameters must be supplied explicitly.
- Auroras, atmospheric fluorescence, repeated scattering, trapped-particle bursts,
  SAA, activation, detector redistribution and saturation are excluded.
- Scattering integrates the atmosphere up to a configurable 1000-km geodetic
  ceiling. Terms attenuated by more than 50 optical depths at every output energy
  are skipped as a numerical floor (<2e-22); this is not an observing cut.
- Single scattering requires validation, particularly when scattering optical
  depth is appreciable. Refine energy, ray, angular and time sampling to establish
  numerical convergence separately from physical model accuracy.

References: [xraylib API](https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions),
[Ajello dark-Earth spectrum](https://arxiv.org/abs/0808.3377), and the source
references in each function's docstring.
