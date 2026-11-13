# DarkNESS-LE

Solar axion flux, coherent conversion, IGRF-14 fields, and RK4 orbit propagation.
Requires Python 3.10+, NumPy 2+, and Astropy. Field, orbit, flux, and conversion
formulas are implemented locally. Astropy supplies time, ephemerides, and frames.

## Setup

From this repository directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m functions.solar_flux
.\.venv\Scripts\python.exe -m functions.conversion
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pytest
```

On Linux/macOS use `.venv/bin/python` instead. `requirements.txt` mirrors
the dependencies in `pyproject.toml` plus pytest; change both together.

## Usage

```python
import numpy as np
from functions.solar_flux import axion_flux, spectrum_summary
from functions.conversion import conversion_probability_uniform

energies_kev = np.linspace(0.1, 10.0, 100)
g_ay = 1e-10  # GeV^-1
flux = axion_flux(energies_kev, g_ay)  # cm^-2 s^-1 keV^-1
total, mean_kev, peak_kev = spectrum_summary(g_ay)
probability = conversion_probability_uniform(b=9.0, length=9.26, g=g_ay)
photon_flux = flux * probability  # cm^-2 s^-1 keV^-1
```

`b` is the transverse field in tesla and `length` is the path length in metres.
The uniform conversion model assumes a uniform field, coherent propagation, negligible
absorption, and weak mixing (probability much less than one). It does not include
finite-mass coherence loss, plasma effects, or a varying geomagnetic field.

The conversion uses 1 T = 195 eV², 1 m = 5.07e6 eV⁻¹, and
1 GeV⁻¹ = 10⁻⁹ eV⁻¹. For 9 T, 9.26 m, and 10⁻¹⁰ GeV⁻¹, the probability
is approximately 1.70 × 10⁻¹⁷. The solar spectrum uses the CAST (2007) fit.

## Field and geometry

```python
from functions.bfield import igrf_field
from functions.lineofsight import los
from functions.geometry import orbit_position, sun_direction

t = "2026-10-19T00:00:00"  # UTC
r = orbit_position(t, h=500, inc=97.4)  # ECEF km
k = sun_direction(t)  # ECEF unit vector toward Sun
b = igrf_field(r, t, report=False)  # ECEF tesla
ray = los(r, k, t, length=10000, n=501, report=False)
print(ray["amplitude_Tm"])  # |integral B_perp ds| in T m
```

Function names and ray dictionary keys are retained. Keyword inputs now use
short physical names: `e`, `g`, `b`, `length`, `r`, `k`, `t`, `n`, `h`, `inc`,
`raan`, and `phase`. Units and output shapes are defined in each docstring.

`orbit_position` integrates Cartesian position and velocity with classical RK4,
using central gravity and optional J2 acceleration (`j2=True` by default).
The initial state is a two-body circular orbit defined by `h`, `inc`, `raan`,
and `phase`; J2 then permits radius variations. The J2 axis is fixed along
GCRS z. Drag, third-body forces, and higher gravity harmonics are omitted.

`step=10.0` sets the maximum integration step in seconds. Steps shorten to
reach each requested time exactly. Times may be unsorted, repeated, or before
the epoch; output order and shape match the input. Smaller steps reduce
integration error; check convergence for the duration and accuracy needed.
RK4 uses a fixed maximum step and has no automatic error control.
References: [RK4 tableau](https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/ClassicalRungeKuttaIntegrator.html)
and [ESA J2 potential](https://gssc.esa.int/navipedia/index.php/Perturbed_Motion).

IGRF coefficients are bundled in `functions/data/igrf14coeffs.txt`, downloaded
directly from the [official NOAA/IAGA text file](https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt).
Primary data reference: [IAGA IGRF-14 (2024)](https://doi.org/10.5281/zenodo.14012302).
Downloaded October 19, 2026. SHA-256:
`8f8d88403028fc4ee92c4f38d97b46e0a87e2cfc496045b43c9e26c1d6b0903c`.

The local field calculation interpolates the Gauss coefficients between
five-year epochs, applies secular variation from 2025 through 2030, and sums
Schmidt-normalized spherical harmonics through degree 13. Dates outside
1900–2030 raise an error. Degree 1 gives the tilted dipole approximation.
Astropy frame transforms use bundled IERS data without runtime downloads.
The ray's 60 km absorption shell is a modeling cutoff, not an IGRF boundary.

Tests include the published reference values from
[NOAA/BGS pyIGRF14](https://www.ngdc.noaa.gov/IAGA/vmod/pyIGRF14.zip),
coefficient interpolation, dipole scaling, pole continuity, and ray integration.

## Atmosphere and path conversion

`functions/atmosphere.py` provides `air_density(h, lat, lon, t)` in kg/m^3
using [NRLMSIS 2.1 through pymsis](https://swxtrec.github.io/pymsis/).
This is an atmosphere model fitted to observations, not direct measurements.
Inputs are WGS84 altitude in km, geodetic latitude and longitude in degrees,
and UTC time. Scalars repeat to match arrays; arrays are matched point by
point. The output has the shared input shape.

By default, pymsis uses historical solar and geomagnetic activity data,
downloading and caching it on first use. Dates outside the available history
require explicit activity inputs. For offline or assumed conditions, supply
all of `f107` (previous day's radio flux in sfu), `f107a` (centered 81-day mean
in sfu), and `ap` (daily geomagnetic index). For example,
`air_density([60, 100, 400], 40, -105, "2020-01-01T00:00:00",
f107=150, f107a=150, ap=4)` uses assumed activity, not observed activity.
Install the added dependency with `python -m pip install -e .`.

`air_attenuation(rho, mac)` returns the intensity loss coefficient in 1/m,
using density in kg/m^3 and an energy-specific mass attenuation coefficient
in cm^2/g. At 4 keV, dry air has `mac=77.88` from
[NIST](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html).
`photon_plasma_mass(rho)` returns the effective photon mass in eV using all
dry-air electrons, appropriate for X-rays well above binding energies and
away from absorption edges; see
[van Bibber et al. (1989)](https://doi.org/10.1103/PhysRevD.39.2089).

Use the `ray` from the field example above:

```python
from functions.atmosphere import air_density, air_attenuation, photon_plasma_mass
from functions.conversion import conversion_probability_path
from astropy.coordinates import EarthLocation
import astropy.units as u

xyz = ray["path_ecef_km"]
location = EarthLocation.from_geocentric(xyz[:, 0], xyz[:, 1], xyz[:, 2], unit=u.km)
rho = air_density(location.height.to_value(u.km), location.lat.deg,
                  location.lon.deg, t)  # historical activity; may download data
p = conversion_probability_path(
    s=ray["distance_km"] * 1000,  # m, detector first
    b=ray["B_perp_ecef_T"],       # signed vectors in the same frame, T
    e=4.0, g=1e-10, ma=1e-4,     # keV, GeV^-1, eV
    mg=photon_plasma_mass(rho),   # eV
    attenuation=air_attenuation(rho, mac=77.88),  # 1/m at 4 keV
)
```

The path function includes phase changes, field direction, and photon loss
between each conversion point and the detector. `mg` and `attenuation` default
to zero (vacuum); each can be a scalar or one value per path sample. `b` can
also be a signed scalar field per sample for a fixed transverse direction.
The result is a dimensionless probability in the weak-conversion limit
(much less than one). Increase the sample count until the result converges.
Only the supplied path is included, so `los` still ends at its 60 km cutoff.
The sampled integral generalizes the constant-medium expression in
[Davoudiasl & Huber (2006), Eqs. (2)-(3)](https://arxiv.org/pdf/hep-ph/0509293).

MSIS also models changing atmospheric composition. The attenuation and photon
mass helpers still assume dry-air composition; using MSIS mass density alone
does not account for composition changes in the upper atmosphere.

## Instrument response

`functions/instrument.py` provides `silicon_qe`, `bare_effective_area`, and
`lobster_effective_area`, with input types and units in their docstrings.
Silicon efficiency uses the bundled photoelectric absorption table from
2 to 50 keV, with log-log interpolation and no extrapolation. The default
active silicon depth is 250 micrometres. It assumes ideal charge collection;
filter/window transmission can be supplied separately to the area functions.
See [NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html).

```python
from functions.instrument import silicon_qe, bare_effective_area, lobster_effective_area

qe = silicon_qe(e=10.0, thickness=250.0)
bare = bare_effective_area(e=10.0, area=100.0)  # cm^2
# Illustrative assumptions, not a measured telescope response:
lobster = lobster_effective_area(
    e=10.0, area=100.0, reflectivity=0.8,
    open_fraction=0.6, focus_fraction=0.25,
)  # cm^2
```

For the bare detector, `area` is exposed detector area. For the lobster optic,
it is projected entrance aperture area. The lobster model counts the central
spot with two reflections, so reflectivity enters squared. `focus_fraction`
is the geometric fraction of open-pore rays reaching that spot before
reflection losses; supply it from the actual geometry or calibration.
Cross arms and unreflected light are excluded. This is a simplified model
based on [Angel (1979)](https://doi.org/10.1086/157397), not a ray tracer.

## Backgrounds and magnetic cutoff

`functions/background/` retains these reference functions:

- `cxb(e)` returns cosmic X-ray photon intensity in photons/cm^2/s/sr/keV,
  using [Gruber et al. (1999)](https://arxiv.org/abs/astro-ph/9903492).
  It converts the paper's energy intensity to photon intensity by dividing
  by energy in keV. Valid inputs are 3 keV to 100 GeV; it rejects lower
  energies rather than extending the fit into the soft X-ray band.
- `earth_albedo(e)` returns the dimensionless reflected-CXB ratio from
  [Churazov et al. (2008), Eq. (9)](https://arxiv.org/abs/astro-ph/0608252),
  valid from 1 to 1000 keV. For energies at least 3 keV, multiply by `cxb(e)`
  for reflected photon intensity. This is a full-disk average; it excludes
  cosmic-ray atmospheric emission, aurora, and sunlight.
- `particle_background(spectra, exposure)` uses the exposure-weighted
  measured-background method of
  [Tawa et al. (2008), Eq. (1)](https://arxiv.org/abs/0803.0616).
  Supply spectra in counts/s/keV with shape (rigidity bins, energy samples),
  and the target observation's exposure in seconds per rigidity bin.
  Output is counts/s/keV per energy sample. Calibration spectra must match
  the detector, region, and event selection; no universal spectrum is assumed.

`functions/bfield.py` contains `cutoff_rigidity(r, t)` for ECEF positions in
km and scalar UTC time. It returns GV using a centered, tilted dipole from
the date's IGRF degree-1 coefficients, with latitude and radius dependence
as discussed by [Smart & Shea (2005)](https://doi.org/10.1016/j.asr.2004.09.015).
It is an approximate vertical cutoff, not particle trajectory tracing or
the Suzaku COR2 calibration. Do not use it as a precise SAA/storm model.

```python
from functions.background import cxb, earth_albedo, particle_background
from functions.bfield import cutoff_rigidity

sky = cxb([4, 10, 30])  # photons/cm^2/s/sr/keV
reflected = earth_albedo([4, 10, 30]) * sky
rc = cutoff_rigidity([6971.2, 0, 0], "2026-01-01T00:00:00")  # GV
# Illustrative calibration spectra and target exposures, not measured data:
nxb = particle_background([[0.01, 0.02], [0.005, 0.01]], [100, 300])
```

Photon backgrounds require solid angle and photon effective area to predict
detected count rates. The particle background already represents detector
counts and must not be multiplied by photon QE or telescope effective area.

## Counts and coupling limits

`functions/sensitivity.py` provides `signal_counts`, `poisson_upper_limit`,
and `limiting_coupling`. Input types, units, and assumptions are in their
docstrings. Signal counts integrate the axion flux times conversion
probability and photon effective area over energy, multiplied by live time.
For changing orbital conditions, calculate each time interval and sum.
The energy range below is chosen to stay inside the silicon table.

```python
from functions.solar_flux import axion_flux
from functions.conversion import conversion_probability_uniform
from functions.instrument import bare_effective_area
from functions.sensitivity import signal_counts, poisson_upper_limit, limiting_coupling

e = np.linspace(2, 10, 501)  # keV
g = 1e-10  # GeV^-1; use the same g for production and conversion
flux = axion_flux(e, g)
p = conversion_probability_uniform(b=3e-5, length=600e3, g=g)
area = bare_effective_area(e, area=100)  # cm^2, includes QE
counts = signal_counts(e, flux, p, area, exposure=1e6)
# Illustrative observation, not real data:
upper = poisson_upper_limit(n=3, background=2.0, confidence=0.95)
g_limit = limiting_coupling(upper, counts, g)
```

The Poisson limit uses a uniform prior for nonnegative signal and assumes
the expected background is known exactly, following the known-background
case of [Helene (1983)](https://doi.org/10.1016/0167-5087(83)90709-3).
It returns signal counts, not signal plus background; zero observed counts
give 2.996 at 95%. It is an upper credible bound, not a detection significance.
An expected sensitivity needs an explicitly chosen procedure for possible
observations; the function does not round background into observed counts.

The coupling limit uses the fourth root of the signal count ratio because
solar production and weak conversion each scale as coupling squared; see
[Davoudiasl & Huber, arXiv:0804.3543](https://arxiv.org/abs/0804.3543).
This simple total-count limit does not reproduce their full time/energy fit.

## Detector counts across an orbit

Install dependencies with `python -m pip install -e .`, including xraylib for
atomic attenuation and scattering. `simulate_counts` propagates the whole orbit
and observes during eclipse and atmospheric grazing. A directional response
callable leaves room for future optics; the example uses bare silicon.

```python
import numpy as np
from astropy import units as u
from astropy.time import Time
from functions.geometry import MU_EARTH_KM3_S2, R_EQUATOR_KM
from functions.instrument import bare_effective_area
from functions.orbit_counts import simulate_counts

e = np.geomspace(3, 20, 64)  # supported combined band: 3-50 keV
start = Time("2026-10-19T00:00:00", scale="utc")
period = 2*np.pi*np.sqrt((R_EQUATOR_KM+500)**3/MU_EARTH_KM3_S2)
seconds = np.append(np.arange(0, period, 30.0), period)
time_edges = start + seconds*u.s

def response(e, mu):
    return bare_effective_area(e, area=10, thickness=250)*max(mu, 0)

# ILLUSTRATIVE calibration and activity assumptions, not measured data:
result = simulate_counts(
    e, time_edges, response=response, det_area=10, fov=1.0,
    g=1e-10, ma=0, goes=1e-6, temperature=10, earth_c=0.01,
    particle_level=1e-3, particle_alpha=0.5,
    particle_ref=10, particle_min=0.1,
    lines=[(6.4, 0.1, 1e-4)],
    orbit=dict(h=500, inc=97.4, epoch=start),
    activity=dict(f107=150, f107a=150, ap=4),
)
print(result["totals"])
```

`response(e, mu)` returns effective photon area [cm^2], scalar or matching the
energy grid, including QE. The loop separately enforces the circular field of
view (`fov` is its half-angle in degrees). Particle events use `det_area` only.
No uncalibrated source or detector parameters receive implicit normalizations.

`rates` contains differential rates [counts/s/keV], shaped (interval, energy).
`counts` contains expected counts per interval; `cumulative` accumulates them,
and `totals` sums the orbit. Components are `signal`, `cxb`, `dark_earth`,
`solar_direct`, `solar_scattered`, `particle`, `background`, and `total`.
Background excludes signal; total includes both. No Poisson events are drawn.

Diagnostics include midpoint ECEF positions, `observing`, `live_exposure_s`,
`tangent_altitude_km`, `earth_fraction`, `solar_visible_fraction` (geometric,
before absorption), and `cutoff_rigidity_gv`. Geometry uses WGS84. For rays
passing through solid Earth the reported tangent altitude is a negative
spherical impact diagnostic, not an interior geodetic altitude.
Observation requires an Earthward center ray below `grazing=120` km. Other
intervals retain geometry but collect no counts. `live_fraction` accounts for
rejected time; no automatic SAA rejection is invented.

Solar, particle, Earth-normalization and activity parameters accept constants
or one value per interval. Lines accept one calibration list or one per interval.
Supply all three MSIS activity inputs for offline use; omitting them uses
historical lookup, potentially downloading data. The positive rigidity floor
prevents inverse-power divergence near the magnetic poles.

Defaults: 8x24 field-of-view rays, 4x16 uniform solar-disk rays (16 arcmin),
and <=5-km atmospheric path spacing. Scattering can be expensive. Refine angular,
path, energy and time sampling before using quantitative results; midpoint time
integration especially needs fine bins at eclipse contacts and grazing windows.
The 30-second example is exploratory, not a convergence guarantee. In a checked
90-km grazing case, two-second bins changed direct solar counts by 6.5% on
refinement; use one-second or finer bins around the limb and verify convergence
for the actual configuration. Coarser bins can be retained in deep eclipse.
Incoming-column interpolation uses `column_tolerance=0.001`, checked adaptively
at segment midpoints; set zero for direct evaluation of every incoming ray.

`background_budget(e, exposure, components)` now integrates already folded
detector differential rates. Its former area/fill API and the exponential
`transmission()` shortcut were removed. `ray_column` provides reusable MSIS
columns; `solar_scattered` adds separate single-scattered solar intensity.

Direct sunlight uses a finite uniform disk; scattered illumination and the axion
signal use the Sun center. Scattering includes Rayleigh and Compton continuum
with recoil. The signal retains the existing 60-km magnetic-path cutoff. Dark
Earth remains a uniform empirical spectrum, uncertain below 15 keV. The fixed
xraylib dry-air mixture at MSIS density is an upper-atmosphere approximation.
No aurora, fluorescence, repeated scattering, SAA/trapped bursts, activation,
detector redistribution or saturation is modeled. This is a forecasting pipeline,
not a calibrated complete LEO background prediction.
