# DarkNESS-LE

Solar axion flux, coherent conversion, IGRF-14 fields, and RK4 orbit propagation.
Requires Python 3.10+, NumPy 2+, and Astropy. Field, orbit, flux, and conversion
formulas are implemented locally. Astropy supplies time, ephemerides, and frames.

## Setup

From this repository directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m functions.solar_flux
.\.venv\Scripts\python.exe -m functions.conversion
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux/macOS use `.venv/bin/python` instead.

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
The conversion model assumes a uniform field, coherent propagation, negligible
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
