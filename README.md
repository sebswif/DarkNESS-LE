# DarkNESS-LE

Solar Primakoff axion flux at Earth and uniform-field axion-photon conversion
utilities, imported from `sebswif/ALP` (`jupyter/solar_flux.py` and
`jupyter/conversion.py`). Requires Python 3.10 or newer and NumPy.

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
probability = conversion_probability_uniform(B_T=9.0, L_m=9.26, g_ay=g_ay)
photon_flux = flux * probability  # cm^-2 s^-1 keV^-1
```

`B_T` is the transverse field in tesla and `L_m` is the path length in metres.
The conversion model assumes a uniform field, coherent propagation, negligible
absorption, and weak mixing (probability much less than one). It does not include
finite-mass coherence loss, plasma effects, or a varying geomagnetic field.

The copied conversion module corrects the original unit factors to
1 T = 195 eV² and 1 GeV⁻¹ = 10⁻⁹ eV⁻¹; the example now runs only when
executed directly. See the [natural-unit conversion reference](https://github.com/ymzhong/naturalunit).
For 9 T, 9.26 m, and 10⁻¹⁰ GeV⁻¹, the result is approximately 1.70 × 10⁻¹⁷.
The solar flux fit and coefficients are preserved from the source file.
