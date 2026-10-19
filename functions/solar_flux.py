"""Solar Primakoff axion spectrum at Earth."""

from math import gamma

import numpy as np

NORM = 6.02e10  # cm^-2 s^-1 keV^-1 at g = 1e-10 GeV^-1
ALPHA = 2.481
T_NORM = 1.205  # keV


def axion_flux(e: float | list | np.ndarray, g: float) -> float | np.ndarray:
    """Calculate differential Primakoff flux at Earth.

    Inputs: e, energy [keV], scalar or array; g, coupling [GeV^-1].
    Output: flux [cm^-2 s^-1 keV^-1], matching e; zero for e <= 0.
    Reference: CAST (2007), Eq. 15, doi:10.1088/1475-7516/2007/04/010.
    """
    e = np.maximum(np.asarray(e, dtype=float), 0)
    flux = NORM * (g / 1e-10) ** 2 * e**ALPHA * np.exp(-e / T_NORM)
    return float(flux) if e.ndim == 0 else flux


def spectrum_summary(g: float) -> tuple[float, float, float]:
    """Integrate the fitted spectrum from zero to infinite energy.

    Input: g, coupling [GeV^-1].
    Output: (total flux [cm^-2 s^-1], mean [keV], peak [keV]).
    Mean and peak describe the fitted shape, including at g = 0.
    Reference: CAST (2007), Eq. 15, doi:10.1088/1475-7516/2007/04/010.
    """
    flux = NORM * (g / 1e-10) ** 2 * T_NORM**(ALPHA + 1) * gamma(ALPHA + 1)
    return flux, (ALPHA + 1) * T_NORM, ALPHA * T_NORM


if __name__ == "__main__":
    print(spectrum_summary(1e-10))
