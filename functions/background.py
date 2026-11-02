"""Cosmic X-ray background, Earth reflection, and detector background."""

import numpy as np
from numpy.typing import ArrayLike


def cxb(e: ArrayLike) -> float | np.ndarray:
    """Return cosmic X-ray photon intensity [photons/cm^2/s/sr/keV].

    Input: e, energy [keV], number, list, or NumPy array, 3 to 1e8 keV.
    Output: float or NumPy array matching e. Uses Gruber's piecewise fit,
    divided by e to convert the published ENERGY intensity to photons.
    Rejects energies below 3 keV; this fit is not a soft X-ray foreground
    model. No Galactic emission, absorption, or source masking is included.
    Multiply by sky solid angle [sr] and photon effective area [cm^2] to
    get counts/s/keV for a uniform response, then integrate energy/time.
    Reference: Gruber et al. (1999), https://arxiv.org/abs/astro-ph/9903492.
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any((e < 3) | (e > 1e8)):
        raise ValueError("e must be finite and between 3 and 1e8 keV")
    low = 7.877 * e**-0.29 * np.exp(-e / 41.13)
    high = (0.0259 * (e / 60)**-5.5 + 0.504 * (e / 60)**-1.58
            + 0.0288 * (e / 60)**-1.05)
    intensity = np.where(e <= 60, low, high) / e
    return float(intensity) if intensity.ndim == 0 else intensity


def earth_albedo(e: ArrayLike) -> float | np.ndarray:
    """Return the dimensionless, full-disk average CXB reflection ratio.

    Input: e, energy [keV], number, list, or NumPy array, 1 to 1000 keV.
    Output: A(E) = reflected intensity / incident CXB intensity, matching e.
    Uses Churazov Eq. (9), for a Gruber-shaped incident spectrum. For e >= 3,
    reflected photon intensity = earth_albedo(e) * cxb(e), in
    photons/cm^2/s/sr/keV. Multiply by Earth's solid angle [sr] for flux.
    This is a disk average, not the brightness of a particular limb/ray.
    Assumes the whole disk is observed with uniform response. The smooth
    fit does not resolve fluorescent lines. Cosmic-ray atmospheric emission,
    aurora, and scattered sunlight are separate and are not included.
    Reference: Churazov et al., MNRAS 385, 719 (2008), Eq. (9),
    https://arxiv.org/abs/astro-ph/0608252.
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any((e < 1) | (e > 1000)):
        raise ValueError("e must be finite and between 1 and 1000 keV")
    albedo = (1.22 / ((e / 28.5)**-2.54 + (e / 51.3)**1.57 - 0.37)
              * (2.93 + (e / 3.08)**4) / (1 + (e / 3.08)**4)
              * (0.123 + (e / 91.83)**3.44) / (1 + (e / 91.83)**3.44))
    return float(albedo) if albedo.ndim == 0 else albedo


def particle_background(spectra: ArrayLike, exposure: ArrayLike) -> np.ndarray:
    """Return an exposure-weighted measured detector background spectrum.

    Inputs: spectra, numeric nested list or NumPy array (n, m), measured
    background COUNT RATES [counts/s/keV], n cutoff-rigidity (or PIN-UD)
    bins and m energy samples; exposure, numeric list or NumPy array (n,),
    time [s] spent by the TARGET observation in each corresponding bin.
    Output: NumPy array (m,), sum(exposure * spectra)/sum(exposure), in
    counts/s/keV. Spectra must use identical energy bins and the same
    detector region, event cuts, and operating conditions as the target.

    Supply detector-specific calibration spectra, already divided by their
    own calibration exposures and energy-bin widths. This follows Tawa's
    weighting method; the paper does not supply a universal particle law.
    Output already consists of detector counts: do not apply photon QE,
    telescope effective area, or sky solid angle again. Multiply by total
    target exposure and integrate over energy to obtain background counts.
    Reference: Tawa et al., PASJ 60, S11 (2008), Eq. (1),
    https://arxiv.org/abs/0803.0616.
    """
    spectra, exposure = np.asarray(spectra, float), np.asarray(exposure, float)
    if (spectra.ndim != 2 or 0 in spectra.shape
            or exposure.shape != (spectra.shape[0],)):
        raise ValueError("spectra must have shape (n, m), exposure shape (n,), n and m >= 1")
    if (not np.isfinite(spectra).all() or not np.isfinite(exposure).all()
            or np.any(spectra < 0) or np.any(exposure < 0) or exposure.sum() <= 0):
        raise ValueError("spectra and exposure must be finite and >= 0, with positive total exposure")
    return (exposure / exposure.sum()) @ spectra
