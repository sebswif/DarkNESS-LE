"""Compact analytical models from the GECOSAX function breakdown.

These are forecasting approximations, with caller-supplied calibration.
No grid, integration, instrument class, or zero-valued stubs are needed.
"""

import numpy as np
from numpy.typing import ArrayLike

# 1 W/m^2 = 1 J/s/m^2 = 6.241509074e15 keV/s / 1e4 cm^2.
W_M2_TO_KEV_CM2_S = 6.241509074e11


def dark_earth(
    e: ArrayLike, c: float, eb: float = 33.7,
    g1: float = -5.0, g2: float = 1.72,
) -> float | np.ndarray:
    """Return Earth photon intensity [ph/cm^2/s/sr/keV].

    Inputs: e, positive energy [keV], scalar/list/array; c, nonnegative
    normalization in output units; eb, positive break energy [keV];
    g1, g2, finite slopes. Output matches e; I(eb) = c/2.
    I = c / ((e/eb)**g1 + (e/eb)**g2).

    Default shape: Ajello et al. (2008), arXiv:0808.3377 (15-200 keV).
    For the PDF's Sazonov/Churazov shape use eb=44, g2=1.4 and a fitted c.
    c is deliberately required: the PDF flags its normalization as unverified.
    Soft-band extrapolation is uncalibrated. This models total dark-Earth
    emission; do not add CXB reflection a second time.
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any(e <= 0):
        raise ValueError("e must be finite and > 0 keV")
    if not np.isfinite([c, eb, g1, g2]).all() or c < 0 or eb <= 0:
        raise ValueError("c must be >= 0, eb > 0, and slopes finite")
    x = np.log(e / eb)
    result = c * np.exp(-np.logaddexp(g1 * x, g2 * x))
    return float(result) if result.ndim == 0 else result


def nxb(
    e: ArrayLike, level: float, lines: ArrayLike = (), scale: float = 1.0,
) -> float | np.ndarray:
    """Return detector background [counts/cm^2/s/keV], matching e.

    Inputs: e, positive energy [keV], scalar/list/array; level, nonnegative
    flat continuum [counts/cm^2/s/keV]; lines, sequence of (energy [keV],
    sigma [keV], integrated rate [counts/cm^2/s]); scale, nonnegative
    continuum multiplier from rigidity/calibration. Gaussian lines have
    full-line integral equal to rate; finite bands retain only their tails.
    b = level*scale + sum(rate*exp(-0.5*((e-center)/sigma)**2)
                             / (sqrt(2*pi)*sigma)).
    As in the PDF, scale applies only to the continuum. Supply measured
    detector parameters; this is not a universal cosmic-ray prediction.
    Multiply by detector region area, never photon QE/effective area.
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any(e <= 0):
        raise ValueError("e must be finite and > 0 keV")
    if not np.isfinite([level, scale]).all() or level < 0 or scale < 0:
        raise ValueError("level and scale must be finite and >= 0")
    lines = np.asarray(lines, dtype=float)
    if lines.shape == (0,):
        lines = lines.reshape(0, 3)
    if (lines.ndim != 2 or lines.shape[1] != 3
            or not np.isfinite(lines).all() or np.any(lines[:, :2] <= 0)
            or np.any(lines[:, 2] < 0)):
        raise ValueError("lines must contain finite (center > 0, sigma > 0, rate >= 0)")
    result = np.full(e.shape, level * scale, dtype=float)
    for center, sigma, rate in lines:
        result += rate / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((e-center)/sigma)**2)
    return float(result) if result.ndim == 0 else result


def solar_spectrum(
    e: ArrayLike, goes: float, temperature: float = 10.0,
    f0: float = 0.0, e0: float = 10.0, gamma: float = 3.0,
) -> float | np.ndarray:
    """Return incident solar photon flux [ph/cm^2/s/keV], matching e.

    Inputs: e, positive energy [keV], scalar/list/array; goes, thermal GOES
    1-8 Angstrom energy flux [W/m^2] >= 0; temperature [MK] > 0;
    f0, optional nonthermal photon flux at e0 in output units >= 0;
    e0 [keV] > 0; gamma, finite tail index > 0.
    Thermal F=N/e*exp(-e/kT), kT=0.08617333262*temperature keV.
    N is normalized analytically over 1.55-12.4 keV, using
    1 W/m^2 = 6.241509074e11 keV/cm^2/s. Tail is f0*(e/e0)**-gamma
    for e >= e0 and zero below. GOES normalizes only the thermal term.
    This is a continuum proxy, omitting solar lines and GOES response;
    the optional tail adds energy beyond the specified thermal flux.
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any(e <= 0):
        raise ValueError("e must be finite and > 0 keV")
    if (not np.isfinite([goes, temperature, f0, e0, gamma]).all()
            or goes < 0 or temperature <= 0 or f0 < 0 or e0 <= 0 or gamma <= 0):
        raise ValueError("goes/f0 must be >= 0 and temperature/e0/gamma > 0, all finite")
    kt = 0.08617333262 * temperature
    # Shift the exponential by the band edge to avoid normalization underflow.
    norm = kt * -np.expm1(-(12.4 - 1.55) / kt)
    result = np.zeros_like(e)
    if goes > 0:
        with np.errstate(over="raise"):
            try:
                result = goes * W_M2_TO_KEV_CM2_S / norm / e * np.exp((1.55 - e) / kt)
            except FloatingPointError as exc:
                raise ValueError("thermal extrapolation overflows; increase temperature or energy") from exc
    tail = e >= e0
    result = result + f0 * np.exp(-gamma * np.maximum(np.log(e / e0), 0)) * tail
    return float(result) if result.ndim == 0 else result


def cxb(e: ArrayLike) -> float | np.ndarray:
    """Return cosmic X-ray photon intensity [photons/cm^2/s/sr/keV].

    Input: e, energy [keV], number, list, or NumPy array, 1 to 1e8 keV.
    Output: float or NumPy array matching e. From 3 keV up, Gruber's
    piecewise fit, divided by e to convert the published ENERGY intensity
    to photons. From 1 to 3 keV, a photon power law of index 1.41 (De Luca
    & Molendi 2004, XMM 2-10 keV), normalized to Gruber at 3 keV so the
    spectrum is continuous; their own normalization is ~40% higher there,
    a systematic to bracket. Rejects energies below 1 keV, where Galactic
    soft emission (not modelled) becomes important. No Galactic emission,
    absorption, or source masking is included.
    Multiply by sky solid angle [sr] and photon effective area [cm^2] to
    get counts/s/keV for a uniform response, then integrate energy/time.
    References: Gruber et al. (1999), https://arxiv.org/abs/astro-ph/9903492;
    De Luca & Molendi, A&A 419, 837 (2004).
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any((e < 1) | (e > 1e8)):
        raise ValueError("e must be finite and between 1 and 1e8 keV")
    gruber_3kev = 7.877 * 3**-0.29 * np.exp(-3 / 41.13)  # energy intensity at 3 keV
    soft = gruber_3kev * (e / 3)**(1 - 1.41)  # energy intensity of the photon index 1.41 law
    low = np.where(e < 3, soft, 7.877 * e**-0.29 * np.exp(-e / 41.13))
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


