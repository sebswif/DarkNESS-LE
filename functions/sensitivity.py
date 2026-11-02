"""Expected signal counts, Poisson upper limits, and coupling limits."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp


def signal_counts(
    e: ArrayLike, flux: ArrayLike, probability: ArrayLike,
    area: ArrayLike, exposure: float,
) -> float:
    """Return expected detected signal counts over the supplied energy range.

    Inputs: e, increasing energy samples [keV], numeric list or NumPy array
    (n,), n >= 2; flux, differential axion flux [axions/cm^2/s/keV];
    probability, axion-to-photon conversion probability [0 to 1];
    area, photon effective area [cm^2]; exposure, live observing time [s],
    float or int >= 0. flux, probability, area accept a number, numeric
    list, or NumPy array; each is scalar or (n,). Output: float counts.

    N = exposure * integral flux(E) * probability(E) * area(E) dE,
    using trapezoids on sampled differential values, not bin totals.
    Flux and conversion must use the same coupling and axion mass.
    Effective area already includes QE; do not apply it twice. Flux must
    describe the selected source region, or include its acceptance in area.
    No sky solid-angle factor is needed for source-integrated solar flux.
    Assumes constant conditions during exposure. For a varying orbit,
    call for each time interval with its own probability/area and sum.
    Increase energy sampling until counts converge.
    Reference: Davoudiasl & Huber, arXiv:0804.3543, counting model Eq. (25).
    """
    e = np.asarray(e, dtype=float)
    if (e.ndim != 1 or e.size < 2 or not np.isfinite(e).all()
            or np.any(e <= 0) or np.any(np.diff(e) <= 0)):
        raise ValueError("e must have >= 2 finite, positive, increasing energy samples")
    if not np.isfinite(exposure) or exposure < 0:
        raise ValueError("exposure must be finite and >= 0 s")
    values = []
    for name, value in (("flux", flux), ("probability", probability), ("area", area)):
        value = np.asarray(value, dtype=float)
        if value.shape not in ((), e.shape) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and >= 0, scalar or shape (n,)")
        if name == "probability" and np.any(value > 1):
            raise ValueError("probability must be <= 1")
        values.append(value)
    flux, probability, area = values
    rate = np.broadcast_to(flux * probability * area, e.shape)
    return float(exposure * np.trapezoid(rate, e))


def poisson_upper_limit(n: int, background: float, confidence: float = 0.95) -> float:
    """Return a one-sided upper limit on expected SIGNAL counts.

    Inputs: n, observed total counts, Python/NumPy integer >= 0;
    background, known expected background counts, float or int >= 0;
    confidence, probability between 0 and 1, float (default 0.95).
    Output: float signal counts, not signal plus background.

    Uses a Poisson likelihood for n with mean signal + background and
    a uniform prior for signal >= 0 (Helene known-background method).
    Solves CDF_Poisson(n; background + limit) / CDF_Poisson(n; background)
    = 1 - confidence. This is a Bayesian upper credible bound; it is not
    a Gaussian significance or a Feldman-Cousins interval. Background
    uncertainty is not included. For n = 0, limit = -log(1-confidence),
    regardless of background. n must be an actual integer observation;
    an expected background count is not automatically an observation.
    Reference: Helene, NIM 212, 319 (1983),
    doi:10.1016/0167-5087(83)90709-3.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be an integer >= 0")
    if not np.isfinite(background) or background < 0:
        raise ValueError("background must be finite and >= 0 counts")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    if n == 0:
        return float(-np.log1p(-confidence))
    k = np.arange(n + 1, dtype=float)
    log_factorial = gammaln(k + 1)

    def log_cdf(mean):
        if mean == 0:
            return 0.0
        return float(logsumexp(k * np.log(mean) - log_factorial) - mean)

    # Logarithms avoid underflow when the observed count is below background.
    baseline = log_cdf(background)
    log_tail = np.log1p(-confidence)

    def equation(signal):
        return log_cdf(background + signal) - baseline - log_tail

    upper = max(1.0, n + 1.0 - background) + 10 * np.sqrt(n + 1.0)
    while equation(upper) > 0:
        upper *= 2
    return float(brentq(equation, 0.0, upper))


def limiting_coupling(limit: float, counts: float, g: float = 1e-10) -> float:
    """Return the solar Primakoff axion coupling upper limit [GeV^-1].

    Inputs: limit, upper limit on signal counts, float or int >= 0;
    counts, expected signal counts at reference coupling g, float or int > 0;
    g, reference coupling [GeV^-1], float or int > 0 (default 1e-10).
    Output: float, g_limit = g * (limit / counts)^(1/4).

    Assumes solar production scales as g^2 and weak conversion as g^2,
    so counts scale as g^4. Keep mass, orbit, source acceptance, exposure,
    and detector response fixed. Both production and conversion used for
    counts must be evaluated at g. This scaling does not apply to a source
    flux independent of g or to strong conversion. This translates a count
    limit; it does not itself calculate discovery significance or expected
    sensitivity. Zero reference counts cannot constrain the coupling.
    Reference: Davoudiasl & Huber, arXiv:0804.3543, Eqs. (25), (27).
    """
    if not np.isfinite([limit, counts, g]).all() or limit < 0 or counts <= 0 or g <= 0:
        raise ValueError("limit must be >= 0, counts and g > 0, all finite")
    return float(g * (limit / counts)**0.25)
