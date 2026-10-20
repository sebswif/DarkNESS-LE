"""IGRF-14 magnetic fields in Earth-fixed ECEF."""

from pathlib import Path

import numpy as np
from astropy.time import Time

Array = np.ndarray
Vector = np.ndarray | list | tuple
R_EARTH_KM = 6371.2
COEFFS = Path(__file__).with_name("data") / "igrf14coeffs.txt"


def load_igrf(year: float) -> tuple[Array, Array]:
    """Interpolate the bundled IGRF-14 coefficients.

    Input: year, decimal UTC year in [1900, 2030].
    Output: (g, h), Gauss coefficients [nT], each shaped (14, 14).
    Uses linear interpolation through 2025, then secular variation.
    Reference: IAGA IGRF-14 (2024), doi:10.5281/zenodo.14012302.
    """
    rows = [s.split() for s in COEFFS.read_text().splitlines()
            if s.strip() and not s.startswith("#")]
    years = np.array(rows[1][3:-1], dtype=float)
    if not years[0] <= year <= years[-1] + 5:
        raise ValueError("year must be in [1900, 2030]")
    j = min(int(np.searchsorted(years, year, side="right")) - 1,
            len(years) - 1)
    dt = year - years[j]
    g, h = np.zeros((2, 14, 14))
    for row in rows[2:]:
        n, m = int(row[1]), int(row[2])
        values = np.array(row[3:], dtype=float)
        rate = ((values[j + 1] - values[j]) / 5
                if j < len(years) - 1 else values[-1])
        (g if row[0] == "g" else h)[n, m] = values[j] + dt * rate
    return g, h


def _legendre(theta: float | Array, degree: int) -> tuple[Array, Array]:
    """Evaluate Schmidt-normalized Legendre functions without a phase sign.

    Inputs: theta, colatitude(s) [rad]; degree, maximum degree (1 to 13).
    Output: (p, dp/dtheta), shape (degree+1, degree+1, *theta.shape).
    Reference: NIST DLMF, https://dlmf.nist.gov/14.10.
    """
    c, s = np.cos(theta), np.sin(theta)
    p = np.zeros((degree + 1, degree + 1) + theta.shape)
    dp = np.zeros_like(p)
    p[0, 0] = 1
    p[1, 0], p[1, 1] = c, s
    dp[1, 0], dp[1, 1] = -s, c
    for n in range(2, degree + 1):
        p[n, n] = (2 * n - 1) * s * p[n - 1, n - 1]
        dp[n, n] = (2 * n - 1) * (
            c * p[n - 1, n - 1] + s * dp[n - 1, n - 1]
        )
        for m in range(n):
            p[n, m] = ((2 * n - 1) * c * p[n - 1, m]
                       - (n + m - 1) * p[n - 2, m]) / (n - m)
            dp[n, m] = ((2 * n - 1) * (
                c * dp[n - 1, m] - s * p[n - 1, m]
            ) - (n + m - 1) * dp[n - 2, m]) / (n - m)
    for n in range(1, degree + 1):
        norm = np.sqrt(2.0)
        for m in range(1, n + 1):
            norm /= np.sqrt((n - m + 1) * (n + m))
            p[n, m] *= norm
            dp[n, m] *= norm
    return p, dp


def igrf_field(
    r: Vector, t: str | Time, report: bool = True, degree: int = 13,
) -> Array:
    """Calculate the IGRF-14 main field from its spherical harmonics.

    Inputs: r, ECEF positions [km], (..., 3); t, scalar UTC ISO/Time;
    report, print field; degree, maximum harmonic degree (1 to 13).
    Output: ECEF field [T], same shape as r.
    Reference: IAGA IGRF-14 (2024), doi:10.5281/zenodo.14012302.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim < 1 or r.shape[-1] != 3 or not np.isfinite(r).all():
        raise ValueError("r must contain finite ECEF vectors (..., 3)")
    radius = np.linalg.norm(r, axis=-1)
    if np.any(radius == 0):
        raise ValueError("Field undefined at Earth's center")
    if not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 13:
        raise ValueError("degree must be an integer in [1, 13]")
    t = t if isinstance(t, Time) else Time(t, scale="utc")
    if not t.isscalar:
        raise ValueError("t must be scalar")
    g, h = load_igrf(t.utc.decimalyear)
    theta = np.arctan2(np.hypot(r[..., 0], r[..., 1]), r[..., 2])
    phi = np.arctan2(r[..., 1], r[..., 0])
    c, s = np.cos(theta), np.sin(theta)
    p, dp = _legendre(theta, degree)
    br, bt, bp = np.zeros((3,) + radius.shape)
    for n in range(1, degree + 1):
        scale = (R_EARTH_KM / radius) ** (n + 2)
        for m in range(n + 1):
            cm, sm = np.cos(m * phi), np.sin(m * phi)
            b = scale * (g[n, m] * cm + h[n, m] * sm)
            br += (n + 1) * b * p[n, m]
            bt -= b * dp[n, m]
            # At a pole, P_n1 / sin(theta) tends to cos(theta) * dP_n1.
            q = np.divide(p[n, m], s, out=np.zeros_like(s), where=s > 1e-10)
            q = np.where(s > 1e-10, q, c * dp[n, m] if m == 1 else 0)
            bp += scale * m * (g[n, m] * sm - h[n, m] * cm) * q
    cp, sp = np.cos(phi), np.sin(phi)
    b = np.stack([
        (br * s + bt * c) * cp - bp * sp,
        (br * s + bt * c) * sp + bp * cp,
        br * c - bt * s,
    ], axis=-1) * 1e-9
    if report:
        print("B [ECEF T]:", b)
    return b


field_ecef = igrf_field


def dipole_field(r: Vector, t: str | Time, report: bool = True) -> Array:
    """Calculate the tilted degree-1 IGRF dipole for visualization.

    Inputs: r, ECEF positions [km], (..., 3); t, scalar UTC ISO/Time;
    report, print field.
    Output: approximate ECEF field [T], same shape as r.
    Reference: IAGA IGRF-14 (2024), doi:10.5281/zenodo.14012302.
    """
    return igrf_field(r, t, report, degree=1)
