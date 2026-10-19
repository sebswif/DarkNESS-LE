"""IGRF-14 fields and coherent ray integrals in Earth-fixed ECEF."""

from pathlib import Path
from typing import TypedDict

import numpy as np
from astropy.time import Time

Array = np.ndarray
Vector = np.ndarray | list | tuple
R_EARTH_KM = 6371.2
COEFFS = Path(__file__).with_name("data") / "igrf14coeffs.txt"


class Ray(TypedDict):
    """Ray samples and field integral; all vectors are ECEF."""

    distance_km: Array  # (n,), distance toward source [km]
    path_ecef_km: Array  # (n, 3), geocentric position [km]
    look_ecef_unit: Array  # (3,), unit direction toward source
    zenith_deg: float  # 0 outward, 90 horizontal, 180 nadir [deg]
    minimum_altitude_km: float  # Minimum height above IGRF sphere [km]
    stopped_at_cutoff: bool  # True if ray reaches the 60 km shell
    B_ecef_T: Array  # (n, 3), field [T]
    B_perp_ecef_T: Array  # (n, 3), transverse field [T]
    integral_ecef_Tm: Array  # (3,), integral of transverse field [T m]
    amplitude_Tm: float  # Norm of the vector integral [T m]


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


def line_of_sight(
    r: Vector, k: Vector, t: str | Time,
    length: float = 10000.0, n: int = 501, report: bool = True,
) -> Ray:
    """Integrate transverse IGRF along a ray toward the source, q = 0.

    Inputs: r, spacecraft ECEF position [km], (3,); k, source direction,
    (3,), normalized internally; t, scalar UTC ISO/Time; length, range [km];
    n, sample count >= 2; report, print zenith and integral amplitude.
    Output: Ray (keys, units, shapes above), ending at length or the 60 km
    absorption shell. Integrates vectors before taking their norm.
    The shell is a modeling cutoff; photons propagate along -k.
    Reference: Raffelt & Stodolsky (1988), doi:10.1103/PhysRevD.37.1237;
    field coefficients: IAGA (2024), doi:10.5281/zenodo.14012302.
    """
    r, k = np.asarray(r, dtype=float), np.asarray(k, dtype=float)
    if r.shape != (3,) or k.shape != (3,) or not np.isfinite([r, k]).all():
        raise ValueError("r and k must be finite vectors (3,)")
    if np.linalg.norm(k) == 0:
        raise ValueError("k must be nonzero")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    k = k / np.linalg.norm(k)
    radius = np.linalg.norm(r)
    shell = R_EARTH_KM + 60
    if radius <= shell:
        raise ValueError("r must be above the 60 km shell")
    projection = np.dot(r, k)
    discriminant = projection**2 - radius**2 + shell**2
    stopped = False
    if projection < 0 and discriminant >= -1e-8:
        entry = -projection - np.sqrt(max(0.0, discriminant))
        if entry <= length:
            length, stopped = entry, True
    s = np.linspace(0, length, n)
    path = r + s[:, None] * k
    zenith = float(np.degrees(np.arccos(np.clip(projection / radius, -1, 1))))
    closest = r + np.clip(-projection, 0, length) * k
    height = np.linalg.norm(closest) - R_EARTH_KM
    b = igrf_field(path, t, report=False)
    b_perp = b - (b @ k)[:, None] * k
    integral = np.trapezoid(b_perp, s * 1000, axis=0)
    amplitude = float(np.linalg.norm(integral))
    if report:
        print(f"Zenith [deg]: {zenith:.6f}; amplitude [T m]: {amplitude}")
    return Ray(
        distance_km=s, path_ecef_km=path, look_ecef_unit=k,
        zenith_deg=zenith, minimum_altitude_km=float(height),
        stopped_at_cutoff=stopped, B_ecef_T=b, B_perp_ecef_T=b_perp,
        integral_ecef_Tm=integral, amplitude_Tm=amplitude,
    )
