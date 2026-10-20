"""Line-of-sight geometry and coherent transverse field integrals."""

from typing import TypedDict

import numpy as np
from astropy.time import Time

from .bfield import Array, R_EARTH_KM, Vector, igrf_field


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


def los(
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
