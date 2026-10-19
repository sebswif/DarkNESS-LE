"""Sun directions and circular J2 orbits: ECI = GCRS, ECEF = ITRS."""

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_sun
from astropy.time import Time
from astropy.utils import iers

Array = np.ndarray
Date = str | list[str] | Time
MU_EARTH_KM3_S2 = 398600.4418
R_EQUATOR_KM = 6378.137
J2 = 1.08263e-3


def _time(t: Date) -> Time:
    """Convert dates to Astropy times.

    Input: t, UTC ISO string(s) or Time.
    Output: Time, preserving input shape and existing Time scale.
    Reference: Astropy (2022), doi:10.3847/1538-4357/ac7c74.
    """
    return t if isinstance(t, Time) else Time(t, scale="utc")


def _frame(frame: str) -> str:
    """Validate a coordinate frame name.

    Input: frame, 'eci' (GCRS) or 'ecef' (ITRS), case insensitive.
    Output: lowercase frame name; raises ValueError if unsupported.
    Reference: Astropy (2022), doi:10.3847/1538-4357/ac7c74.
    """
    frame = frame.lower()
    if frame not in ("eci", "ecef"):
        raise ValueError("frame must be 'eci' or 'ecef'")
    return frame


def _ecef(r: Array, t: Time) -> Array:
    """Rotate geocentric positions into the Earth-fixed frame, offline.

    Inputs: r, GCRS positions [km], (..., 3); t, matching Time(s).
    Output: ITRS positions [km], same shape as r; bundled IERS data.
    Reference: Astropy (2022), doi:10.3847/1538-4357/ac7c74.
    """
    r = GCRS(CartesianRepresentation(np.moveaxis(r, -1, 0) * u.km), obstime=t)
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
    ):
        r = r.transform_to(ITRS(obstime=t))
    return np.moveaxis(r.cartesian.xyz.to_value(u.km), 0, -1)


def sun_direction(t: Date, frame: str = "ecef") -> Array:
    """Calculate the Earth-to-Sun unit vector.

    Inputs: t, UTC ISO string(s) or Time; frame, 'eci' or 'ecef'.
    Output: unit vector(s), (3,) for scalar t or (..., 3) for time arrays.
    Reference: Astropy get_sun (ERFA epv00, simplified VSOP2000):
    https://docs.astropy.org/en/stable/api/astropy.coordinates.get_sun.html
    """
    t, frame = _time(t), _frame(frame)
    r = np.moveaxis(get_sun(t).cartesian.xyz.to_value(u.km), 0, -1)
    if frame == "ecef":
        r = _ecef(r, t)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def orbit_position(
    t: Date, epoch: str | Time = "2026-10-19T00:00:00",
    h: float = 500.0, inc: float = 97.4, raan: float = 0.0,
    phase: float = 0.0, frame: str = "ecef", j2: bool = True,
) -> Array:
    """Propagate a circular orbit with first-order secular J2 drift.

    Inputs: t, UTC ISO string(s)/Time; epoch, scalar UTC ISO/Time;
    h, equatorial altitude [km]; inc, inclination [deg]; raan, ascending
    node [deg]; phase, argument of latitude [deg] at epoch in GCRS;
    frame, 'eci'/'ecef'; j2, include J2 (no drag or periodic terms).
    Output: geocentric positions [km], (3,) or (..., 3), matching t.
    Reference: Brouwer (1959), doi:10.1086/107958 (secular J2 terms).
    """
    t, epoch, frame = _time(t), _time(epoch), _frame(frame)
    if not np.isfinite([h, inc, raan, phase]).all():
        raise ValueError("Orbit parameters must be finite")
    if h < 0 or not 0 <= inc <= 180:
        raise ValueError("Require h >= 0 and inc in [0, 180]")
    if not epoch.isscalar:
        raise ValueError("epoch must be scalar")
    a = R_EQUATOR_KM + h
    n = np.sqrt(MU_EARTH_KM3_S2 / a**3)
    dt = (t - epoch).to_value(u.s)
    inc = np.deg2rad(inc)
    rate = J2 * n * (R_EQUATOR_KM / a) ** 2 if j2 else 0.0
    raan = np.deg2rad(raan) - 1.5 * rate * np.cos(inc) * dt
    phase = np.deg2rad(phase) + (
        n + 1.5 * rate * (4 * np.cos(inc) ** 2 - 1)
    ) * dt
    co, so = np.cos(raan), np.sin(raan)
    cu, su = np.cos(phase), np.sin(phase)
    ci, si = np.cos(inc), np.sin(inc)
    r = a * np.stack([co * cu - so * su * ci,
                      so * cu + co * su * ci, su * si], axis=-1)
    return _ecef(r, t) if frame == "ecef" else r


if __name__ == "__main__":
    print(sun_direction("2026-10-19T00:00:00"))
    print(orbit_position("2026-10-19T00:00:00"))
