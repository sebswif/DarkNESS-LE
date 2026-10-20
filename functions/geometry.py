"""Sun direction and RK4 orbits. ECI = GCRS; ECEF = ITRS."""

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_sun
from astropy.time import Time
from astropy.utils import iers

MU_EARTH_KM3_S2 = 398600.4418
R_EQUATOR_KM = 6378.137
J2 = 1.08263e-3


def _frame(r: np.ndarray, t: Time, frame: str) -> np.ndarray:
    """Return positions in the requested frame.

    Inputs: r, ECI positions [km], (..., 3); t, times; frame, eci/ecef.
    Output: positions [km], same shape as r.
    Reference: Astropy (2022), doi:10.3847/1538-4357/ac7c74.
    """
    if frame.lower() == "eci":
        return r
    if frame.lower() != "ecef":
        raise ValueError("frame must be 'eci' or 'ecef'")
    r = GCRS(CartesianRepresentation(np.moveaxis(r, -1, 0) * u.km), obstime=t)
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
    ):
        r = r.transform_to(ITRS(obstime=t))
    return np.moveaxis(r.cartesian.xyz.to_value(u.km), 0, -1)


def sun_direction(
    t: str | list[str] | Time, frame: str = "ecef",
) -> np.ndarray:
    """Return the unit vector from Earth toward the Sun.

    Inputs: t, UTC dates or Time; frame, eci/ecef.
    Output: unit vectors, shape t.shape + (3,).
    Reference: Astropy (2022), doi:10.3847/1538-4357/ac7c74.
    """
    t = Time(t, scale="utc")
    r = np.moveaxis(get_sun(t).cartesian.xyz.to_value(u.km), 0, -1)
    r = _frame(r, t, frame)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def _motion(state: np.ndarray, j2: bool) -> np.ndarray:
    """Return velocity and acceleration from gravity.

    Inputs: state, ECI [x, y, z, vx, vy, vz] [km, km/s]; j2, enable J2.
    Output: [vx, vy, vz, ax, ay, az] [km/s, km/s^2]. J2 axis: ECI z.
    Reference: ESA Navipedia, Perturbed Motion, Eq. 2.
    """
    r = state[:3]
    radius = np.linalg.norm(r)
    a = -MU_EARTH_KM3_S2 * r / radius**3
    if j2:
        z2 = (r[2] / radius) ** 2
        a += (1.5 * J2 * MU_EARTH_KM3_S2 * R_EQUATOR_KM**2 / radius**5
              * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3]))
    return np.concatenate([state[3:], a])


def _rk4(state: np.ndarray, dt: float, j2: bool) -> np.ndarray:
    """Advance the orbit by one RK4 step.

    Inputs: state, ECI position/velocity [km, km/s], (6,); dt, step [s];
    j2, enable J2. Output: updated state [km, km/s], (6,).
    Reference: Hipparchus, ClassicalRungeKuttaIntegrator.
    """
    k1 = _motion(state, j2)
    k2 = _motion(state + dt * k1 / 2, j2)
    k3 = _motion(state + dt * k2 / 2, j2)
    k4 = _motion(state + dt * k3, j2)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def orbit_position(
    t: str | list[str] | Time, epoch: str | Time = "2026-10-19T00:00:00",
    h: float = 500.0, inc: float = 97.4, raan: float = 0.0,
    phase: float = 0.0, frame: str = "ecef", j2: bool = True,
    step: float = 10.0,
) -> np.ndarray:
    """Calculate orbit positions using RK4, starting from a circular orbit.

    Inputs: t, UTC dates/Time; epoch, initial UTC date/Time;
    h, equatorial altitude [km]; inc, inclination [deg]; raan, ascending
    node [deg]; phase, initial argument of latitude [deg], in ECI;
    frame, eci/ecef; j2, enable J2; step, maximum step [s].
    Output: positions from Earth's center [km], shape t.shape + (3,).
    Reference: Hipparchus, ClassicalRungeKuttaIntegrator;
    ESA Navipedia, Perturbed Motion, Eq. 2.
    """
    t, epoch = Time(t, scale="utc"), Time(epoch, scale="utc")
    if not np.isfinite([h, inc, raan, phase, step]).all():
        raise ValueError("Orbit parameters and step must be finite")
    if h < 0 or not 0 <= inc <= 180:
        raise ValueError("Require h >= 0 and inc in [0, 180]")
    if not epoch.isscalar:
        raise ValueError("epoch must be scalar")
    if step <= 0:
        raise ValueError("step must be positive")
    a = R_EQUATOR_KM + h
    inc, raan, phase = np.deg2rad([inc, raan, phase])
    co, so = np.cos(raan), np.sin(raan)
    cu, su = np.cos(phase), np.sin(phase)
    ci, si = np.cos(inc), np.sin(inc)
    r0 = a * np.array([co * cu - so * su * ci,
                       so * cu + co * su * ci, su * si])
    v0 = np.sqrt(MU_EARTH_KM3_S2 / a) * np.array([
        -co * su - so * cu * ci, -so * su + co * cu * ci, cu * si,
    ])
    dt = np.asarray((t - epoch).to_value(u.s)).ravel()
    if not np.isfinite(dt).all():
        raise ValueError("t must contain finite times")
    r = np.empty((dt.size, 3))
    for indices in (np.flatnonzero(dt >= 0), np.flatnonzero(dt < 0)):
        state, elapsed = np.concatenate([r0, v0]), 0.0
        for index in indices[np.argsort(np.abs(dt[indices]))]:
            duration = float(dt[index] - elapsed)
            n = int(np.ceil(abs(duration) / step))
            for _ in range(n):
                state = _rk4(state, duration / n, j2)
            r[index] = state[:3]
            elapsed = dt[index]
    r = r.reshape(t.shape + (3,))
    return _frame(r, t, frame)


if __name__ == "__main__":
    print(sun_direction("2026-10-19T00:00:00"))
    print(orbit_position("2026-10-19T00:00:00"))
