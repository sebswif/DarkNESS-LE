"""MSIS air density, X-ray attenuation, and photon mass."""

import numpy as np
import pymsis
from numpy.typing import ArrayLike
from scipy.constants import alpha, electron_mass, hbar, c, electron_volt, Avogadro


def air_density(
    h: ArrayLike, lat: ArrayLike, lon: ArrayLike,
    t: str | np.datetime64 | list[str] | list[np.datetime64] | np.ndarray,
    *, f107: ArrayLike | None = None, f107a: ArrayLike | None = None,
    ap: ArrayLike | None = None,
) -> float | np.ndarray:
    """Return NRLMSIS 2.1 mass density [kg/m^3], using pymsis.

    Inputs: h, WGS84 altitude [km], >= 0; lat, geodetic latitude [deg],
    -90 to 90; lon, longitude [deg], -180 to 180; t, UTC ISO time or
    numpy datetime64. Scalars or arrays are matched point by point;
    scalar inputs repeat to match arrays. Output has their shared shape.
    Types: h, lat, lon and supplied activity inputs accept numbers, lists,
    or NumPy arrays (ArrayLike). t accepts a string, numpy datetime64,
    or a list/NumPy array of those. Returns a float or NumPy array.

    By default pymsis loads historical solar/geomagnetic activity data,
    downloading and caching it if needed (internet required initially).
    For offline use or dates without data, supply all three: f107, previous
    day's solar radio flux [sfu]; f107a, centered 81-day mean [sfu]; ap,
    daily geomagnetic index. Each is a scalar or matches the input shape.
    Supplied values must describe the requested date, or be stated assumptions.
    MSIS is fitted to observations; its output is a model estimate, not a
    direct measurement. Uses daily Ap mode, not storm-time 3-hour indices.
    Source: https://swxtrec.github.io/pymsis/reference/generated/pymsis.msis.calculate.html
    """
    h, lat, lon, dates = np.broadcast_arrays(
        np.asarray(h, float), np.asarray(lat, float), np.asarray(lon, float),
        np.asarray(t, dtype="datetime64[us]"),
    )
    if (h.size == 0 or not all(np.isfinite(v).all() for v in (h, lat, lon))
            or np.any(h < 0) or np.any(np.abs(lat) > 90)
            or np.any(np.abs(lon) > 180) or np.isnat(dates).any()):
        raise ValueError("need finite h >= 0, lat in [-90, 90], lon in [-180, 180], and valid UTC t")
    activity = (f107, f107a, ap)
    if any(v is not None for v in activity) and not all(v is not None for v in activity):
        raise ValueError("supply f107, f107a, and ap together, or omit all three")
    if f107 is not None:
        f107, f107a, ap = [np.broadcast_to(np.asarray(v, float), h.shape).ravel()
                          for v in activity]
        if (not all(np.isfinite(v).all() for v in (f107, f107a, ap))
                or np.any(f107 <= 0) or np.any(f107a <= 0) or np.any(ap < 0)):
            raise ValueError("f107 and f107a must be > 0 and ap >= 0, all finite")
        ap = np.repeat(ap[:, None], 7, axis=1)  # Only daily column is used.
    result = pymsis.calculate(
        dates.ravel(), lon.ravel(), lat.ravel(), h.ravel(),
        f107s=f107, f107as=f107a, aps=ap, version=2.1, geomagnetic_activity=1,
    )
    rho = result[..., pymsis.Variable.MASS_DENSITY].reshape(h.shape)
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("MSIS returned invalid density; check the coordinates and activity inputs")
    return float(rho) if rho.ndim == 0 else rho


def air_attenuation(rho: ArrayLike, mac: ArrayLike) -> float | np.ndarray:
    """Return the photon intensity loss coefficient Gamma in 1/m.

    Inputs: rho, dry-air density [kg/m^3]; mac, mass attenuation
    coefficient [cm^2/g] at the photon energy. Scalars or matching arrays.
    Types: rho and mac accept numbers, lists, or NumPy arrays (ArrayLike).
    Returns a float for scalar inputs, otherwise a NumPy array.
    Output: Gamma = 0.1 * rho * mac; intensity after length L [m] is
    exp(-Gamma * L) for constant density. This is not transmission itself.
    Example: NIST dry air at 4 keV has mac = 77.88 cm^2/g.
    Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html
    """
    rho, mac = np.broadcast_arrays(np.asarray(rho, float), np.asarray(mac, float))
    if not np.isfinite(rho).all() or not np.isfinite(mac).all() or np.any(rho < 0) or np.any(mac < 0):
        raise ValueError("rho and mac must be finite and >= 0")
    loss = 0.1 * rho * mac
    return float(loss) if loss.ndim == 0 else loss


def photon_plasma_mass(rho: ArrayLike) -> float | np.ndarray:
    """Return the effective X-ray photon mass in dry air [eV].

    Input: rho, density [kg/m^3], scalar or array, >= 0. Output matches rho.
    Types: rho accepts a number, list, or NumPy array (ArrayLike).
    Returns a float for a scalar input, otherwise a NumPy array.
    Uses m_gamma^2 = 4*pi*alpha*n_e*(hbar*c)^3/(m_e*c^2), in eV units,
    with n_e = rho * N_A * 0.49919 * 1000 [electrons/m^3].
    0.49919 is the dry-air electron/mass ratio Z/A [mol/g] (NIST).
    Counts bound as well as free electrons: the X-ray approximation applies
    well above binding energies, away from absorption edges, near 4 keV.
    It is not an optical/radio refractive-index model.
    Reference: van Bibber et al. (1989), doi:10.1103/PhysRevD.39.2089.
    """
    rho = np.asarray(rho, dtype=float)
    if not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("rho must be finite and >= 0 kg/m^3")
    electrons = rho * Avogadro * 0.49919 * 1000
    mass = np.sqrt(4 * np.pi * alpha * electrons * (hbar * c / electron_volt)**3
                   / (electron_mass * c**2 / electron_volt))
    return float(mass) if mass.ndim == 0 else mass
