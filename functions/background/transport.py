"""Single-scattered sunlight through a spherical-shell-bounded WGS84 atmosphere.

Mass density is MSIS; composition is fixed xraylib NIST dry air, an explicit
approximation above the well-mixed atmosphere. No multiple scattering,
fluorescence, polarization, or Compton broadening is included.
"""

import numpy as np
import xraylib
import xraylib_np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar

from ..atmosphere import air_density
from .models import solar_spectrum

_AXES = np.array([6378.137, 6378.137, 6356.752314245])  # WGS84 [km]
_AIR = xraylib.GetCompoundDataNISTByName("Air, Dry (near sea level)")
_Z = np.asarray(_AIR["Elements"], dtype=np.int64)
_W = np.asarray(_AIR["massFractions"])
_ATOMIC_WEIGHT = np.array([xraylib.AtomicWeight(int(z)) for z in _Z])


def _unit(v):
    v = np.asarray(v, float)
    if v.shape[-1:] != (3,) or not np.isfinite(v).all() or np.any(np.linalg.norm(v, axis=-1) == 0):
        raise ValueError("directions must be finite nonzero (...,3) vectors")
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _geodetic(points):
    p = np.asarray(points)
    lon, lat, height = EarthLocation.from_geocentric(*np.moveaxis(p, -1, 0), unit=u.km).to_geodetic()
    return height.to_value(u.km), lat.to_value(u.deg), lon.to_value(u.deg)


def _density(points, t, activity, top):
    h, lat, lon = _geodetic(points)
    result = np.zeros(h.shape)
    mask = (h >= -1e-6) & (h <= top)
    if mask.any():
        result[mask] = air_density(np.maximum(h[mask], 0), lat[mask], lon[mask],
                                   Time(t).utc.isot, **(activity or {}))
    return result


def _geometry(r, k, top):
    """Return integration entry/exit [km] and solid-Earth obstruction."""
    q, v = r / _AXES, k / _AXES
    a, b, c = v @ v, q @ v, q @ q - 1
    if c < -1e-8:
        raise ValueError("ray origin must be on or above WGS84 Earth")
    discriminant = b*b-a*c
    hit = np.inf
    if discriminant >= 0:
        roots = np.array([-b-np.sqrt(discriminant), -b+np.sqrt(discriminant)]) / a
        forward = roots[roots >= -1e-7]
        if forward.size and (c > 1e-8 or b < 0):
            hit = max(0.0, float(forward.min()))
    # Enclosing sphere bounds the exact geodetic-height ceiling. Density is
    # evaluated only below top; it is zero in the thin excess shell.
    dot = r @ k
    disc = dot*dot - r@r + (_AXES[0]+top)**2
    if disc <= 0:
        return 0.0, 0.0, np.isfinite(hit)
    start = max(0.0, -dot-np.sqrt(disc))
    end = min(max(0.0, -dot+np.sqrt(disc)), hit)
    return start, max(start, end), np.isfinite(hit)


def _tangent(r, k):
    """Minimum geodetic altitude along a forward ray, including solid Earth."""
    if r @ k >= 0:
        return float(_geodetic(r)[0])
    # Deep-Earth geodetic coordinates are ill-conditioned. For blocked rays
    # a negative spherical impact altitude is a useful diagnostic instead.
    if _geometry(r, k, 1000)[2]:
        return min(-1e-6, float(np.linalg.norm(r-(r@k)*k)-6371.2))
    fit = minimize_scalar(lambda s: float(_geodetic(r+s*k)[0]),
                          bounds=(0, 2*np.linalg.norm(r)), method="bounded")
    return float(fit.fun)


def _mac(e):
    """Dry-air total attenuation [cm^2/g] at arbitrary-shaped energies."""
    e = np.asarray(e, float)
    values = (_W @ xraylib_np.CS_Total(_Z, np.ascontiguousarray(e.ravel()))).reshape(e.shape)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("energies are outside xraylib's total-attenuation data range")
    return values


def _dcs(e, theta, kind):
    # Bound-electron incoherent scattering vanishes at zero momentum
    # transfer; the scalar xraylib API reports q=0 as a domain error.
    if kind == "compton" and theta == 0:
        return np.zeros_like(e)
    if kind == "compton":
        # Native DCS_Compt rejects very small momentum transfers encountered
        # in Sun-tracking views. Use its equivalent KN*S(q,Z) expression,
        # with the low-q S proportional to q^2 limit below 0.001 Angstrom^-1.
        q = np.ascontiguousarray(e*np.sin(theta/2)/xraylib.KEV2ANGST)
        factors = xraylib_np.SF_Compt(_Z, np.maximum(q, 0.001))
        factors *= np.minimum(q/0.001, 1)**2
        kn = xraylib_np.DCS_KN(np.ascontiguousarray(e), np.array([theta]))[:, 0]
        return kn * np.einsum("z,ze->e", _W*xraylib.AVOGNUM/_ATOMIC_WEIGHT, factors)
    return np.einsum("z,ze->e", _W, xraylib_np.DCS_Rayl(
        _Z, np.ascontiguousarray(e), np.array([theta]))[:, :, 0])


def ray_column(r, look, t, *, activity=None, top=1000.0, step=5.0):
    """Sample atmospheric rays and return columns and reusable path profiles.

    r: ECEF origin(s) [km], (3,) or (m,3); look: outgoing viewing unit
    direction(s), same shapes, broadcast pointwise; t: scalar UTC time.
    activity: optional f107/f107a/ap mapping forwarded to air_density.
    top: geodetic atmosphere ceiling [km]; step: maximum path spacing [km].
    All rays end at first solid-Earth intersection or exit the atmosphere.
    Output always has a ray axis: column (m,) [g/cm^2], blocked (m,) bool,
    and paths, a list of dicts with s [km], points [ECEF km], rho [kg/m^3],
    and column [g/cm^2] accumulated from the detector-side boundary.
    A blocked ray's column describes only the visible atmosphere; its
    THROUGH-EARTH transmission must still be set to zero by the caller.
    Samples explicitly include closest approach and solid intersections.
    """
    if not np.isfinite([top, step]).all() or top <= 0 or step <= 0:
        raise ValueError("top and step must be finite and positive km")
    if not Time(t).isscalar:
        raise ValueError("t must be scalar")
    r, k = np.broadcast_arrays(np.asarray(r, float), _unit(look))
    if r.ndim not in (1, 2) or r.shape[-1] != 3 or not np.isfinite(r).all():
        raise ValueError("origins must be finite (3,) or (m,3)")
    r, k = np.atleast_2d(r), np.atleast_2d(k)
    paths, blocked = [], []
    for origin, direction in zip(r, k):
        start, end, hit = _geometry(origin, direction, top)
        closest = np.clip(-origin@direction, start, end)
        bounds = np.unique([start, closest, end])
        s = np.unique(np.concatenate([np.linspace(a, b, max(2, int(np.ceil((b-a)/step))+1))
                                      for a, b in zip(bounds[:-1], bounds[1:])])) if len(bounds)>1 else np.array([start])
        paths.append({"s": s, "points": origin+s[:, None]*direction})
        blocked.append(hit)
    if paths:
        all_points = np.concatenate([p["points"] for p in paths])
        density = _density(all_points, t, activity, top)
        offset = 0
        for path in paths:
            size = len(path["s"])
            path["rho"] = density[offset:offset+size]
            # rho kg/m^3 * ds km * 100 = g/cm^2.
            path["column"] = cumulative_trapezoid(path["rho"]*100, path["s"], initial=0)
            offset += size
    return {"column": np.array([p["column"][-1] for p in paths]),
            "blocked": np.array(blocked, dtype=bool), "paths": paths}


def _incoming_columns(points, sun, t, activity, top, step, tolerance):
    """Adaptively interpolate log columns along a smooth viewing ray.

    Each evaluated column still uses <=step path spacing. Midpoint checks
    refine interpolation until relative column error is below tolerance.
    tolerance=0 evaluates every incoming ray (reference calculation).
    """
    if tolerance == 0 or len(points) <= 17:
        return ray_column(points, sun, t, activity=activity, top=top, step=step)["column"]
    anchors = np.unique(np.linspace(0, len(points)-1, 17).astype(int))
    distance = np.linalg.norm(points-points[0], axis=1)
    values = {}

    def evaluate(indices):
        missing = [int(i) for i in indices if int(i) not in values]
        if missing:
            cols = ray_column(points[missing], sun, t, activity=activity, top=top, step=step)["column"]
            values.update(zip(missing, np.log(np.maximum(cols, 1e-300))))

    evaluate(anchors)
    while True:
        pairs = [(int(a), int(b)) for a, b in zip(anchors[:-1], anchors[1:]) if b-a > 1]
        middle = [(a+b)//2 for a, b in pairs]
        evaluate(middle)
        refine = [m for (a, b), m in zip(pairs, middle)
                  if abs(values[m] - (values[a]+(values[b]-values[a])
                                      *(distance[m]-distance[a])/(distance[b]-distance[a]))) > np.log1p(tolerance)]
        if not refine:
            break
        anchors = np.sort(np.concatenate([anchors, refine]))
    # Retain evaluated check points as well as anchors for interpolation.
    indices = sorted(values)
    return np.exp(np.interp(distance, distance[indices], [values[i] for i in indices]))


def solar_scattered(e, r, sun, look, t, goes, temperature, *,
                    activity=None, top=1000.0, step=5.0, paths=None,
                    column_tolerance=1e-3):
    """Return single-scattered solar intensity [ph/cm^2/s/sr/keV], (m,n).

    e: output photon energies [keV], positive increasing (n,); r: detector
    ECEF position [km], (3,); sun: direction TO Sun (3,); look: viewing
    directions (m,3) or (3,); t: scalar UTC; goes [W/m^2], temperature [MK].
    activity/top/step follow ray_column. Optional paths: its reusable path
    list for exactly these r/look/time/settings; omit for standalone use.
    column_tolerance: relative log-column interpolation tolerance along each
    viewing ray, default 0.001; 0 evaluates every incoming ray directly.

    Rayleigh is elastic. Compton uses Ein=Eout/(1-Eout*(1-cos(theta))/511)
    with Jacobian (Ein/Eout)^2. The solar source is evaluated at Ein, even
    outside the output band. Differential cross sections [cm^2/g/sr] and
    total attenuation come from the same xraylib NIST dry-air mass mixture.
    Incoming rays intersecting solid Earth are dark. Illumination uses
    the Sun center (parallel rays), not its finite disk at shadow edges.
    Below momentum transfer 0.001 Angstrom^-1, the incoherent scattering
    factor is continued proportional to q^2 to its zero forward limit.
    """
    e = np.asarray(e, float)
    if not np.isfinite(column_tolerance) or column_tolerance < 0:
        raise ValueError("column_tolerance must be finite and >=0")
    if (e.ndim != 1 or not e.size or not np.isfinite(e).all()
            or np.any(e <= 0) or np.any(np.diff(e) <= 0)):
        raise ValueError("e must be finite positive increasing (n,)")
    sun, looks = _unit(sun), np.atleast_2d(_unit(look))
    if sun.shape != (3,) or np.asarray(r).shape != (3,):
        raise ValueError("r and sun must be (3,)")
    source = solar_spectrum(e, goes, temperature)
    result = np.zeros((len(looks), len(e)))
    if not np.any(source):
        return result
    if paths is None:
        paths = ray_column(r, looks, t, activity=activity, top=top, step=step)["paths"]
    if len(paths) != len(looks):
        raise ValueError("paths must match look directions")
    mu_out = _mac(e)
    for j, (direction, path) in enumerate(zip(looks, paths)):
        cosine = np.clip(sun@direction, -1, 1)
        theta = float(np.arccos(cosine))
        denominator = 1-e*(1-cosine)/510.99895
        if np.any(denominator <= 0):
            raise ValueError("output energies exceed the Compton kinematic limit")
        incident = e/denominator
        # Ignore contributions attenuated by >50 optical depths at every
        # output energy (relative numerical floor <2e-22, not a physical cut).
        keep = (path["rho"] > 0) & (path["column"]*mu_out.min() < 50)
        if not keep.any():
            continue
        points = path["points"][keep]
        # Check shadows before the expensive incoming-column sampling.
        lit = np.array([not _geometry(p, sun, top)[2] for p in points])
        if not lit.any():
            continue
        indices = np.flatnonzero(keep)[lit]
        incoming = _incoming_columns(points[lit], sun, t, activity, top, step, column_tolerance)
        outgoing = path["column"][indices]
        elastic = (source * _dcs(e, theta, "rayleigh")
                   * np.exp(-incoming[:, None]*mu_out))
        compton = (solar_spectrum(incident, goes, temperature)
                   * _dcs(incident, theta, "compton") / denominator**2
                   * np.exp(-incoming[:, None]*_mac(incident)))
        integrand = np.zeros((len(path["s"]), len(e)))
        integrand[indices] = ((elastic+compton) * np.exp(-outgoing[:, None]*mu_out)
                              * path["rho"][indices, None]*100)
        result[j] = np.trapezoid(integrand, path["s"], axis=0)
    return result
