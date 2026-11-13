"""Expected detector counts along an orbit; no plotting or random events."""

from collections.abc import Callable

import numpy as np
from astropy import units as u
from astropy.time import Time

from .atmosphere import air_attenuation, photon_plasma_mass
from .background import background_budget, cxb, dark_earth, nxb, ray_column, solar_scattered, solar_spectrum
from .background.transport import _density, _geometry, _mac, _tangent, _unit
from .bfield import cutoff_rigidity
from .conversion import conversion_probability_path
from .geometry import orbit_position, sun_direction
from .lineofsight import los
from .solar_flux import axion_flux


def _cone(axis, radius, nr, na):
    """Gauss-Legendre solid-angle quadrature about axis, angles in radians."""
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in (nr, na)):
        raise ValueError("radial and azimuth sample counts must be positive integers")
    x, w = np.polynomial.legendre.leggauss(nr)
    # Stable 1-cos(radius) for small apparent solar disks.
    extent = 2*np.sin(radius/2)**2
    mu = 1-(x+1)*extent/2
    phi = (np.arange(na)+0.5)*2*np.pi/na
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    east = _unit(np.cross(axis, reference))
    north = np.cross(axis, east)
    transverse = np.cos(phi)[:, None]*east + np.sin(phi)[:, None]*north
    directions = (mu[:, None, None]*axis
                  + np.sqrt(np.maximum(0, 1-mu**2))[:, None, None]*transverse)
    weights = np.repeat(w*extent/2 * 2*np.pi/na, na)
    return directions.reshape(-1, 3), weights


def _series(value, n, name, *, positive=False):
    value = np.asarray(value, float)
    if value.shape not in ((), (n,)) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite, scalar or shape ({n},)")
    if np.any(value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be {'>0' if positive else '>=0'}")
    return np.broadcast_to(value, (n,))


def _areas(response, e, cosines, fov):
    areas = []
    for mu in np.atleast_1d(cosines):
        if mu < np.cos(fov)-1e-14:
            areas.append(np.zeros_like(e))
            continue
        value = np.asarray(response(e, float(mu)), float)
        if value.shape not in ((), e.shape) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("response(e, mu) must return finite nonnegative cm^2, scalar or shape e")
        areas.append(np.broadcast_to(value, e.shape))
    return np.asarray(areas)


def simulate_counts(
    e, time_edges, *, response: Callable, det_area: float, fov: float,
    g: float, goes, temperature, earth_c, particle_level,
    particle_alpha, particle_ref, particle_min, lines=(), ma: float = 0.0,
    core: tuple | None = None,
    orbit: dict | None = None, activity: dict | None = None,
    live_fraction=1.0, grazing: float = 120.0, solar_radius: float = 16.0,
    top: float = 1000.0, path_step: float = 5.0,
    fov_radial: int = 8, fov_azimuth: int = 24,
    solar_radial: int = 4, solar_azimuth: int = 16,
    signal_length: float = 10000.0, column_tolerance: float = 1e-3,
) -> dict:
    """Return time-resolved expected signal and background detector counts.

    e: positive increasing energy grid [keV], (ne,), >=2; restricted to
    1-50 keV (CXB/Si domain). time_edges: UTC Time or ISO sequence,
    (nt+1,), strictly increasing; midpoint rates are used in each interval.
    response(e, mu): effective photon area [cm^2], scalar or (ne,), already
    including QE; mu=cos(angle to Sun-tracking boresight). Hard circular FOV
    is applied here. det_area: detector region area [cm^2]; fov: half-angle
    [deg], (0,90]. g [GeV^-1], ma [eV]. No saturation/dead-time model.
    core: optional (solid angle [sr], area [cm^2] (ne,)) narrow on-axis
    term (e.g. a focusing optic's spot), applied to the boresight for
    diffuse sources and solar-disk points inside it, and added to the
    signal area. payload.bare_ccd/lobster_ccd supply response, det_area,
    fov and core together.

    goes [W/m^2], temperature [MK], earth_c [ph/cm^2/s/sr/keV],
    particle_level [counts/cm^2/s/keV], particle_alpha (>=0), particle_ref
    and particle_min [GV] (>0) accept scalar or (nt,). Rigidity continuum
    scaling = (max(Rc,particle_min)/particle_ref)**(-particle_alpha).
    lines: (nlines,3) or (nt,nlines,3), as in nxb; lines are not scaled.
    All uncalibrated normalizations are required explicitly.

    orbit: kwargs for orbit_position; defaults h=500, inc=97.4,
    epoch=time_edges[0], ECEF. activity: optional f107/f107a/ap mapping,
    each scalar or (nt,); omit all for MSIS historical lookup, which may
    require network/data. live_fraction: scalar or (nt,), [0,1].
    grazing: Sun-center maximum tangent altitude [km], default120; observe
    only for Earthward rays. solar_radius: uniform solar disk [arcmin],
    default16. top/path_step: atmospheric ceiling/maximum spacing [km].
    Quadrature resolutions are configurable. signal_length: magnetic
    integration limit [km]; los also stops at its existing 60-km shell.
    column_tolerance: incoming-column interpolation error control, default
    0.001; set 0 for direct evaluation of every scattering source point.

    Output: energy_keV, time_edges and time_mid (Time), positions_ecef_km
    (nt,3), observing (nt,), live_exposure_s, tangent_altitude_km,
    earth_fraction, solar_visible_fraction (geometric, before absorption),
    cutoff_rigidity_gv; rates (dict of (nt,ne), counts/s/keV); counts and
    cumulative (dict of (nt,), counts); totals (dict of float counts).
    Components: signal, cxb, dark_earth, solar_direct, solar_scattered,
    particle, background, total. Ineligible intervals have zero rates and
    counts but retain geometry. Zero-live intervals have zero counts.

    Dark Earth is uniform empirical emission on solid-Earth-covered rays.
    Signal uses the solar center, direct sunlight a uniform finite disk,
    scattered illumination the Sun center. No aurora, fluorescence,
    multiple scattering, SAA/trapped bursts, activation or redistribution.
    Dry-air composition at MSIS density is a forecasting approximation.
    Refine time/energy/angular/path samples until relevant counts converge.
    """
    e = np.asarray(e, float)
    if (e.ndim != 1 or e.size < 2 or not np.isfinite(e).all()
            or np.any((e < 1) | (e > 50)) or np.any(np.diff(e) <= 0)):
        raise ValueError("e must be increasing finite (ne,), >=2 samples in 1-50 keV")
    edges = Time(time_edges, scale="utc")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges.jd)):
        raise ValueError("time_edges must be a finite UTC sequence of >=2 times")
    durations = np.asarray((edges[1:]-edges[:-1]).to_value(u.s))
    if np.any(durations <= 0):
        raise ValueError("time_edges must strictly increase")
    if (not np.isfinite([det_area, fov, g, ma, grazing, solar_radius, top, path_step, signal_length, column_tolerance]).all()
            or det_area < 0 or not 0 < fov <= 90 or g < 0 or ma < 0
            or not 0 <= grazing < top or not 0 < solar_radius < 5400
            or path_step <= 0 or signal_length <= 0 or column_tolerance < 0 or not callable(response)):
        raise ValueError("invalid detector, source, grazing, or path parameters")
    nt = len(durations)
    params = {name: _series(value, nt, name, positive=name in ("temperature", "particle_ref", "particle_min"))
              for name, value in dict(goes=goes, temperature=temperature, earth_c=earth_c,
                                      particle_level=particle_level, particle_alpha=particle_alpha,
                                      particle_ref=particle_ref, particle_min=particle_min).items()}
    live = _series(live_fraction, nt, "live_fraction")
    if np.any(live > 1):
        raise ValueError("live_fraction must be <=1")
    activity = {} if activity is None else activity
    if activity and set(activity) != {"f107", "f107a", "ap"}:
        raise ValueError("activity requires exactly f107, f107a, and ap")
    activity_series = {k: _series(v, nt, k, positive=k != "ap") for k, v in activity.items()}
    lines = np.asarray(lines, float)
    if lines.shape == (0,):
        lines = np.empty((0, 3))
    if lines.ndim == 2:
        nxb(e[:1], 0, lines)  # Validate constant calibration even outside observing windows.
        lines = np.broadcast_to(lines, (nt,)+lines.shape)
    elif lines.ndim == 3 and lines.shape[0] == nt:
        for row in lines:
            nxb(e[:1], 0, row)
    else:
        raise ValueError("lines must be (nlines,3) or (nt,nlines,3)")
    fov_angle = np.deg2rad(fov)
    disk_angle = np.deg2rad(solar_radius/60)
    # Validate quadrature and evaluate fixed instrumental response just once.
    ref = np.array([0., 0., 1.])
    cone, omega = _cone(ref, fov_angle, fov_radial, fov_azimuth)
    area_view = _areas(response, e, cone@ref, fov_angle)
    disk, disk_weights = _cone(ref, disk_angle, solar_radial, solar_azimuth)
    disk_weights /= disk_weights.sum()
    area_disk = _areas(response, e, disk@ref, fov_angle)
    area_center = _areas(response, e, [1.0], fov_angle)[0]
    omega_all, area_all = omega, area_view
    if core is not None:
        omega_core, area_core = float(core[0]), np.broadcast_to(np.asarray(core[1], float), e.shape)
        if not 0 < omega_core < 2*np.pi or not np.isfinite(area_core).all() or np.any(area_core < 0):
            raise ValueError("core must be (solid angle in (0, 2pi) sr, finite area >= 0 cm^2)")
        area_center = area_center + area_core
        area_disk = area_disk + np.outer(disk@ref >= 1 - omega_core/(2*np.pi), area_core)
        omega_all = np.append(omega, omega_core)
        area_all = np.vstack([area_view, area_core])
    mu = _mac(e)
    mids = edges[:-1]+durations/2*u.s
    orbit_options = dict(h=500.0, inc=97.4, epoch=edges[0])
    orbit_options.update(orbit or {})
    if orbit_options.pop("frame", "ecef").lower() != "ecef":
        raise ValueError("orbit frame must be ecef")
    positions = orbit_position(mids, frame="ecef", **orbit_options)
    suns = sun_direction(mids, frame="ecef")
    names = ("signal", "cxb", "dark_earth", "solar_direct", "solar_scattered", "particle")
    rates = {k: np.zeros((nt, len(e))) for k in names}
    tangent, earth_fill, visible, rigidity = np.zeros((4, nt))
    observing = np.zeros(nt, dtype=bool)
    for i, (r, sun, t) in enumerate(zip(positions, suns, mids)):
        act = {k: v[i] for k, v in activity_series.items()}
        views, _ = _cone(sun, fov_angle, fov_radial, fov_azimuth)
        if core is not None:
            views = np.vstack([views, sun])  # boresight carries the core term
        solar_views, _ = _cone(sun, disk_angle, solar_radial, solar_azimuth)
        covered = np.array([_geometry(r, direction, top)[2] for direction in views])
        solar_blocked = np.array([_geometry(r, direction, top)[2] for direction in solar_views])
        earth_fill[i] = omega@covered[:len(omega)]/omega.sum()
        visible[i] = disk_weights@(~solar_blocked)
        tangent[i] = _tangent(r, sun)
        rigidity[i] = cutoff_rigidity(r, t)
        observing[i] = r@sun < 0 and tangent[i] < grazing
        if not observing[i]:
            continue
        p = {k: v[i] for k, v in params.items()}
        paths = ray_column(r, views, t, activity=act, top=top, step=path_step)
        transmission = np.exp(-paths["column"][:, None]*mu)
        transmission[covered] = 0
        rates["cxb"][i] = np.sum(omega_all[:, None]*area_all*transmission, axis=0)*cxb(e)
        rates["dark_earth"][i] = np.sum(omega_all[covered, None]*area_all[covered], axis=0)*dark_earth(e, p["earth_c"])
        if p["goes"] > 0:
            solar_paths = ray_column(r, solar_views, t, activity=act, top=top, step=path_step)
            direct = np.exp(-solar_paths["column"][:, None]*mu)
            direct[solar_blocked] = 0
            rates["solar_direct"][i] = (np.sum(disk_weights[:, None]*area_disk*direct, axis=0)
                                         * solar_spectrum(e, p["goes"], p["temperature"]))
            scattered = solar_scattered(e, r, sun, views, t, p["goes"], p["temperature"],
                                         activity=act, top=top, step=path_step, paths=paths["paths"],
                                         column_tolerance=column_tolerance)
            rates["solar_scattered"][i] = np.sum(omega_all[:, None]*area_all*scattered, axis=0)
        scale = (max(rigidity[i], p["particle_min"])/p["particle_ref"])**-p["particle_alpha"]
        rates["particle"][i] = det_area*nxb(e, p["particle_level"], lines[i], scale)
        if g != 0 and np.any(area_center):
            ray = los(r, sun, t, length=signal_length,
                      n=max(2, int(np.ceil(signal_length/path_step))+1), report=False)
            rho = _density(ray["path_ecef_km"], t, act, top)
            mg = photon_plasma_mass(rho)
            probabilities = np.array([
                conversion_probability_path(ray["distance_km"]*1000, ray["B_perp_ecef_T"],
                                            energy, g, ma, mg, air_attenuation(rho, coefficient))
                for energy, coefficient in zip(e, mu)])
            rates["signal"][i] = area_center*axion_flux(e, g)*probabilities
    rates["background"] = sum(rates[k] for k in names if k != "signal")
    rates["total"] = rates["signal"]+rates["background"]
    exposure = durations*live*observing
    counts = {k: np.zeros(nt) for k in rates}
    for i in range(nt):
        budget = background_budget(e, exposure[i], {k: rates[k][i] for k in names if k != "signal"})
        for k in names:
            counts[k][i] = (float(exposure[i]*np.trapezoid(rates[k][i], e))
                            if k == "signal" else budget[k])
        counts["background"][i] = budget["total"]
        counts["total"][i] = budget["total"]+counts["signal"][i]
    return dict(energy_keV=e.copy(), time_edges=edges, time_mid=mids,
                positions_ecef_km=positions, observing=observing,
                live_exposure_s=exposure, tangent_altitude_km=tangent,
                earth_fraction=earth_fill, solar_visible_fraction=visible,
                cutoff_rigidity_gv=rigidity, rates=rates, counts=counts,
                cumulative={k: np.cumsum(v) for k, v in counts.items()},
                totals={k: float(v.sum()) for k, v in counts.items()})
