"""Square-pore lobster-eye (micropore) optic, built from geometry and Fresnel.

Model (on-axis point source, spherical optic of radius R = 2f):

- A pore at aperture position x points at the centre of curvature, so an
  on-axis ray meets its walls at grazing angle theta = arcsin(|x| / R).
- Square pores separate into two independent axes. In one axis a ray drifts
  L*tan(theta) across a pore of width d while travelling its length L. With
  u = (L/d)*tan(theta) and a uniform entry point, it reflects
  floor(u) times with probability 1 - frac(u) and floor(u)+1 times with
  probability frac(u). Each reflection multiplies intensity by R(theta, E).
- An odd number of reflections mirrors the ray about the pore axis, sending
  it towards the focus at f = R/2; an even number leaves its direction
  unchanged. Odd in both axes -> central spot; odd in one -> cross arms;
  even in both -> unfocused (includes straight-through rays).
- Because the axes separate, a rectangular aperture of sides Dx, Dy gives
  A_spot = open_fraction * Gx_odd * Gy_odd, with G_odd = integral of the
  odd-reflection weight over that side.
- Reflectance is the Fresnel reflectance of a thick mirror with optical
  constants from xraylib, times a Debye-Waller roughness factor.

Geometric focal spot: within a pore all focused rays leave parallel and
fill the pore exit, so each axis gives a top-hat of full width d centred on
the focus. A pore-axis misalignment of delta tilts the reflected ray by
2*delta, adding 2*f*delta of blur.

Not modelled: off-axis sources, pore shear or figure errors beyond the
single slope_error blur, tile gaps and frames (fold into open_fraction or
scale the result), multilayers, polarization differences (negligible at
grazing incidence), and detector QE (multiply by instrument.silicon_qe).

References: Angel, ApJ 233, 364 (1979), doi:10.1086/157397 (square-pore
lobster geometry); Fresnel reflectance and Debye-Waller roughness factor as
in Als-Nielsen & McMorrow, Elements of Modern X-ray Physics, 2nd ed. (2011),
ch. 3; optical constants from xraylib, Schoonjans et al. (2011),
doi:10.1016/j.sab.2011.09.011.
"""

import numpy as np
import xraylib
from numpy.typing import ArrayLike

HC_KEV_NM = 1.239841984  # photon wavelength [nm] = HC_KEV_NM / E [keV]
E_MIN_KEV, E_MAX_KEV = 0.1, 100.0


def _energies(e):
    e = np.atleast_1d(np.asarray(e, dtype=float))
    if e.ndim != 1 or not np.isfinite(e).all() or np.any((e < E_MIN_KEV) | (e > E_MAX_KEV)):
        raise ValueError(f"e must be finite (n,) energies within {E_MIN_KEV:g}-{E_MAX_KEV:g} keV")
    return e


def optical_constants(e: ArrayLike, material: str = "Ir",
                      density: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta, beta) of the refractive index n = 1 - delta + i*beta.

    Inputs: e, energies [keV], (n,), 0.1 to 100; material, element symbol
    or chemical formula accepted by xraylib (e.g. "Ir", "Au", "SiO2");
    density [g/cm^3], optional for elements (xraylib value), required
    for compounds. Output: two (n,) arrays, dimensionless.
    Reference: xraylib Refractive_Index_Re/Im, Schoonjans et al. (2011).
    """
    e = _energies(e)
    if density is None:
        try:
            density = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(material))
        except ValueError as exc:
            raise ValueError("density is required for compounds") from exc
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and > 0 g/cm^3")
    delta = np.array([1 - xraylib.Refractive_Index_Re(material, x, density) for x in e])
    beta = np.array([xraylib.Refractive_Index_Im(material, x, density) for x in e])
    return delta, beta


def fresnel_reflectance(theta: ArrayLike, delta: ArrayLike, beta: ArrayLike,
                        sigma_over_wavelength: ArrayLike = 0.0) -> np.ndarray:
    """Return thick-mirror X-ray reflectance |r|^2 at grazing angle theta.

    Inputs (broadcast together): theta, grazing angle [rad], >= 0; delta,
    beta, optical constants; sigma_over_wavelength, rms roughness divided by
    wavelength. r = (sin(theta) - k) / (sin(theta) + k) with
    k = sqrt(n^2 - cos^2(theta)), n = 1 - delta + i*beta; times the
    Debye-Waller factor exp(-(4*pi*sigma*sin(theta)/lambda)^2).
    Output: reflectance from 0 to 1.
    Reference: Als-Nielsen & McMorrow (2011), ch. 3.
    """
    theta, delta, beta, rough = np.broadcast_arrays(*(np.asarray(v, float) for v in
                                                      (theta, delta, beta, sigma_over_wavelength)))
    s = np.sin(theta)
    # n^2 - cos^2 written without cancelling 1 - 1.
    k = np.sqrt(s**2 - 2*delta + delta**2 - beta**2 + 2j*beta*(1 - delta))
    r = np.abs((s - k) / (s + k))**2
    return r * np.exp(-(4*np.pi*rough*s)**2)


def mirror_reflectivity(e: ArrayLike, theta: ArrayLike, material: str = "Ir",
                        density: float | None = None, roughness: float = 0.0) -> np.ndarray:
    """Return reflectance for energies e [keV] (n,) and angles theta [rad] (m,).

    roughness: rms [nm] >= 0. Output shape (n, m). See fresnel_reflectance.
    """
    e = _energies(e)
    theta = np.atleast_1d(np.asarray(theta, float))
    if not np.isfinite(roughness) or roughness < 0:
        raise ValueError("roughness must be finite and >= 0 nm")
    delta, beta = optical_constants(e, material, density)
    return fresnel_reflectance(theta[None, :], delta[:, None], beta[:, None],
                               (roughness / (HC_KEV_NM / e))[:, None])


def _axis(e, side, radius, ratio, material, density, roughness, samples):
    """Integrate odd/even reflection weights over one aperture side [cm]."""
    x = np.linspace(0, side / 2, samples)
    theta = np.arcsin(np.minimum(x / radius, 1.0))
    u = ratio * np.tan(theta)
    n0 = np.floor(u)
    frac = u - n0
    if material is None:
        refl = np.ones((len(e), samples))
    else:
        refl = mirror_reflectivity(e, theta, material, density, roughness)
    low, high = (1 - frac) * refl**n0, frac * refl**(n0 + 1)
    low_odd = (n0 % 2 == 1)
    odd = np.where(low_odd, low, high)
    even = np.where(low_odd, high, low)
    return 2*np.trapezoid(odd, x, axis=1), 2*np.trapezoid(even, x, axis=1)


def lobster_eye(
    e: ArrayLike, *, focal_length: float, aperture: float | tuple[float, float],
    pore: float, wall: float, length: float, material: str | None = "Ir",
    density: float | None = None, roughness: float = 0.0,
    slope_error: float = 0.0, samples: int = 4001,
) -> dict:
    """Return on-axis effective areas and sizing figures of a square-pore optic.

    Inputs: e, energies [keV], (n,), 0.1 to 100; focal_length f [cm] > 0
    (optic radius 2f); aperture, square side or (width, height) [cm] > 0,
    each <= 4f; pore, pore width d [um] > 0; wall, wall thickness [um] >= 0;
    length, pore length L [mm] > 0; material, mirror coating for xraylib
    (None = perfect reflector, an upper bound); density [g/cm^3]; roughness,
    rms [nm] >= 0; slope_error, pore-axis misalignment FWHM [arcmin] >= 0;
    samples, integration points per half side.

    Output dict (areas [cm^2], shape (n,), WITHOUT detector QE):
    area_focus (central spot), area_arms (both cross arms), area_unfocused,
    open_fraction = (d/(d+w))^2, geometric_area, length_ratio L/d,
    fov_deg (width, height) = full angle the aperture subtends at the
    centre of curvature, critical_angle_deg = sqrt(2*delta) per energy,
    live_halfwidth_cm = 2f*sin(critical angle), the aperture half-width
    inside which walls reflect efficiently, and spot_mm = focal spot full
    width ~ hypot(d, 2*f*slope_error), with spot_arcmin = spot / f.
    With material=None, focus + arms + unfocused = open_fraction * area.
    See the module docstring for assumptions and references.
    """
    e = _energies(e)
    width, height = (aperture, aperture) if np.ndim(aperture) == 0 else aperture
    values = [focal_length, width, height, pore, wall, length, roughness, slope_error]
    if not np.isfinite(values).all():
        raise ValueError("geometry inputs must be finite")
    if focal_length <= 0 or pore <= 0 or length <= 0 or wall < 0 or roughness < 0 or slope_error < 0:
        raise ValueError("need focal_length, pore, length > 0 and wall, roughness, slope_error >= 0")
    radius = 2 * focal_length
    if not (0 < width <= 2*radius and 0 < height <= 2*radius):
        raise ValueError("aperture sides must be > 0 and <= 4 * focal_length")
    if not isinstance(samples, (int, np.integer)) or samples < 3:
        raise ValueError("samples must be an integer >= 3")
    ratio = length * 1000 / pore
    gx_odd, gx_even = _axis(e, width, radius, ratio, material, density, roughness, samples)
    gy_odd, gy_even = _axis(e, height, radius, ratio, material, density, roughness, samples)
    open_fraction = (pore / (pore + wall))**2
    if material is None:
        critical = np.full(len(e), np.nan)
    else:
        critical = np.sqrt(2 * optical_constants(e, material, density)[0])
    spot = float(np.hypot(pore * 1e-3, 2 * focal_length * 10 * np.deg2rad(slope_error / 60)))
    return dict(
        energy_keV=e,
        area_focus=open_fraction * gx_odd * gy_odd,
        area_arms=open_fraction * (gx_odd * gy_even + gx_even * gy_odd),
        area_unfocused=open_fraction * gx_even * gy_even,
        open_fraction=open_fraction,
        geometric_area=width * height,
        length_ratio=ratio,
        fov_deg=tuple(float(np.degrees(2 * np.arcsin(s / 2 / radius))) for s in (width, height)),
        critical_angle_deg=np.degrees(critical),
        live_halfwidth_cm=radius * np.sin(critical),
        spot_mm=spot,
        spot_arcmin=float(np.degrees(spot / (focal_length * 10)) * 60),
    )
