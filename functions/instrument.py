"""Silicon detection efficiency and simple collecting areas."""

import numpy as np
import xraylib_np
from numpy.typing import ArrayLike

E_MIN_KEV, E_MAX_KEV = 0.1, 100.0  # silicon_qe energy domain
_SI = np.array([14])


def _fraction(value: ArrayLike, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def silicon_qe(e: ArrayLike, thickness: float = 250.0) -> float | np.ndarray:
    """Return the fraction of photons absorbed photoelectrically in silicon.

    Inputs: e, energy [keV], number, list, or NumPy array, 0.1 to 100 keV;
    thickness, active silicon depth [micrometres], float or int >= 0.
    Output: efficiency from 0 to 1, float or array matching e.

    QE = 1 - exp(-mac * rho * thickness), with thickness converted to cm
    and rho = 2.33 g/cm^3. mac is the silicon photoelectric cross section
    [cm^2/g] from xraylib (CS_Photo), evaluated at each energy, so the
    K edge at 1.839 keV is exact rather than interpolated across.
    Checked against the bundled NIST table data/SiPhotoElectricMAC.txt
    (2-50 keV); below 2 keV a published check is still pending.
    Assumes normal incidence and full charge collection in the active layer.
    Windows, dead layers, charge loss, and fluorescence escape are omitted;
    this is an ideal absorption efficiency, not a calibrated detector response.
    Reference: xraylib, Schoonjans et al. (2011), doi:10.1016/j.sab.2011.09.011;
    NIST XCOM, https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
    Density: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html
    """
    e = np.asarray(e, dtype=float)
    if not np.isfinite(thickness) or thickness < 0:
        raise ValueError("thickness must be finite and >= 0 micrometres")
    if not np.isfinite(e).all() or np.any((e < E_MIN_KEV) | (e > E_MAX_KEV)):
        raise ValueError(f"e must be finite and within {E_MIN_KEV:g}-{E_MAX_KEV:g} keV")
    coefficient = xraylib_np.CS_Photo(_SI, np.ascontiguousarray(e.ravel()))[0].reshape(e.shape)
    qe = -np.expm1(-coefficient * 2.33 * thickness * 1e-4)
    return float(qe) if qe.ndim == 0 else qe


def bare_effective_area(
    e: ArrayLike, area: float, thickness: float = 250.0,
    transmission: ArrayLike = 1.0,
) -> float | np.ndarray:
    """Return bare silicon detector effective area [cm^2].

    Inputs: e, energy [keV], number/list/NumPy array, 0.1 to 100;
    area, exposed detector area [cm^2], float or int >= 0;
    thickness, active silicon depth [micrometres], float or int >= 0;
    transmission, window/filter transmission [0 to 1], number or array
    matching e. Default 1 means no window/filter loss.
    Output: float or NumPy array; A_eff = area * transmission * silicon_qe.
    Assumes normal incidence. Area excludes masked or inactive regions.
    Energy dependence of transmission must be supplied by the caller.
    Reference: NIST XCOM absorption law; see silicon_qe.
    """
    if not np.isfinite(area) or area < 0:
        raise ValueError("area must be finite and >= 0 cm^2")
    transmission = np.broadcast_to(_fraction(transmission, "transmission"), np.shape(e))
    result = area * transmission * silicon_qe(e, thickness)
    return float(result) if np.ndim(result) == 0 else result


def lobster_effective_area(
    e: ArrayLike, area: float, reflectivity: ArrayLike,
    open_fraction: float, focus_fraction: ArrayLike,
    thickness: float = 250.0, transmission: ArrayLike = 1.0,
) -> float | np.ndarray:
    """Return a simple two-reflection lobster-eye central-spot area [cm^2].

    Inputs: e, energy [keV], number/list/NumPy array, 0.1 to 100;
    area, projected entrance aperture [cm^2], float or int >= 0;
    reflectivity, probability per wall reflection [0 to 1], number or array;
    open_fraction, fraction of aperture not blocked by pore walls [0 to 1],
    float or int; focus_fraction, fraction of rays entering open pores that
    geometrically reach the selected central spot with two reflections
    [0 to 1], number or array, BEFORE reflection losses;
    thickness, active silicon depth [micrometres], float or int >= 0;
    transmission, window/filter transmission [0 to 1], number or array.
    Array inputs must match e. Output: float or array matching e.

    A_eff = area * open_fraction * focus_fraction * reflectivity^2
            * transmission * silicon_qe(e, thickness).
    Assumes the same reflectivity for each of two perpendicular reflections.
    Supply reflectivity for the energy, coating, and grazing angle, and a
    geometry-derived or measured focus_fraction; neither is universal.
    Includes only the central spot, not single-reflection cross arms or
    unreflected photons. No ray tracing or automatic off-axis correction.
    Do not pass an already calibrated effective area as the aperture area.
    Reference: Angel, ApJ 233, 364 (1979), doi:10.1086/157397;
    this product is a simplified response model based on that geometry.
    """
    reflectivity = np.broadcast_to(_fraction(reflectivity, "reflectivity"), np.shape(e))
    focus_fraction = np.broadcast_to(_fraction(focus_fraction, "focus_fraction"), np.shape(e))
    open_fraction = _fraction(open_fraction, "open_fraction")
    if open_fraction.ndim != 0:
        raise ValueError("open_fraction must be a single number")
    result = (bare_effective_area(e, area, thickness, transmission)
              * open_fraction * focus_fraction * reflectivity**2)
    return float(result) if np.ndim(result) == 0 else result
