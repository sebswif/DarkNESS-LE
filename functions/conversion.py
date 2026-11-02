"""Axion-photon conversion in uniform fields and sampled paths."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid


def conversion_probability_uniform(b: float, length: float, g: float) -> float:
    """Calculate vacuum conversion in a uniform transverse field.

    Inputs: b, field [T]; length, path length [m]; g, coupling [GeV^-1].
    Output: dimensionless probability; assumes coherence and weak mixing.
    Reference: Raffelt & Stodolsky (1988), doi:10.1103/PhysRevD.37.1237.
    """
    b = b * 195.0  # eV^2
    length = length * 5.07e6  # eV^-1
    g = g * 1e-9  # eV^-1
    return (g * b * length / 2) ** 2


def conversion_probability_path(
    s: ArrayLike, b: ArrayLike, e: float, g: float, ma: float = 0.0,
    mg: ArrayLike = 0.0, attenuation: ArrayLike = 0.0,
) -> float:
    """Return axion-to-photon probability along a sampled straight path.

    Inputs: s, increasing distances [m], (n,), n >= 2, starting at zero
    at the detector and extending toward the source (same order as los);
    b, signed transverse field [T], (n,) for fixed direction or (n, 3)
    for vectors in one fixed frame; e, photon energy [keV], scalar > 0;
    g, coupling [GeV^-1]; ma, axion mass [eV], scalar >= 0;
    mg, photon mass [eV], scalar or (n,); attenuation, intensity loss
    coefficient Gamma [1/m], scalar or (n,), both >= 0.
    Types: s and b accept numeric lists or NumPy arrays; mg and attenuation
    also accept single numbers (ArrayLike). e, g, ma are single numbers
    (float or int). Returns a float.

    Computes P = (g/2)^2 * |integral B(s)*exp(i*phase(s)-tau(s)/2) ds|^2,
    phase(s) = integral_0^s (ma^2-mg^2)/(2*E) ds, tau(s) = integral_0^s
    Gamma ds, converting inputs to eV units. Vector amplitudes are summed
    before taking their squared norm; pass transverse vectors, not |B|.
    Absorption is from each conversion point back to the detector.
    Assumes weak conversion (P << 1), straight travel, and X-ray energies
    much larger than either mass. Uses trapezoids: refine samples until P
    converges, especially when phase or absorption changes rapidly.
    Reference: Davoudiasl & Huber (2006), hep-ph/0509293, Eqs. (2)-(3),
    doi:10.1103/PhysRevLett.97.141302; extended to sampled varying fields.
    """
    s, b = np.asarray(s, float), np.asarray(b, float)
    if s.ndim != 1 or s.size < 2 or not np.isfinite(s).all() or s[0] != 0 or np.any(np.diff(s) <= 0):
        raise ValueError("s must start at zero and increase, with >= 2 finite samples")
    if b.shape not in ((s.size,), (s.size, 3)) or not np.isfinite(b).all():
        raise ValueError("b must be finite with shape (n,) or (n, 3)")
    if not np.isfinite([e, g, ma]).all() or e <= 0 or ma < 0:
        raise ValueError("e must be > 0, ma >= 0, and e, g, ma finite scalars")
    mg, attenuation = np.asarray(mg, float), np.asarray(attenuation, float)
    for name, value in (("mg", mg), ("attenuation", attenuation)):
        if value.shape not in ((), s.shape) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and >= 0, scalar or shape (n,)")
    q = np.broadcast_to((ma**2 - mg**2) / (2 * e * 1000) * 5.07e6, s.shape)
    phase = cumulative_trapezoid(q, s, initial=0)
    tau = cumulative_trapezoid(np.broadcast_to(attenuation, s.shape), s, initial=0)
    weight = np.exp(1j * phase - tau / 2)
    if b.ndim == 2:
        weight = weight[:, None]
    amplitude = np.trapezoid(b * weight, s, axis=0)
    return float((g * 1e-9 * 195.0 * 5.07e6 / 2)**2
                 * np.sum(np.abs(amplitude)**2))


if __name__ == "__main__":
    print(conversion_probability_uniform(9, 9.26, 1e-10))
