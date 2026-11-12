"""Integration of spectra already folded through detector response."""

import numpy as np
from numpy.typing import ArrayLike


def background_budget(e: ArrayLike, exposure: float,
                      components: dict[str, ArrayLike]) -> dict[str, float]:
    """Return expected component counts and their total for one interval.

    e: increasing energy samples [keV], (n,), n>=2. exposure: live time
    [s] >=0. components: named differential DETECTOR rates [counts/s/keV],
    each scalar or (n,). Geometry, area, QE and transmission must already
    be included. Output maps component names to float counts, plus 'total'
    (a reserved name). Replaces the former area/fill interface.
    """
    e = np.asarray(e, float)
    if (e.ndim != 1 or e.size < 2 or not np.isfinite(e).all()
            or np.any(e <= 0) or np.any(np.diff(e) <= 0)):
        raise ValueError("e must contain >=2 finite positive increasing samples")
    if not np.isfinite(exposure) or exposure < 0 or "total" in components:
        raise ValueError("exposure must be finite and >=0; 'total' is reserved")
    result = {}
    for name, rate in components.items():
        rate = np.asarray(rate, float)
        if rate.shape not in ((), e.shape) or not np.isfinite(rate).all() or np.any(rate < 0):
            raise ValueError(f"{name}: rate must be finite, nonnegative, scalar or shape {e.shape}")
        result[name] = float(exposure * np.trapezoid(np.broadcast_to(rate, e.shape), e))
    result["total"] = sum(result.values())
    return result
