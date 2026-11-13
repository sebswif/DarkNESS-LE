"""Bare and lobster-eye CCD payloads as keyword arguments for simulate_counts.

Usage: simulate_counts(e, time_edges, **bare_ccd(...), ...) or
**lobster_ccd(e, ...). Each returns response(e, mu) [cm^2, incl. QE],
det_area (particle-collecting area [cm^2]), fov (half-angle [deg]) and
core: None, or (solid angle [sr], area [cm^2] (n,)), a narrow on-axis term
too small for the field-of-view quadrature; simulate_counts applies it to
the boresight for diffuse sources and adds it to the signal area.

Lobster (square counting region of side s at the focus, area S, focal
length f, aperture footprint Dx*Dy, optic field of view Dx*Dy/(2f)^2 sr):
- core: focused photons, solid angle S/f^2, area A_focus * QE * enclosed,
  with the spot taken as a Gaussian of FWHM spot_mm per axis.
- response: arm and unfocused photons travel straight in their unfocused
  axis and spread over Dx*Dy. Averaged over the field of view, a fraction
  2*S/(Dx*Dy) of arm photons and S/(Dx*Dy) of unfocused photons from each
  direction lands in the region.
- det_area = S: particles are counted only in the region.
Arm photons from the on-axis source itself (~s/D of the arm area) are
ignored, slightly underestimating the signal. Approximate near the field edge.
"""

import numpy as np
from scipy.special import erf

from .instrument import silicon_qe
from .lobster import lobster_eye


def bare_ccd(area: float, fov: float, thickness: float = 50.0) -> dict:
    """CCD of area [cm^2] and depletion thickness [um] behind a collimator
    of half-angle fov [deg]; ideal Si QE, no filter, no optics."""
    return dict(response=lambda e, mu: area * silicon_qe(e, thickness) * max(mu, 0.0),
                det_area=area, fov=fov, core=None)


def lobster_ccd(e, optic: dict, thickness: float = 50.0,
                extraction_mm: float | None = None) -> dict:
    """CCD at the focus of lobster.lobster_eye(e, **optic).

    e: the energy grid [keV] that simulate_counts will use. thickness:
    depletion [um]. extraction_mm: side of the square counting region
    [mm], default 2 * spot_mm.
    """
    o = lobster_eye(e, **optic)
    f = optic["focal_length"]  # cm
    side = 2 * o["spot_mm"] if extraction_mm is None else extraction_mm
    if not np.isfinite(side) or side <= 0:
        raise ValueError("extraction_mm must be finite and > 0")
    sigma = o["spot_mm"] / (2 * np.sqrt(2 * np.log(2)))
    enclosed = erf(side / (2 * np.sqrt(2) * sigma))**2
    region = (side / 10)**2  # cm^2
    qe = silicon_qe(o["energy_keV"], thickness)
    spread = (2 * o["area_arms"] + o["area_unfocused"]) * region / o["geometric_area"] * qe
    omega_fov = o["geometric_area"] / (2 * f)**2
    return dict(response=lambda e, mu: spread, det_area=region,
                fov=float(np.degrees(np.arccos(1 - omega_fov / (2 * np.pi)))),
                core=(region / f**2, o["area_focus"] * enclosed * qe))
