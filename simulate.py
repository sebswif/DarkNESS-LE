"""Trade-study simulation: bare vs lobster-eye CCD payloads in low Earth orbit.

Chain: solar axion flux -> RK4 orbit and Sun direction -> IGRF field along
the Sun line of sight -> MSIS air (plasma mass, absorption) -> conversion
probability -> payload response (silicon QE, lobster optic) -> CXB, dark
Earth, solar direct/scattered and particle backgrounds -> expected counts
-> Poisson upper limit -> coupling limit.

Each epoch simulates one orbit; epochs are spread evenly through a year and
averaged, then scaled to the mission length. Background levels and optic
quality are ILLUSTRATIVE inputs, not calibrated values (see memory/).

Limit procedure (explicitly chosen): observed counts = round(expected
background), 95% Helene upper limit on signal, then g ∝ counts^(1/4).

Run: python simulate.py            (about 1-2 min per payload per epoch)
     python simulate.py --epochs 1 --step 120   (quick look)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time

from functions.geometry import MU_EARTH_KM3_S2, R_EQUATOR_KM
from functions.orbit_counts import simulate_counts
from functions.payload import bare_ccd, lobster_ccd
from functions.sensitivity import limiting_coupling, poisson_upper_limit

OPTIC = dict(pore=40, wall=8, length=2.4, material="Ir", roughness=0.0, slope_error=3.0)


def payloads(e, thickness):
    """Configurations under study. Focal length from the bus long axis."""
    return {
        "bare 10 cm2, 5 deg": bare_ccd(10, 5.0, thickness),
        "lobster 6U (2x3x1) f=30": lobster_ccd(e, {**OPTIC, "focal_length": 30, "aperture": (20, 9)}, thickness),
        "lobster 8U (2x4x1) f=40": lobster_ccd(e, {**OPTIC, "focal_length": 40, "aperture": (20, 9)}, thickness),
        "lobster linear 6U f=60": lobster_ccd(e, {**OPTIC, "focal_length": 60, "aperture": (9, 9)}, thickness),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=4, help="orbits sampled through the year")
    p.add_argument("--step", type=float, default=60.0, help="time bin [s]")
    p.add_argument("--emin", type=float, default=1.0, help="keV, >= 1")
    p.add_argument("--emax", type=float, default=10.0, help="keV")
    p.add_argument("--ne", type=int, default=32, help="energy samples")
    p.add_argument("--g", type=float, default=1e-10, help="reference coupling [GeV^-1]")
    p.add_argument("--goes", type=float, default=0.0,
                   help="GOES 1-8 A flux [W/m^2]; > 0 adds solar direct + scattered (slow)")
    p.add_argument("--h", type=float, default=500.0, help="altitude [km]")
    p.add_argument("--inc", type=float, default=51.6, help="inclination [deg]")
    p.add_argument("--years", type=float, default=1.0, help="mission length")
    p.add_argument("--live", type=float, default=1.0, help="live fraction [0-1]")
    p.add_argument("--thickness", type=float, default=50.0, help="CCD depletion [um]")
    p.add_argument("--out", default="output/trade_results.json")
    a = p.parse_args()
    warnings.filterwarnings("ignore")

    e = np.geomspace(a.emin, a.emax, a.ne)
    period = 2*np.pi*np.sqrt((R_EQUATOR_KM + a.h)**3 / MU_EARTH_KM3_S2)
    orbits = a.years * 365.25 * 86400 / period
    starts = Time("2026-01-01T00:00:00", scale="utc") + np.arange(a.epochs) * (365.25 / a.epochs) * u.day
    common = dict(g=a.g, goes=a.goes, temperature=10.0, live_fraction=a.live,
                  earth_c=1.48e-2,            # dark-Earth normalization (Ajello 2008 shape)
                  particle_level=1.26e-3,     # Suzaku XIS0-level NXB [ct/cm^2/s/keV]
                  particle_alpha=0.5, particle_ref=10.0, particle_min=0.1,
                  activity=dict(f107=150.0, f107a=150.0, ap=4.0),  # offline MSIS
                  fov_radial=4, fov_azimuth=8)

    print(f"{a.emin:g}-{a.emax:g} keV, g={a.g:g}, {a.h:g} km / {a.inc:g} deg, "
          f"{a.epochs} epoch(s), {a.step:g} s bins, {orbits:.0f} orbits in {a.years:g} yr. ILLUSTRATIVE backgrounds.")
    results = {}
    for name, payload in payloads(e, a.thickness).items():
        t0, per_orbit = time.time(), []
        for start in starts:
            edges = start + np.append(np.arange(0, period, a.step), period) * u.s
            run = simulate_counts(e, edges, **payload, orbit=dict(h=a.h, inc=a.inc, epoch=start), **common)
            per_orbit.append(run["totals"])
        mean = {k: float(np.mean([t[k] for t in per_orbit])) for k in per_orbit[0]}
        total = {k: v * orbits for k, v in mean.items()}
        upper = poisson_upper_limit(int(round(total["background"])), total["background"])
        g_limit = limiting_coupling(upper, total["signal"], a.g) if total["signal"] > 0 else float("inf")
        results[name] = dict(fov_deg=payload["fov"], particle_area_cm2=payload["det_area"],
                             counts_per_orbit=mean, counts_mission=total,
                             upper_limit_counts=upper, g_limit=g_limit)
        print(f"{name:26s} signal {total['signal']:9.3g}  background {total['background']:9.3g} "
              f"(particle {total['particle']:.2g}, cxb {total['cxb']:.2g}, earth {total['dark_earth']:.2g}, "
              f"solar {total['solar_direct'] + total['solar_scattered']:.2g})  "
              f"g < {g_limit:.3g} GeV^-1   [{time.time() - t0:.0f} s]")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(settings=vars(a), results=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
