"""Figure: how strongly axions convert as the spacecraft crosses the night side.

Run from this folder:  python fig_conversion_map.py
Output: conversion_map.png beside this script

The idea in one line: axions from the Sun pass through the Earth and turn
into X-rays in the magnetic field between the atmosphere and the
spacecraft. The conversion probability is

    P = (g / 2 * |integral of B_perp ds|)^2,

where B_perp is the part of the field at right angles to the ray. So we
only need the field integral along each ray, |int B_perp ds| [T m].

Steps:
  1. Fly the spacecraft for two orbits and compute the field integral
     every 30 s (only while the Earth hides the Sun).
  2. Pick one complete pass behind the Earth.
  3. At the middle of that pass, compute the field integral for a grid
     of positions all over the globe -> a map.
  4. Plot (a) the map with the pass on top, (b) the time series.

Not included yet: X-ray absorption in the atmosphere (rays just stop at a
60 km shell) and the axion mass. Values near the shadow edge are upper
bounds.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import LogNorm

from functions.bfield import R_EARTH_KM
from functions.conversion import conversion_probability_uniform
from functions.geometry import (
    MU_EARTH_KM3_S2, R_EQUATOR_KM, orbit_position, sun_direction,
)
from functions.lineofsight import los

# ---- Settings -------------------------------------------------------------
G = 1e-10                      # axion-photon coupling [GeV^-1]
EPOCH = "2026-10-19T00:00:00"  # orbit start time [UTC]
ALTITUDE_KM = 500.0
INCLINATION_DEG = 97.4
STEP_S = 30                    # time between orbit samples [s]
GRID_DEG = 3.0                 # map resolution [deg]
OUT = Path(__file__).resolve().parent

# P is proportional to (field integral)^2. This constant converts between
# them: P = P_PER_TM2 * (field integral in T m)^2.
P_PER_TM2 = conversion_probability_uniform(b=1.0, length=1.0, g=G)


# ---- Helpers --------------------------------------------------------------
def sun_hidden(r, k):
    """True if the Earth (plus a 60 km atmosphere) blocks the Sun.

    r: spacecraft position [km]; k: unit vector toward the Sun.
    The Sun is hidden when the ray from r toward k passes closer to
    Earth's centre than the 60 km shell, on the far side of the Earth.
    """
    along = np.dot(r, k)                       # distance along the ray to closest approach
    closest = np.sqrt(np.dot(r, r) - along**2)  # closest distance to Earth's centre
    return along < 0 and closest < R_EARTH_KM + 60


def field_integral(r, k, t):
    """|int B_perp ds| [T m] toward the Sun, or NaN if the Sun is visible."""
    if not sun_hidden(r, k):
        return np.nan
    ray = los(r, k, t, report=False)
    return ray["amplitude_Tm"]


def lat_lon(r):
    """Latitude and longitude [deg] of Earth-fixed positions r [km]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


# ---- Main -----------------------------------------------------------------
def main():
    # 1. Two orbits. Orbit period from Kepler's third law.
    radius_km = R_EQUATOR_KM + ALTITUDE_KM
    period_s = 2 * np.pi * np.sqrt(radius_km**3 / MU_EARTH_KM3_S2)
    times = Time(EPOCH) + np.arange(0, 2 * period_s, STEP_S) * u.s
    positions = orbit_position(times, epoch=EPOCH, h=ALTITUDE_KM,
                               inc=INCLINATION_DEG)  # Earth-fixed [km]
    sun_dirs = sun_direction(times)                  # Earth-fixed unit vectors

    series = np.array([field_integral(r, k, t)
                       for r, k, t in zip(positions, sun_dirs, times)])
    hidden = np.isfinite(series)  # True while the Sun is behind the Earth

    # 2. First complete pass: starts when the Sun goes from visible to hidden.
    start = 1
    while not (hidden[start] and not hidden[start - 1]):
        start += 1
    stop = start
    while hidden[stop]:
        stop += 1
    middle = (start + stop) // 2
    t_map, k_map = times[middle], sun_dirs[middle]

    # 3. Map: put the spacecraft at every grid point and compute the integral.
    lats = np.arange(-90 + GRID_DEG / 2, 90, GRID_DEG)
    lons = np.arange(-180 + GRID_DEG / 2, 180, GRID_DEG)
    grid = np.full((lats.size, lons.size), np.nan)
    for i, lat in enumerate(np.radians(lats)):
        for j, lon in enumerate(np.radians(lons)):
            r = radius_km * np.array([np.cos(lat) * np.cos(lon),
                                      np.cos(lat) * np.sin(lon),
                                      np.sin(lat)])
            grid[i, j] = field_integral(r, k_map, t_map)

    # 4. Plot.
    OUT.mkdir(exist_ok=True)
    fig, (ax_map, ax_time) = plt.subplots(2, 1, figsize=(7, 7.5),
                                          height_ratios=[1.3, 1])
    colors = LogNorm(np.nanmin(grid), np.nanmax(grid))  # shared colour scale

    # (a) Map; grey background = Sun visible (no value computed).
    ax_map.set_facecolor("0.9")
    mesh = ax_map.pcolormesh(lons, lats, grid, norm=colors, shading="nearest")
    pass_lat, pass_lon = lat_lon(positions[start:stop])
    ax_map.plot(pass_lon, pass_lat, color="w", lw=0.8)
    ax_map.scatter(pass_lon, pass_lat, c=series[start:stop], norm=colors,
                   s=14, edgecolors="w", linewidths=0.6, zorder=3)
    anti_lat, anti_lon = lat_lon(-k_map)  # point directly opposite the Sun
    ax_map.plot(anti_lon, anti_lat, "wx", ms=8, mew=2)
    ax_map.annotate("anti-Sun", (anti_lon, anti_lat), xytext=(6, 6),
                    textcoords="offset points", color="w")
    ax_map.set(xlim=(-180, 180), ylim=(-90, 90),
               xticks=range(-180, 181, 60), yticks=range(-90, 91, 30),
               xlabel="Longitude [deg]", ylabel="Latitude [deg]",
               title="(a) Conversion over the night side")
    fig.colorbar(mesh, ax=ax_map, label=r"$|\int B_\perp\,ds|$ [T m]")

    # (b) Time series; grey bands = Sun visible.
    hours = (times - times[0]).to_value(u.h)
    ax_time.plot(hours, series, color="C0")
    ax_time.fill_between(hours, 0, 1, where=~hidden, color="0.9", step="mid",
                         transform=ax_time.get_xaxis_transform(),
                         label="Sun in view")
    ax_time.axvspan(hours[start], hours[stop - 1], color="C1", alpha=0.15,
                    label="pass shown in (a)")
    ax_time.set(yscale="log", xlim=(0, hours[-1]),
                xlabel=f"Hours after {EPOCH}",
                ylabel=r"$|\int B_\perp\,ds|$ [T m]",
                title="(b) Conversion along the orbit")
    ax_time.legend(loc="upper right", fontsize=8)
    # Right axis: the same curve read as probability P.
    right = ax_time.secondary_yaxis("right", functions=(
        lambda tm: P_PER_TM2 * tm**2,
        lambda p: np.sqrt(np.maximum(p, 0) / P_PER_TM2),
    ))
    right.set_ylabel(f"P at g = {G:g} GeV$^{{-1}}$")

    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(OUT / f"conversion_map.{ext}", dpi=200)

    print(f"Map time {t_map.isot}")
    print(f"Sun hidden for {hidden.mean():.0%} of the orbit")
    print(f"P along orbit: {P_PER_TM2 * np.nanmin(series)**2:.1e} "
          f"to {P_PER_TM2 * np.nanmax(series)**2:.1e}")
    print(f"Wrote {OUT / 'conversion_map.png'}")


if __name__ == "__main__":
    main()
