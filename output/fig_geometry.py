"""Figure: why the conversion map looks the way it does.

Run from this folder:  python fig_geometry.py
Output: geometry.png beside this script

We slice the Earth with one flat plane: the plane that contains the
direction to the Sun and the Earth's magnetic (dipole) axis. In that
plane we place the spacecraft at many points on its 500 km orbit shell,
keep the points where the Earth hides the Sun, and draw each ray toward
the Sun down to the 60 km shell.

Each ray is coloured by its field integral |int B_perp ds| [T m], and a
red arrow shows the field direction at the spacecraft. Two things set
the colour:
  - path length: short straight down, long when skimming the atmosphere;
  - angle between ray and field: if the field points along the ray,
    B_perp = 0 and nothing converts (the "blind spots").

The thin blue curves are ideal dipole field lines, for orientation only.
The colours use the full IGRF field.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import LogNorm

from functions.bfield import R_EARTH_KM, igrf_field, load_igrf
from functions.geometry import sun_direction
from functions.lineofsight import los

# ---- Settings -------------------------------------------------------------
TIME = "2026-10-19T01:36:30"  # UTC; same moment as the map in fig_conversion_map
ALTITUDE_KM = 500.0
STEP_DEG = 4.0                # spacing of spacecraft positions around the orbit
OUT = Path(__file__).resolve().parent

R_SHELL = R_EARTH_KM + 60           # rays stop here [km]
R_ORBIT = R_EARTH_KM + ALTITUDE_KM  # spacecraft shell [km]


def main():
    t = Time(TIME)
    sun = sun_direction(t)  # unit vector toward the Sun, Earth-fixed

    # 1. Build the plane. The dipole axis comes from the first three IGRF
    #    coefficients (g10, g11, h11); minus sign makes it point north.
    g, h = load_igrf(t.decimalyear)
    up = -np.array([g[1, 1], h[1, 1], g[1, 0]])
    up = up / np.linalg.norm(up)
    # "side" is the part of the Sun direction at right angles to "up".
    side = sun - np.dot(sun, up) * up
    side = side / np.linalg.norm(side)

    def to_plane(v):
        """Flatten 3D vectors to (x, y) = (along side, along up)."""
        return np.stack([v @ side, v @ up], axis=-1)

    # 2. Spacecraft positions around the orbit circle in this plane; keep
    #    those where the Earth hides the Sun, and trace each ray.
    rays = []
    for angle in np.radians(np.arange(0, 360, STEP_DEG)):
        r = R_ORBIT * (np.cos(angle) * side + np.sin(angle) * up)
        along = np.dot(r, sun)
        closest = np.sqrt(np.dot(r, r) - along**2)
        if along < 0 and closest < R_SHELL:  # Sun hidden
            rays.append(los(r, sun, t, n=201, report=False))

    integral = np.array([ray["amplitude_Tm"] for ray in rays])
    length = np.array([ray["distance_km"][-1] for ray in rays])
    height = np.array([to_plane(ray["path_ecef_km"][0])[1] for ray in rays])

    # Rays to label: straight down, longest, and the weakest in each half.
    straight_down = np.argmin(length)
    grazing = np.argmax(length)
    north_blind = np.argmin(np.where(height > 0, integral, np.inf))
    south_blind = np.argmin(np.where(height < 0, integral, np.inf))

    # 3. Plot.
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = plt.get_cmap("viridis")
    colors = LogNorm(integral.min(), integral.max())

    # Earth, 60 km shell (dotted), orbit (solid).
    circle = np.linspace(0, 2 * np.pi, 600)
    cx, cy = np.cos(circle), np.sin(circle)
    ax.fill(R_EARTH_KM * cx, R_EARTH_KM * cy, color="0.8", zorder=2)
    ax.plot(R_SHELL * cx, R_SHELL * cy, "k:", lw=0.8, zorder=3)
    ax.plot(R_ORBIT * cx, R_ORBIT * cy, "k-", lw=0.8, zorder=3)

    # Shadow edge (dashed): rays toward the Sun that just touch the shell.
    sx, sy = to_plane(sun)
    for sign in (-1, 1):
        x0, y0 = sign * R_SHELL * -sy, sign * R_SHELL * sx
        ax.plot([x0, x0 - 2 * R_EARTH_KM * sx], [y0, y0 - 2 * R_EARTH_KM * sy],
                "k--", lw=0.7, zorder=3)

    # Ideal dipole field lines, r = L cos^2(latitude), one through each
    # spacecraft position so every red arrow sits on a line. L is where the
    # line crosses the magnetic equator.
    lat = np.linspace(-np.pi / 2, np.pi / 2, 2000)
    for ray in rays:
        x0, y0 = to_plane(ray["path_ecef_km"][0])
        L = np.hypot(x0, y0) / np.cos(np.arctan2(y0, -x0)) ** 2
        r = L * np.cos(lat) ** 2
        above = r > R_EARTH_KM
        ax.plot(-r[above] * np.cos(lat[above]), r[above] * np.sin(lat[above]),
                color="C0", lw=0.5, alpha=0.6, zorder=1)

    # Rays, coloured by field integral, plus field direction (red arrows).
    for ray, value in zip(rays, integral):
        start, end = to_plane(ray["path_ecef_km"][[0, -1]])
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=cmap(colors(value)), lw=2.5, zorder=4)
        b = igrf_field(ray["path_ecef_km"][0], t, report=False)
        bx, by = to_plane(b / np.linalg.norm(b)) * 350  # arrow length 350 km
        x, y = start * (R_ORBIT + 150) / R_ORBIT        # just outside the orbit
        ax.annotate("", xy=(x + bx / 2, y + by / 2), xytext=(x - bx / 2, y - by / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="C3",
                                    shrinkA=0, shrinkB=0), zorder=5)

    # Labels for the four example rays.
    labels = [
        (straight_down, f"anti-Sun: {length[straight_down]:.0f} km path"),
        (grazing, f"grazing: {length[grazing]:.0f} km path"),
        (north_blind, "blind spot: field along ray"),
        (south_blind, "blind spot: field along ray"),
    ]
    for i, text in labels:
        x, y = to_plane(rays[i]["path_ecef_km"][100])  # middle of the ray
        ax.annotate(text, (x, y), xytext=(-1.22 * R_EARTH_KM, y), fontsize=8,
                    va="center", ha="right", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

    # Sun arrow and dipole axis.
    tail = np.array([-1.15, -0.95]) * R_EARTH_KM
    ax.annotate("", xy=tail + 1200 * np.array([sx, sy]), xytext=tail,
                arrowprops=dict(arrowstyle="->", lw=2.5, color="orange"))
    ax.text(*(tail + 1300 * np.array([sx, sy])), " to Sun", color="orange",
            va="center")
    ax.plot([0, 0], [-1.25 * R_EARTH_KM, 1.25 * R_EARTH_KM], color="C0",
            lw=1, ls="-.")
    ax.text(0, 1.26 * R_EARTH_KM, "dipole axis", ha="center", color="C0",
            fontsize=9)

    ax.set(aspect="equal", xlim=(-1.9 * R_EARTH_KM, 0.05 * R_EARTH_KM),
           ylim=(-1.3 * R_EARTH_KM, 1.3 * R_EARTH_KM),
           xlabel="km", ylabel="km", title="Conversion geometry")
    fig.colorbar(plt.cm.ScalarMappable(norm=colors, cmap=cmap), ax=ax,
                 shrink=0.7, label=r"$|\int B_\perp\,ds|$ [T m]")
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(OUT / f"geometry.{ext}", dpi=200)

    for i, text in labels:
        print(f"{text:30s} |int B_perp ds| = {integral[i]:6.2f} T m")
    print(f"Wrote {OUT / 'geometry.png'}")


if __name__ == "__main__":
    main()
