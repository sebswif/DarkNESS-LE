"""Trade-study figures: effective area, one-orbit timeline, backgrounds, limits.

Run from the repo root after simulate.py:  python output/fig_trade.py
Writes output/fig_area.png, fig_orbit.png, fig_background.png, fig_limits.png.
Reads output/trade_results.json; reruns one orbit for the timeline (~1 min).
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.geometry import MU_EARTH_KM3_S2, R_EQUATOR_KM  # noqa: E402
from functions.orbit_counts import simulate_counts  # noqa: E402
from functions.payload import kwargs  # noqa: E402
from functions.solar_flux import axion_flux  # noqa: E402
from simulate import payloads  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
# Reference palette, categorical slots in fixed order (one per payload).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COMPONENTS = {"particle": "#6250d6", "cxb": "#2a78d6", "dark_earth": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAST_2017 = 6.6e-11  # GeV^-1, CAST, Nature Phys. 13, 584 (2017)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 8,
})

SHORT = {0: "Bare, 10° aperture", 1: "Lobster 6U, f=30 cm", 2: "Lobster 8U, f=40 cm", 3: "Lobster linear 6U, f=60 cm"}


def signal_area(p, e):
    area = np.broadcast_to(np.asarray(p["response"](e, 1.0), float), e.shape)
    return area + (0 if p["core"] is None else p["core"][1])


def fig_area(e, configs):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, height_ratios=[1, 1.6])
    top.plot(e, axion_flux(e, 1e-10) / 1e10, color=INK2)
    top.set_ylabel("Axion flux at Earth\n[10¹⁰ cm⁻² s⁻¹ keV⁻¹]")
    top.set_title("Solar axion spectrum (g = 10⁻¹⁰ GeV⁻¹) and payload signal area", loc="left", color=INK)
    for i, (name, p) in enumerate(configs.items()):
        area = signal_area(p, e)
        bottom.plot(e, area, color=SERIES[i], label=SHORT[i])
        bottom.annotate(SHORT[i], (e[-1], area[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)
    bottom.set_yscale("log")
    bottom.set_xscale("log")
    bottom.set_xlabel("Energy [keV]")
    bottom.set_ylabel("On-axis effective area,\nincl. 50 µm Si QE [cm²]")
    bottom.legend(loc="lower left")
    bottom.set_xlim(e[0], e[-1] * 1.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_area.png", dpi=200)


def fig_orbit(e, configs):
    start = Time("2026-01-01T00:00:00", scale="utc")
    period = 2*np.pi*np.sqrt((R_EQUATOR_KM + 500)**3 / MU_EARTH_KM3_S2)
    edges = start + np.append(np.arange(0, period, 30.0), period) * u.s
    common = dict(g=1e-10, goes=0.0, temperature=10.0, earth_c=1.48e-2, particle_level=1.26e-3,
                  particle_alpha=0.5, particle_ref=10.0, particle_min=0.1,
                  orbit=dict(h=500.0, inc=51.6, epoch=start),
                  activity=dict(f107=150.0, f107a=150.0, ap=4.0), fov_radial=4, fov_azimuth=8)
    runs = {name: simulate_counts(e, edges, **kwargs(p), **common) for name, p in configs.items()}
    first = next(iter(runs.values()))
    minutes = (first["time_mid"] - start).to_value(u.min)
    durations = np.diff((edges - start).to_value(u.s))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True, height_ratios=[1, 1.3, 1.3])
    for ax in axes:
        for lo, hi in _windows(minutes, first["observing"]):
            ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
    axes[0].plot(minutes, first["tangent_altitude_km"], color=INK2)
    axes[0].axhline(120, color=MUTED, lw=0.8, ls="--")
    axes[0].text(minutes[-1], 120, "grazing limit 120 km ", ha="right", va="bottom", fontsize=7, color=MUTED)
    axes[0].set_ylim(-3000, 1000)
    axes[0].set_ylabel("Sun-ray tangent\naltitude [km]")
    axes[0].set_title("One orbit, 500 km / 51.6°: shaded = observing (eclipse, grazing < 120 km)",
                      loc="left", color=INK)
    for i, (name, run) in enumerate(runs.items()):
        live = np.where(first["observing"], durations, np.nan)
        axes[1].plot(minutes, run["counts"]["background"] / live, color=SERIES[i], label=SHORT[i])
        axes[2].plot(minutes, run["counts"]["signal"] / live, color=SERIES[i], label=SHORT[i])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Background\n[counts/s]")
    axes[1].legend(loc="center right")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Axion signal\n[counts/s]")
    axes[2].set_xlabel("Time since epoch [min]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_orbit.png", dpi=200)


def _windows(t, mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    step = t[1] - t[0]
    return [(t[a] - step/2, t[b - 1] + step/2) for a, b in zip(edges[::2], edges[1::2])]


def fig_background(results):
    names = list(results)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(names))
    for k, (comp, color) in enumerate(COMPONENTS.items()):
        values = [max(results[n]["counts_mission"][comp], 1e-4) for n in names]
        ax.scatter(values, y + (k - 1) * 0.22, s=36, color=color, label=comp.replace("_", " "),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    signal = [results[n]["counts_mission"]["signal"] for n in names]
    ax.scatter(signal, y, marker="D", s=30, color=INK, label="axion signal", zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(y, [SHORT[i] for i in range(len(names))])
    ax.invert_yaxis()
    ax.set_xlabel("Counts per year (1–10 keV)")
    ax.set_title("Background components vs signal, 1 yr", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_background.png", dpi=200)


def fig_limits(results):
    names = list(results)
    g = [results[n]["g_limit"] / 1e-10 for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(names))
    ax.hlines(y, 0, g, color=[SERIES[i] for i in y], lw=6, capstyle="round")
    for i, value in enumerate(g):
        ax.text(value, i, f"  {value:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(CAST_2017 / 1e-10, color=MUTED, ls="--", lw=0.8)
    ax.text(CAST_2017 / 1e-10, -0.75, " CAST 2017 (0.66)", fontsize=7.5, color=MUTED, va="center")
    ax.set_yticks(y, [SHORT[i] for i in y])
    ax.set_ylim(len(names) - 0.5, -1.0)
    ax.set_xlim(0, max(g) * 1.25)
    ax.set_xlabel("95% upper limit on g_aγ [10⁻¹⁰ GeV⁻¹], 1 yr (lower is better)")
    ax.set_title("Coupling reach by payload", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_limits.png", dpi=200)


if __name__ == "__main__":
    e = np.geomspace(1, 10, 32)
    configs = payloads(e, 50.0)
    results = json.loads((OUT / "trade_results.json").read_text())["results"]
    fig_area(e, configs)
    fig_background(results)
    fig_limits(results)
    fig_orbit(e, configs)
    print("wrote", ", ".join(f"fig_{n}.png" for n in ("area", "background", "limits", "orbit")))
