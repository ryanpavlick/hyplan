"""Generate all JOSS paper figures from a single Catalina Island campaign.

Runs the full HyPlan pipeline — study area, flight lines, terrain-aware
swath, wind-aware Dubins paths, cloud/phenology timing — and produces
four publication figures from the same scenario.
"""

import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from matplotlib.colors import LightSource
from shapely.geometry import shape

from hyplan import (
    AVIRIS3, DynamicAviation_B200,
    Airport, initialize_data,
    FlightLine, box_around_polygon,
    compute_flight_plan,
    generate_swath_polygon, calculate_swath_widths,
    DubinsPath3D,
    ConstantWindField,
    ureg,
)
from hyplan.terrain import generate_demfile

# ── Shared setup ──
sensor = AVIRIS3()
aircraft = DynamicAviation_B200()
flight_altitude = ureg.Quantity(20000, "feet")
alt_m = flight_altitude.m_as("meter")

with open("notebooks/exampledata/catalina.geojson") as f:
    geojson = json.load(f)
study_area = shape(geojson["features"][0]["geometry"])
centroid = study_area.centroid

# Generate flight lines
flight_lines = box_around_polygon(
    instrument=sensor,
    altitude_msl=flight_altitude,
    polygon=study_area,
    box_name="CAT",
    overlap=20.0,
    alternate_direction=True,
    clip_to_polygon=False,
)
print(f"Generated {len(flight_lines)} flight lines")

# Download DEM covering the study area
bounds = study_area.bounds  # (minx, miny, maxx, maxy)
dem_file = generate_demfile(
    np.array([bounds[1], bounds[3]]),
    np.array([bounds[0], bounds[2]]),
)
print(f"DEM: {dem_file}")


# =====================================================================
# Figure 1: Workflow overview (keep the diagram version)
# =====================================================================
exec(open("paper/figures/fig1_workflow.py").read())


# =====================================================================
# Figure 2: Flight box generation over Catalina
# =====================================================================
print("\n=== Figure 2: Flight box ===")
exec(open("paper/figures/fig2_flight_box.py").read())


# =====================================================================
# Figure 3: Wind-aware trochoidal Dubins paths over Catalina
# =====================================================================
print("\n=== Figure 3: Wind + Dubins ===")

cruise_speed = aircraft.cruise_speed_at(flight_altitude)
bank_angle = aircraft.max_bank_angle

# Use a subset of lines for visual clarity
subset = flight_lines[:4]
wind_mps = (40 * ureg.knot).m_as("m/s")
wind_vec = (wind_mps, 0.0)  # 40 kt from west

# Compute Dubins paths
paths_still = []
paths_wind = []
for i in range(len(subset) - 1):
    wp_from = subset[i].waypoint2 if i % 2 == 0 else subset[i].waypoint1
    wp_to = subset[i + 1].waypoint1 if (i + 1) % 2 == 0 else subset[i + 1].waypoint2

    p_still = DubinsPath3D(
        start=wp_from, end=wp_to,
        speed=cruise_speed, bank_angle=bank_angle,
        pitch_min=-5, pitch_max=5, step_size=100.0,
    )
    p_wind = DubinsPath3D(
        start=wp_from, end=wp_to,
        speed=cruise_speed, bank_angle=bank_angle,
        pitch_min=-5, pitch_max=5, step_size=100.0,
        wind=wind_vec,
    )
    paths_still.append(p_still)
    paths_wind.append(p_wind)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

for ax, paths, title in [
    (ax3a, paths_still, "Still air (circular arcs)"),
    (ax3b, paths_wind, "40 kt westerly (trochoidal arcs)"),
]:
    # Study area outline
    sx_area, sy_area = study_area.exterior.xy
    ax.plot(sx_area, sy_area, color="gray", linewidth=0.8, linestyle=":",
            alpha=0.4, zorder=1)
    ax.fill(sx_area, sy_area, alpha=0.05, color="green", zorder=0)

    # Flight lines
    for i, fl in enumerate(subset):
        x = [fl.waypoint1.longitude, fl.waypoint2.longitude]
        y = [fl.waypoint1.latitude, fl.waypoint2.latitude]
        if i % 2 == 0:
            ax.annotate("", xy=(x[1], y[1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2),
                        zorder=5)
        else:
            ax.annotate("", xy=(x[0], y[0]), xytext=(x[1], y[1]),
                        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2),
                        zorder=5)

    # Dubins paths
    for p in paths:
        pts = p.points
        ax.plot(pts[:, 1], pts[:, 0], color="#e67e22", linewidth=1.8, zorder=3)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=7)
    ax.tick_params(labelsize=6)

    # Wind arrows (right panel only)
    if paths is paths_wind:
        x_range = ax.get_xlim() if ax.get_xlim()[1] > ax.get_xlim()[0] else (-118.7, -118.3)
        y_range = ax.get_ylim() if ax.get_ylim()[1] > ax.get_ylim()[0] else (33.3, 33.5)
        for wlon in np.arange(x_range[0], x_range[1], 0.04):
            for wlat in np.arange(y_range[0], y_range[1], 0.012):
                ax.annotate("", xy=(wlon + 0.012, wlat), xytext=(wlon, wlat),
                            arrowprops=dict(arrowstyle="-|>", color="#e74c3c",
                                            lw=0.5, alpha=0.3), zorder=1)

ax3a.set_ylabel("Latitude", fontsize=7)

# Crab angle inset
ax_inset = fig3.add_axes([0.73, 0.72, 0.20, 0.20])
ax_inset.set_xlim(-1.2, 1.5)
ax_inset.set_ylim(-0.5, 1.5)
ax_inset.set_aspect("equal")
ax_inset.axis("off")
ax_inset.add_patch(mpatches.FancyBboxPatch(
    (-1.15, -0.45), 2.6, 1.9,
    boxstyle="round,pad=0.05", facecolor="white", edgecolor="#bdc3c7",
    linewidth=0.8, zorder=0))
crab_angle = 8.0
ax_inset.annotate("", xy=(1.2, 0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5), zorder=4)
ax_inset.text(1.3, 0, "Ground\ntrack", fontsize=5, color="#27ae60", va="center", ha="left")
hx = np.cos(np.radians(crab_angle))
hy = np.sin(np.radians(crab_angle))
ax_inset.annotate("", xy=(hx, hy), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5), zorder=4)
ax_inset.text(hx + 0.08, hy + 0.1, "Heading", fontsize=5, color="#2c3e50", va="bottom")
arc_theta = np.linspace(0, crab_angle, 20)
ax_inset.plot(0.5 * np.cos(np.radians(arc_theta)),
              0.5 * np.sin(np.radians(arc_theta)),
              color="#e74c3c", linewidth=1, zorder=4)
ax_inset.text(0.55, 0.12, f"{crab_angle}\u00b0", fontsize=6, color="#e74c3c", fontweight="bold")
ax_inset.annotate("", xy=(0.4, 1.1), xytext=(-0.4, 1.1),
                  arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1), zorder=4)
ax_inset.text(0.0, 1.25, "Wind", fontsize=5, color="#e74c3c", ha="center", fontweight="bold")
ax_inset.set_title("Crab angle", fontsize=6, fontweight="bold", pad=2)

# Legend
legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#2c3e50", linewidth=2, label="Flight lines"),
    plt.Line2D([0], [0], color="#e67e22", linewidth=1.8, label="Dubins transit"),
]
ax3a.legend(handles=legend_elements, loc="lower left", fontsize=6, framealpha=0.9)

fig3.text(0.5, 0.01,
          f"King Air B200, {cruise_speed.to(ureg.knot):.0f} cruise, "
          f"{bank_angle}\u00b0 bank, R = {paths_still[0].min_turn_radius.to(ureg.km):.1f}"
          f" \u2014 Catalina Island",
          ha="center", fontsize=6.5, color="#555555")

fig3.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.92, wspace=0.08)
fig3.savefig("paper/figures/fig3_wind_dubins.png", dpi=300,
             bbox_inches="tight", facecolor="white")
fig3.savefig("paper/figures/fig3_wind_dubins.pdf",
             bbox_inches="tight", facecolor="white")
plt.close(fig3)
print("Figure 3 saved.")


# =====================================================================
# Figure 4: Cloud + phenology timing (synthetic, Catalina-region)
# =====================================================================
print("\n=== Figure 4: Cloud + phenology ===")
exec(open("paper/figures/fig4_cloud_phenology.py").read())


print("\n=== All figures generated ===")
