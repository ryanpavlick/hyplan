#%%
"""Figure 3: Terrain-aware flight box — Rincón de la Vieja National Park.

Four-panel figure showing the spatial planning pipeline over a volcanic
national park with irregular boundaries and significant terrain relief:
  (a) Study area polygon with terrain
  (b) Minimum rotated bounding rectangle + sensor parameters
  (c) Flight lines with terrain-aware spacing
  (d) Clipped lines with terrain-aware swath polygons
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from matplotlib.colors import LightSource
from shapely.geometry import shape

from hyplan import (
    AVIRIS3, box_around_polygon_terrain,
    generate_swath_polygon, calculate_swath_widths,
    ureg,
)
from hyplan.geometry import minimum_rotated_rectangle, rectangle_dimensions
from hyplan.terrain import generate_demfile

# ── Setup ──
sensor = AVIRIS3()
flight_altitude = ureg.Quantity(25000, "feet")

# Rincón de la Vieja National Park, Costa Rica
gdf = gpd.read_file("rincon_de_la_vieja.geojson")
study_area = gdf.geometry.iloc[0]
centroid = study_area.centroid

# Rotated rectangle
min_rect = minimum_rotated_rectangle(study_area)
lat0, lon0, azimuth, length_m, width_m = rectangle_dimensions(min_rect, azimuth=None)

# Swath width (flat-earth reference)
swath_w = sensor.swath_width(flight_altitude)

# DEM
bounds = study_area.bounds
dem_file = generate_demfile(
    np.array([bounds[1], bounds[3]]),
    np.array([bounds[0], bounds[2]]),
)

# ── Terrain-aware flight lines ──
lines_unclipped = box_around_polygon_terrain(
    instrument=sensor,
    altitude_msl=flight_altitude,
    polygon=study_area,
    box_name="RDV",
    overlap=20.0,
    alternate_direction=True,
    clip_to_polygon=False,
)

# Clipped to buffered polygon for full swath coverage at edges
clip_buffer_deg = 0.012
buffered_area = study_area.buffer(clip_buffer_deg)

lines_clipped = box_around_polygon_terrain(
    instrument=sensor,
    altitude_msl=flight_altitude,
    polygon=study_area,
    box_name="RDV",
    overlap=20.0,
    alternate_direction=True,
    clip_to_polygon=True,
    clip_polygon=buffered_area,
)

# Terrain-aware swaths
swaths = []
for fl in lines_clipped:
    try:
        sw = generate_swath_polygon(fl, sensor, along_precision=500.0, dem_file=dem_file)
        swaths.append(sw)
    except Exception:
        swaths.append(None)

# DEM shaded relief
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)
    dem_bounds = src.bounds
    valid = dem_data[dem_data > 0]
    elev_min, elev_max = int(valid.min()), int(valid.max())

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(dem_data, cmap=plt.cm.terrain, vert_exag=2, blend_mode="soft")
# Blend toward white for a more transparent/faded background
fade = 0.4  # 0=original, 1=fully white
rgb[..., :3] = rgb[..., :3] * (1 - fade) + fade
dem_extent = [dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top]

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
(ax_a, ax_b, ax_c, ax_d) = axes.flat

pad = 0.015
xlim = (bounds[0] - pad, bounds[2] + pad)
ylim = (bounds[1] - pad, bounds[3] + pad)

for ax in axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5.5)

ann_bbox = dict(boxstyle="round,pad=0.12", facecolor="white",
                edgecolor="none", alpha=0.85)

sx, sy = study_area.exterior.xy

# ── Panel (a): Study area with terrain ──
ax_a.imshow(rgb, extent=dem_extent, origin="upper", aspect="auto", zorder=0)
ax_a.plot(sx, sy, color="black", linewidth=1.5, zorder=3)
ax_a.set_title("(a) Study area", fontsize=8, fontweight="bold")
ax_a.set_ylabel("Latitude", fontsize=7)
ax_a.text(centroid.x, bounds[1] + 0.006, "Rincón de la Vieja\nNational Park",
          ha="center", va="bottom", fontsize=5.5, fontstyle="italic",
          bbox=ann_bbox, zorder=4)
ax_a.text(0.97, 0.03, f"Elev: {elev_min}\u2013{elev_max} m",
          transform=ax_a.transAxes, fontsize=5.5, ha="right", va="bottom",
          bbox=ann_bbox, zorder=5)

# ── Panel (b): Bounding rectangle + sensor info ──
ax_b.imshow(rgb, extent=dem_extent, origin="upper", aspect="auto", zorder=0)
ax_b.plot(sx, sy, color="black", linewidth=1, zorder=2)
rx, ry = min_rect.exterior.xy
ax_b.plot(rx, ry, color="#e74c3c", linewidth=1.5, linestyle="-", zorder=3)
ax_b.set_title("(b) Bounding rectangle + sensor", fontsize=8, fontweight="bold")
ax_b.text(0.03, 0.97,
          f"AVIRIS-3 @ {flight_altitude.to(ureg.feet):.0f}\n"
          f"Swath: {swath_w.to(ureg.km):.1f}\n"
          f"Overlap: 20%\n"
          f"Azimuth: {azimuth:.0f}\u00b0\n"
          f"Lines: {len(lines_unclipped)} (terrain-aware)",
          transform=ax_b.transAxes, fontsize=5.5, va="top",
          bbox=ann_bbox, zorder=5)

# ── Panel (c): Unclipped flight lines ──
ax_c.imshow(rgb, extent=dem_extent, origin="upper", aspect="auto", zorder=0)
ax_c.plot(sx, sy, color="black", linewidth=1, zorder=2)

for fl in lines_unclipped:
    x = [fl.waypoint1.longitude, fl.waypoint2.longitude]
    y = [fl.waypoint1.latitude, fl.waypoint2.latitude]
    ax_c.plot(x, y, color="white", linewidth=1.5, zorder=3)
    ax_c.plot(x, y, color="#2c3e50", linewidth=0.8, zorder=4)
    mx, my = np.mean(x), np.mean(y)
    ax_c.annotate("", xy=(x[1], y[1]), xytext=(mx, my),
                  arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=0.8),
                  zorder=5)

ax_c.set_title("(c) Terrain-aware line spacing", fontsize=8, fontweight="bold")
ax_c.set_ylabel("Latitude", fontsize=7)
ax_c.set_xlabel("Longitude", fontsize=7)

# ── Panel (d): Clipped lines + terrain-aware swaths ──
ax_d.imshow(rgb, extent=dem_extent, origin="upper", aspect="auto", zorder=0)
ax_d.plot(sx, sy, color="black", linewidth=1, linestyle="-", alpha=0.8, zorder=2)

for sw in swaths:
    if sw is not None and sw.is_valid:
        swx, swy = sw.exterior.xy
        ax_d.fill(swx, swy, alpha=0.45, color="#3498db", edgecolor="#2980b9",
                  linewidth=0.5, zorder=3)

for fl in lines_clipped:
    x = [fl.waypoint1.longitude, fl.waypoint2.longitude]
    y = [fl.waypoint1.latitude, fl.waypoint2.latitude]
    ax_d.plot(x, y, color="white", linewidth=1.2, zorder=4)
    ax_d.plot(x, y, color="#2c3e50", linewidth=0.6, zorder=5)

ax_d.set_title("(d) Clipped + terrain swaths", fontsize=8, fontweight="bold")
ax_d.set_xlabel("Longitude", fontsize=7)

all_widths = [calculate_swath_widths(sw) for sw in swaths if sw is not None and sw.is_valid]
if all_widths:
    w_min = min(w["min_width"] for w in all_widths)
    w_max = max(w["max_width"] for w in all_widths)
    ax_d.text(0.03, 0.03,
              f"Swath width: {w_min:.0f}\u2013{w_max:.0f} m",
              transform=ax_d.transAxes, fontsize=5.5,
              bbox=ann_bbox, zorder=6)

fig.suptitle("Terrain-Aware Flight Box \u2014 Rinc\u00f3n de la Vieja, Costa Rica",
             fontsize=10, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig2_flight_box.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("fig2_flight_box.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Figure 2 saved.")

# %%
