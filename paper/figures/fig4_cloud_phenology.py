"""Figure 4: Cloud climatology + vegetation phenology — when to fly.

Uses real data:
- Cloud fraction from Open-Meteo ERA5 (no auth required)
- NDVI from MODIS MOD13A1 via NASA AppEEARS (EarthData credentials)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box

# Load credentials
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# ── Month ticks ──
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# ── Study sites (Costa Rica) ──
sites = gpd.GeoDataFrame([
    {"Name": "Rincon de la Vieja", "geometry": box(-85.40, 10.72, -85.25, 10.87)},
    {"Name": "Guanacaste lowlands", "geometry": box(-85.70, 10.40, -85.40, 10.60)},
], crs="EPSG:4326")

tmp_poly = os.path.join(tempfile.gettempdir(), "fig4_sites.geojson")
sites.to_file(tmp_poly, driver="GeoJSON")

# ── Fetch real cloud fraction ──
print("Fetching cloud fraction from Open-Meteo ERA5...")
from hyplan.clouds import fetch_cloud_fraction, summarize_cloud_fraction_by_doy

cloud_df = fetch_cloud_fraction(
    polygon_file=tmp_poly,
    year_start=2010,
    year_stop=2022,
    day_start=1,
    day_stop=365,
    source="openmeteo",
)
cloud_summary = summarize_cloud_fraction_by_doy(cloud_df, window=7)
print(f"  Cloud: {len(cloud_df)} rows, {cloud_df['polygon_id'].nunique()} sites")

# ── Fetch real NDVI via AppEEARS ──
import pandas as pd
ndvi_cache = os.path.join(os.path.dirname(__file__), "fig4_ndvi_cache.csv")

if os.path.exists(ndvi_cache):
    print(f"Loading cached NDVI from {ndvi_cache}")
    ndvi_df = pd.read_csv(ndvi_cache, parse_dates=["date"])
else:
    print("Fetching NDVI from MODIS via AppEEARS...")
    from hyplan.phenology import fetch_phenology

    ndvi_df = fetch_phenology(
        polygon_file=tmp_poly,
        product="ndvi",
        year_start=2010,
        year_stop=2022,
        source="appeears",
    )
    # Cache for future runs
    ndvi_df.to_csv(ndvi_cache, index=False)
    print(f"  Cached to {ndvi_cache}")

from hyplan.phenology import summarize_phenology_by_doy
ndvi_summary = summarize_phenology_by_doy(ndvi_df, window=3)
print(f"  NDVI: {len(ndvi_df)} rows, {ndvi_df['polygon_id'].nunique()} sites")

# ── Figure ──
fig, (ax_cloud, ax_ndvi) = plt.subplots(
    2, 1, figsize=(6, 4.5), sharex=True,
    gridspec_kw={"hspace": 0.08},
)

site_colors = {
    "Rincon de la Vieja": "#2ecc71",
    "Guanacaste lowlands": "#e67e22",
}

# Dry season window (Dec-Apr)
for ax in (ax_cloud, ax_ndvi):
    ax.axvspan(335, 365, alpha=0.10, color="#e67e22", zorder=0)
    ax.axvspan(1, 105, alpha=0.10, color="#e67e22", zorder=0)

# ── Top: Cloud fraction ──
for name, grp in cloud_summary.groupby("polygon_id"):
    grp = grp.sort_values("day_of_year")
    color = site_colors.get(name, "gray")
    ax_cloud.plot(grp["day_of_year"], grp["cloud_fraction_mean"],
                  color=color, linewidth=1.5, label=name)
    if "cloud_fraction_std" in grp.columns:
        ax_cloud.fill_between(
            grp["day_of_year"],
            grp["cloud_fraction_mean"] - grp["cloud_fraction_std"],
            grp["cloud_fraction_mean"] + grp["cloud_fraction_std"],
            alpha=0.12, color=color,
        )

ax_cloud.set_ylabel("Cloud Fraction", fontsize=8)
ax_cloud.set_ylim(0, 1.0)
ax_cloud.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
ax_cloud.set_title("Cloud Climatology and Vegetation Phenology \u2014 Costa Rica",
                    fontsize=10, fontweight="bold", pad=8)
ax_cloud.tick_params(labelsize=7)
ax_cloud.annotate("Dry season", xy=(60, 0.12), fontsize=6, ha="center",
                   color="#c0392b", fontstyle="italic")

# ── Bottom: NDVI ──
for name, grp in ndvi_summary.groupby("polygon_id"):
    grp = grp.sort_values("day_of_year")
    color = site_colors.get(name, "gray")
    ax_ndvi.plot(grp["day_of_year"], grp["value_mean"],
                 color=color, linewidth=1.5, label=name)
    if "value_std" in grp.columns:
        ax_ndvi.fill_between(
            grp["day_of_year"],
            grp["value_mean"] - grp["value_std"],
            grp["value_mean"] + grp["value_std"],
            alpha=0.12, color=color,
        )

ax_ndvi.set_ylabel("NDVI", fontsize=8)
ax_ndvi.set_ylim(0, 1.0)
ax_ndvi.set_xlabel("Month", fontsize=8)
ax_ndvi.set_xlim(1, 365)
ax_ndvi.set_xticks(month_starts)
ax_ndvi.set_xticklabels(month_labels)
ax_ndvi.tick_params(labelsize=7)

fig.text(0.15, 0.52, "Dry season\n(optimal)",
         fontsize=7, ha="center", va="center",
         color="#c0392b", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdebd0",
                   edgecolor="#c0392b", linewidth=0.8))

fig.text(0.5, 0.01,
         "ERA5 cloud fraction (Open-Meteo) and MODIS NDVI (AppEEARS), 2010\u20132022",
         ha="center", fontsize=6, color="#999999")

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, hspace=0.08)
fig.savefig("paper/figures/fig4_cloud_phenology.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("paper/figures/fig4_cloud_phenology.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)

if os.path.exists(tmp_poly):
    os.remove(tmp_poly)
print("Figure 4 saved.")
