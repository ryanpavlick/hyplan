#%%
"""Figure 3: Complete mission — still air vs wind, map + altitude profile.

Uses compute_flight_plan over Rincón de la Vieja National Park with
MRLB (Liberia) as departure/return airport.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from hyplan import (
    AVIRIS3, KingAirB200,
    Airport, initialize_data,
    box_around_polygon_terrain,
    compute_flight_plan, ConstantWindField,
    ureg,
)

# ── Setup ──
sensor = AVIRIS3()
aircraft = KingAirB200()
flight_altitude = ureg.Quantity(25000, "feet")
cruise_speed = aircraft.cruise_speed_at(flight_altitude)
bank_angle = 25.0

gdf = gpd.read_file("rincon_de_la_vieja.geojson")
study_area = gdf.geometry.iloc[0]

initialize_data(countries=["CR"])
airport = Airport("MRLB")

# Terrain-aware clipped lines (same parameters as fig2_flight_box.py)
clip_buffer_deg = 0.012
buffered_area = study_area.buffer(clip_buffer_deg)

lines = box_around_polygon_terrain(
    instrument=sensor,
    altitude_msl=flight_altitude,
    polygon=study_area,
    box_name="RDV",
    overlap=20.0,
    alternate_direction=True,
    clip_to_polygon=True,
    clip_polygon=buffered_area,
)

# Wind: 60 kt northeasterly wind (from NE)
wind_speed = 60 * ureg.knot
wind_field = ConstantWindField(wind_speed=wind_speed, wind_from_deg=45.0)

# ── Flight plans ──
plan_still = compute_flight_plan(
    aircraft=aircraft, flight_sequence=lines,
    takeoff_airport=airport, return_airport=airport,
)
plan_wind = compute_flight_plan(
    aircraft=aircraft, flight_sequence=lines,
    takeoff_airport=airport, return_airport=airport,
    wind_source=wind_field,
)

# ── Segment colors ──
seg_colors = {
    "takeoff": "#8e44ad", "climb": "#8e44ad",
    "transit": "#e67e22",
    "flight_line": "#2c3e50",
    "descent": "#8e44ad", "approach": "#8e44ad",
}
seg_lw = {
    "takeoff": 1.2, "climb": 1.2,
    "transit": 1.8,
    "flight_line": 2.5,
    "descent": 1.2, "approach": 1.2,
}

# ── Figure ──
fig = plt.figure(figsize=(10, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.28, wspace=0.08)
ax_map1 = fig.add_subplot(gs[0, 0])
ax_map2 = fig.add_subplot(gs[0, 1], sharey=ax_map1)
ax_alt = fig.add_subplot(gs[1, :])

# ── Top row: Map views ──
for ax, plan, title in [
    (ax_map1, plan_still, "Still air"),
    (ax_map2, plan_wind, "60 kt northeasterly wind"),
]:
    # Study area
    sx, sy = study_area.exterior.xy
    ax.plot(sx, sy, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, zorder=1)
    ax.fill(sx, sy, alpha=0.05, color="green", zorder=0)

    # Airport
    ax.plot(airport.longitude, airport.latitude, "s", color="#8e44ad",
            markersize=6, zorder=6)
    ax.text(airport.longitude - 0.02, airport.latitude + 0.008,
            airport.icao_code, fontsize=8, color="#8e44ad",
            fontweight="bold", zorder=6, ha="right")

    # Plot each segment
    for _, seg in plan.iterrows():
        geom = seg.geometry
        if geom is None or geom.is_empty:
            continue
        seg_type = seg["segment_type"]
        color = seg_colors.get(seg_type, "#bdc3c7")
        lw = seg_lw.get(seg_type, 1.0)
        xs, ys = geom.xy
        ax.plot(xs, ys, color=color, linewidth=lw,
                zorder=3 if seg_type == "flight_line" else 2)

        if seg_type == "flight_line":
            mid = len(xs) // 2
            if mid > 0:
                ax.annotate("", xy=(xs[mid], ys[mid]),
                            xytext=(xs[mid - 1], ys[mid - 1]),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                            zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.tick_params(labelsize=7)

    # Wind arrows (right panel only)
    if plan is plan_wind:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        for wlon in np.arange(xlims[0], xlims[1], 0.05):
            for wlat in np.arange(ylims[0], ylims[1], 0.015):
                ax.annotate("", xy=(wlon - 0.011, wlat - 0.011),
                            xytext=(wlon, wlat),
                            arrowprops=dict(arrowstyle="-|>", color="#e74c3c",
                                            lw=0.5, alpha=0.2), zorder=1)

ax_map1.set_ylabel("Latitude", fontsize=9)
plt.setp(ax_map2.get_yticklabels(), visible=False)

# Legends on both panels
legend_elements = [
    plt.Line2D([0], [0], color="#2c3e50", linewidth=2.5, label="Data collection"),
    plt.Line2D([0], [0], color="#e67e22", linewidth=1.8, label="Transit (Dubins)"),
    plt.Line2D([0], [0], color="#8e44ad", linewidth=1.2, label="Takeoff / landing"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#8e44ad",
               markersize=5, label=f"Airport ({airport.icao_code})"),
]
ax_map1.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)
ax_map2.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)

# ── Bottom: Altitude profile ──
alt_offset = 500  # ft visual offset so traces don't overlap
for plan, label, ls, alpha, offset in [
    (plan_still, "Still air", "-", 0.8, alt_offset),
    (plan_wind, "With wind", "--", 0.5, 0),
]:
    cum_time = 0.0
    for _, seg in plan.iterrows():
        t = seg["time_to_segment"]
        seg_type = seg["segment_type"]
        color = seg_colors.get(seg_type, "#bdc3c7")
        h_start = seg["start_altitude"] + offset
        h_end = seg["end_altitude"] + offset
        ax_alt.plot([cum_time, cum_time + t], [h_start, h_end],
                    color=color, linewidth=1.5 if ls == "-" else 1.0,
                    linestyle=ls, alpha=alpha)
        cum_time += t

alt_legend = [
    plt.Line2D([0], [0], color="#8e44ad", lw=1.5, label="Takeoff / landing"),
    plt.Line2D([0], [0], color="#e67e22", lw=1.5, label="Transit"),
    plt.Line2D([0], [0], color="#2c3e50", lw=1.5, label="Data collection"),
    plt.Line2D([0], [0], color="black", lw=1.5, ls="-", label=f"Still air (+{alt_offset} ft offset)"),
    plt.Line2D([0], [0], color="black", lw=1.0, ls="--", label="With wind"),
]
ax_alt.legend(handles=alt_legend, loc="upper right", fontsize=7, framealpha=0.9)
ax_alt.set_xlabel("Mission time (minutes)", fontsize=9)
ax_alt.set_ylabel("Altitude (ft MSL)", fontsize=9)
ax_alt.set_title("Altitude profile", fontsize=11, fontweight="bold")
ax_alt.set_ylim(-500, flight_altitude.m_as("feet") * 1.15)
ax_alt.tick_params(labelsize=7)
ax_alt.axhline(flight_altitude.m_as("feet"), color="#bdc3c7", linewidth=0.5,
               linestyle=":", zorder=0)

t_still = plan_still["time_to_segment"].sum()
t_wind = plan_wind["time_to_segment"].sum()
ax_alt.text(0.98, 0.15,
            f"Still air: {t_still:.0f} min\nWith wind: {t_wind:.0f} min "
            f"({t_wind - t_still:+.0f})",
            transform=ax_alt.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", linewidth=0.5))

fig.text(0.5, 0.99,
         f"King Air B200, {cruise_speed.to(ureg.knot):.0f} cruise, "
         f"{bank_angle}\u00b0 bank"
         f" \u2014 {airport.icao_code} \u2192 Rinc\u00f3n de la Vieja \u2192 {airport.icao_code}",
         ha="center", fontsize=8, color="#555555", va="top")

fig.savefig("fig3_wind_dubins.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("fig3_wind_dubins.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure 3 saved.")

# %%
