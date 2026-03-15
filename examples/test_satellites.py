#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from shapely.geometry import box

from hyplan.satellites import (
    SATELLITE_REGISTRY,
    get_satellite,
    compute_ground_track,
    compute_swath_footprint,
    find_overpasses,
    find_all_overpasses,
    overpasses_to_kml,
)

#%% List all registered satellites
print("=== Registered Satellites ===")
for name, info in SATELLITE_REGISTRY.items():
    print(f"  {name:20s}  NORAD: {info.norad_id:6d}  Swath: {info.swath_width_km:7.1f} km  Max SZA: {info.max_sza:.0f}")

#%% Compute ground track for a single satellite over 2 hours
sat_name = "Landsat-9"
start = datetime.utcnow()
end = start + timedelta(hours=2)

print(f"\n=== Ground Track: {sat_name} ===")
print(f"Time window: {start} to {end} UTC")

track = compute_ground_track(sat_name, start, end, time_step_s=30.0)
print(f"Ground track points: {len(track)}")
print(track[["timestamp", "latitude", "longitude", "altitude_km", "solar_zenith"]].head(10))

#%% Plot global ground track
fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

ax.scatter(
    track["longitude"], track["latitude"],
    c=track["solar_zenith"], cmap="RdYlBu_r", s=5,
    transform=ccrs.PlateCarree(), zorder=5,
)
ax.set_title(f"{sat_name} Ground Track ({start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')} UTC)")
plt.colorbar(ax.collections[0], ax=ax, label="Solar Zenith Angle (deg)", shrink=0.6)
plt.tight_layout()
plt.show()

#%% Compute swath footprint
swath = compute_swath_footprint(track)
print(f"\n=== Swath Footprint ===")
print(f"Pass segments: {len(swath)}")
for _, row in swath.iterrows():
    print(f"  {row['pass_start']} to {row['pass_end']}  Ascending: {row['ascending']}")

#%% Find overpasses over a region (Southern California)
region = box(-119.0, 33.5, -117.0, 34.5)  # lon_min, lat_min, lon_max, lat_max

start_window = datetime.utcnow()
end_window = start_window + timedelta(days=3)

print(f"\n=== Overpasses: {sat_name} over SoCal ===")
print(f"Search window: {start_window} to {end_window} UTC")

overpasses = find_overpasses(
    sat_name,
    region=region,
    start_time=start_window,
    end_time=end_window,
    time_step_s=10.0,
)

if overpasses.empty:
    print("No overpasses found.")
else:
    print(f"Found {len(overpasses)} overpass(es):")
    for _, row in overpasses.iterrows():
        print(
            f"  {row['pass_start']} - {row['pass_end']}  "
            f"SZA: {row['solar_zenith_at_center']:.1f}  "
            f"Usable: {row['is_usable']}  "
            f"Ascending: {row['ascending']}"
        )

#%% Find overpasses for multiple satellites
print(f"\n=== All Satellite Overpasses over SoCal (3 days) ===")

# Use a subset of satellites for faster results
target_sats = ["Landsat-8", "Landsat-9", "Sentinel-2A", "Sentinel-2B", "PACE"]

all_overpasses = find_all_overpasses(
    satellites=target_sats,
    region=region,
    start_time=start_window,
    end_time=end_window,
    time_step_s=10.0,
)

if all_overpasses.empty:
    print("No overpasses found for any satellite.")
else:
    print(f"Found {len(all_overpasses)} total overpass(es):")
    for _, row in all_overpasses.iterrows():
        print(
            f"  {row['satellite_name']:20s}  "
            f"{row['pass_start']}  "
            f"SZA: {row['solar_zenith_at_center']:.1f}  "
            f"Usable: {row['is_usable']}"
        )

#%% Plot overpasses on map
if not all_overpasses.empty:
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot region of interest
    rx, ry = region.exterior.xy
    ax.plot(rx, ry, "k--", linewidth=2, label="Region of Interest", transform=ccrs.PlateCarree())

    # Plot each overpass
    colors = {"Landsat-8": "red", "Landsat-9": "blue", "Sentinel-2A": "green",
              "Sentinel-2B": "orange", "PACE": "purple"}

    for _, row in all_overpasses.iterrows():
        geom = row["geometry"]
        color = colors.get(row["satellite_name"], "gray")
        label = f"{row['satellite_name']} ({row['pass_start'].strftime('%m-%d %H:%M')})"

        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.2, color=color, transform=ccrs.PlateCarree())
            ax.plot(x, y, color=color, linewidth=1, label=label, transform=ccrs.PlateCarree())
        elif geom.geom_type == "LineString":
            x, y = geom.xy
            ax.plot(x, y, color=color, linewidth=2, label=label, transform=ccrs.PlateCarree())

    ax.set_extent([-120.5, -115.5, 32.5, 35.5])
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Satellite Overpasses — Southern California")
    plt.tight_layout()
    plt.show()

#%% Export to KML
if not all_overpasses.empty:
    overpasses_to_kml(all_overpasses, "satellite_overpasses.kml")
    print("Exported to satellite_overpasses.kml")

# %%
