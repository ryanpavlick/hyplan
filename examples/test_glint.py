#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from hyplan.flight_line import FlightLine
from hyplan.sensors import AVIRIS3
from hyplan.glint import calculate_target_and_glint_vectorized, compute_glint_vectorized
from hyplan.units import ureg



#%% Example Inputs for calculate_target_and_glint_vectorized

# Sensor parameters
sensor_lat = np.array([34.05, 34.06, 34.07])
sensor_lon = np.array([-118.25, -118.26, -118.27])
sensor_alt = np.array([1000, 1000, 1000])  # Altitude in meters
viewing_azimuth = np.array([45, 90, 135])  # Azimuth angles
tilt_angle = np.array([10, 20, 30])  # Tilt angles
observation_datetime = np.array([
    pd.Timestamp("2025-02-17T17:00:00"),
    pd.Timestamp("2025-02-17T17:05:00"),
    pd.Timestamp("2025-02-17T17:10:00"),
])

# Calculate glint results
results = calculate_target_and_glint_vectorized(
    sensor_lat=sensor_lat,
    sensor_lon=sensor_lon,
    sensor_alt=sensor_alt,
    viewing_azimuth=viewing_azimuth,
    tilt_angle=tilt_angle,
    observation_datetime=observation_datetime
)

# Print the results
print("Results from calculate_target_and_glint_vectorized:")
print(f"Target Latitudes: {results[0]}")
print(f"Target Longitudes: {results[1]}")
print(f"Glint Angles: {results[2]}")

#%% Setup for compute_glint_vectorized

# Initialize flight line parameters
altitude = ureg.Quantity(5000, "meter")
lat0 = 9.5  # Center latitude
lon0 = -84.75  # Center longitude
azimuth = 0.0  # Azimuth in degrees
length = ureg.Quantity(50000, "meter")  # Flight line length in meters

# Create a flight line object
flight_line = FlightLine.start_length_azimuth(
    lat1=lat0,
    lon1=lon0,
    length=length,
    az=azimuth,
    altitude_msl=altitude,
)

# Initialize sensor
sensor = AVIRIS3()

# Observation datetime
observation_datetime = datetime(2025, 2, 17, 18, 0, tzinfo=timezone.utc)

# Compute glint for the flight line
gdf = compute_glint_vectorized(flight_line, sensor, observation_datetime)

# Print the head of the GeoDataFrame
print("\nGeoDataFrame from compute_glint_vectorized:")
print(gdf.head())

# Export to GeoJSON
gdf.to_file("glint_results.geojson", driver="GeoJSON")

#%% Plotting

from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

# Compute glint
gdf = compute_glint_vectorized(flight_line, sensor, observation_datetime, output_geometry="along_track")

# Get the swath polygon outline
# swath_polygon = generate_swath_polygon(flight_line, sensor)

# Define the normalization and colormap
norm = TwoSlopeNorm(vmin=0, vcenter=35, vmax=45)  # Fixed range with midpoint at 25
cmap = plt.cm.viridis_r  # Use the inverted colormap

# Plot the GeoDataFrame with the swath polygon outline
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the glint angles as points
gdf.plot(
    ax=ax,
    column='glint_angle',  # Use glint_angle for color mapping
    cmap=cmap,             # Custom colormap
    norm=norm,             # Apply normalization
    legend=True,           # Add a legend
    legend_kwds={'label': "Glint Angle (degrees)", 'orientation': "vertical"}
)

# # Plot the swath polygon outline (uncomment if you have the polygon)
# if isinstance(swath_polygon, Polygon):
#     x, y = swath_polygon.exterior.xy
#     ax.plot(x, y, color='red', linewidth=2)
# else:
#     for poly in swath_polygon:  # In case it's a MultiPolygon
#         x, y = poly.exterior.xy
#         ax.plot(x, y, color='red', linewidth=2)

# Add title and axis labels
ax.set_title("Glint Angles Across Flight Line with Swath Outline", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Adjust aspect ratio to reflect real-world scales
ratio = 4
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_high - y_low)) * ratio, adjustable="box")

# Show the plot
plt.show()

# %%
