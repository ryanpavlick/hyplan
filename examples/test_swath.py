#%%
import matplotlib.pyplot as plt

from hyplan.units import ureg
from hyplan.flight_line import FlightLine
from hyplan.sensors import AVIRIS3
from hyplan.swath import generate_swath_polygon, calculate_swath_widths, export_polygon_to_kml

# Initialize test parameters
altitude = ureg.Quantity(5000, "meter")
lat0 = 34.296  # Center latitude
lon0 = -117.593  # Center longitude
azimuth = 90.0  # Azimuth in degrees
length = ureg.Quantity(60000, "meter")  # Flight line length in meters

# Create a flight line
flight_line = FlightLine.start_length_azimuth(
    lat1=lat0,
    lon1=lon0,
    length=length,
    az=azimuth,
    altitude_msl=altitude,
)

# Initialize the spectrometer
instrument = AVIRIS3()

# Generate a swath polygon
swath_polygon = generate_swath_polygon(
    flight_line=flight_line,
    sensor=instrument,
    along_precision=100.0,
    across_precision=10.0,
    dem_file=None,
)

# Visualize the swath polygon
fig, ax = plt.subplots()
x, y = swath_polygon.exterior.xy
ax.plot(x, y, label="Swath Polygon", color="blue")
ax.set_title("Swath Polygon Visualization")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

# Get limits
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()

# Adjust aspect ratio
ratio = 0.5
ax.set_aspect(abs((x_right - x_left) / (y_high - y_low)) * ratio, adjustable="box")

# Show the plot
plt.show()

# Calculate swath widths
swath_widths = calculate_swath_widths(swath_polygon)

# Print the swath widths
print("Swath Widths:")
print(f"Minimum Width: {swath_widths['min_width']:.2f} meters")
print(f"Mean Width: {swath_widths['mean_width']:.2f} meters")
print(f"Maximum Width: {swath_widths['max_width']:.2f} meters")

# Example usage
kml_filename = "swath_polygon.kml"
export_polygon_to_kml(swath_polygon, kml_filename)




# %%
