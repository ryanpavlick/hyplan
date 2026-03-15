#%%
import logging
from shapely.geometry import Polygon
from hyplan.flight_box import (
    box_around_center_line,
    box_around_polygon
)
from hyplan.units import ureg
from hyplan.sensors import AVIRIS3
from hyplan.swath import generate_swath_polygon
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define test inputs
instrument = AVIRIS3()
altitude = ureg.Quantity(5000, "meter")
polygon = Polygon([
    (-118.7, 34.0),  # Bottom-left
    (-118.6, 34.3),  # Top-left
    (-118.3, 34.2),  # Top-right
    (-118.4, 34.0),  # Bottom-right
    (-118.5, 34.1),  # Middle
])


def plot_flight_lines_and_swaths(
    flight_lines, polygon, bounding_box=None, swaths=None, centroid=None, title="Flight Lines and Swaths"
):
    """
    Plot flight lines, swaths, bounding box, and the input polygon.

    Args:
        flight_lines (List[FlightLine]): List of FlightLine objects.
        polygon (Polygon): Input Shapely Polygon.
        bounding_box (Optional[Polygon]): Bounding box as a Shapely Polygon.
        swaths (Optional[List[Polygon]]): List of swath polygons.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the input polygon
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, label="Input Polygon", color="blue", linewidth=2)

    # Plot the bounding box if provided
    if bounding_box:
        x, y = bounding_box.exterior.xy
        ax.plot(x, y, label="Bounding Box", color="green", linestyle="--", linewidth=2)

    # Plot the flight lines
    for line in flight_lines:
        line_geom = line.geometry
        if isinstance(line_geom, LineString):
            x, y = line_geom.xy
            ax.plot(x, y, label=line.site_name, color="red", alpha=0.6)

    # Plot swaths
    if swaths:
        for swath in swaths:
            if isinstance(swath, Polygon):
                x, y = swath.exterior.xy
                ax.fill(x, y, color="orange", alpha=0.3, label="Swath")

    if centroid:
        ax.plot(centroid.x, centroid.y, "ro", label="Centroid")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # ax.legend()
    ax.grid(True)
    plt.show()


#%% Test box_around_center_line
logging.info("Testing box_around_center_line...")
center_lines = box_around_center_line(
    instrument=instrument,
    altitude_msl=altitude,
    lat0=34.0,
    lon0=-118.5,
    azimuth=45.0,
    box_length=ureg.Quantity(60000, "meter"),
    box_width=ureg.Quantity(15000, "meter"),
    overlap=20.0
)

# Generate swaths for center-aligned lines
center_swaths = [generate_swath_polygon(line, instrument) for line in center_lines]

# Plot center-aligned flight lines and swaths
plot_flight_lines_and_swaths(
    flight_lines=center_lines,
    polygon=None,
    bounding_box=None,
    swaths=center_swaths,
    title="Center-Aligned Flight Lines and Swaths",
)

#%% Test box_around_minimum_rectangle
logging.info("Testing box_around_minimum_rectangle...")
min_rect_lines = box_around_polygon(
    instrument=instrument,
    altitude_msl=altitude,
    polygon=polygon,
    box_name="MinRectTest",
    overlap=20.0,
)

# Generate swaths for minimum rectangle lines
min_rect_swaths = [generate_swath_polygon(line, instrument) for line in min_rect_lines]

# Plot minimum rotated rectangle flight lines and swaths
plot_flight_lines_and_swaths(
    flight_lines=min_rect_lines,
    polygon=polygon,
    swaths=min_rect_swaths,
    title="Minimum Rotated Rectangle Flight Lines and Swaths",
)

#%% Test box_around_rotated_rectangle
logging.info("Testing box_around_rotated_rectangle...")
rotated_rect_lines = box_around_polygon(
    instrument=instrument,
    altitude_msl=altitude,
    polygon=polygon,
    azimuth=160.0,
    box_name="RotatedRectTest",
    overlap=20.0,
)

# Generate swaths for rotated rectangle lines
rotated_rect_swaths = [generate_swath_polygon(line, instrument) for line in rotated_rect_lines]

# Plot rotated rectangle flight lines and swaths
plot_flight_lines_and_swaths(
    flight_lines=rotated_rect_lines,
    polygon=polygon,
    swaths=rotated_rect_swaths,
    title="Rotated Rectangle Flight Lines and Swaths",
)


# %%


