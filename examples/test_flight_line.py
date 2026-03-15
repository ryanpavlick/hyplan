#%%
import logging
from shapely.geometry import Polygon
from hyplan.units import ureg
from hyplan.flight_line import FlightLine, to_gdf
import geopandas as gpd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Helper function to plot flight lines
def plot_flight_lines(title, flight_lines, polygons=None):
    """
    Plot flight lines and optional polygons.

    Args:
        title (str): Title of the plot.
        flight_lines (list): List of (FlightLine, label) tuples.
        polygons (list, optional): List of polygons to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    for flight_line, label in flight_lines:
        x, y = zip(*flight_line.geometry.coords)
        ax.plot(x, y, marker="o", label=label)

    if polygons:
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, linestyle="--")

    ax.legend()
    plt.grid()
    plt.show()

# Helper function to plot with arrows
def add_arrow(ax, line, color, label, arrow_length=0.02):
    """
    Add an arrow along the line to indicate its direction.

    Args:
        ax: The matplotlib axis to plot on.
        line: The FlightLine object.
        color: Color of the arrow.
        label: Label for the legend.
        arrow_length: Fraction of the line length for the arrow.
    """
    coords = list(line.geometry.coords)
    mid_idx = len(coords) // 2
    start = coords[mid_idx - 1]
    end = coords[mid_idx]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="-|>", lw=1.5),
    )

# %% Example 1: Create a FlightLine using start_range_azimuth
flight_line_1 = FlightLine.start_length_azimuth(
    lat1=34.05,
    lon1=-118.25,
    length=ureg.Quantity(50000, "meter"),
    az=45.0,
    altitude_msl=ureg.Quantity(1000, "meter"),
    site_name="LA Northeast",
    investigator="Dr. Smith"
)
plot_flight_lines("Example 1: Start Range Azimuth", [(flight_line_1, "LA Northeast")])

# %% Example 2: Create a FlightLine using center_radius_azimuth
flight_line_2 = FlightLine.center_length_azimuth(
    lat=34.15,
    lon=-118.55,
    length=ureg.Quantity(80000, "meter"),
    az=80.0,
    altitude_msl=ureg.Quantity(1200, "meter"),
    site_name="LA East-West",
    investigator="Dr. Jones"
)
plot_flight_lines("Example 2: Center Radius Azimuth", [(flight_line_2, "LA East-West")])

# %% Example 3: Clip a FlightLine to a polygon

def plot_clipped_flight_lines_with_original(
    title, original_flight_line, clipped_flight_lines, polygon
):
    """
    Plot the original flight line, clipped flight lines, and the clipping polygon.

    Args:
        title (str): Title of the plot.
        original_flight_line (FlightLine): The original flight line.
        clipped_flight_lines (list): List of tuples with (FlightLine, label) for clipped segments.
        polygon (Polygon): The clipping polygon.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # Plot the original flight line
    x, y = zip(*original_flight_line.geometry.coords)
    ax.plot(x, y, linestyle="--", color="blue", label="Original Flight Line", alpha=0.7)

    # Plot the clipped flight lines
    for flight_line, label in clipped_flight_lines:
        x, y = zip(*flight_line.geometry.coords)
        ax.plot(x, y, marker="o", label=label)

    # Highlight the clipping polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, linestyle="--", color="red", label="Clipping Polygon")

    # Add annotations for clipped segments
    for flight_line, label in clipped_flight_lines:
        midpoint = flight_line.geometry.interpolate(0.5, normalized=True)
        ax.annotate(label, (midpoint.x, midpoint.y), fontsize=9, color="green")

    ax.legend()
    plt.grid()
    plt.show()

# Create a clipping polygon
clip_polygon = Polygon([
    (-118.7, 34.0),  # Bottom-left
    (-118.6, 34.3),  # Top-left
    (-118.3, 34.2),  # Top-right
    (-118.4, 34.0),  # Bottom-right
    (-118.5, 34.1),  # Middle
])

# Clip the flight line
clipped_flight_lines = flight_line_2.clip_to_polygon(clip_polygon)

if clipped_flight_lines:
    # Prepare the clipped flight lines with labels
    labeled_clipped_flight_lines = [
        (segment, segment.site_name or f"Segment {i}") for i, segment in enumerate(clipped_flight_lines)
    ]

    # Plot the original and clipped flight lines
    plot_clipped_flight_lines_with_original(
        "Example 3: Clip to Polygon",
        flight_line_2,
        labeled_clipped_flight_lines,
        clip_polygon
    )
else:
    print("No valid segments after clipping.")


# %% Example 4: Rotate a FlightLine around its midpoint
angles = [30, 60, 90, 120, 150, 180]
rotated_lines = [
    (flight_line_1.rotate_around_midpoint(angle=angle), f"Rotated {angle}°")
    for angle in angles
]
plot_flight_lines("Example 4: Rotate Around Midpoint", [(flight_line_1, "Original")] + rotated_lines)

# %% Example 5: Split a FlightLine into segments with a gap
segments_with_gap = flight_line_1.split_by_length(
    max_length=ureg.Quantity(5000, "meter"),
    gap_length=ureg.Quantity(2000, "meter")
)
plot_flight_lines(
    "Example 5: Flight Line Segments with Gaps",
    [(seg, f"Segment {i}") for i, seg in enumerate(segments_with_gap)]
)

# %% Example 6: Offset FlightLine along its direction
offset_line = flight_line_1.offset_along(
    offset_start=ureg.Quantity(10000, "meter"),
    offset_end=ureg.Quantity(-5000, "meter")
)
plot_flight_lines(
    "Example 6: Offset Along Direction",
    [(flight_line_1, "Original"), (offset_line, "Offset Along")]
)

# %% Example 7: Offset FlightLine perpendicular to its direction
offset_across_line = flight_line_1.offset_across(ureg.Quantity(5000, "meter"))
plot_flight_lines(
    "Example 7: Offset Across",
    [(flight_line_1, "Original"), (offset_across_line, "Offset Across")]
)

# %% Example 8: Reverse a FlightLine
reversed_line = offset_across_line.reverse()
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Example 8: Reverse Flight Line with Directional Arrows")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")

ax.plot(*flight_line_1.geometry.xy, label="Original", color="blue")
ax.plot(*reversed_line.geometry.xy, label="Reversed and Offset", color="orange")

add_arrow(ax, flight_line_1, color="blue", label="Original")
add_arrow(ax, reversed_line, color="orange", label="Reversed")

ax.legend()
plt.grid()
plt.show()

# %% Example 9: Convert a list of FlightLines to a GeoDataFrame
flight_lines = [flight_line_1, flight_line_2]
gdf = to_gdf(flight_lines)
print(gdf)

# %% Example 10: Export FlightLines to GeoJSON
gdf.to_file("flight_lines.geojson", driver="GeoJSON")
print("Exported GeoJSON to flight_lines.geojson")

# %% Example 11: Import FlightLines from GeoJSON
imported_gdf = gpd.read_file("flight_lines.geojson")
print("Imported GeoJSON:")
print(imported_gdf)

# %%
from leafmap import leafmap

m = leafmap.Map()
m.edit_vector("flight_lines.geojson")
# %%
