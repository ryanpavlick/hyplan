import numpy as np
import simplekml
from shapely.geometry import Polygon
import pymap3d.vincenty

from .flight_line import FlightLine
from .sensors import LineScanner
from .terrain import ray_terrain_intersection
from .geometry import process_linestring

__all__ = [
    "generate_swath_polygon",
    "calculate_swath_widths",
    "export_polygon_to_kml",
]


def generate_swath_polygon(
    flight_line: FlightLine,
    sensor: LineScanner,
    along_precision: float = 100.0,
    across_precision: float = 10.0,
    dem_file=None,
) -> Polygon:
    """
    Generate a swath polygon for a given flight line and line scanning imager.

    Args:
        flight_line (FlightLine): The flight line object containing geometry and altitude (MSL).
        sensor (LineScanner): The LineScanner object with field of view (FOV).
        along_precision (float): Precision of the interpolation along the flight line in meters.
        across_precision (float): Precision of the ray-terrain intersection sampling in meters.
        dem_file (str, optional): Path to the DEM file. If None, it will be generated.

    Returns:
        Polygon: A Shapely Polygon representing the swath.
    """
    # Get flight line altitude (MSL) — ray_terrain_intersection expects MSL
    altitude_msl = flight_line.altitude_msl.magnitude

    # Interpolate points along the flight line
    lats, lons, azimuths, *_ = process_linestring(flight_line.track(precision=along_precision))

    # Calculate the half-angle for port and starboard.
    # half_angle is a scalar; ray_terrain_intersection broadcasts it
    # across all along-track points via np.atleast_1d.
    half_angle = sensor.half_angle

    # Compute azimuths for port and starboard sides
    az_port = (azimuths + 270.0) % 360.0
    az_starboard = (azimuths + 90.0) % 360.0

    # Perform ray-terrain intersection for port side 
    port_lats, port_lons, _ = ray_terrain_intersection(
        lats, lons, altitude_msl, az=az_port, tilt=half_angle, 
        precision=across_precision, dem_file=dem_file
    )

    # Perform ray-terrain intersection for starboard side 
    starboard_lats, starboard_lons, _ = ray_terrain_intersection(
        lats, lons, altitude_msl, az=az_starboard, tilt=half_angle, 
        precision=across_precision, dem_file=dem_file
    )

    # Concatenate the two sides
    swath_lats = np.concatenate([port_lats, starboard_lats[::-1]])
    swath_lons = np.concatenate([port_lons, starboard_lons[::-1]])

    # Create a Shapely polygon
    swath_polygon = Polygon(zip(swath_lons, swath_lats))

    return swath_polygon

def calculate_swath_widths(swath_polygon: Polygon) -> dict:
    """Calculate the minimum, mean, and maximum width of a swath polygon.

    Args:
        swath_polygon (Polygon): The swath polygon generated for a flight line.

    Returns:
        dict: A dictionary containing the min, mean, and max widths in meters.
    """
    coords = np.array(swath_polygon.exterior.coords)
    mid_index = len(coords) // 2

    # Split into port and starboard points
    port_coords = coords[:mid_index]
    starboard_coords = coords[mid_index:][::-1]  # Reverse to align correctly

    # Ensure equal lengths for port and starboard
    if len(port_coords) > len(starboard_coords):
        port_coords = port_coords[:len(starboard_coords)]
    elif len(starboard_coords) > len(port_coords):
        starboard_coords = starboard_coords[:len(port_coords)]

    # Extract latitudes and longitudes
    port_lats, port_lons = port_coords[:, 1], port_coords[:, 0]
    starboard_lats, starboard_lons = starboard_coords[:, 1], starboard_coords[:, 0]

    # Vectorized vincenty distance calculation
    distances, _ = pymap3d.vincenty.vdist(
        port_lats, port_lons, starboard_lats, starboard_lons
    )

    # Filter out invalid or zero distances
    valid_distances = distances[distances > 0]

    # Handle edge case where no valid widths are found
    if valid_distances.size == 0:
        return {"min_width": 0.0, "mean_width": 0.0, "max_width": 0.0}

    return {
        "min_width": np.min(valid_distances),
        "mean_width": np.mean(valid_distances),
        "max_width": np.max(valid_distances),
    }

def export_polygon_to_kml(swath_polygon: Polygon, kml_filename: str, name="Swath Polygon") -> None:
    """
    Export a Shapely polygon to a KML file with an unfilled style using simplekml.

    Args:
        swath_polygon (Polygon): A Shapely Polygon representing the swath.
        kml_filename (str): Output KML file path.
        name (str): Name for the KML placemark.
    """
    # Create a KML object
    kml = simplekml.Kml()

    # Convert Shapely polygon coordinates to a list of (lon, lat) tuples
    coords = [(lon, lat) for lon, lat in swath_polygon.exterior.coords]

    # Add the polygon to the KML
    pol = kml.newpolygon(name=name, outerboundaryis=coords)

    # Set the style for the polygon
    pol.style.polystyle.color = simplekml.Color.changealpha("00", simplekml.Color.blue)  # Transparent fill
    pol.style.linestyle.color = simplekml.Color.blue  # Blue border
    pol.style.linestyle.width = 2  # Border width

    # Save the KML to a file
    kml.save(kml_filename)
    print(f"Polygon exported to KML file: {kml_filename}")

