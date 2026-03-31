"""Sensor swath polygon generation and width analysis.

Computes the ground footprint of a line-scanning sensor along a flight line,
accounting for cross-track field of view and altitude.
:func:`generate_swath_polygon` returns a Shapely polygon of the swath;
:func:`calculate_swath_widths` measures port/starboard widths along the track.
"""

from typing import Optional

import numpy as np
import simplekml
from shapely.geometry import Polygon
import pymap3d.vincenty

from .flight_line import FlightLine
from .terrain import ray_terrain_intersection
from .geometry import process_linestring

__all__ = [
    "generate_swath_polygon",
    "calculate_swath_widths",
    "export_polygon_to_kml",
]


def generate_swath_polygon(
    flight_line: FlightLine,
    sensor,
    along_precision: float = 100.0,
    across_precision: float = 10.0,
    dem_file: Optional[str] = None,
) -> Polygon:
    """
    Generate a swath polygon for a given flight line and sensor.

    Works with any sensor that implements ``swath_offset_angles()``
    returning ``(port_edge_angle, starboard_edge_angle)`` in degrees
    from nadir (negative = port, positive = starboard). This includes
    nadir-looking line scanners, tilted line scanners, LVIS, and
    side-looking radar.

    Args:
        flight_line (FlightLine): The flight line object containing geometry and altitude (MSL).
        sensor: A sensor with a ``swath_offset_angles()`` method.
        along_precision (float): Precision of the interpolation along the flight line in meters.
        across_precision (float): Precision of the ray-terrain intersection sampling in meters.
        dem_file (str, optional): Path to the DEM file. If None, it will be generated.

    Returns:
        Polygon: A Shapely Polygon representing the swath.
    """
    altitude_msl = flight_line.altitude_msl.magnitude
    lats, lons, azimuths, *_ = process_linestring(
        flight_line.track(precision=along_precision)
    )

    port_angle, starboard_angle = sensor.swath_offset_angles()

    # Azimuths perpendicular to track
    az_port = (azimuths + 270.0) % 360.0      # left of track
    az_starboard = (azimuths + 90.0) % 360.0   # right of track

    # Each swath edge angle is measured from nadir.
    # Negative = port side, positive = starboard side.
    # Map each edge to (azimuth_array, tilt_from_nadir).
    def _edge_ray(angle):
        if angle < 0:
            return az_port, abs(angle)
        else:
            return az_starboard, angle

    edge1_az, edge1_tilt = _edge_ray(port_angle)
    edge2_az, edge2_tilt = _edge_ray(starboard_angle)

    edge1_lats, edge1_lons, _ = ray_terrain_intersection(
        lats, lons, altitude_msl, az=edge1_az, tilt=edge1_tilt,
        precision=across_precision, dem_file=dem_file
    )
    edge2_lats, edge2_lons, _ = ray_terrain_intersection(
        lats, lons, altitude_msl, az=edge2_az, tilt=edge2_tilt,
        precision=across_precision, dem_file=dem_file
    )

    # Filter out NaN values from failed terrain intersections
    valid1 = ~(np.isnan(edge1_lats) | np.isnan(edge1_lons))
    valid2 = ~(np.isnan(edge2_lats) | np.isnan(edge2_lons))
    edge1_lats, edge1_lons = edge1_lats[valid1], edge1_lons[valid1]
    edge2_lats, edge2_lons = edge2_lats[valid2], edge2_lons[valid2]

    swath_lats = np.concatenate([edge1_lats, edge2_lats[::-1]])
    swath_lons = np.concatenate([edge1_lons, edge2_lons[::-1]])
    return Polygon(zip(swath_lons, swath_lats))

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

