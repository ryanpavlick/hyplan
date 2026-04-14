"""Sensor swath polygon generation and width analysis.

Computes the ground footprint of a line-scanning sensor along a flight line,
accounting for cross-track field of view and altitude.
:func:`generate_swath_polygon` returns a Shapely polygon of the swath;
:func:`calculate_swath_widths` measures port/starboard widths along the track.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import simplekml
from shapely.geometry import Polygon
from shapely.ops import transform
import pymap3d.vincenty

from .flight_line import FlightLine
from .instruments import ScanningSensor
from .terrain import ray_terrain_intersection
from .geometry import get_utm_transforms, process_linestring

__all__ = [
    "generate_swath_polygon",
    "calculate_swath_widths",
    "analyze_swath_gaps_overlaps",
    "export_polygon_to_kml",
]


def _resolve_swath_boresight_azimuths(
    track_azimuths: np.ndarray,
    heading_mode: str = "track",
    crab_angle_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
) -> np.ndarray:
    """Compute instrument boresight azimuths for swath edge computation.

    In ``"track"`` mode (default), the instrument is assumed aligned with
    the ground track.  In ``"crabbed"`` mode, the boresight is rotated
    to match the aircraft heading, which differs from the track when the
    aircraft crabs into a crosswind.

    Args:
        track_azimuths: Per-point track azimuths (degrees true).
        heading_mode: ``"track"`` or ``"crabbed"``.
        crab_angle_deg: Crab angle to add to track azimuths
            (used when ``heading_mode="crabbed"`` and ``heading_deg``
            is not provided).
        heading_deg: Constant aircraft heading (used when
            ``heading_mode="crabbed"`` for a uniform heading along the
            entire line).

    Returns:
        Array of boresight azimuths (same length as *track_azimuths*).
    """
    if heading_mode == "track":
        return track_azimuths
    if heading_mode != "crabbed":
        raise ValueError(
            f"heading_mode must be 'track' or 'crabbed', got {heading_mode!r}"
        )
    if heading_deg is not None:
        return np.full_like(track_azimuths, heading_deg % 360.0)
    if crab_angle_deg is not None:
        return (track_azimuths + crab_angle_deg) % 360.0
    raise ValueError(
        "heading_mode='crabbed' requires either crab_angle_deg or heading_deg"
    )


def generate_swath_polygon(
    flight_line: FlightLine,
    sensor: ScanningSensor,
    along_precision: float = 100.0,
    across_precision: float = 10.0,
    dem_file: Optional[str] = None,
    heading_mode: str = "track",
    crab_angle_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
) -> Polygon:
    """Generate a swath polygon for a given flight line and sensor.

    Works with any sensor satisfying the
    :class:`~hyplan.instruments.ScanningSensor` protocol — i.e. exposing
    ``swath_offset_angles()`` returning ``(port_edge_angle, starboard_edge_angle)``
    in degrees from nadir (negative = port, positive = starboard). This
    includes nadir-looking line scanners, tilted line scanners, LVIS, and
    side-looking radar.

    Args:
        flight_line: The flight line containing geometry and altitude (MSL).
        sensor: Any object satisfying :class:`~hyplan.instruments.ScanningSensor`.
        along_precision: Along-track interpolation spacing (meters).
        across_precision: Ray-terrain intersection sampling (meters).
        dem_file: Path to the DEM file.  If *None*, one is generated.
        heading_mode: ``"track"`` (default) orients the swath
            perpendicular to the ground track.  ``"crabbed"`` orients it
            perpendicular to the aircraft heading, which differs from
            the track when the aircraft crabs into a crosswind.
        crab_angle_deg: Crab angle (degrees) to add to track azimuths.
            Only used when ``heading_mode="crabbed"``.
        heading_deg: Constant aircraft heading (degrees true).
            Only used when ``heading_mode="crabbed"``.

    Returns:
        A Shapely Polygon representing the swath.
    """
    altitude_msl = flight_line.altitude_msl.magnitude
    lats, lons, azimuths, *_ = process_linestring(
        flight_line.track(precision=along_precision)
    )

    # Resolve boresight direction (track or crabbed heading)
    boresight = _resolve_swath_boresight_azimuths(
        azimuths, heading_mode, crab_angle_deg, heading_deg,
    )

    port_angle, starboard_angle = sensor.swath_offset_angles()

    # Swath edges perpendicular to boresight direction
    az_port = (boresight + 270.0) % 360.0
    az_starboard = (boresight + 90.0) % 360.0

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

def analyze_swath_gaps_overlaps(
    swath_polygons: List[Polygon],
) -> pd.DataFrame:
    """Pairwise gap/overlap analysis between adjacent swath polygons.

    For each consecutive pair ``(swath_polygons[i], swath_polygons[i+1])``,
    reports the area of overlap and the area of any gap between them. Either
    value is zero when the polygons abut or overlap entirely. The caller is
    responsible for ordering the polygons (e.g. in the perpendicular-offset
    order produced by :func:`hyplan.flight_box.box_around_center_line`).

    Each pair is projected to a local UTM zone before computing areas, so
    the resulting values are in square meters regardless of the input CRS
    (must be lon/lat in WGS84).

    Args:
        swath_polygons: Polygons in adjacency order. Must be in WGS84
            (EPSG:4326). Returns an empty DataFrame if fewer than two
            polygons are provided.

    Returns:
        DataFrame with columns:
            - ``pair_index``: index ``i`` of the first polygon in the pair.
            - ``overlap_area_m2``: area of intersection in m².
            - ``gap_area_m2``: area between the two polygons in m² when
              they don't overlap; 0.0 if they do.
            - ``overlap_fraction``: ``overlap_area_m2`` divided by the
              mean of the two polygon areas.
    """
    if len(swath_polygons) < 2:
        return pd.DataFrame(
            columns=["pair_index", "overlap_area_m2", "gap_area_m2", "overlap_fraction"]
        )

    rows = []
    for i in range(len(swath_polygons) - 1):
        a = swath_polygons[i]
        b = swath_polygons[i + 1]

        # Project this pair to a local UTM CRS so areas come out in m².
        to_utm, _ = get_utm_transforms([a.centroid, b.centroid])
        a_m = transform(to_utm, a)
        b_m = transform(to_utm, b)

        overlap_area = a_m.intersection(b_m).area
        if overlap_area > 0:
            gap_area = 0.0
        else:
            # Gap = area inside the convex hull of the pair but in neither polygon.
            hull = a_m.union(b_m).convex_hull
            gap_area = hull.area - a_m.area - b_m.area
            gap_area = max(gap_area, 0.0)

        mean_area = 0.5 * (a_m.area + b_m.area)
        overlap_fraction = overlap_area / mean_area if mean_area > 0 else 0.0

        rows.append({
            "pair_index": i,
            "overlap_area_m2": overlap_area,
            "gap_area_m2": gap_area,
            "overlap_fraction": overlap_fraction,
        })

    return pd.DataFrame(rows)


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

