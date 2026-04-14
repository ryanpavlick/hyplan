"""Elevation sampling, min/max queries, along-track profiles, and aspect."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..geometry import process_linestring
from ._demgrid import DEMGrid
from .io import generate_demfile, load_dem

logger = logging.getLogger(__name__)


def get_elevations(lats: np.ndarray, lons: np.ndarray, dem_file: str) -> np.ndarray:
    """
    Extract elevation values for given latitudes and longitudes from a DEM file.

    Reads the entire raster band once and indexes it in bulk, rather than
    querying pixel-by-pixel. The raster is cached per ``(path, mtime)`` so
    repeated calls against the same DEM (the common case in flight planning)
    pay the read cost only once.
    """
    dem = load_dem(dem_file)
    return get_elevations_from_grid(lats, lons, dem)


def get_elevations_from_grid(
    lats: np.ndarray, lons: np.ndarray, dem: DEMGrid
) -> np.ndarray:
    """Extract elevation values from a :class:`DEMGrid` (no file I/O)."""
    gt = dem.geotransform
    raster = dem.array

    xs = np.round((lons - gt[0]) / gt[1]).astype(int)
    ys = np.round((lats - gt[3]) / gt[5]).astype(int)

    out_of_bounds = (xs < 0) | (xs >= raster.shape[1]) | (ys < 0) | (ys >= raster.shape[0])
    if np.any(out_of_bounds):
        n_oob = int(out_of_bounds.sum())
        logger.warning(
            f"{n_oob} query point(s) fall outside the DEM extent. "
            "Edge pixel elevations will be used for these points."
        )

    xs = np.clip(xs, 0, raster.shape[1] - 1)
    ys = np.clip(ys, 0, raster.shape[0] - 1)

    return raster[ys, xs]


def get_min_max_elevations(dem_file: str) -> Tuple[float, float]:
    """
    Get the minimum and maximum elevation values from a DEM file.

    Args:
        dem_file (str): Path to the DEM file.

    Returns:
        Tuple[float, float]: (min_elevation, max_elevation) in the DEM file.
    """
    dem = load_dem(dem_file)
    return dem.raster_min, dem.raster_max


def terrain_elevation_along_track(flight_line, dem_file: str,
                                   precision: float = 100.0) -> dict:
    """Min, mean, and max terrain elevation (m MSL) along a flight line's nadir track.

    Samples the DEM at evenly-spaced points along the flight line and returns
    summary statistics useful for Mode 3 per-line altitude planning.

    Args:
        flight_line: A FlightLine object with a ``track(precision)`` method.
        dem_file: Path to a DEM GeoTIFF covering the flight line.
        precision: Along-track sampling interval in meters. Default 100 m.

    Returns:
        Dict with keys ``"min"``, ``"mean"``, and ``"max"`` (all in meters MSL).
    """
    lats, lons, *_ = process_linestring(flight_line.track(precision=precision))
    elevations = get_elevations(lats, lons, dem_file).astype(float)
    return {
        "min": float(np.nanmin(elevations)),
        "mean": float(np.nanmean(elevations)),
        "max": float(np.nanmax(elevations)),
    }


def terrain_aspect_azimuth(polygon, dem_file: str = None) -> float:
    """Dominant terrain gradient direction (degrees from north) for a polygon.

    Computes the dominant downslope azimuth from the DEM gradient over the
    polygon area.  To minimise altitude variation along each flight line
    (as recommended by Zhao et al. 2021 for Mode 3), orient flight lines
    *perpendicular* to the returned azimuth::

        flight_azimuth = (terrain_aspect_azimuth(polygon) + 90) % 360

    Args:
        polygon: Shapely Polygon defining the survey area (WGS84 lon/lat).
        dem_file: Path to a DEM GeoTIFF.  If ``None``, one is downloaded and
            cached from the Copernicus GLO-30 archive.

    Returns:
        Dominant downslope azimuth in degrees clockwise from true north
        (range 0-360).
    """
    if dem_file is None:
        coords = np.array(polygon.exterior.coords)
        lons_poly = coords[:, 0]
        lats_poly = coords[:, 1]
        dem_file = generate_demfile(lats_poly, lons_poly)

    dem = load_dem(dem_file)
    elevations = dem.array.astype(float)

    # np.gradient returns (d/d_row, d/d_col).  In a north-up GeoTIFF rows
    # increase southward, so the north component is the *negative* row gradient.
    dy, dx = np.gradient(elevations)
    north_component = -dy   # positive = elevation increases going north
    east_component = dx     # positive = elevation increases going east

    # Aspect: direction of steepest ascent, clockwise from north.
    aspect = np.degrees(np.arctan2(east_component, north_component))
    downslope = (aspect + 180.0) % 360.0  # reverse to get downslope direction

    # Circular mean, ignoring flat pixels (lowest quartile of gradient magnitude)
    magnitude = np.sqrt(north_component ** 2 + east_component ** 2)
    threshold = np.percentile(magnitude, 25)
    significant = magnitude > threshold

    downslope_rad = np.radians(downslope[significant])
    mean_sin = float(np.nanmean(np.sin(downslope_rad)))
    mean_cos = float(np.nanmean(np.cos(downslope_rad)))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
