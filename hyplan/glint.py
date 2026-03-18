from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pymap3d import los
from sunposition import sunpos

from .flight_line import FlightLine
from .sensors import LineScanner
from .geometry import process_linestring
from .exceptions import HyPlanValueError

__all__ = [
    "calculate_target_and_glint_vectorized",
    "glint_angle",
    "compute_glint_vectorized",
]


def calculate_target_and_glint_vectorized(
    sensor_lat: np.ndarray, sensor_lon: np.ndarray, sensor_alt: np.ndarray,
    viewing_azimuth: np.ndarray, tilt_angle: np.ndarray,
    observation_datetime: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of target locations and glint angles for a set of sensors.

    Args:
        sensor_lat (np.ndarray): Latitudes of the sensors (decimal degrees).
        sensor_lon (np.ndarray): Longitudes of the sensors (decimal degrees).
        sensor_alt (np.ndarray): Altitudes of the sensors above sea level (meters).
        viewing_azimuth (np.ndarray): Viewing azimuths relative to true north (degrees).
        tilt_angle (np.ndarray): Tilt angles of the sensors from nadir (degrees).
        observation_datetime (np.ndarray): Timestamps of observations (UTC).

    Returns:
        tuple: (target_lat, target_lon, glint_angles)
    """
    # Ensure observation_datetime is a NumPy array
    observation_datetime = np.asarray(observation_datetime)

    # Step 1: Calculate the target locations on the surface using pymap3d
    target_lat, target_lon, _ = los.lookAtSpheroid(
        sensor_lat,
        sensor_lon,
        sensor_alt,
        viewing_azimuth,
        tilt_angle
    )

    # Step 2: Calculate the solar positions using `sunpos`
    solar_azimuth, solar_zenith, _, _, _ = sunpos(
        dt=observation_datetime,
        latitude=sensor_lat,
        longitude=sensor_lon,
        elevation=sensor_alt,
        radians=False  # Output in degrees
    )

    # Step 3: Calculate the glint angles
    glint_angles = glint_angle(solar_azimuth, solar_zenith, viewing_azimuth, np.abs(tilt_angle))

    return target_lat, target_lon, glint_angles

def glint_angle(solar_azimuth: np.ndarray, solar_zenith: np.ndarray, view_azimuth: np.ndarray, view_zenith: np.ndarray) -> np.ndarray:
    """
    Calculate the specular glint angle between the sun and sensor viewing directions.

    The glint angle is the angular difference between the specular reflection
    direction and the sensor viewing direction, assuming a flat surface. A glint
    angle of 0 degrees indicates perfect specular reflection (sun glint).

    Args:
        solar_azimuth (np.ndarray): Solar azimuth angles in degrees.
        solar_zenith (np.ndarray): Solar zenith angles in degrees.
        view_azimuth (np.ndarray): Sensor viewing azimuth angles in degrees.
        view_zenith (np.ndarray): Sensor viewing zenith angles in degrees (from nadir).

    Returns:
        np.ndarray: Glint angles in degrees. Values near 0 indicate strong sun glint.
    """
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)
    view_zenith_rad = np.deg2rad(view_zenith)
    view_azimuth_rad = np.deg2rad(view_azimuth)

    phi = solar_azimuth_rad - view_azimuth_rad
    glint_cos = (
        np.cos(view_zenith_rad) * np.cos(solar_zenith_rad) -
        np.sin(view_zenith_rad) * np.sin(solar_zenith_rad) * np.cos(phi)
    )

    # Clamp to [-1, 1] to avoid numerical issues
    glint_cos = np.clip(glint_cos, -1, 1)

    glint_array = np.degrees(np.arccos(glint_cos))
    return glint_array

def compute_glint_vectorized(flight_line: FlightLine, sensor: LineScanner, observation_datetime: datetime, output_geometry: str = "geographic") -> gpd.GeoDataFrame:
    """
    Computes glint angles across a flight line and returns the results as a GeoDataFrame.

    Args:
        flight_line (FlightLine): FlightLine object defining the flight path.
        sensor (LineScanner): LineScanner object defining sensor characteristics.
        observation_datetime (datetime): The observation timestamp.

    Returns:
        GeoDataFrame: Results containing target locations and glint angles.
    """
    # Get track coordinates, altitude (MSL), and azimuth
    latitudes, longitudes, azimuths, along_track_distance = process_linestring(flight_line.track())
    altitude_msl = flight_line.altitude_msl.magnitude

    # Define tilt angles from -half_angle to +half_angle in 1-degree increments
    half_angle = sensor.half_angle  # Extract half angle from sensor
    tilt_angles = np.arange(-half_angle, half_angle + 1, 1)  # Shape: (T,)

    if len(tilt_angles) == 0:
        raise HyPlanValueError(f"Sensor half_angle {half_angle} produced an empty tilt angle array.")

    # Cross-track line scanner: the scan is perpendicular to the flight direction.
    # Positive tilt looks starboard (azimuth + 90°), negative tilt looks port (azimuth - 90°).
    # Build per-point azimuths that match the tilt sign.
    n_tilts = len(tilt_angles)
    view_azimuths = np.empty(len(azimuths) * n_tilts)
    for j, az in enumerate(azimuths):
        for k, t in enumerate(tilt_angles):
            view_azimuths[j * n_tilts + k] = (az + 90.0) % 360.0 if t >= 0 else (az - 90.0) % 360.0

    # Tile tilt angles for all azimuths and take absolute value since
    # the port/starboard direction is now encoded in view_azimuths
    tilt_angles = np.tile(np.abs(tilt_angles), len(latitudes))

    # Repeat latitudes, longitudes, and altitudes to match the number of angle combinations
    latitudes = np.repeat(latitudes, len(tilt_angles) // len(latitudes))
    longitudes = np.repeat(longitudes, len(tilt_angles) // len(longitudes))
    altitudes = np.full_like(latitudes, altitude_msl)
    along_track_distance = np.repeat(along_track_distance, len(tilt_angles) // len(along_track_distance))
    
    # Expand observation_datetime to match the shape of latitudes
    observation_datetimes = np.full(latitudes.shape, observation_datetime)

    # Call the vectorized glint calculation function
    target_lat, target_lon, glint_angles = calculate_target_and_glint_vectorized(
        sensor_lat=latitudes,
        sensor_lon=longitudes,
        sensor_alt=altitudes,
        viewing_azimuth=view_azimuths,
        tilt_angle=tilt_angles,
        observation_datetime=observation_datetimes
    )

    # Include tilt_angle and viewing_azimuth in the GeoDataFrame
    data = {
        "target_latitude": target_lat,
        "target_longitude": target_lon,
        "glint_angle": glint_angles,
        "tilt_angle": tilt_angles,
        "viewing_azimuth": view_azimuths,
        "along_track_distance": along_track_distance
    }

    if output_geometry == "geographic":
        geometry = [Point(lon, lat) for lon, lat in zip(target_lon, target_lat)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 CRS
    elif output_geometry == "along_track":
        geometry = [Point(tilt_angles, along_track_distance) for tilt_angles,along_track_distance  in zip(tilt_angles,along_track_distance)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=None)  # Assuming WGS84 CRS
    else:
        raise HyPlanValueError("Invalid output_geometry parameter. Must be 'geographic' or 'along_track'.")

    return gdf
