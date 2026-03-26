from datetime import datetime
from typing import Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pint import Quantity
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pymap3d import los
import pymap3d.vincenty
from sunposition import sunpos

from .flight_line import FlightLine
from .sensors import LineScanner
from .waypoint import Waypoint
from .geometry import process_linestring, get_utm_transforms, wrap_to_360
from .units import ureg
from .exceptions import HyPlanValueError, HyPlanTypeError

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_target_and_glint_vectorized",
    "glint_angle",
    "compute_glint_vectorized",
    "GlintArc",
    "compute_glint_arc",
]


class GlintArc:
    """Arc-shaped flight path that tilts the sensor to maximize specular glint on a target.

    The aircraft flies a circular arc so that the bank angle tilts a nadir-pointing
    sensor to match the solar zenith angle (SZA). At the arc midpoint, the sensor
    view zenith equals SZA and the view azimuth is opposite the solar azimuth,
    achieving a glint angle of zero (perfect specular reflection).

    This is designed for offshore targets (oil/gas platforms) over water where
    terrain effects are negligible.

    Args:
        target_lat: Latitude of the target in decimal degrees.
        target_lon: Longitude of the target in decimal degrees.
        observation_datetime: UTC datetime for solar position computation.
        altitude_msl: Aircraft altitude above mean sea level.
        speed: Aircraft speed.
        arc_extent: Total angular extent of the arc in degrees (default 30).
        site_name: Optional name for the target site.
        bank_direction: Bank direction, "left" or "right" (default "right").
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        observation_datetime: datetime,
        altitude_msl: Quantity,
        speed: Quantity,
        arc_extent: float = 30.0,
        site_name: Optional[str] = None,
        bank_direction: str = "right",
    ):
        if not (-90.0 <= target_lat <= 90.0):
            raise HyPlanValueError("target_lat must be between -90 and 90 degrees.")
        if not (-180.0 <= target_lon <= 180.0):
            raise HyPlanValueError("target_lon must be between -180 and 180 degrees.")
        if bank_direction not in ("left", "right"):
            raise HyPlanValueError("bank_direction must be 'left' or 'right'.")
        if arc_extent <= 0:
            raise HyPlanValueError("arc_extent must be positive.")

        self.target_lat = target_lat
        self.target_lon = target_lon
        self.observation_datetime = observation_datetime
        self.site_name = site_name
        self.bank_direction = bank_direction
        self.arc_extent = float(arc_extent)

        # Validate and store altitude
        if not isinstance(altitude_msl, Quantity):
            altitude_msl = ureg.Quantity(altitude_msl, "meter")
        else:
            altitude_msl = altitude_msl.to("meter")
        self.altitude_msl = altitude_msl

        # Validate and store speed
        if not isinstance(speed, Quantity):
            raise HyPlanTypeError("speed must be a pint Quantity with speed units.")
        self.speed = speed
        self._speed_mps = speed.to(ureg.meter / ureg.second).magnitude

        self._compute_arc()

    def _compute_arc(self):
        """Compute solar geometry, turn parameters, and arc geometry."""
        # Step 1: Solar position at target
        solar_az, solar_zen, _, _, _ = sunpos(
            dt=self.observation_datetime,
            latitude=self.target_lat,
            longitude=self.target_lon,
            elevation=0.0,
            radians=False,
        )
        self.solar_azimuth = float(solar_az)
        self.solar_zenith = float(solar_zen)

        # Validate solar zenith
        if self.solar_zenith < 5.0:
            raise HyPlanValueError(
                f"Solar zenith angle is {self.solar_zenith:.1f}° (sun near zenith). "
                "Bank angle would be too small and turn radius too large for a glint arc."
            )
        if self.solar_zenith > 75.0:
            logger.warning(
                "Solar zenith angle is %.1f° — bank angle will be steep and "
                "turn radius very tight.", self.solar_zenith
            )

        # Step 2: Bank angle = SZA
        self.bank_angle = self.solar_zenith

        # Step 3: Turn radius
        g = 9.80665  # m/s^2
        bank_rad = np.radians(self.bank_angle)
        self._turn_radius_m = self._speed_mps ** 2 / (g * np.tan(bank_rad))

        # Step 4: Heading at midpoint
        if self.bank_direction == "right":
            self.heading_at_midpoint = float(wrap_to_360(self.solar_azimuth + 90.0))
        else:
            self.heading_at_midpoint = float(wrap_to_360(self.solar_azimuth + 270.0))

        # Step 5: Aircraft position at arc midpoint
        # Sensor looks perpendicular to heading at bank_angle from nadir.
        # Target is on the ground; aircraft is offset from target toward the sun
        # by ground_distance = altitude * tan(bank_angle).
        altitude_m = self.altitude_msl.magnitude
        ground_distance = altitude_m * np.tan(bank_rad)
        aircraft_mid_lat, aircraft_mid_lon = pymap3d.vincenty.vreckon(
            self.target_lat, self.target_lon, ground_distance, self.solar_azimuth
        )
        self._aircraft_mid_lat = float(aircraft_mid_lat)
        self._aircraft_mid_lon = float(aircraft_mid_lon)

        # Step 6: Circle center from aircraft midpoint
        if self.bank_direction == "right":
            center_azimuth = float(wrap_to_360(self.heading_at_midpoint + 90.0))
        else:
            center_azimuth = float(wrap_to_360(self.heading_at_midpoint - 90.0))

        center_lat, center_lon = pymap3d.vincenty.vreckon(
            self._aircraft_mid_lat, self._aircraft_mid_lon,
            self._turn_radius_m, center_azimuth,
        )

        # Step 7: Build arc geometry in UTM
        mid_point = Point(self._aircraft_mid_lon, self._aircraft_mid_lat)
        center_point = Point(float(center_lon), float(center_lat))
        to_utm, from_utm = get_utm_transforms([mid_point, center_point])

        mid_utm = transform(to_utm, mid_point)
        center_utm = transform(to_utm, center_point)

        # Angle of aircraft midpoint relative to circle center
        dx = mid_utm.x - center_utm.x
        dy = mid_utm.y - center_utm.y
        mid_angle = np.arctan2(dy, dx)

        # Generate arc points
        half_ext = np.radians(self.arc_extent / 2.0)
        n_points = max(int(self.arc_extent * 2), 30)

        # Right bank: aircraft moves clockwise (decreasing angle in math coords)
        # Left bank: aircraft moves counterclockwise (increasing angle)
        if self.bank_direction == "right":
            angles = np.linspace(mid_angle + half_ext, mid_angle - half_ext, n_points)
        else:
            angles = np.linspace(mid_angle - half_ext, mid_angle + half_ext, n_points)

        xs = center_utm.x + self._turn_radius_m * np.cos(angles)
        ys = center_utm.y + self._turn_radius_m * np.sin(angles)

        lons, lats = from_utm(xs, ys)
        self.geometry = LineString(np.column_stack([lons, lats]))

    @property
    def turn_radius(self) -> Quantity:
        """Turn radius of the arc in meters."""
        return ureg.Quantity(self._turn_radius_m, "meter")

    @property
    def length(self) -> Quantity:
        """Arc length in meters."""
        arc_length = self._turn_radius_m * np.radians(self.arc_extent)
        return ureg.Quantity(round(arc_length, 2), "meter")

    @property
    def duration(self) -> Quantity:
        """Time to fly the arc."""
        return (self.length / self.speed).to(ureg.second)

    @property
    def waypoint1(self) -> Waypoint:
        """Start point of the arc as a Waypoint."""
        coords = self.geometry.coords
        lat, lon = coords[0][1], coords[0][0]
        # Compute heading tangent to arc at start
        lat2, lon2 = coords[1][1], coords[1][0]
        _, az = pymap3d.vincenty.vdist(lat, lon, lat2, lon2)
        name = f"{self.site_name}_start" if self.site_name else "arc_start"
        return Waypoint(latitude=lat, longitude=lon, heading=float(az),
                        altitude_msl=self.altitude_msl, name=name)

    @property
    def waypoint2(self) -> Waypoint:
        """End point of the arc as a Waypoint."""
        coords = self.geometry.coords
        lat, lon = coords[-1][1], coords[-1][0]
        lat2, lon2 = coords[-2][1], coords[-2][0]
        _, az = pymap3d.vincenty.vdist(lat2, lon2, lat, lon)
        name = f"{self.site_name}_end" if self.site_name else "arc_end"
        return Waypoint(latitude=lat, longitude=lon, heading=float(az),
                        altitude_msl=self.altitude_msl, name=name)

    def track(self, precision: Union[Quantity, float] = 100.0) -> LineString:
        """Return the arc as a densified LineString.

        Args:
            precision: Desired distance between interpolated points.
                Accepts a Quantity with length units or a plain float (assumed meters).

        Returns:
            LineString: The arc track.
        """
        if isinstance(precision, Quantity):
            precision_m = precision.to("meter").magnitude
        else:
            precision_m = float(precision)

        arc_length_m = self.length.magnitude
        n_points = max(int(np.ceil(arc_length_m / precision_m)) + 1, 3)

        # Regenerate arc at requested resolution if needed
        if n_points <= len(self.geometry.coords):
            return self.geometry

        # Rebuild arc with finer sampling
        mid_point = Point(self._aircraft_mid_lon, self._aircraft_mid_lat)
        to_utm, from_utm = get_utm_transforms(mid_point)
        mid_utm = transform(to_utm, mid_point)

        # Recompute circle center in UTM
        if self.bank_direction == "right":
            center_azimuth = float(wrap_to_360(self.heading_at_midpoint + 90.0))
        else:
            center_azimuth = float(wrap_to_360(self.heading_at_midpoint - 90.0))

        center_lat, center_lon = pymap3d.vincenty.vreckon(
            self._aircraft_mid_lat, self._aircraft_mid_lon,
            self._turn_radius_m, center_azimuth,
        )
        center_utm = transform(to_utm, Point(float(center_lon), float(center_lat)))

        dx = mid_utm.x - center_utm.x
        dy = mid_utm.y - center_utm.y
        mid_angle = np.arctan2(dy, dx)
        half_ext = np.radians(self.arc_extent / 2.0)

        if self.bank_direction == "right":
            angles = np.linspace(mid_angle + half_ext, mid_angle - half_ext, n_points)
        else:
            angles = np.linspace(mid_angle - half_ext, mid_angle + half_ext, n_points)

        xs = center_utm.x + self._turn_radius_m * np.cos(angles)
        ys = center_utm.y + self._turn_radius_m * np.sin(angles)
        lons, lats = from_utm(xs, ys)
        return LineString(np.column_stack([lons, lats]))

    def to_dict(self) -> dict:
        """Convert the glint arc to a dictionary representation."""
        return {
            "geometry": list(self.geometry.coords),
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "aircraft_mid_lat": self._aircraft_mid_lat,
            "aircraft_mid_lon": self._aircraft_mid_lon,
            "altitude_msl": self.altitude_msl.magnitude,
            "speed_mps": self._speed_mps,
            "solar_azimuth": self.solar_azimuth,
            "solar_zenith": self.solar_zenith,
            "bank_angle": self.bank_angle,
            "bank_direction": self.bank_direction,
            "heading_at_midpoint": self.heading_at_midpoint,
            "turn_radius": self._turn_radius_m,
            "arc_extent": self.arc_extent,
            "arc_length": self.length.magnitude,
            "site_name": self.site_name,
        }

    def to_geojson(self) -> dict:
        """Convert the glint arc to a GeoJSON Feature dictionary."""
        coords = list(self.geometry.coords)
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "target_lat": self.target_lat,
                "target_lon": self.target_lon,
                "altitude_msl": self.altitude_msl.magnitude,
                "solar_azimuth": self.solar_azimuth,
                "solar_zenith": self.solar_zenith,
                "bank_angle": self.bank_angle,
                "bank_direction": self.bank_direction,
                "heading_at_midpoint": self.heading_at_midpoint,
                "turn_radius": self._turn_radius_m,
                "arc_extent": self.arc_extent,
                "site_name": self.site_name,
            },
        }


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


def compute_glint_arc(
    glint_arc: GlintArc,
    sensor: LineScanner,
    output_geometry: str = "geographic",
) -> gpd.GeoDataFrame:
    """Compute glint angles across a glint arc's sensor swath.

    Analogous to :func:`compute_glint_vectorized` but for arc-shaped flight
    paths where the aircraft is banked. The sensor FOV sweeps cross-track
    relative to the banked aircraft, so the effective Earth-frame view zenith
    is ``bank_angle + sensor_tilt``.

    Args:
        glint_arc: GlintArc object defining the arc flight path.
        sensor: LineScanner object defining sensor characteristics.
        output_geometry: "geographic" for target lat/lon Points, or
            "along_track" for (tilt, along_track_distance) Points.

    Returns:
        GeoDataFrame with columns: target_latitude, target_longitude,
        glint_angle, tilt_angle (sensor frame), view_zenith (Earth frame),
        viewing_azimuth, along_track_distance.
    """
    # Get track coordinates and azimuths along the arc
    latitudes, longitudes, azimuths, along_track_distance = process_linestring(
        glint_arc.track()
    )
    altitude_msl = glint_arc.altitude_msl.magnitude
    bank_angle = glint_arc.bank_angle

    # Sensor tilt angles in the sensor frame (cross-track sweep)
    half_angle = sensor.half_angle
    sensor_tilts = np.arange(-half_angle, half_angle + 1, 1)

    if len(sensor_tilts) == 0:
        raise HyPlanValueError(
            f"Sensor half_angle {half_angle} produced an empty tilt angle array."
        )

    n_track = len(latitudes)
    n_tilts = len(sensor_tilts)

    # Compute Earth-frame view zenith for each sensor tilt.
    # In the banked frame, tilt=0 corresponds to the bank direction (VZA = bank_angle).
    # Positive tilt looks further from nadir (toward bank), negative toward nadir.
    # Earth-frame VZA = bank_angle + sensor_tilt
    earth_vza = bank_angle + sensor_tilts  # shape (n_tilts,)

    # Build view azimuths and absolute VZA arrays for all (track_point, tilt) pairs
    view_zeniths = np.tile(earth_vza, n_track)
    sensor_tilt_tiled = np.tile(sensor_tilts, n_track)

    # View azimuth depends on bank direction and whether VZA crossed nadir
    view_azimuths = np.empty(n_track * n_tilts)
    for j, az in enumerate(azimuths):
        for k in range(n_tilts):
            idx = j * n_tilts + k
            vz = view_zeniths[idx]
            if glint_arc.bank_direction == "right":
                if vz >= 0:
                    view_azimuths[idx] = (az + 90.0) % 360.0
                else:
                    view_azimuths[idx] = (az - 90.0) % 360.0
            else:  # left
                if vz >= 0:
                    view_azimuths[idx] = (az - 90.0) % 360.0
                else:
                    view_azimuths[idx] = (az + 90.0) % 360.0

    # Take absolute value of VZA (direction now encoded in azimuth)
    abs_view_zeniths = np.abs(view_zeniths)

    # Repeat track coordinates to match
    lats_tiled = np.repeat(latitudes, n_tilts)
    lons_tiled = np.repeat(longitudes, n_tilts)
    alts_tiled = np.full_like(lats_tiled, altitude_msl)
    atd_tiled = np.repeat(along_track_distance, n_tilts)

    observation_datetimes = np.full(lats_tiled.shape, glint_arc.observation_datetime)

    # Compute target locations and glint angles
    target_lat, target_lon, glint_angles = calculate_target_and_glint_vectorized(
        sensor_lat=lats_tiled,
        sensor_lon=lons_tiled,
        sensor_alt=alts_tiled,
        viewing_azimuth=view_azimuths,
        tilt_angle=abs_view_zeniths,
        observation_datetime=observation_datetimes,
    )

    data = {
        "target_latitude": target_lat,
        "target_longitude": target_lon,
        "glint_angle": glint_angles,
        "tilt_angle": sensor_tilt_tiled,
        "view_zenith": view_zeniths,
        "viewing_azimuth": view_azimuths,
        "along_track_distance": atd_tiled,
    }

    if output_geometry == "geographic":
        geometry = [Point(lon, lat) for lon, lat in zip(target_lon, target_lat)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
    elif output_geometry == "along_track":
        geometry = [
            Point(t, d) for t, d in zip(sensor_tilt_tiled, atd_tiled)
        ]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=None)
    else:
        raise HyPlanValueError(
            "Invalid output_geometry parameter. Must be 'geographic' or 'along_track'."
        )

    return gdf
