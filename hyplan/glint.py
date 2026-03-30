"""
Glint geometry and arc planning.

The geometry is derived from optics first, then kinematics:

1. **Specular reflection** defines the aircraft midpoint position
   (offset from target along the solar azimuth by ``alt * tan(SZA)``).
2. The **turn center** is placed on the opposite side of the aircraft
   from the target (toward the sun), guaranteeing the target is always
   outside the turn circle.
3. The aircraft flies a **180-degree banked turn** around that center.

At the arc midpoint the view zenith equals SZA and the view azimuth
opposes the solar azimuth, giving a glint angle of zero (perfect
specular reflection).
"""

import logging
import numpy as np
import geopandas as gpd

from datetime import datetime
from typing import Optional, Tuple

from shapely.geometry import Point, LineString
from shapely.ops import transform

import pymap3d
import pymap3d.vincenty
from pymap3d import los

from sunposition import sunpos

from .units import ureg
from .exceptions import HyPlanValueError, HyPlanTypeError
from .geometry import process_linestring, get_utm_transforms, wrap_to_360
from .sensors import LineScanner
from .flight_line import FlightLine
from .waypoint import Waypoint

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# GLINT ARC
# -----------------------------------------------------------------------------

class GlintArc:
    """Arc-shaped flight path that tilts the sensor to maximize specular glint.

    The aircraft position is derived from specular reflection geometry, then
    the turn center is placed on the **opposite side** of the aircraft from
    the target (toward the sun).  This guarantees the target is always outside
    the turn circle.

    At the arc midpoint the bank angle tilts the nadir-pointing sensor so
    that VZA = bank_angle and the view azimuth opposes the solar azimuth,
    producing a glint angle of zero (perfect specular reflection).

    Args:
        target_lat: Latitude of the target in decimal degrees.
        target_lon: Longitude of the target in decimal degrees.
        observation_datetime: UTC datetime for solar position computation.
        altitude_msl: Aircraft altitude above mean sea level (pint Quantity).
        speed: Aircraft speed (pint Quantity).
        bank_angle: Bank angle in degrees.  Defaults to SZA when SZA <= 60.
        site_name: Optional name for the target site.
        bank_direction: ``"left"`` or ``"right"`` (default ``"right"``).
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        observation_datetime: datetime,
        altitude_msl,
        speed,
        bank_angle: Optional[float] = None,
        site_name: Optional[str] = None,
        bank_direction: str = "right",
    ):
        if bank_direction not in ("left", "right"):
            raise HyPlanValueError("bank_direction must be 'left' or 'right'")
        if bank_angle is not None and not (0.0 < bank_angle < 90.0):
            raise HyPlanValueError("bank_angle must be between 0 and 90 degrees (exclusive).")

        self.target_lat = target_lat
        self.target_lon = target_lon
        self.observation_datetime = observation_datetime
        self.site_name = site_name
        self.bank_direction = bank_direction
        self._bank_angle_override = bank_angle

        self.altitude_msl = altitude_msl.to(ureg.meter)
        self.speed = speed
        self._speed_mps = speed.to(ureg.meter / ureg.second).magnitude

        self._compute_arc()

    # -------------------------------------------------------------------------
    # CORE GEOMETRY
    # -------------------------------------------------------------------------

    def _compute_arc(self):

        # --- Solar geometry ---
        solar_az, solar_zen, *_ = sunpos(
            dt=self.observation_datetime,
            latitude=self.target_lat,
            longitude=self.target_lon,
            elevation=0.0,
            radians=False,
        )

        self.solar_azimuth = float(solar_az)
        self.solar_zenith = float(solar_zen)

        if self.solar_zenith < 5.0:
            raise HyPlanValueError(
                f"Solar zenith angle is {self.solar_zenith:.1f}° (sun near zenith). "
                "Bank angle would be too small for a stable glint arc."
            )

        # --- Bank angle ---
        if self._bank_angle_override is None:
            if self.solar_zenith > 60.0:
                raise HyPlanValueError(
                    f"Solar zenith angle is {self.solar_zenith:.1f}° > 60°. "
                    "Specify bank_angle explicitly to override."
                )
            self.bank_angle = self.solar_zenith
        else:
            self.bank_angle = self._bank_angle_override

        self.arc_extent = 180.0

        # --- Turn radius ---
        g = 9.80665
        bank_rad = np.radians(self.bank_angle)
        self._turn_radius_m = self._speed_mps**2 / (g * np.tan(bank_rad))

        # --- Aircraft midpoint (specular geometry) ---
        altitude_m = self.altitude_msl.magnitude
        ground_distance = altitude_m * np.tan(np.radians(self.solar_zenith))

        A_lat, A_lon = pymap3d.vincenty.vreckon(
            self.target_lat,
            self.target_lon,
            ground_distance,
            self.solar_azimuth,
        )

        self._aircraft_mid_lat = float(A_lat)
        self._aircraft_mid_lon = float(A_lon)

        # --- Heading (informational) ---
        if self.bank_direction == "right":
            self.heading_at_midpoint = wrap_to_360(self.solar_azimuth + 90)
        else:
            self.heading_at_midpoint = wrap_to_360(self.solar_azimuth - 90)

        # --- Turn center ---
        # Center is on the opposite side of the aircraft from the target
        # (toward the sun).  This guarantees the target is always outside
        # the turn circle: dist(target, center) = ground_distance + R > R.
        C_lat, C_lon = pymap3d.vincenty.vreckon(
            A_lat,
            A_lon,
            self._turn_radius_m,
            self.solar_azimuth,
        )

        # --- Build arc in UTM ---
        A = Point(A_lon, A_lat)
        C = Point(C_lon, C_lat)

        to_utm, from_utm = get_utm_transforms([A, C])

        A_utm = transform(to_utm, A)
        C_utm = transform(to_utm, C)

        dx = A_utm.x - C_utm.x
        dy = A_utm.y - C_utm.y

        theta_mid = np.arctan2(dy, dx)

        half_ext = np.radians(self.arc_extent / 2)
        n_points = max(int(self.arc_extent * 2), 60)

        if self.bank_direction == "right":
            angles = np.linspace(theta_mid - half_ext, theta_mid + half_ext, n_points)
        else:
            angles = np.linspace(theta_mid + half_ext, theta_mid - half_ext, n_points)

        xs = C_utm.x + self._turn_radius_m * np.cos(angles)
        ys = C_utm.y + self._turn_radius_m * np.sin(angles)

        lons, lats = from_utm(xs, ys)
        self.geometry = LineString(np.column_stack([lons, lats]))

    # -------------------------------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------------------------------

    @property
    def turn_radius(self):
        return ureg.Quantity(self._turn_radius_m, "meter")

    @property
    def length(self):
        return ureg.Quantity(
            self._turn_radius_m * np.radians(self.arc_extent), "meter"
        )

    @property
    def duration(self):
        return (self.length / self.speed).to(ureg.second)

    @property
    def waypoint1(self):
        coords = self.geometry.coords
        lat, lon = coords[0][1], coords[0][0]
        lat2, lon2 = coords[1][1], coords[1][0]
        _, az = pymap3d.vincenty.vdist(lat, lon, lat2, lon2)

        return Waypoint(
            latitude=lat,
            longitude=lon,
            heading=float(az),
            altitude_msl=self.altitude_msl,
            name="arc_start",
        )

    @property
    def waypoint2(self):
        coords = self.geometry.coords
        lat, lon = coords[-1][1], coords[-1][0]
        lat2, lon2 = coords[-2][1], coords[-2][0]
        _, az = pymap3d.vincenty.vdist(lat2, lon2, lat, lon)

        return Waypoint(
            latitude=lat,
            longitude=lon,
            heading=float(az),
            altitude_msl=self.altitude_msl,
            name="arc_end",
        )

    def track(self, precision=100.0) -> LineString:
        """Return the arc as a densified LineString.

        Args:
            precision: Desired distance between interpolated points.
                Accepts a Quantity with length units or a plain float (assumed meters).
        """
        if hasattr(precision, "to"):
            precision_m = precision.to("meter").magnitude
        else:
            precision_m = float(precision)

        arc_length_m = self.length.magnitude
        n_points = max(int(np.ceil(arc_length_m / precision_m)) + 1, 3)

        if n_points <= len(self.geometry.coords):
            return self.geometry

        # Rebuild arc with finer sampling using stored circle center
        A = Point(self._aircraft_mid_lon, self._aircraft_mid_lat)
        C_lat, C_lon = pymap3d.vincenty.vreckon(
            self._aircraft_mid_lat, self._aircraft_mid_lon,
            self._turn_radius_m, self.solar_azimuth,
        )
        C = Point(float(C_lon), float(C_lat))

        to_utm, from_utm = get_utm_transforms([A, C])
        A_utm = transform(to_utm, A)
        C_utm = transform(to_utm, C)

        dx = A_utm.x - C_utm.x
        dy = A_utm.y - C_utm.y
        theta_mid = np.arctan2(dy, dx)
        half_ext = np.radians(self.arc_extent / 2.0)

        if self.bank_direction == "right":
            angles = np.linspace(theta_mid - half_ext, theta_mid + half_ext, n_points)
        else:
            angles = np.linspace(theta_mid + half_ext, theta_mid - half_ext, n_points)

        xs = C_utm.x + self._turn_radius_m * np.cos(angles)
        ys = C_utm.y + self._turn_radius_m * np.sin(angles)
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
            "bank_angle_auto": self._bank_angle_override is None,
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
                "bank_angle_auto": self._bank_angle_override is None,
                "bank_direction": self.bank_direction,
                "heading_at_midpoint": self.heading_at_midpoint,
                "turn_radius": self._turn_radius_m,
                "arc_extent": self.arc_extent,
                "site_name": self.site_name,
            },
        }


# -----------------------------------------------------------------------------
# GLINT CALCULATIONS
# -----------------------------------------------------------------------------

def glint_angle(solar_azimuth, solar_zenith, view_azimuth, view_zenith):
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)
    view_zenith_rad = np.deg2rad(view_zenith)
    view_azimuth_rad = np.deg2rad(view_azimuth)

    phi = solar_azimuth_rad - view_azimuth_rad

    glint_cos = (
        np.cos(view_zenith_rad) * np.cos(solar_zenith_rad)
        - np.sin(view_zenith_rad) * np.sin(solar_zenith_rad) * np.cos(phi)
    )

    glint_cos = np.clip(glint_cos, -1, 1)
    return np.degrees(np.arccos(glint_cos))


def calculate_target_and_glint_vectorized(
    sensor_lat,
    sensor_lon,
    sensor_alt,
    viewing_azimuth,
    tilt_angle,
    observation_datetime,
):
    target_lat, target_lon, _ = los.lookAtSpheroid(
        sensor_lat,
        sensor_lon,
        sensor_alt,
        viewing_azimuth,
        tilt_angle,
    )

    solar_azimuth, solar_zenith, *_ = sunpos(
        dt=observation_datetime,
        latitude=sensor_lat,
        longitude=sensor_lon,
        elevation=sensor_alt,
        radians=False,
    )

    glint_angles = glint_angle(
        solar_azimuth,
        solar_zenith,
        viewing_azimuth,
        np.abs(tilt_angle),
    )

    return target_lat, target_lon, glint_angles


def compute_glint_vectorized(
    flight_line: FlightLine,
    sensor: LineScanner,
    observation_datetime: datetime,
    output_geometry: str = "geographic",
) -> gpd.GeoDataFrame:
    """Compute glint angles across a flight line's sensor swath.

    Args:
        flight_line: FlightLine object defining the flight path.
        sensor: LineScanner object defining sensor characteristics.
        observation_datetime: The observation timestamp (UTC).
        output_geometry: "geographic" for target lat/lon Points, or
            "along_track" for (tilt, along_track_distance) Points.

    Returns:
        GeoDataFrame with columns: target_latitude, target_longitude,
        glint_angle, tilt_angle, viewing_azimuth, along_track_distance.
    """
    latitudes, longitudes, azimuths, along_track_distance = process_linestring(
        flight_line.track()
    )
    altitude_msl = flight_line.altitude_msl.magnitude

    half_angle = sensor.half_angle
    tilt_angles = np.arange(-half_angle, half_angle + 1, 1)
    n_tilts = len(tilt_angles)

    view_azimuths = np.empty(len(azimuths) * n_tilts)
    for j, az in enumerate(azimuths):
        for k, t in enumerate(tilt_angles):
            view_azimuths[j * n_tilts + k] = (
                (az + 90.0) % 360.0 if t >= 0 else (az - 90.0) % 360.0
            )

    tilt_angles_tiled = np.tile(np.abs(tilt_angles), len(latitudes))

    latitudes = np.repeat(latitudes, n_tilts)
    longitudes = np.repeat(longitudes, n_tilts)
    altitudes = np.full_like(latitudes, altitude_msl)
    along_track_distance = np.repeat(along_track_distance, n_tilts)

    observation_datetimes = np.full(latitudes.shape, observation_datetime)

    target_lat, target_lon, glint_angles = calculate_target_and_glint_vectorized(
        sensor_lat=latitudes,
        sensor_lon=longitudes,
        sensor_alt=altitudes,
        viewing_azimuth=view_azimuths,
        tilt_angle=tilt_angles_tiled,
        observation_datetime=observation_datetimes,
    )

    data = {
        "target_latitude": target_lat,
        "target_longitude": target_lon,
        "glint_angle": glint_angles,
        "tilt_angle": tilt_angles_tiled,
        "viewing_azimuth": view_azimuths,
        "along_track_distance": along_track_distance,
    }

    if output_geometry == "geographic":
        geometry = [Point(lon, lat) for lon, lat in zip(target_lon, target_lat)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
    elif output_geometry == "along_track":
        geometry = [
            Point(t, d) for t, d in zip(tilt_angles_tiled, along_track_distance)
        ]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=None)
    else:
        raise HyPlanValueError(
            "Invalid output_geometry parameter. Must be 'geographic' or 'along_track'."
        )

    return gdf


def compute_glint_arc(
    glint_arc: GlintArc,
    sensor: LineScanner,
    output_geometry: str = "geographic",
) -> gpd.GeoDataFrame:
    """Compute glint angles across a glint arc's sensor swath.

    Args:
        glint_arc: GlintArc object defining the arc flight path.
        sensor: LineScanner object defining sensor characteristics.
        output_geometry: "geographic" for target lat/lon Points, or
            "along_track" for (tilt, along_track_distance) Points.

    Returns:
        GeoDataFrame with columns: target_latitude, target_longitude,
        glint_angle, tilt_angle, view_zenith, viewing_azimuth,
        along_track_distance.
    """
    latitudes, longitudes, azimuths, along_track_distance = process_linestring(
        glint_arc.track()
    )

    altitude_msl = glint_arc.altitude_msl.magnitude
    bank_angle = glint_arc.bank_angle

    half_angle = sensor.half_angle
    sensor_tilts = np.arange(-half_angle, half_angle + 1, 1)

    n_track = len(latitudes)
    n_tilts = len(sensor_tilts)

    earth_vza = bank_angle + sensor_tilts

    view_zeniths = np.tile(earth_vza, n_track)
    sensor_tilt_tiled = np.tile(sensor_tilts, n_track)

    view_azimuths = np.empty(n_track * n_tilts)
    for j, az in enumerate(azimuths):
        for k in range(n_tilts):
            idx = j * n_tilts + k
            vz = view_zeniths[idx]

            if glint_arc.bank_direction == "right":
                view_azimuths[idx] = (az + 90.0) % 360.0 if vz >= 0 else (az - 90.0) % 360.0
            else:
                view_azimuths[idx] = (az - 90.0) % 360.0 if vz >= 0 else (az + 90.0) % 360.0

    latitudes = np.repeat(latitudes, n_tilts)
    longitudes = np.repeat(longitudes, n_tilts)
    altitudes = np.full_like(latitudes, altitude_msl)
    atd_tiled = np.repeat(along_track_distance, n_tilts)

    observation_datetimes = np.full(latitudes.shape, glint_arc.observation_datetime)

    target_lat, target_lon, glint_angles = calculate_target_and_glint_vectorized(
        sensor_lat=latitudes,
        sensor_lon=longitudes,
        sensor_alt=altitudes,
        viewing_azimuth=view_azimuths,
        tilt_angle=np.abs(view_zeniths),
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

