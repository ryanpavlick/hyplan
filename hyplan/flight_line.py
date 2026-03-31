"""Flight line definition and manipulation.

A :class:`FlightLine` represents a single straight-and-level data-collection
segment defined by two geographic endpoints, an altitude, and metadata. The
class provides factory constructors for common creation patterns
(:meth:`~FlightLine.start_length_azimuth`,
:meth:`~FlightLine.center_length_azimuth`) and methods for geometric
operations such as splitting, clipping, rotating, and offsetting.

Waypoint properties at each endpoint (heading, altitude, position) are
derived automatically from the geodesic geometry via Vincenty's formulae.
"""

from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from pint import Quantity
from typing import Optional, List, Dict, Union, Tuple
import pymap3d
import pymap3d.vincenty
import geopandas as gpd
import numpy as np
import logging

from .units import ureg
from .geometry import wrap_to_180
from .waypoint import Waypoint, is_waypoint
from .exceptions import HyPlanTypeError, HyPlanValueError

logger = logging.getLogger(__name__)

__all__ = [
    "FlightLine",
    "to_gdf",
]


class FlightLine:
    """
    Represents a geospatial flight line with properties, validations, and operations.

    A FlightLine is defined by two Waypoint objects (start and end).  The
    Shapely LineString geometry and geodesic properties (length, azimuths)
    are derived from those waypoints.

    Altitude is stored as MSL (above mean sea level), which is the standard
    aviation reference. Sensor calculations that depend on height above ground
    (AGL) must account for terrain elevation separately.
    """
    def __init__(
        self,
        waypoint1: Waypoint,
        waypoint2: Waypoint,
        site_name: Optional[str] = None,
        site_description: Optional[str] = None,
        investigator: Optional[str] = None,
    ):
        if not is_waypoint(waypoint1) or not is_waypoint(waypoint2):
            raise HyPlanTypeError("waypoint1 and waypoint2 must be Waypoint objects.")

        self._waypoint1 = waypoint1
        self._waypoint2 = waypoint2
        self.site_name = site_name
        self.site_description = site_description
        self.investigator = investigator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_geometry(self, geometry: LineString, site_name: Optional[str] = None) -> "FlightLine":
        """Create a new FlightLine from a LineString, inheriting metadata.

        Altitude, speed, and segment_type propagate from the source.
        Delay is reset to None (loiter is specific to a geographic point).
        """
        _validate_linestring(geometry)
        coords = list(geometry.coords)
        lon1, lat1 = coords[0]
        lon2, lat2 = coords[-1]

        _, az12 = pymap3d.vincenty.vdist(lat1, lon1, lat2, lon2)
        _, az21 = pymap3d.vincenty.vdist(lat2, lon2, lat1, lon1)

        wp1 = Waypoint(
            latitude=lat1, longitude=lon1,
            heading=float(az12),
            altitude_msl=self.altitude_msl,
            name=f"{site_name}_start" if site_name else f"{self.site_name}_start" if self.site_name else "start",
            speed=self._waypoint1.speed,
            segment_type=self._waypoint1.segment_type,
        )
        wp2 = Waypoint(
            latitude=lat2, longitude=lon2,
            heading=(float(az21) + 180.0) % 360.0,
            altitude_msl=self.altitude_msl,
            name=f"{site_name}_end" if site_name else f"{self.site_name}_end" if self.site_name else "end",
            speed=self._waypoint1.speed,
            segment_type=self._waypoint1.segment_type,
        )
        return FlightLine(
            waypoint1=wp1, waypoint2=wp2,
            site_name=site_name or self.site_name,
            site_description=self.site_description,
            investigator=self.investigator,
        )

    # ------------------------------------------------------------------
    # Properties — same external API as before
    # ------------------------------------------------------------------

    @property
    def geometry(self) -> LineString:
        """Shapely LineString derived from the two waypoints."""
        return LineString([
            (self._waypoint1.longitude, self._waypoint1.latitude),
            (self._waypoint2.longitude, self._waypoint2.latitude),
        ])

    @property
    def altitude_msl(self) -> Quantity:
        """Flight altitude MSL (from waypoint1)."""
        return self._waypoint1.altitude_msl

    @altitude_msl.setter
    def altitude_msl(self, value: Quantity):
        """Set altitude on both waypoints."""
        validated = self._validate_altitude(value)
        self._waypoint1.altitude_msl = validated
        self._waypoint2.altitude_msl = validated

    @property
    def lat1(self) -> float:
        """Latitude of the start point in decimal degrees."""
        return self._waypoint1.latitude

    @property
    def lon1(self) -> float:
        """Longitude of the start point in decimal degrees."""
        return self._waypoint1.longitude

    @property
    def lat2(self) -> float:
        """Latitude of the end point in decimal degrees."""
        return self._waypoint2.latitude

    @property
    def lon2(self) -> float:
        """Longitude of the end point in decimal degrees."""
        return self._waypoint2.longitude

    @property
    def length(self) -> Quantity:
        """Geodesic length of the flight line (Vincenty formula) in meters."""
        length, _ = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(round(length, 2), "meter")

    @property
    def az12(self) -> Quantity:
        """Forward azimuth from start to end point in degrees."""
        _, az12 = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(az12, "degree")

    @property
    def az21(self) -> Quantity:
        """Forward azimuth from end to start point in degrees."""
        _, az21 = pymap3d.vincenty.vdist(self.lat2, self.lon2, self.lat1, self.lon1)
        return ureg.Quantity(az21, "degree")

    @property
    def waypoint1(self) -> Waypoint:
        """Start point Waypoint."""
        return self._waypoint1

    @property
    def waypoint2(self) -> Waypoint:
        """End point Waypoint."""
        return self._waypoint2

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_altitude(altitude: Quantity) -> Quantity:
        if not isinstance(altitude, Quantity):
            altitude = ureg.Quantity(altitude, "meter")
        else:
            altitude = altitude.to("meter")

        if altitude.magnitude < 0:
            raise HyPlanValueError(
                f"Altitude must be non-negative, got {altitude.magnitude} meters"
            )
        if altitude.magnitude > 22000:
            logger.warning(
                f"Altitude {altitude.magnitude} meters is above 22,000 m. "
                "Verify this is intended (ER-2/WB-57 range)."
            )
        return altitude

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def start_length_azimuth(
        cls,
        lat1: float,
        lon1: float,
        length: Quantity,
        az: float,
        altitude_msl: Quantity = None,
        site_name: Optional[str] = None,
        site_description: Optional[str] = None,
        investigator: Optional[str] = None,
        **kwargs,
    ) -> "FlightLine":
        """Create a flight line from a start point, length, and azimuth.

        Args:
            lat1: Start latitude in decimal degrees.
            lon1: Start longitude in decimal degrees.
            length: Line length as a Quantity with distance units.
            az: Forward azimuth in degrees from true north.
            altitude_msl: Flight altitude MSL.
            site_name: Optional site name.
            site_description: Optional site description.
            investigator: Optional investigator name.

        Returns:
            A new FlightLine extending from (lat1, lon1) along the given azimuth.
        """
        if not isinstance(length, Quantity) or not length.check("[length]"):
            raise HyPlanValueError("Length must be a Quantity with units of distance.")
        if not isinstance(az, (int, float)):
            raise HyPlanValueError(f"Azimuth must be a numeric value in degrees. Got {type(az)}.")

        length_m = length.to("meter").magnitude
        lat2, lon2 = pymap3d.vincenty.vreckon(lat1, lon1, length_m, az)
        lon2 = wrap_to_180(lon2)

        _, az21 = pymap3d.vincenty.vdist(lat2, lon2, lat1, lon1)

        alt = cls._validate_altitude(altitude_msl)
        wp1 = Waypoint(latitude=lat1, longitude=lon1, heading=float(az) % 360,
                       altitude_msl=alt,
                       name=f"{site_name}_start" if site_name else "start")
        wp2 = Waypoint(latitude=float(lat2), longitude=float(lon2),
                       heading=(float(az21) + 180.0) % 360.0,
                       altitude_msl=alt,
                       name=f"{site_name}_end" if site_name else "end")

        return cls(waypoint1=wp1, waypoint2=wp2,
                   site_name=site_name, site_description=site_description,
                   investigator=investigator)

    @classmethod
    def center_length_azimuth(
        cls,
        lat: float,
        lon: float,
        length: Quantity,
        az: float,
        altitude_msl: Quantity = None,
        site_name: Optional[str] = None,
        site_description: Optional[str] = None,
        investigator: Optional[str] = None,
        **kwargs,
    ) -> "FlightLine":
        """Create a flight line centered on a point, extending equally in both directions.

        Args:
            lat: Center latitude in decimal degrees.
            lon: Center longitude in decimal degrees.
            length: Total line length as a Quantity with distance units.
            az: Forward azimuth in degrees from true north.
            altitude_msl: Flight altitude MSL.
            site_name: Optional site name.
            site_description: Optional site description.
            investigator: Optional investigator name.

        Returns:
            A new FlightLine centered on (lat, lon) along the given azimuth.
        """
        if not isinstance(length, Quantity) or not length.check("[length]"):
            raise HyPlanValueError("Length must be a Quantity with units of distance.")
        if not isinstance(az, (int, float)):
            raise HyPlanValueError(f"Azimuth must be a numeric value in degrees. Got {type(az)}.")

        length_m = length.to("meter").magnitude

        lat2, lon2 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az)
        lat1, lon1 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az - 180)

        lon1, lon2 = wrap_to_180(lon1), wrap_to_180(lon2)

        _, az12 = pymap3d.vincenty.vdist(lat1, lon1, lat2, lon2)
        _, az21 = pymap3d.vincenty.vdist(lat2, lon2, lat1, lon1)

        alt = cls._validate_altitude(altitude_msl)
        wp1 = Waypoint(latitude=float(lat1), longitude=float(lon1),
                       heading=float(az12) % 360,
                       altitude_msl=alt,
                       name=f"{site_name}_start" if site_name else "start")
        wp2 = Waypoint(latitude=float(lat2), longitude=float(lon2),
                       heading=(float(az21) + 180.0) % 360.0,
                       altitude_msl=alt,
                       name=f"{site_name}_end" if site_name else "end")

        return cls(waypoint1=wp1, waypoint2=wp2,
                   site_name=site_name, site_description=site_description,
                   investigator=investigator)

    # ------------------------------------------------------------------
    # Transform methods
    # ------------------------------------------------------------------

    def clip_to_polygon(
        self, clip_polygon: Union[Polygon, MultiPolygon]
    ) -> Optional[List["FlightLine"]]:
        """
        Clip the flight line to a specified polygon.

        Args:
            clip_polygon (Union[Polygon, MultiPolygon]): The polygon to clip the flight line to.

        Returns:
            Optional[List["FlightLine"]]: A list of resulting FlightLine(s), or None if no intersection exists.
        """
        clipped_geometry = self.geometry.intersection(clip_polygon)

        if clipped_geometry.is_empty:
            logger.info(f"FlightLine {self.site_name or '<Unnamed>'} excluded after clipping: No intersection.")
            return None

        if isinstance(clipped_geometry, LineString):
            if clipped_geometry.equals(self.geometry):
                logger.info(f"FlightLine {self.site_name or '<Unnamed>'} is entirely within the polygon.")
                return [self]
            else:
                logger.info(f"FlightLine {self.site_name or '<Unnamed>'} was clipped into a single segment.")
                return [self._from_geometry(clipped_geometry)]

        if isinstance(clipped_geometry, MultiLineString):
            results = []
            for i, segment in enumerate(clipped_geometry.geoms):
                new_site_name = f"{self.site_name}_{i:02d}" if self.site_name else f"Segment_{i:02d}"
                logger.info(f"FlightLine {self.site_name or '<Unnamed>'} was split into segment: {new_site_name}")
                results.append(self._from_geometry(segment, site_name=new_site_name))
            return results

        logger.error(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")
        raise HyPlanTypeError(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")

    def track(self, precision: Union[Quantity, float] = 100.0) -> LineString:
        """
        Generate a LineString representing the flight line.

        Args:
            precision (Union[Quantity, float]): Desired distance between interpolated points.
                Accepts a Quantity with length units or a plain float (assumed meters).

        Returns:
            LineString: A LineString object containing the interpolated track.
        """
        if isinstance(precision, Quantity):
            precision_m = precision.to("meter").magnitude
        else:
            precision_m = float(precision)

        num_points = int(np.ceil(self.length.to("meter").magnitude / precision_m)) + 1

        track_lat, track_lon = pymap3d.vincenty.track2(
            self.lat1, self.lon1, self.lat2, self.lon2, npts=num_points, deg=True
        )

        track_lon = wrap_to_180(track_lon)
        return LineString(zip(track_lon, track_lat))

    def reverse(self) -> "FlightLine":
        """
        Reverse the direction of the flight line.

        Returns:
            FlightLine: A new FlightLine object with reversed direction.
        """
        reversed_geom = LineString(list(reversed(self.geometry.coords)))
        return self._from_geometry(reversed_geom)

    def offset_north_east(self, offset_north: Quantity, offset_east: Quantity) -> "FlightLine":
        """
        Offset the flight line in the north and east directions.

        Args:
            offset_north (Quantity): Distance to offset in the north direction (positive or negative).
            offset_east (Quantity): Distance to offset in the east direction (positive or negative).

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        if not isinstance(offset_north, Quantity):
            offset_north = ureg.Quantity(offset_north, "meter")
        if not isinstance(offset_east, Quantity):
            offset_east = ureg.Quantity(offset_east, "meter")

        offset_north_m = offset_north.to("meter").magnitude
        offset_east_m = offset_east.to("meter").magnitude

        def compute_offset(lat, lon, north, east):
            new_lat, new_lon, _ = pymap3d.ned2geodetic(
                north, east, 0, lat, lon, self.altitude_msl.magnitude
            )
            return new_lat, wrap_to_180(new_lon)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, offset_north_m, offset_east_m)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, offset_north_m, offset_east_m)

        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])
        return self._from_geometry(offset_geometry)

    def offset_across(self, offset_distance: Union[Quantity, float]) -> "FlightLine":
        """
        Offset the flight line perpendicular to its direction by a specified distance.

        Args:
            offset_distance (Union[Quantity, float]): Distance to offset the line
                (positive for right, negative for left). Plain floats are assumed meters.

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        if not isinstance(offset_distance, Quantity):
            offset_distance = ureg.Quantity(offset_distance, "meter")

        perpendicular_az = (self.az12.magnitude + 90) % 360 if offset_distance.magnitude >= 0 else (self.az12.magnitude - 90) % 360

        def compute_offset(lat, lon, distance, azimuth):
            return pymap3d.vincenty.vreckon(lat, lon, distance.to("meter").magnitude, azimuth)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, abs(offset_distance), perpendicular_az)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, abs(offset_distance), perpendicular_az)

        new_lon1, new_lon2 = wrap_to_180(new_lon1), wrap_to_180(new_lon2)
        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])
        return self._from_geometry(offset_geometry)

    def offset_along(self, offset_start: Union[Quantity, float], offset_end: Union[Quantity, float]) -> "FlightLine":
        """
        Offset the flight line along its direction by modifying the start and end points.

        Args:
            offset_start (Union[Quantity, float]): Distance to offset the start point along the line
                (positive or negative). Plain floats are assumed meters.
            offset_end (Union[Quantity, float]): Distance to offset the end point along the line
                (positive or negative). Plain floats are assumed meters.

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        if not isinstance(offset_start, Quantity):
            offset_start = ureg.Quantity(offset_start, "meter")
        if not isinstance(offset_end, Quantity):
            offset_end = ureg.Quantity(offset_end, "meter")

        def compute_offset(lat, lon, offset, azimuth):
            if offset < 0:
                azimuth = (azimuth + 180) % 360
                offset = abs(offset)
            return pymap3d.vincenty.vreckon(lat, lon, offset.to("meter").magnitude, azimuth)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, offset_start, self.az12.magnitude)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, offset_end, wrap_to_180(180.0 + self.az21.magnitude))

        new_lon1, new_lon2 = wrap_to_180(new_lon1), wrap_to_180(new_lon2)
        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])
        return self._from_geometry(offset_geometry)

    def rotate_around_midpoint(self, angle: float) -> "FlightLine":
        """
        Rotate the flight line around its midpoint by a specified angle.

        Args:
            angle (float): Rotation angle in degrees. Positive values indicate counterclockwise rotation.

        Returns:
            FlightLine: A new FlightLine object rotated around its midpoint.
        """
        if not isinstance(angle, (int, float)):
            raise HyPlanValueError(f"Angle must be a number. Received: {angle}")

        angle_rad = np.radians(angle)
        midpoint = self.geometry.interpolate(0.5, normalized=True)

        def rotate_point(x, y, center_x, center_y, angle_radians):
            delta_x = x - center_x
            delta_y = y - center_y
            rotated_x = delta_x * np.cos(angle_radians) - delta_y * np.sin(angle_radians) + center_x
            rotated_y = delta_x * np.sin(angle_radians) + delta_y * np.cos(angle_radians) + center_y
            return rotated_x, rotated_y

        rotated_coords = [
            rotate_point(x, y, midpoint.x, midpoint.y, angle_rad)
            for x, y in self.geometry.coords
        ]

        return self._from_geometry(LineString(rotated_coords))

    def split_by_length(self, max_length: Quantity, gap_length: Optional[Quantity] = None) -> List["FlightLine"]:
        """
        Split the flight line into segments of a specified maximum length with an optional gap between segments.

        Args:
            max_length (Quantity): Maximum length of each segment (meters).
            gap_length (Optional[Quantity]): Length of the gap between segments (meters). Default is None.

        Returns:
            List[FlightLine]: List of FlightLine objects representing the segments.
        """
        total_length_m = self.length.to("meter").magnitude
        max_length_m = max_length.to("meter").magnitude
        gap_length_m = gap_length.to("meter").magnitude if gap_length else 0

        if max_length_m <= 0:
            raise HyPlanValueError("Maximum length must be greater than 0.")
        if gap_length and gap_length_m < 0:
            raise HyPlanValueError("Gap length cannot be negative.")
        if max_length_m + gap_length_m > total_length_m:
            return [self]

        segments = []
        remaining_length_m = total_length_m
        current_start_lat, current_start_lon = self.lat1, self.lon1
        segment_index = 0

        while remaining_length_m > 0:
            current_segment_length_m = min(max_length_m, remaining_length_m)
            remaining_length_m -= current_segment_length_m

            end_lat, end_lon = pymap3d.vincenty.vreckon(
                current_start_lat, current_start_lon, current_segment_length_m, self.az12.magnitude
            )
            end_lon = wrap_to_180(end_lon)

            segment_geometry = LineString([(current_start_lon, current_start_lat), (end_lon, end_lat)])
            seg_name = f"{self.site_name}_seg_{segment_index}" if self.site_name else f"Segment_{segment_index}"
            segments.append(self._from_geometry(segment_geometry, site_name=seg_name))
            segment_index += 1

            if gap_length and remaining_length_m > gap_length_m:
                remaining_length_m -= gap_length_m
                current_start_lat, current_start_lon = pymap3d.vincenty.vreckon(
                    end_lat, end_lon, gap_length_m, self.az12.magnitude
                )
                current_start_lon = wrap_to_180(current_start_lon)
            elif gap_length:
                break  # remaining length is less than the gap
            else:
                current_start_lat, current_start_lon = end_lat, end_lon

        return segments

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Convert the flight line to a dictionary representation.

        Returns:
            Dict: Dictionary with keys for geometry coordinates, endpoints,
                length (meters), altitude (meters), and metadata fields.
        """
        return {
            "geometry": list(self.geometry.coords),
            "lat1": self.lat1,
            "lon1": self.lon1,
            "lat2": self.lat2,
            "lon2": self.lon2,
            "length": self.length.magnitude,
            "altitude_msl": self.altitude_msl.magnitude,
            "site_name": self.site_name,
            "site_description": self.site_description,
            "investigator": self.investigator,
        }

    def to_geojson(self) -> Dict:
        """
        Convert the flight line to a GeoJSON Feature dictionary.

        Returns:
            Dict: GeoJSON Feature with LineString geometry and properties
                including altitude, site name, description, and investigator.
        """
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(self.geometry.coords),
            },
            "properties": {
                "altitude_msl": self.altitude_msl.magnitude,
                "site_name": self.site_name,
                "site_description": self.site_description,
                "investigator": self.investigator,
            },
        }


def _validate_linestring(geometry: LineString):
    """Validate a LineString for use as FlightLine geometry."""
    if not isinstance(geometry, LineString):
        raise HyPlanValueError("Geometry must be a Shapely LineString.")
    if len(geometry.coords) != 2:
        raise HyPlanValueError("LineString must have exactly two points.")
    for lon, lat in geometry.coords:
        if not (-90 <= lat <= 90):
            raise HyPlanValueError(f"Latitude {lat} is out of bounds (-90 to 90).")
        if not (-180 <= lon <= 180):
            raise HyPlanValueError(f"Longitude {lon} is out of bounds (-180 to 180).")


def to_gdf(flight_lines: List[FlightLine], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Convert a list of FlightLine objects to a GeoDataFrame.

    Args:
        flight_lines (List[FlightLine]): Flight lines to convert.
        crs (str): Coordinate reference system (default: "EPSG:4326").

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with flight line attributes and geometries.
    """
    data = [fl.to_dict() for fl in flight_lines]
    geometries = [fl.geometry for fl in flight_lines]
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)
