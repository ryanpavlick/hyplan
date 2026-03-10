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
from .dubins_path import Waypoint

logger = logging.getLogger(__name__)


class FlightLine:
    """
    Represents a geospatial flight line with properties, validations, and operations.
    """
    def __init__(
        self,
        geometry: LineString,
        altitude: Quantity,
        site_name: Optional[str] = None,
        site_description: Optional[str] = None,
        investigator: Optional[str] = None,
    ):
        self._validate_geometry(geometry)
        self.geometry = geometry
        self.altitude = self._validate_altitude(altitude)
        self.site_name = site_name
        self.site_description = site_description
        self.investigator = investigator

    @staticmethod
    def _validate_geometry(geometry: LineString):
        if not isinstance(geometry, LineString):
            raise ValueError("Geometry must be a Shapely LineString.")
        if len(geometry.coords) != 2:
            raise ValueError("LineString must have exactly two points.")
        for lon, lat in geometry.coords:
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} is out of bounds (-90 to 90).")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} is out of bounds (-180 to 180).")

    @staticmethod
    def _validate_altitude(altitude: Quantity) -> Quantity:
        if not isinstance(altitude, Quantity):
            altitude = ureg.Quantity(altitude, "meter")
        else:
            altitude = altitude.to("meter")

        if altitude.magnitude < 0 or altitude.magnitude > 22000:
            logger.warning(
                f"Altitude {altitude.magnitude} meters is outside the typical range (0-22000 meters).")
        return altitude

    @property
    def lat1(self):
        return self.geometry.coords[0][1]

    @property
    def lon1(self):
        return self.geometry.coords[0][0]

    @property
    def lat2(self):
        return self.geometry.coords[-1][1]

    @property
    def lon2(self):
        return self.geometry.coords[-1][0]

    @property
    def length(self) -> Quantity:
        length, _ = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(round(length, 2), "meter")

    @property
    def az12(self) -> Quantity:
        _, az12 = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(az12, "degree")

    @property
    def az21(self) -> Quantity:
        _, az21 = pymap3d.vincenty.vdist(self.lat2, self.lon2, self.lat1, self.lon1)
        return ureg.Quantity(az21, "degree")

    @property
    def waypoint1(self) -> Waypoint:
        name = f"{self.site_name}_start" if self.site_name else "start"
        return Waypoint(latitude=self.lat1, longitude=self.lon1, heading=self.az12.magnitude, altitude=self.altitude, name=name)

    @property
    def waypoint2(self) -> Waypoint:
        heading = (self.az21.magnitude + 180.0) % 360.0
        name = f"{self.site_name}_end" if self.site_name else "end"
        return Waypoint(latitude=self.lat2, longitude=self.lon2, heading=heading, altitude=self.altitude, name=name)

    @classmethod
    def start_length_azimuth(
        cls,
        lat1: float,
        lon1: float,
        length: Quantity,
        az: float,
        **kwargs,
    ) -> "FlightLine":
        if not isinstance(length, Quantity) or not length.check("[length]"):
            raise ValueError("Length must be a Quantity with units of distance.")
        if not isinstance(az, (int, float)):
            raise ValueError(f"Azimuth must be a numeric value in degrees. Got {type(az)}.")

        length_m = length.to("meter").magnitude
        lat2, lon2 = pymap3d.vincenty.vreckon(lat1, lon1, length_m, az)
        lon2 = wrap_to_180(lon2)

        geometry = LineString([(lon1, lat1), (lon2, lat2)])
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def center_length_azimuth(
        cls,
        lat: float,
        lon: float,
        length: Quantity,
        az: float,
        **kwargs,
    ) -> "FlightLine":
        if not isinstance(length, Quantity) or not length.check("[length]"):
            raise ValueError("Length must be a Quantity with units of distance.")
        if not isinstance(az, (int, float)):
            raise ValueError(f"Azimuth must be a numeric value in degrees. Got {type(az)}.")

        length_m = length.to("meter").magnitude

        lat2, lon2 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az)
        lat1, lon1 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az - 180)

        lon1, lon2 = wrap_to_180(lon1), wrap_to_180(lon2)
        geometry = LineString([(lon1, lat1), (lon2, lat2)])
        return cls(geometry=geometry, **kwargs)

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
                return [
                    FlightLine(
                        geometry=clipped_geometry,
                        altitude=self.altitude,
                        site_name=self.site_name,
                        site_description=self.site_description,
                        investigator=self.investigator,
                    )
                ]

        if isinstance(clipped_geometry, MultiLineString):
            results = []
            for i, segment in enumerate(clipped_geometry.geoms):
                new_site_name = f"{self.site_name}_{i:02d}" if self.site_name else f"Segment_{i:02d}"
                logger.info(f"FlightLine {self.site_name or '<Unnamed>'} was split into segment: {new_site_name}")
                results.append(
                    FlightLine(
                        geometry=segment,
                        altitude=self.altitude,
                        site_name=new_site_name,
                        site_description=self.site_description,
                        investigator=self.investigator,
                    )
                )
            return results

        logger.error(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")
        raise TypeError(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")

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
        reversed_geometry = LineString(list(reversed(self.geometry.coords)))
        return FlightLine(
            geometry=reversed_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

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
                north, east, 0, lat, lon, self.altitude.magnitude
            )
            return new_lat, wrap_to_180(new_lon)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, offset_north_m, offset_east_m)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, offset_north_m, offset_east_m)

        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

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

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

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

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

    def rotate_around_midpoint(self, angle: float) -> "FlightLine":
        """
        Rotate the flight line around its midpoint by a specified angle.

        Args:
            angle (float): Rotation angle in degrees. Positive values indicate counterclockwise rotation.

        Returns:
            FlightLine: A new FlightLine object rotated around its midpoint.
        """
        if not isinstance(angle, (int, float)):
            raise ValueError(f"Angle must be a number. Received: {angle}")

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

        return FlightLine(
            geometry=LineString(rotated_coords),
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator,
        )

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
            raise ValueError("Maximum length must be greater than 0.")
        if gap_length and gap_length_m < 0:
            raise ValueError("Gap length cannot be negative.")
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
            segments.append(
                FlightLine(
                    geometry=segment_geometry,
                    altitude=self.altitude,
                    site_name=f"{self.site_name}_seg_{segment_index}" if self.site_name else f"Segment_{segment_index}",
                    site_description=self.site_description,
                    investigator=self.investigator,
                )
            )
            segment_index += 1

            if gap_length and remaining_length_m > gap_length_m:
                remaining_length_m -= gap_length_m
                current_start_lat, current_start_lon = pymap3d.vincenty.vreckon(
                    end_lat, end_lon, gap_length_m, self.az12.magnitude
                )
                current_start_lon = wrap_to_180(current_start_lon)
            else:
                break

        return segments

    def to_dict(self) -> Dict:
        return {
            "geometry": list(self.geometry.coords),
            "lat1": self.lat1,
            "lon1": self.lon1,
            "lat2": self.lat2,
            "lon2": self.lon2,
            "length": self.length.magnitude,
            "altitude_m": self.altitude.magnitude,
            "site_name": self.site_name,
            "site_description": self.site_description,
            "investigator": self.investigator,
        }

    def to_geojson(self) -> Dict:
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(self.geometry.coords),
            },
            "properties": {
                "altitude_m": self.altitude.magnitude,
                "site_name": self.site_name,
                "site_description": self.site_description,
                "investigator": self.investigator,
            },
        }


def to_gdf(flight_lines: List[FlightLine], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    data = [fl.to_dict() for fl in flight_lines]
    geometries = [fl.geometry for fl in flight_lines]
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)
