import numpy as np

import math
from shapely.geometry import Point, LineString
from shapely.ops import transform
from typing import Union, Dict
from dubins import path_sample

from .geometry import get_utm_transforms, wrap_to_360
from .units import ureg


class Waypoint:
    def __init__(self, latitude: float, longitude: float, heading: float, altitude_msl: Union[ureg.Quantity, float, None] = None, name: str = None):
        """
        Initialize a Waypoint object.

        Args:
            latitude (float): Latitude in decimal degrees.
            longitude (float): Longitude in decimal degrees.
            heading (float): Heading in degrees relative to North.
            altitude_msl (Union[Quantity, float, None], optional): Altitude MSL in meters or as a pint Quantity. Defaults to None.
            name (str, optional): Name of the waypoint. Defaults to None.
        """
        # Validate latitude and longitude and process geometry
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        self.geometry = Point(longitude, latitude)

        self.latitude = latitude
        self.longitude = longitude

        if isinstance(heading, (int, float)):
            self.heading = wrap_to_360(float(heading))
        else:
            raise TypeError("Heading must be a float or an int")

        # Validate and process altitude (MSL)
        if altitude_msl is None:
            self.altitude_msl = None
        elif isinstance(altitude_msl, (int, float)):
            self.altitude_msl = float(altitude_msl) * ureg.meter
        elif hasattr(altitude_msl, 'units') and altitude_msl.check('[length]'):
            self.altitude_msl = altitude_msl.to(ureg.meter)
        else:
            raise TypeError("altitude_msl must be None, a float (meters), or a pint Quantity with length units")

        if name is not None:
            self.name = str(name)
        else:
            self.name = f"({self.geometry.y:.2f}, {self.geometry.x:.2f})"

    def to_dict(self) -> Dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "altitude_msl": self.altitude_msl,
            "name": self.name
        }


class DubinsPath:
    def __init__(self, start: Waypoint, end: Waypoint, speed: Union[ureg.Quantity, float], bank_angle: float, step_size: float):
        """
        Initialize a DubinsPath object.

        Args:
            start (Waypoint): The starting waypoint.
            end (Waypoint): The ending waypoint.
            speed (Union[Quantity, float]): Speed as a pint Quantity or a float (meters per second).
            bank_angle (float): Bank angle in degrees.
            step_size (float): Step size for sampling the trajectory.
        """
        if not isinstance(start, Waypoint) or not isinstance(end, Waypoint):
            raise TypeError("start and end must be Waypoint objects")

        self.start = start
        self.end = end

        if isinstance(speed, float):
            self.speed_mps = speed  # Assume meters per second if speed is a float
        elif hasattr(speed, 'units') and speed.check('[speed]'):
            self.speed_mps = speed.to(ureg.meter / ureg.second).magnitude
        else:
            raise TypeError("speed must be a pint Quantity with speed units or a float (meters per second)")

        self.bank_angle = bank_angle
        self.step_size = step_size

        # Calculate Dubins path properties
        self._geometry = None
        self._length = None
        self._calculate_path()

    def _calculate_path(self):
        """
        Calculate the Dubins path and its properties.
        """
        # Convert bank angle to radians
        bank_angle_rad = np.radians(self.bank_angle)

        # Calculate the turn radius
        g = 9.8  # m/s^2
        turn_radius = (self.speed_mps ** 2) / (g * math.tan(bank_angle_rad))

        # Convert azimuths to radians
        heading1 = -np.radians(self.start.heading - 90.0)
        heading2 = -np.radians(self.end.heading  - 90.0)

        # Get UTM transforms
        to_utm, from_utm = get_utm_transforms([self.start.geometry, self.end.geometry])

        # Transform points to UTM
        start_utm = transform(to_utm, self.start.geometry)
        end_utm = transform(to_utm, self.end.geometry)

        # Define the start and end configurations
        q0 = (start_utm.x, start_utm.y, heading1)
        q1 = (end_utm.x, end_utm.y, heading2)

        # Generate the Dubins path
        qs, _ = path_sample(q0, q1, turn_radius, self.step_size)

        # Convert sampled points back to geographic coordinates using bulk transform
        qs_arr = np.array(qs)
        xs_utm = qs_arr[:, 0]
        ys_utm = qs_arr[:, 1]
        lons_geo, lats_geo = from_utm(xs_utm, ys_utm)

        # Create LineString and calculate length
        self._geometry = LineString(np.column_stack([lons_geo, lats_geo]))
        self._length = self._calculate_length(qs)
    
    def _calculate_length(self, qs) -> ureg.Quantity:
        """
        Calculate the length of the Dubins path directly from the sampled points using vectorized operations.

        Args:
            qs (list): List of sampled points [(x1, y1, h1), (x2, y2, h2), ...].

        Returns:
            Quantity: Total length of the path in meters.
        """
        coordinates = np.array([(x, y) for x, y, _ in qs])
        diffs = np.diff(coordinates, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        
        return (distances.sum() * ureg.meter)  # Convert to pint.Quantity


    @property
    def geometry(self) -> LineString:
        """
        Get the Dubins path as a LineString.

        Returns:
            LineString: The Dubins path.
        """
        return self._geometry

    @property
    def length(self) -> float:
        """
        Get the length of the Dubins path in meters.

        Returns:
            float: Length of the path in meters.
        """
        return self._length
    

    def to_dict(self) -> Dict:
        return {
            "geometry": self.geometry,
            "start_lat": self.start.latitude,
            "start_lon": self.start.longitude,
            "end_lat": self.end.latitude,
            "end_lon": self.end.longitude,
            "start_altitude": self.start.altitude_msl.to(ureg.meter).magnitude,
            "end_altitude": self.end.altitude_msl.to(ureg.meter).magnitude,
            "start_heading": self.start.heading,
            "end_heading": self.end.heading,
            "distance": self.length.to(ureg.nautical_mile).magnitude
        }
