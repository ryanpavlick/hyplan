"""Waypoint class for flight planning."""

from __future__ import annotations

import logging
import warnings
from typing import Union, Dict

import pymap3d
from shapely.geometry import Point

from .geometry import wrap_to_180, wrap_to_360
from pint import Quantity
from .units import ureg
from .exceptions import HyPlanTypeError, HyPlanValueError

logger = logging.getLogger(__name__)


class Waypoint:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        heading: float,
        altitude_msl: Union[Quantity, float, None] = None,
        name: str | None = None,
        speed: Union[Quantity, float, None] = None,
        delay: Union[Quantity, float, None] = None,
        segment_type: str | None = None,
    ):
        """
        Initialize a Waypoint object.

        Args:
            latitude (float): Latitude in decimal degrees.
            longitude (float): Longitude in decimal degrees.
            heading (float): Heading in degrees relative to North.
            altitude_msl (Union[Quantity, float, None], optional): Altitude MSL in meters or as a pint Quantity. Defaults to None.
            name (str, optional): Name of the waypoint. Defaults to None.
            speed (Union[Quantity, float, None], optional): Speed override in m/s or as a pint Quantity. Used for the departing leg. Defaults to None.
            delay (Union[Quantity, float, None], optional): Loiter time at waypoint in seconds or as a pint Quantity. Defaults to None.
            segment_type (str, optional): Segment type label for the departing leg (e.g. "pattern", "sampling"). Used by compute_flight_plan. Defaults to None.
        """
        # Validate latitude and longitude and process geometry
        if not (-90.0 <= latitude <= 90.0):
            raise HyPlanValueError("Latitude must be between -90 and 90 degrees")
        if not (-180.0 <= longitude <= 180.0):
            raise HyPlanValueError("Longitude must be between -180 and 180 degrees")
        self.geometry = Point(longitude, latitude)

        self.latitude = latitude
        self.longitude = longitude

        if isinstance(heading, (int, float)):
            self.heading = wrap_to_360(float(heading))
        else:
            raise HyPlanTypeError("Heading must be a float or an int")

        # Validate and process altitude (MSL)
        if altitude_msl is None:
            self.altitude_msl = None
        elif isinstance(altitude_msl, (int, float)):
            self.altitude_msl = float(altitude_msl) * ureg.meter
        elif hasattr(altitude_msl, 'units') and altitude_msl.check('[length]'):
            self.altitude_msl = altitude_msl.to(ureg.meter)
        else:
            raise HyPlanTypeError("altitude_msl must be None, a float (meters), or a pint Quantity with length units")

        if self.altitude_msl is not None:
            alt_m = self.altitude_msl.magnitude
            if alt_m < 0:
                raise HyPlanValueError(f"Altitude must be non-negative, got {alt_m} m")
            if alt_m > 22000:
                warnings.warn(
                    f"Altitude {alt_m} m is above 22,000 m. Verify this is intended.",
                    stacklevel=2,
                )

        if name is not None:
            self.name = str(name)
        else:
            self.name = f"({self.geometry.y:.2f}, {self.geometry.x:.2f})"

        # Optional fields for flight planning
        self.speed = _validate_quantity(speed, '[speed]', ureg.meter / ureg.second, 'speed')
        self.delay = _validate_quantity(delay, '[time]', ureg.second, 'delay')
        self.segment_type = segment_type

    def offset_north_east(
        self,
        offset_north: Union[Quantity, float],
        offset_east: Union[Quantity, float],
    ) -> "Waypoint":
        """Return a new Waypoint translated by geodetic N/E offsets.

        Args:
            offset_north: Distance north (positive) or south (negative).
                Float interpreted as meters; or a pint Quantity with length units.
            offset_east: Distance east (positive) or west (negative).
                Float interpreted as meters; or a pint Quantity with length units.

        Returns:
            A new Waypoint at the translated position, preserving all other attributes.
        """
        if isinstance(offset_north, (int, float)):
            n_m = float(offset_north)
        else:
            n_m = offset_north.m_as(ureg.meter)
        if isinstance(offset_east, (int, float)):
            e_m = float(offset_east)
        else:
            e_m = offset_east.m_as(ureg.meter)

        alt_m = self.altitude_msl.magnitude if self.altitude_msl is not None else 0.0
        new_lat, new_lon, _ = pymap3d.ned2geodetic(
            n_m, e_m, 0, self.latitude, self.longitude, alt_m,
        )
        return Waypoint(
            latitude=round(new_lat, 6),
            longitude=round(wrap_to_180(new_lon), 6),  # type: ignore[arg-type]
            heading=self.heading,  # type: ignore[arg-type]
            altitude_msl=self.altitude_msl,
            name=self.name,
            speed=self.speed,
            delay=self.delay,
            segment_type=self.segment_type,
        )

    def to_dict(self) -> Dict:
        """
        Convert the waypoint to a dictionary representation.

        Returns:
            Dict: Dictionary with latitude, longitude, heading, altitude_msl, and name.
        """
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "altitude_msl": self.altitude_msl,
            "name": self.name,
        }


def is_waypoint(obj) -> bool:
    """Check if an object is a Waypoint (duck-type safe for notebook reloads)."""
    return (
        hasattr(obj, 'latitude') and hasattr(obj, 'longitude')
        and hasattr(obj, 'heading') and hasattr(obj, 'altitude_msl')
        and hasattr(obj, 'geometry')
    )


def _validate_quantity(value, dimensionality, default_unit, field_name):
    """Validate and convert an optional pint Quantity field."""
    if value is None:
        return None
    elif isinstance(value, (int, float)):
        return float(value) * default_unit
    elif hasattr(value, 'units') and value.check(dimensionality):
        return value.to(default_unit)
    else:
        raise HyPlanTypeError(
            f"{field_name} must be None, a float ({default_unit}), or a pint Quantity with {dimensionality} units"
        )
