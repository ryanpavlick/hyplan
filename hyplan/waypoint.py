"""Waypoint class for flight planning."""

from __future__ import annotations

from typing import Any, TypeGuard
import logging
import warnings

import pymap3d
import pymap3d.vincenty
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
        altitude_msl: Quantity | float | None = None,
        name: str | None = None,
        speed: Quantity | float | None = None,
        delay: Quantity | float | None = None,
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
            self.heading: float = float(wrap_to_360(float(heading)))
        else:
            raise HyPlanTypeError("Heading must be a float or an int")

        # Validate and process altitude (MSL)
        self.altitude_msl: Quantity | None
        if altitude_msl is None:
            self.altitude_msl = None
        elif isinstance(altitude_msl, (int, float)):
            self.altitude_msl = float(altitude_msl) * ureg.meter
        elif hasattr(altitude_msl, 'units') and altitude_msl.check('[length]'):
            self.altitude_msl = altitude_msl.to(ureg.meter)
        else:
            raise HyPlanTypeError("altitude_msl must be None, a float (meters), or a pint Quantity with length units")

        if self.altitude_msl is not None:
            alt_m = self.altitude_msl.m_as(ureg.meter)
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
        offset_north: Quantity | float,
        offset_east: Quantity | float,
    ) -> Waypoint:
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
            longitude=round(wrap_to_180(new_lon), 6),  # type: ignore[arg-type]  # pymap3d returns ndarray, round expects float
            heading=self.heading,
            altitude_msl=self.altitude_msl,
            name=self.name,
            speed=self.speed,
            delay=self.delay,
            segment_type=self.segment_type,
        )

    def to_dict(self) -> dict[Any, Any]:
        """Convert the waypoint to a dictionary representation.

        The returned dict round-trips through :meth:`Waypoint.from_dict`
        and includes every field accepted by ``__init__``. Quantity-valued
        fields (``altitude_msl``, ``speed``, ``delay``) are returned as
        :class:`pint.Quantity` instances; for JSON-friendly output the
        caller is responsible for serialization.

        Returns:
            Dict: Dictionary with all eight Waypoint fields:
                ``latitude``, ``longitude``, ``heading``, ``altitude_msl``,
                ``name``, ``speed``, ``delay``, ``segment_type``.
        """
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "altitude_msl": self.altitude_msl,
            "name": self.name,
            "speed": self.speed,
            "delay": self.delay,
            "segment_type": self.segment_type,
        }

    @classmethod
    def from_dict(cls, data: dict[Any, Any]) -> Waypoint:
        """Reconstruct a Waypoint from a :meth:`to_dict` dictionary.

        Required keys are ``latitude``, ``longitude``, and ``heading``.
        Optional keys (``altitude_msl``, ``name``, ``speed``, ``delay``,
        ``segment_type``) default to ``None`` if absent. This is the
        inverse of :meth:`to_dict` and round-trips losslessly when the
        Quantity fields are preserved as ``pint.Quantity`` instances.

        Args:
            data: Mapping produced by :meth:`to_dict`, or a subset thereof.

        Returns:
            A new Waypoint instance.
        """
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            heading=data["heading"],
            altitude_msl=data.get("altitude_msl"),
            name=data.get("name"),
            speed=data.get("speed"),
            delay=data.get("delay"),
            segment_type=data.get("segment_type"),
        )

    @classmethod
    def relative_to(
        cls,
        anchor: Waypoint | tuple[float, float],
        *,
        bearing: float,
        distance: Quantity | float,
        heading: float | None = None,
        altitude_msl: Quantity | float | None = None,
        name: str | None = None,
        speed: Quantity | float | None = None,
        delay: Quantity | float | None = None,
        segment_type: str | None = None,
    ) -> Waypoint:
        """Create a Waypoint as a geodesic offset from an anchor point.

        Computes the destination via Vincenty direct-problem: from
        ``anchor`` along the great-circle initial bearing ``bearing``
        for the given ``distance``.  Equivalent to Lait's ``=FROM(loc,
        az, dist)`` position expression in the GSFC flight planner.

        Args:
            anchor: A :class:`Waypoint` or ``(latitude, longitude)``
                tuple to anchor against.  Latitude and longitude must
                be in decimal degrees.
            bearing: Initial great-circle bearing from ``anchor``,
                in degrees true (clockwise from north).  Wrapped to
                ``[0, 360)``.
            distance: Geodesic distance along that bearing.  ``float``
                values are interpreted as nautical miles (matches the
                common flight-planning convention); pass a pint
                :class:`Quantity` for other units.
            heading: Heading of the new waypoint in degrees true.
                If ``None`` (default), copies ``bearing`` so the
                waypoint faces the direction it was offset toward —
                the ergonomic "fly toward the new point" default.
            altitude_msl: Altitude MSL.  Float interpreted as metres
                or a pint Quantity with length units; ``None``
                leaves it unset.
            name: Optional name for the new waypoint.
            speed: Optional speed override for the departing leg.
            delay: Optional loiter time at the new waypoint.
            segment_type: Optional segment-type label.

        Returns:
            A new Waypoint at the computed destination.

        Examples:
            >>> edw = Waypoint(latitude=34.92, longitude=-117.87,
            ...                heading=0, name="EDW")
            >>> wp_a = Waypoint.relative_to(edw, bearing=90, distance=200)
            >>> # wp_a is 200 nmi true east of EDW

            >>> from hyplan.units import ureg
            >>> wp_b = Waypoint.relative_to(
            ...     (34.92, -117.87),
            ...     bearing=180,
            ...     distance=50 * ureg.kilometer,
            ...     heading=270,
            ...     name="WP_B",
            ... )
        """
        if isinstance(anchor, Waypoint):
            anchor_lat, anchor_lon = anchor.latitude, anchor.longitude
        else:
            anchor_lat, anchor_lon = float(anchor[0]), float(anchor[1])

        # Distance: float → nautical miles (planning convention);
        # Quantity → m_as(meter).
        if isinstance(distance, (int, float)):
            distance_m = float(distance) * 1852.0
        else:
            distance_m = distance.m_as(ureg.meter)

        bearing_deg = float(wrap_to_360(float(bearing)))

        new_lat, new_lon = pymap3d.vincenty.vreckon(
            anchor_lat, anchor_lon, distance_m, bearing_deg,
        )
        # vreckon may return numpy scalars; normalise to plain floats
        # and wrap the longitude into [-180, 180).
        new_lat = float(new_lat)
        new_lon = float(wrap_to_180(float(new_lon)))

        return cls(
            latitude=round(new_lat, 6),
            longitude=round(new_lon, 6),
            heading=float(heading) if heading is not None else bearing_deg,
            altitude_msl=altitude_msl,
            name=name,
            speed=speed,
            delay=delay,
            segment_type=segment_type,
        )


def is_waypoint(obj: Any) -> TypeGuard[Waypoint]:
    """Check if an object is a Waypoint (duck-type safe for notebook reloads)."""
    return (
        hasattr(obj, 'latitude') and hasattr(obj, 'longitude')
        and hasattr(obj, 'heading') and hasattr(obj, 'altitude_msl')
        and hasattr(obj, 'geometry')
    )


def _validate_quantity(
    value: Quantity | float | None,
    dimensionality: str,
    default_unit: Any,
    field_name: str,
) -> Quantity | None:
    """Validate and convert an optional pint Quantity field."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * default_unit
    if hasattr(value, 'units') and value.check(dimensionality):
        return value.to(default_unit)
    raise HyPlanTypeError(
        f"{field_name} must be None, a float ({default_unit}), or a pint Quantity with {dimensionality} units"
    )
