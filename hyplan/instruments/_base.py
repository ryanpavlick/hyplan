"""Base class and structural protocol for HyPlan sensor models."""

from typing import Protocol, Tuple, runtime_checkable

from pint import Quantity

from ..exceptions import HyPlanTypeError
from ..units import ureg  # noqa: F401 — available to subclasses via this module


class Sensor:
    """Base class to represent a generic sensor.

    Args:
        name (str): Human-readable name identifying the sensor.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def _validate_quantity(self, value: Quantity, expected_unit: Quantity) -> Quantity:
        """
        Validate that a value is a pint Quantity and convert it to the expected unit.

        Args:
            value (Quantity): The value to validate.
            expected_unit (Quantity): The target unit to convert to.

        Returns:
            Quantity: The value converted to the expected unit.

        Raises:
            HyPlanTypeError: If value is not a pint Quantity.
        """
        if not isinstance(value, Quantity):
            raise HyPlanTypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
        return value.to(expected_unit)


@runtime_checkable
class ScanningSensor(Protocol):
    """Structural type for sensors with a continuous cross-track swath.

    This is the protocol expected by :func:`hyplan.swath.generate_swath_polygon`,
    :func:`hyplan.flight_box.box_around_center_line`, and the glint helpers in
    :mod:`hyplan.glint`. Any object that exposes these attributes will satisfy
    the protocol; explicit inheritance is not required, but the concrete
    classes :class:`hyplan.instruments.LineScanner`,
    :class:`hyplan.instruments.LVIS`, and
    :class:`hyplan.instruments.SidelookingRadar` all conform to it.

    :class:`hyplan.instruments.FrameCamera` deliberately does **not** conform —
    frame cameras are planned through their own footprint helpers rather than
    via the swath/scanner pipeline.
    """

    name: str

    @property
    def half_angle(self) -> float:
        """Half-FOV from boresight to swath edge, in degrees."""
        ...

    def swath_offset_angles(self) -> Tuple[float, float]:
        """Cross-track angles for the port and starboard swath edges.

        Both angles are measured from nadir in degrees: negative = port
        (left of track), positive = starboard (right of track).
        """
        ...

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Cross-track swath width at the given altitude AGL, in meters."""
        ...
