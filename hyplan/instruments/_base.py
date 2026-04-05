"""Abstract base class for all HyPlan sensor models."""

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
            TypeError: If value is not a pint Quantity.
        """
        if not isinstance(value, Quantity):
            raise HyPlanTypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
        return value.to(expected_unit)
