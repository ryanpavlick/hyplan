from typing import Union

from pint import UnitRegistry, Quantity, set_application_registry
from .exceptions import HyPlanValueError

ureg = UnitRegistry()

# Set application-wide registry
set_application_registry(ureg)


def magnitude_in(value, unit) -> float:
    """Return ``value`` as a float in ``unit``.

    Accepts either a :class:`pint.Quantity` (which is converted to ``unit``
    and stripped to its magnitude) or a bare numeric (which is assumed to
    already be in ``unit`` and is returned as a float). Use this at module
    boundaries where the caller may pass either form.

    For the common case where ``value`` is already known to be a Quantity,
    prefer pint's built-in :meth:`pint.Quantity.m_as` directly --
    ``q.m_as("meter")`` is equivalent to ``q.to("meter").magnitude`` but
    shorter, and doesn't need this helper.
    """
    if hasattr(value, "to"):
        return value.m_as(unit)
    return float(value)

def convert_distance(distance: float, from_unit: str, to_unit: str) -> float:
    """
    Convert distance between specified units.

    Args:
        distance (float): Distance to convert.
        from_unit (str): Unit of the input distance. Must be one of "meters", "kilometers", "miles", "nautical_miles", "feet".
        to_unit (str): Unit of the output distance. Must be one of "meters", "kilometers", "miles", "nautical_miles", "feet".

    Returns:
        float: Distance converted to the target unit.
    """
    units = {
        "meters": ureg.meter,
        "kilometers": ureg.kilometer,
        "miles": ureg.mile,
        "nautical_miles": ureg.nautical_mile,
        "feet": ureg.foot
    }
    if from_unit not in units or to_unit not in units:
        raise HyPlanValueError(f"Unsupported unit. Choose from {list(units.keys())}.")
    
    return (distance * units[from_unit]).m_as(units[to_unit])

def convert_speed(speed: float, from_unit: str, to_unit: str) -> float:
    """
    Convert speed between specified units.

    Args:
        speed (float): Speed to convert.
        from_unit (str): Unit of the input speed. Must be one of "mps", "kph", "mph", "knots", "fps".
        to_unit (str): Unit of the output speed. Must be one of "mps", "kph", "mph", "knots", "fps".

    Returns:
        float: Speed converted to the target unit.
    """
    units = {
        "mps": ureg.meter / ureg.second,   # Meters per second
        "kph": ureg.kilometer / ureg.hour,  # Kilometers per hour
        "mph": ureg.mile / ureg.hour,      # Miles per hour
        "knots": ureg.nautical_mile / ureg.hour,  # Knots (nautical miles per hour)
        "fps": ureg.foot / ureg.second,    # Feet per second
    }
    if from_unit not in units or to_unit not in units:
        raise HyPlanValueError(f"Unsupported unit. Choose from {list(units.keys())}.")

    return (speed * units[from_unit]).m_as(units[to_unit])

def convert_angle(angle: float, from_unit: str, to_unit: str) -> float:
    """
    Convert angle between specified units.

    Args:
        angle (float): Angle to convert.
        from_unit (str): Unit of the input angle. Must be one of "degrees", "radians", "arcminutes", "arcseconds".
        to_unit (str): Unit of the output angle. Must be one of "degrees", "radians", "arcminutes", "arcseconds".

    Returns:
        float: Angle converted to the target unit.
    """
    units = {
        "degrees": ureg.degree,
        "radians": ureg.radian,
        "arcminutes": ureg.arcminute,
        "arcseconds": ureg.arcsecond,
    }
    if from_unit not in units or to_unit not in units:
        raise HyPlanValueError(f"Unsupported unit. Choose from {list(units.keys())}.")

    return (angle * units[from_unit]).m_as(units[to_unit])

def convert_time(time: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a time duration between specified units.

    Args:
        time (float): Time duration to convert.
        from_unit (str): Unit of the input duration. Must be one of "seconds", "minutes", "hours", "days".
        to_unit (str): Unit of the output duration. Must be one of "seconds", "minutes", "hours", "days".

    Returns:
        float: Duration converted to the target unit.
    """
    units = {
        "seconds": ureg.second,
        "minutes": ureg.minute,
        "hours": ureg.hour,
        "days": ureg.day,
    }
    if from_unit not in units or to_unit not in units:
        raise HyPlanValueError(f"Unsupported unit. Choose from {list(units.keys())}.")

    return (time * units[from_unit]).m_as(units[to_unit])

def altitude_to_flight_level(altitude: Union[float, int, Quantity], pressure: Union[float, int, Quantity] = 1013.25) -> str:
    """
    Converts altitude to flight level (FL), considering atmospheric pressure.

    Args:
        altitude (float or pint.Quantity): Altitude to convert. If a float, it is assumed to be in meters.
        pressure (float or pint.Quantity): Atmospheric pressure. If a float, it is assumed to be in hPa.

    Returns:
        str: Flight level (FL) as a string (e.g., "FL030", "FL350").

    Raises:
        ValueError: If altitude is not a float or a pint length.
        ValueError: If pressure is not a float or a pint pressure unit.
    """
    # Validate altitude
    if isinstance(altitude, ureg.Quantity):
        if not altitude.check("[length]"):
            raise HyPlanValueError("Altitude must have units of length.")
        altitude_ft = altitude.m_as("feet")
    elif isinstance(altitude, (int, float)):
        # Assume numeric value is in meters
        altitude_ft = (altitude * ureg.meter).m_as("feet")
    else:
        raise HyPlanValueError("Altitude must be a pint length or a number (assumed meters).")

    # Validate pressure
    if isinstance(pressure, ureg.Quantity):
        if not pressure.check("[pressure]"):
            raise HyPlanValueError("Pressure must have units of pressure.")
        pressure_hpa = pressure.m_as("hPa")
    elif isinstance(pressure, (int, float)):
        pressure_hpa = pressure  # Assume numeric value is in hPa
    else:
        raise HyPlanValueError("Pressure must be a pint pressure unit or a number (assumed hPa).")

    # Adjust altitude for atmospheric pressure deviation
    if pressure_hpa != 1013.25:
        pressure_ratio = pressure_hpa / 1013.25
        altitude_ft /= pressure_ratio

    # Convert to flight level
    flight_level = int(round(altitude_ft / 100))

    # Format flight level as a string with leading zeros
    return f"FL{flight_level:03d}"


