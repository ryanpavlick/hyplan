import numpy as np
import pymap3d.vincenty
from typing import Optional, List, Callable, Dict, Union
from shapely.geometry import Polygon
import logging

from . import flight_line
from .units import ureg, altitude_to_flight_level
from .geometry import wrap_to_180, rotated_rectangle, minimum_rotated_rectangle, buffer_polygon_along_azimuth, _validate_polygon
from .exceptions import HyPlanValueError


logger = logging.getLogger(__name__)


def _validate_inputs(**kwargs) -> None:
    """
    Validate input parameters for various operations using dynamic rules.

    Args:
        **kwargs: Arbitrary keyword arguments representing parameters to validate.
            Supported parameters and their rules:
            - altitude: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - box_length: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - box_width: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - overlap: Must be a float between 0 and 100 (inclusive).
            - starting_point: Must be either "edge" or "center".
            - azimuth: Must be a float, wrapped to [-180, 180] degrees.
            - polygon: If provided, must be a valid Shapely Polygon.
            - clip_to_polygon: Must be a boolean.

    Raises:
        ValueError: If any parameter fails its validation rule.

    Notes:
        - Length-related parameters (`altitude`, `box_length`, `box_width`) are checked for dimensionality if they are `ureg.Quantity` and converted to meters.
        - Unknown parameters will be ignored, with a warning logged.
    """
    rules: Dict[str, Callable[[Union[float, Polygon, bool, None]], bool]] = {
        'altitude': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'box_length': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'box_width': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'overlap': lambda x: isinstance(x, (float, int)) and 0 <= x <= 100,
        'starting_point': lambda x: x in {"edge", "center"},
        'azimuth': lambda x: isinstance(x, float),
        'polygon': lambda x: x is None or _validate_polygon(x),
        'clip_to_polygon': lambda x: isinstance(x, bool),
    }

    for key, value in kwargs.items():
        if key in {'altitude', 'box_length', 'box_width'}:
            # Validate and process length-related parameters
            if isinstance(value, ureg.Quantity):
                if not value.check("[length]"):
                    raise HyPlanValueError(f"Invalid unit for '{key}': Expected a length unit. Got {value.dimensionality}.")
                value = value.to("meter").magnitude  # Convert to meters
            elif not isinstance(value, float):
                raise HyPlanValueError(f"Invalid type for '{key}': Expected float (meters) or ureg.Quantity. Got {type(value)}.")
            
            if value <= 0:
                raise HyPlanValueError(f"Invalid value for '{key}': {value}. Must be greater than 0.")

        elif key == 'azimuth':
            # Validate and wrap azimuth
            if not isinstance(value, float):
                raise HyPlanValueError(f"Invalid type for 'azimuth': Expected float. Got {type(value)}.")
            kwargs[key] = wrap_to_180(value)

        # elif key == 'polygon' and value is not None:
        #     # Validate polygon
        #     _validate_polygon(value)

        elif key in rules:
            # Validate other parameters using rules
            if not rules[key](value):
                raise HyPlanValueError(f"Invalid value for '{key}': {value}. Check documentation for valid inputs.")
        else:
            # Warn about unknown parameters
            logger.warning(f"Unknown parameter '{key}' provided. No validation rule exists.")
    
    logger.debug("All inputs passed validation.")

        
def box_around_center_line(
    instrument: object,
    altitude_msl: ureg.Quantity,
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length: ureg.Quantity,
    box_width: ureg.Quantity,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    starting_point: str = "center",
    polygon: Optional[Polygon] = None,
) -> List[flight_line.FlightLine]:
    """
    Create a series of flight lines around a center line based on the given box dimensions and instrument properties.

    Args:
        instrument (object): An object with a ``swath_width(altitude_agl)`` method.
        altitude_msl (ureg.Quantity): Flight altitude MSL. Must be a positive length Quantity.
            Note: this is passed directly to ``instrument.swath_width()`` which expects
            AGL. Over flat terrain near sea level the difference is negligible; for
            mountainous terrain, consider adjusting for mean terrain elevation.
        lat0 (float): Latitude of the box center in decimal degrees (-90 to 90).
        lon0 (float): Longitude of the box center in decimal degrees (-180 to 180).
        azimuth (float): Orientation of the box in degrees. Will be wrapped to [-180, 180].
        box_length (ureg.Quantity): Length of the box as a positive length Quantity.
        box_width (ureg.Quantity): Width of the box as a positive length Quantity.
        box_name (str): Name prefix for the flight lines.
        start_numbering (int): Starting number for flight line naming. Must be a positive integer.
        overlap (float): Percentage overlap between adjacent swaths. Must be between 0 and 100.
        alternate_direction (bool): Whether to alternate flight line directions.
        starting_point (str): Whether to start the first line from the "edge" or "center" of the box.
        polygon (Optional[Polygon]): Optional polygon to clip flight lines. Must be valid.

    Returns:
        List[flight_line.FlightLine]: A list of generated flight lines.

    Raises:
        ValueError: If inputs do not meet validation criteria.

    Notes:
        - Flight lines are generated around the center of the box, with distances calculated from the centerline.
        - Clipping is applied to each line if a polygon is provided.
    """
    # Validate inputs
    _validate_inputs(
        altitude=altitude_msl,
        box_length=box_length,
        box_width=box_width,
        overlap=overlap,
        azimuth=azimuth,
        polygon=polygon
    )

    if not hasattr(instrument, "swath_width") or not callable(instrument.swath_width):
        raise HyPlanValueError("Instrument must have a callable method `swath_width(altitude_agl)`.")

    # Compute swath spacing and number of lines
    # Note: swath_width expects AGL; using MSL as approximation
    swath = instrument.swath_width(altitude_msl)
    if not isinstance(swath, ureg.Quantity):
        swath = ureg.Quantity(swath, "meter")
    if swath <= 0:
        raise HyPlanValueError(f"Invalid swath width {swath}. Must be positive.")

    swath_spacing = swath * (1 - (overlap / 100))
    if swath_spacing <= 0:
        raise HyPlanValueError(f"Invalid swath spacing {swath_spacing}. Adjust overlap or instrument parameters.")

    if polygon:
        along_track_buffer = 2000.0
        polygon = buffer_polygon_along_azimuth(polygon, along_track_buffer, swath.magnitude/2, azimuth)
        box_length += ureg.Quantity(along_track_buffer, "meter")

    nlines = max(1, int(np.ceil(box_width / swath_spacing)))

    logger.info(f"Calculated swath spacing: {swath_spacing:.2f} meters.")
    logger.info(f"Number of lines: {nlines}.")

    # Generate flight lines
    dists_from_center = np.arange(-nlines // 2, nlines // 2 + 1) * swath_spacing

    if starting_point == "edge":
        first_line = flight_line.FlightLine.start_length_azimuth(
            lat1=lat0, lon1=lon0, length=box_length, az=azimuth, altitude_msl=altitude_msl
        )
    elif starting_point == "center":
        first_line = flight_line.FlightLine.center_length_azimuth(
            lat=lat0, lon=lon0, length=box_length, az=azimuth, altitude_msl=altitude_msl
        )

    flight_level = altitude_to_flight_level(altitude_msl)
    lines = []

    for idx, dist in enumerate(dists_from_center):
        line = first_line.offset_across(dist)
        if alternate_direction and idx % 2 == 0:
            line = line.reverse()

        line.site_name = f"{box_name}_L{idx + start_numbering:02d}_{flight_level}"

        if polygon:
            clipped_lines = line.clip_to_polygon(polygon)
            if clipped_lines:
                lines.extend(clipped_lines)
                logger.debug(f"Line {line.site_name} clipped into {len(clipped_lines)} segments.")
            else:
                logger.info(f"Line {line.site_name} fully excluded after clipping.")
        else:
            lines.append(line)

    if not lines:
        logger.warning("No flight lines were generated after clipping.")

    return lines


def box_around_polygon(
    instrument: object,
    altitude_msl: ureg.Quantity,
    polygon: Polygon,
    azimuth: Optional[float] = None,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    clip_to_polygon: bool = True,
) -> List[flight_line.FlightLine]:
    """
    Generate flight lines based on either:
    - The minimum rotated rectangle of a polygon (if `azimuth=None`)
    - A rotated rectangle at a specified azimuth (if `azimuth` is given)

    Args:
        instrument (object): Object with ``swath_width(altitude_agl)`` method.
        altitude_msl (ureg.Quantity): Flight altitude MSL.
        polygon (Polygon): Input polygon to generate flight lines within.
        azimuth (Optional[float]): If provided, uses a user-defined azimuth instead of the minimum rotated rectangle.
        box_name (str): Prefix for flight line names.
        start_numbering (int): Starting number for flight line names.
        overlap (float): Overlap percentage between adjacent swaths.
        alternate_direction (bool): Whether to alternate flight line directions.
        clip_to_polygon (bool): Whether to clip flight lines to the convex hull of the polygon.
        starting_point (str): Whether to start the first line from the "edge" or "center".
        
    Returns:
        List[flight_line.FlightLine]: A list of generated flight lines.

    Raises:
        ValueError: If inputs are invalid.

    Notes:
        - If `azimuth` is `None`, the function will automatically compute the **minimum rotated rectangle**.
        - If `azimuth` is provided, a **rotated rectangle at the given azimuth** will be used.
    """

    # Validate inputs
    if not isinstance(polygon, Polygon):
        raise HyPlanValueError("Input must be a Shapely Polygon.")

    # Compute bounding rectangle based on the provided azimuth
    try:
        if azimuth is None:
            logger.info("Using minimum rotated rectangle for polygon bounding box.")
            bounding_box = minimum_rotated_rectangle(polygon)
        else:
            logger.info(f"Using rotated rectangle at azimuth {azimuth:.2f}°.")
            bounding_box = rotated_rectangle(polygon, azimuth)
    except Exception as e:
        raise HyPlanValueError(f"Failed to calculate bounding box: {e}")

    # Extract centroid, dimensions, and azimuth
    lon0, lat0 = bounding_box.centroid.coords[0]
    minx, miny, maxx, maxy = bounding_box.bounds
    box_length = pymap3d.vincenty.vdist(miny, minx, maxy, minx)[0] * ureg.meter
    box_width = pymap3d.vincenty.vdist(miny, minx, miny, maxx)[0] * ureg.meter

    if azimuth is None:
        # Get exterior coordinates of the bounding box
        lons, lats = list(bounding_box.exterior.coords.xy)

        # Compute distances along both rectangle axes
        length1, az1 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[3], lons[3])
        length2, az2 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[1], lons[1])

        # Use the azimuth corresponding to the longer side
        if length1 >= length2:
            azimuth = wrap_to_180(az1)
        else:
            azimuth = wrap_to_180(az2)


    logger.info(
        f"Bounding box derived: Center=({lat0:.6f}, {lon0:.6f}), Azimuth={azimuth:.2f}°, "
        f"Length={box_length.magnitude:.2f} m, Width={box_width.magnitude:.2f} m."
    )

    # Call `box_around_center_line` to generate flight lines
    return box_around_center_line(
        instrument=instrument,
        altitude_msl=altitude_msl,
        lat0=lat0,
        lon0=lon0,
        azimuth=azimuth,
        box_length=box_length,
        box_width=box_width,
        box_name=box_name,
        start_numbering=start_numbering,
        overlap=overlap,
        alternate_direction=alternate_direction,
        polygon=polygon if clip_to_polygon else None,
        starting_point="center",
    )