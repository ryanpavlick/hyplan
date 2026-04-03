"""Flight box generation — parallel flight lines covering a geographic area.

Provides functions to lay out parallel, evenly-spaced flight lines over a
study area.  Lines can be generated from a center line and swath overlap
(:func:`box_around_center_line`), from an arbitrary polygon boundary
(:func:`box_around_polygon`), or with terrain-aware altitude adjustments
(:func:`box_around_center_terrain`).
"""

import numpy as np
import pymap3d.vincenty
from typing import Optional, List, Callable, Dict, Union
from pint import Quantity
from shapely.geometry import Polygon
import logging

from . import flight_line
from . import terrain
from .sensors import LineScanner
from .swath import generate_swath_polygon, calculate_swath_widths
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

    nlines = max(1, int(np.ceil((box_width / swath_spacing).to("dimensionless").magnitude)))

    logger.info(f"Calculated swath spacing: {swath_spacing:.2f} meters.")
    logger.info(f"Number of lines: {nlines}.")

    # Generate flight lines — symmetric offsets centered at 0
    dists_from_center = (np.arange(nlines) - (nlines - 1) / 2.0) * swath_spacing

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
        raise HyPlanValueError(
            "All flight lines were clipped away by the polygon. "
            "Check polygon coverage and orientation."
        )

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

    # Extract centroid and rectangle edge dimensions
    lon0, lat0 = bounding_box.centroid.coords[0]
    lons, lats = list(bounding_box.exterior.coords.xy)

    # Compute distances and azimuths along both rectangle axes
    length1, az1 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[1], lons[1])
    length2, az2 = pymap3d.vincenty.vdist(lats[1], lons[1], lats[2], lons[2])

    if azimuth is None:
        # Use the azimuth corresponding to the longer side
        if length1 >= length2:
            azimuth = wrap_to_180(az1)
            box_length = float(length1) * ureg.meter
            box_width = float(length2) * ureg.meter
        else:
            azimuth = wrap_to_180(az2)
            box_length = float(length2) * ureg.meter
            box_width = float(length1) * ureg.meter
    else:
        # Assign box_length to the edge most aligned with the user azimuth
        az1_diff = abs(wrap_to_180(float(az1) - azimuth))
        az2_diff = abs(wrap_to_180(float(az2) - azimuth))
        if az1_diff <= az2_diff:
            box_length = float(length1) * ureg.meter
            box_width = float(length2) * ureg.meter
        else:
            box_length = float(length2) * ureg.meter
            box_width = float(length1) * ureg.meter


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


def box_around_polygon_terrain(
    instrument,
    altitude_msl: Quantity,
    polygon: Polygon,
    azimuth: Optional[float] = None,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    clip_to_polygon: bool = True,
    safe_altitude: Quantity = ureg.Quantity(300, "meter"),
    min_line_length: Quantity = ureg.Quantity(200, "meter"),
) -> List[flight_line.FlightLine]:
    """Generate terrain-aware flight lines covering a polygon.

    Works for both ``LineScanner`` and ``SidelookingRadar`` sensors.
    The flight altitude MSL is provided explicitly by the caller.
    Line spacing is computed by ray-terrain intersection at each
    candidate line position, ensuring the requested overlap is
    maintained even over variable terrain.

    Unlike :func:`box_around_polygon`, which uses a flat-earth
    ``swath_width()`` for line spacing, this function calls
    :func:`~hyplan.swath.generate_swath_polygon` with a DEM at each
    candidate line to determine the actual terrain-projected swath
    width before stepping to the next line.

    Unlike :func:`box_around_center_terrain`, this function accepts a
    polygon boundary (instead of explicit box dimensions), works with
    any sensor that implements ``swath_width()`` and
    ``swath_offset_angles()``, and always uses a caller-supplied
    altitude rather than deriving one from a target pixel size.

    Args:
        instrument: Sensor with ``swath_width(altitude_agl)`` and
            ``swath_offset_angles()`` methods. Accepts both
            ``LineScanner`` and ``SidelookingRadar``.
        altitude_msl: Flight altitude above mean sea level.
        polygon: Study-area boundary polygon (WGS84 lon/lat).
        azimuth: Flight-line orientation in degrees from true north.
            If ``None``, uses the minimum rotated rectangle of the
            polygon.
        box_name: Prefix for flight-line site names.
        start_numbering: First line number used in site names.
        overlap: Swath overlap percentage between adjacent lines (0–100).
        alternate_direction: Reverse every other line direction.
        clip_to_polygon: Clip lines to the polygon boundary.
        safe_altitude: Minimum required clearance above the highest
            terrain point in the survey area.
        min_line_length: Drop clipped segments shorter than this value.

    Returns:
        List of :class:`~hyplan.flight_line.FlightLine` objects with
        terrain-aware spacing.

    Raises:
        HyPlanValueError: For invalid inputs or if ``altitude_msl``
            does not provide sufficient clearance above terrain.
    """
    if not isinstance(polygon, Polygon):
        raise HyPlanValueError("polygon must be a Shapely Polygon.")
    if not (
        hasattr(instrument, "swath_width") and callable(instrument.swath_width)
        and hasattr(instrument, "swath_offset_angles") and callable(instrument.swath_offset_angles)
    ):
        raise HyPlanValueError(
            "instrument must implement swath_width(altitude_agl) and swath_offset_angles()."
        )
    _validate_inputs(altitude=altitude_msl, overlap=overlap)

    # Compute bounding rectangle from polygon
    try:
        if azimuth is None:
            logger.info("Using minimum rotated rectangle for polygon bounding box.")
            bounding_box = minimum_rotated_rectangle(polygon)
        else:
            logger.info(f"Using rotated rectangle at azimuth {azimuth:.2f}°.")
            bounding_box = rotated_rectangle(polygon, azimuth)
    except Exception as e:
        raise HyPlanValueError(f"Failed to calculate bounding box: {e}")

    lon0, lat0 = bounding_box.centroid.coords[0]
    lons, lats = list(bounding_box.exterior.coords.xy)

    length1, az1 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[1], lons[1])
    length2, az2 = pymap3d.vincenty.vdist(lats[1], lons[1], lats[2], lons[2])

    if azimuth is None:
        if length1 >= length2:
            azimuth = wrap_to_180(az1)
            box_length = float(length1) * ureg.meter
            box_width  = float(length2) * ureg.meter
        else:
            azimuth = wrap_to_180(az2)
            box_length = float(length2) * ureg.meter
            box_width  = float(length1) * ureg.meter
    else:
        az1_diff = abs(wrap_to_180(float(az1) - azimuth))
        az2_diff = abs(wrap_to_180(float(az2) - azimuth))
        if az1_diff <= az2_diff:
            box_length = float(length1) * ureg.meter
            box_width  = float(length2) * ureg.meter
        else:
            box_length = float(length2) * ureg.meter
            box_width  = float(length1) * ureg.meter

    logger.info(
        f"Bounding box: center=({lat0:.6f}, {lon0:.6f}), az={azimuth:.2f}°, "
        f"length={box_length.magnitude:.0f} m, width={box_width.magnitude:.0f} m."
    )

    box_length_m      = box_length.to("meter").magnitude
    box_width_m       = box_width.to("meter").magnitude
    safe_altitude_m   = safe_altitude.to("meter").magnitude
    min_line_length_m = min_line_length.to("meter").magnitude

    dem_file = _generate_box_dem(lat0, lon0, azimuth, box_length_m, box_width_m)

    _, max_elev = terrain.get_min_max_elevations(dem_file)
    clearance = altitude_msl.to("meter").magnitude - max_elev
    if clearance < safe_altitude_m:
        raise HyPlanValueError(
            f"Minimum clearance {clearance:.0f} m is below safe_altitude "
            f"{safe_altitude_m:.0f} m. Increase altitude_msl or safe_altitude."
        )
    logger.info(
        f"Altitude {altitude_msl:.0f}, max terrain {max_elev:.0f} m, clearance {clearance:.0f} m."
    )

    center_line = flight_line.FlightLine.center_length_azimuth(
        lat=lat0, lon=lon0, length=box_length, az=azimuth,
        altitude_msl=altitude_msl,
    )
    edge_line = center_line.offset_across(ureg.Quantity(-box_width_m / 2, "meter"))
    flight_level = altitude_to_flight_level(altitude_msl)

    lines          = []
    current_offset = 0.0
    line_index     = 0

    while current_offset <= box_width_m:
        candidate  = edge_line.offset_across(ureg.Quantity(current_offset, "meter"))
        swath_poly = generate_swath_polygon(candidate, instrument, dem_file=dem_file)
        widths     = calculate_swath_widths(swath_poly)
        min_swath  = widths["min_width"]

        step = min_swath * (1.0 - overlap / 100.0)
        if step <= 0:
            logger.warning(
                f"Non-positive swath step ({step:.1f} m) at offset {current_offset:.0f} m. "
                "Terrain may be too close to aircraft. Stopping walk."
            )
            break

        if clip_to_polygon:
            clipped = candidate.clip_to_polygon(polygon)
            output_segments = (
                [seg for seg in clipped
                 if seg.length.to("meter").magnitude >= min_line_length_m]
                if clipped else []
            )
        else:
            output_segments = [candidate]

        for seg in output_segments:
            seg.site_name = f"{box_name}_L{line_index + start_numbering:02d}_{flight_level}"
            if alternate_direction and line_index % 2 == 1:
                seg = seg.reverse()
            lines.append(seg)

        current_offset += step
        line_index += 1

    if not lines:
        logger.warning("No flight lines generated. Check polygon coverage and altitude.")

    return lines


def _generate_box_dem(
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length_m: float,
    box_width_m: float,
) -> str:
    """Generate a DEM file covering the survey box.

    Computes the four corners of the box and generates a merged DEM
    that covers the entire area.

    Args:
        lat0: Center latitude in decimal degrees.
        lon0: Center longitude in decimal degrees.
        azimuth: Box orientation in degrees from true north.
        box_length_m: Along-track box length in meters.
        box_width_m: Across-track box width in meters.

    Returns:
        Path to the generated DEM file.
    """
    half_len = box_length_m / 2
    half_wid = box_width_m / 2

    corner_lats = []
    corner_lons = []
    for along_az in (azimuth, azimuth + 180):
        for across_az in (azimuth + 90, azimuth - 90):
            lat, lon = pymap3d.vincenty.vreckon(
                lat0, lon0, half_len, along_az
            )
            lat, lon = pymap3d.vincenty.vreckon(
                lat, lon, half_wid, across_az
            )
            corner_lats.append(lat)
            corner_lons.append(lon)

    return terrain.generate_demfile(
        np.array(corner_lats), wrap_to_180(np.array(corner_lons))
    )


def _compute_altitude_msl(
    instrument: LineScanner,
    pixel_size: Quantity,
    dem_file: str,
) -> Quantity:
    """Compute the MSL altitude that achieves the desired pixel size over terrain.

    Sets altitude so the nadir GSD equals ``pixel_size`` at the lowest
    terrain point in the DEM, guaranteeing the pixel size requirement
    everywhere in the box.

    Args:
        instrument: A LineScanner with ``altitude_agl_for_ground_sample_distance``.
        pixel_size: Desired ground sample distance.
        dem_file: Path to the DEM file covering the survey area.

    Returns:
        Flight altitude MSL as a Quantity.
    """
    min_elev, _ = terrain.get_min_max_elevations(dem_file)
    altitude_agl = instrument.altitude_agl_for_ground_sample_distance(pixel_size)
    return altitude_agl + ureg.Quantity(float(min_elev), "meter")


def box_around_center_terrain(
    instrument: LineScanner,
    pixel_size: Quantity,
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length: Quantity,
    box_width: Quantity,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    safe_altitude: Quantity = ureg.Quantity(300, "meter"),
    polygon: Optional[Polygon] = None,
    min_line_length: Quantity = ureg.Quantity(200, "meter"),
) -> List[flight_line.FlightLine]:
    """Create flight lines around a center point with terrain-aware spacing.

    Unlike ``box_around_center_line``, which uses a constant swath width,
    this function uses ray-terrain intersection to compute the actual swath
    width at each line position. Lines are spaced so that the minimum
    terrain-aware swath width guarantees the requested overlap.

    The flight altitude (MSL) is chosen so the nadir pixel size equals
    ``pixel_size`` at the lowest terrain point in the survey box.

    Args:
        instrument: A LineScanner sensor object.
        pixel_size: Desired nadir ground sample distance (e.g. ``ureg.Quantity(3, "meter")``).
        lat0: Latitude of the box center in decimal degrees.
        lon0: Longitude of the box center in decimal degrees.
        azimuth: Orientation of the box in degrees from true north.
        box_length: Along-track length of the box.
        box_width: Across-track width of the box.
        box_name: Name prefix for flight lines.
        start_numbering: Starting number for flight line naming.
        overlap: Percentage overlap between adjacent swaths (0–100).
        alternate_direction: Whether to alternate flight line directions.
        safe_altitude: Minimum clearance above ground level.
        polygon: Optional polygon to clip flight lines to.
        min_line_length: Minimum flight line length after clipping.

    Returns:
        A list of FlightLine objects with terrain-adjusted spacing.

    Raises:
        HyPlanValueError: If inputs fail validation or terrain is too high
            for the requested safe altitude.
    """
    _validate_inputs(
        box_length=box_length,
        box_width=box_width,
        overlap=overlap,
        azimuth=azimuth,
        polygon=polygon,
    )
    azimuth = wrap_to_180(azimuth)

    if not isinstance(instrument, LineScanner):
        raise HyPlanValueError("instrument must be a LineScanner instance.")

    box_length_m = box_length.to("meter").magnitude
    box_width_m = box_width.to("meter").magnitude
    safe_altitude_m = safe_altitude.to("meter").magnitude
    min_line_length_m = min_line_length.to("meter").magnitude

    # 1. Generate DEM covering the survey area
    dem_file = _generate_box_dem(lat0, lon0, azimuth, box_length_m, box_width_m)

    # 2. Compute altitude MSL for desired pixel size over terrain
    altitude_msl = _compute_altitude_msl(instrument, pixel_size, dem_file)

    # Check safe altitude against highest terrain
    _, max_elev = terrain.get_min_max_elevations(dem_file)
    clearance = altitude_msl.to("meter").magnitude - max_elev
    if clearance < safe_altitude_m:
        raise HyPlanValueError(
            f"Minimum clearance {clearance:.0f} m is below the safe altitude "
            f"of {safe_altitude_m:.0f} m. Increase pixel_size or safe_altitude."
        )

    logger.info(f"Terrain-adjusted altitude: {altitude_msl:.0f}")

    # 3. Create the center line and left-edge reference line
    center_line = flight_line.FlightLine.center_length_azimuth(
        lat=lat0, lon=lon0, length=box_length, az=azimuth,
        altitude_msl=altitude_msl,
    )
    edge_line = center_line.offset_across(ureg.Quantity(-box_width_m / 2, "meter"))

    flight_level = altitude_to_flight_level(altitude_msl)

    # 4. Walk across the box, placing lines with terrain-aware spacing
    lines = []
    current_offset = 0.0
    line_index = 0

    while current_offset <= box_width_m:
        candidate = edge_line.offset_across(ureg.Quantity(current_offset, "meter"))

        # Compute terrain-aware swath width via ray-terrain intersection
        swath_poly = generate_swath_polygon(
            candidate, instrument, dem_file=dem_file
        )
        widths = calculate_swath_widths(swath_poly)
        min_swath = widths["min_width"]

        step = min_swath * (1.0 - overlap / 100.0)
        if step <= 0:
            logger.warning(
                f"Swath step is non-positive ({step:.1f} m) at offset "
                f"{current_offset:.0f} m. Terrain may be too close to aircraft."
            )
            break

        # Clip to polygon if provided
        if polygon:
            clipped = candidate.clip_to_polygon(polygon)
            if clipped:
                output_segments = [
                    seg for seg in clipped
                    if seg.length.to("meter").magnitude >= min_line_length_m
                ]
            else:
                output_segments = []
        else:
            output_segments = [candidate]

        # Name and collect output lines
        for seg in output_segments:
            seg.site_name = (
                f"{box_name}_L{line_index + start_numbering:02d}_{flight_level}"
            )
            if alternate_direction and line_index % 2 == 1:
                seg = seg.reverse()
            lines.append(seg)

        current_offset += step
        line_index += 1

    if not lines:
        logger.warning("No flight lines were generated.")

    return lines