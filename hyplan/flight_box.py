"""Flight box generation — parallel flight lines covering a geographic area.

Provides functions to lay out parallel, evenly-spaced flight lines over a
study area.  Lines can be generated from a center line and swath overlap
(:func:`box_around_center_line`), from an arbitrary polygon boundary
(:func:`box_around_polygon`), or with terrain-aware spacing
(:func:`box_around_polygon_terrain`, :func:`box_around_center_terrain`).
All planning functions accept any sensor that implements ``swath_width()``
and ``swath_offset_angles()`` — including ``LineScanner``, ``SidelookingRadar``,
and ``LVIS``.  The utility :func:`altitude_msl_for_pixel_size` is provided
for ``LineScanner`` users who need to derive flight altitude from a target GSD.
"""

import numpy as np
import pymap3d.vincenty
from typing import Optional, List, Callable, Dict, Union
from pint import Quantity
from shapely.geometry import Polygon
import logging

from . import flight_line
from . import terrain
from .instruments import LineScanner, ScanningSensor
from .swath import generate_swath_polygon, calculate_swath_widths
from .units import ureg, altitude_to_flight_level
from .geometry import (
    wrap_to_180,
    rotated_rectangle,
    minimum_rotated_rectangle,
    buffer_polygon_along_azimuth,
    rectangle_dimensions,
    _validate_polygon,
)
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
    rules: Dict[str, Callable[[Union[float, Quantity, Polygon, bool, None]], Optional[bool]]] = {
        'altitude': lambda x: isinstance(x, (float, Quantity)) and x > 0,
        'box_length': lambda x: isinstance(x, (float, Quantity)) and x > 0,
        'box_width': lambda x: isinstance(x, (float, Quantity)) and x > 0,
        'overlap': lambda x: isinstance(x, (float, int)) and 0 <= x <= 100,
        'starting_point': lambda x: x in {"edge", "center"},
        'azimuth': lambda x: isinstance(x, float),
        'polygon': lambda x: x is None or _validate_polygon(x),
        'clip_to_polygon': lambda x: isinstance(x, bool),
    }

    for key, value in kwargs.items():
        if key in {'altitude', 'box_length', 'box_width'}:
            # Validate and process length-related parameters
            if isinstance(value, Quantity):
                if not value.check("[length]"):
                    raise HyPlanValueError(f"Invalid unit for '{key}': Expected a length unit. Got {value.dimensionality}.")
                value = value.m_as("meter")  # Convert to meters
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
    instrument: ScanningSensor,
    altitude_msl: Quantity,
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length: Quantity,
    box_width: Quantity,
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
    if not isinstance(swath, Quantity):
        swath = ureg.Quantity(swath, "meter")
    if swath <= 0:
        raise HyPlanValueError(f"Invalid swath width {swath}. Must be positive.")

    swath_spacing = swath * (1 - (overlap / 100))
    if swath_spacing <= 0:
        raise HyPlanValueError(f"Invalid swath spacing {swath_spacing}. Adjust overlap or instrument parameters.")

    if polygon:
        along_track_buffer = 2000.0
        polygon = buffer_polygon_along_azimuth(polygon, along_track_buffer, swath.magnitude/2, azimuth)
        box_length += ureg.Quantity(along_track_buffer, "meter")  # type: ignore[misc]

    nlines = max(1, int(np.ceil((box_width / swath_spacing).m_as("dimensionless"))))

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
    instrument: ScanningSensor,
    altitude_msl: Quantity,
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

    lat0, lon0, azimuth, length_m, width_m = rectangle_dimensions(bounding_box, azimuth)
    box_length = length_m * ureg.meter
    box_width = width_m * ureg.meter

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
    instrument: ScanningSensor,
    altitude_msl: Quantity,
    polygon: Polygon,
    azimuth: Optional[float] = None,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    clip_to_polygon: bool = True,
    clip_polygon: Optional[Polygon] = None,
    safe_altitude: Quantity = ureg.Quantity(300, "meter"),
    min_line_length: Quantity = ureg.Quantity(200, "meter"),
    target_agl: Optional[Quantity] = None,
) -> List[flight_line.FlightLine]:
    """Generate terrain-aware flight lines covering a polygon.

    Works for both ``LineScanner`` and ``SidelookingRadar`` sensors.
    Supports two altitude modes:

    **Mode 2 (default):** A fixed ``altitude_msl`` is used for every flight
    line.  Line spacing is computed by ray-terrain intersection at each
    candidate line position, ensuring the requested overlap is maintained even
    over variable terrain.

    **Mode 3:** Pass ``target_agl`` instead of relying solely on
    ``altitude_msl``.  Each flight line is assigned an individual altitude
    derived from the mean terrain elevation along its nadir track plus
    ``target_agl``, so GSD and overlap remain stable across mountainous
    terrain (Zhao et al. 2021).  Raise :class:`~hyplan.exceptions.HyPlanValueError`
    if both ``altitude_msl`` and ``target_agl`` are provided.

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
        altitude_msl: Flight altitude above mean sea level (Mode 2).
            Used as a reference altitude for box geometry in Mode 3.
        polygon: Study-area boundary polygon (WGS84 lon/lat).
        azimuth: Flight-line orientation in degrees from true north.
            If ``None``, uses the minimum rotated rectangle of the
            polygon.
        box_name: Prefix for flight-line site names.
        start_numbering: First line number used in site names.
        overlap: Swath overlap percentage between adjacent lines (0–100).
        alternate_direction: Reverse every other line direction.
        clip_to_polygon: Clip lines to the polygon boundary.
        clip_polygon: If provided, clip lines to this polygon instead of
            ``polygon``. Useful when the bounding box and the clip boundary
            differ (e.g. when delegating from :func:`box_around_center_terrain`).
        safe_altitude: Minimum required clearance above terrain.  In Mode 2
            this is checked globally; in Mode 3 it is checked per line against
            the maximum terrain elevation along that line's nadir track.
        min_line_length: Drop clipped segments shorter than this value.
        target_agl: Desired altitude above ground level for Mode 3.  When
            provided each flight line receives an individual ``altitude_msl``
            computed as ``mean nadir terrain + target_agl``.  Cannot be used
            together with a custom ``altitude_msl`` — raise
            :class:`~hyplan.exceptions.HyPlanValueError` if both are supplied.

    Returns:
        List of :class:`~hyplan.flight_line.FlightLine` objects with
        terrain-aware spacing.  In Mode 3 each line carries a distinct
        ``altitude_msl`` derived from local terrain.

    Raises:
        HyPlanValueError: For invalid inputs, conflicting altitude arguments,
            or insufficient terrain clearance.
    """
    if not isinstance(polygon, Polygon):
        raise HyPlanValueError("polygon must be a Shapely Polygon.")
    if not isinstance(instrument, ScanningSensor):
        raise HyPlanValueError(
            "instrument must satisfy the ScanningSensor protocol "
            "(swath_width(altitude_agl), swath_offset_angles(), half_angle)."
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

    lat0, lon0, azimuth, length_m, width_m = rectangle_dimensions(bounding_box, azimuth)
    box_length = length_m * ureg.meter
    box_width = width_m * ureg.meter

    logger.info(
        f"Bounding box: center=({lat0:.6f}, {lon0:.6f}), az={azimuth:.2f}°, "
        f"length={box_length.magnitude:.0f} m, width={box_width.magnitude:.0f} m."
    )

    box_length_m      = box_length.m_as("meter")
    box_width_m       = box_width.m_as("meter")
    safe_altitude_m   = safe_altitude.m_as("meter")
    min_line_length_m = min_line_length.m_as("meter")
    mode3             = target_agl is not None
    target_agl_m      = target_agl.m_as("meter") if mode3 else None  # type: ignore[union-attr]

    dem_file = _generate_box_dem(lat0, lon0, azimuth, box_length_m, box_width_m)

    if not mode3:
        # Mode 2: single fixed altitude — check global terrain clearance up front
        _, max_elev = terrain.get_min_max_elevations(dem_file)
        clearance = altitude_msl.m_as("meter") - max_elev
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

    lines          = []
    current_offset = 0.0
    line_index     = 0

    while current_offset <= box_width_m:
        candidate = edge_line.offset_across(ureg.Quantity(current_offset, "meter"))

        if mode3:
            # Mode 3: compute per-line altitude from mean nadir terrain + target AGL
            elev_stats = terrain.terrain_elevation_along_track(candidate, dem_file)
            line_alt_m = elev_stats["mean"] + target_agl_m
            candidate.altitude_msl = ureg.Quantity(line_alt_m, "meter")
            clearance = line_alt_m - elev_stats["max"]
            if clearance < safe_altitude_m:
                logger.warning(
                    f"Line {line_index + start_numbering:02d}: clearance {clearance:.0f} m "
                    f"is below safe_altitude {safe_altitude_m:.0f} m "
                    f"(mean terrain {elev_stats['mean']:.0f} m, "
                    f"max terrain {elev_stats['max']:.0f} m). "
                    "Consider increasing target_agl or safe_altitude."
                )
            logger.info(
                f"Line {line_index + start_numbering:02d}: mean terrain "
                f"{elev_stats['mean']:.0f} m, altitude_msl {line_alt_m:.0f} m."
            )

        flight_level = altitude_to_flight_level(candidate.altitude_msl)

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

        effective_clip = clip_polygon if clip_polygon is not None else polygon
        if clip_to_polygon:
            clipped = candidate.clip_to_polygon(effective_clip)
            output_segments = (
                [seg for seg in clipped
                 if seg.length.m_as("meter") >= min_line_length_m]
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

    return terrain.generate_demfile(  # type: ignore[no-any-return]
        np.array(corner_lats), wrap_to_180(np.array(corner_lons))  # type: ignore[arg-type]
    )


def _rectangle_polygon(
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length_m: float,
    box_width_m: float,
) -> Polygon:
    """Build a Shapely Polygon for a georeferenced rectangle.

    Args:
        lat0: Center latitude in decimal degrees.
        lon0: Center longitude in decimal degrees.
        azimuth: Box orientation in degrees from true north.
        box_length_m: Along-track length in meters.
        box_width_m: Across-track width in meters.

    Returns:
        A Shapely Polygon with corners in WGS84 lon/lat.
    """
    half_len = box_length_m / 2
    half_wid = box_width_m / 2

    def _corner(along_az, across_az):
        lat, lon = pymap3d.vincenty.vreckon(lat0, lon0, half_len, along_az)
        lat, lon = pymap3d.vincenty.vreckon(float(lat), float(lon), half_wid, across_az)
        return (wrap_to_180(float(lon)), float(lat))

    return Polygon([
        _corner(azimuth,       azimuth + 90),
        _corner(azimuth,       azimuth - 90),
        _corner(azimuth + 180, azimuth - 90),
        _corner(azimuth + 180, azimuth + 90),
    ])


def altitude_msl_for_pixel_size(
    instrument: LineScanner,
    pixel_size: Quantity,
    dem_file: str,
) -> Quantity:
    """Compute the MSL altitude for a ``LineScanner`` to achieve a target pixel size.

    Sets altitude so the nadir GSD equals ``pixel_size`` at the lowest terrain
    point in the DEM, guaranteeing the pixel size requirement is met everywhere
    in the survey area.  Use this to compute ``altitude_msl`` before calling
    :func:`box_around_center_terrain` or :func:`box_around_polygon_terrain`.

    Args:
        instrument: A ``LineScanner`` with ``altitude_agl_for_ground_sample_distance``.
        pixel_size: Desired nadir ground sample distance.
        dem_file: Path to a DEM file covering the survey area (see
            :func:`hyplan.terrain.generate_demfile`).

    Returns:
        Flight altitude MSL as a Quantity.
    """
    min_elev, _ = terrain.get_min_max_elevations(dem_file)
    altitude_agl = instrument.altitude_agl_for_ground_sample_distance(pixel_size)
    return altitude_agl + ureg.Quantity(float(min_elev), "meter")  # type: ignore[no-any-return]


def box_around_center_terrain(
    instrument: ScanningSensor,
    altitude_msl: Quantity,
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
    target_agl: Optional[Quantity] = None,
) -> List[flight_line.FlightLine]:
    """Create terrain-aware flight lines around an explicit center point.

    Constructs a rectangular survey box from the center coordinates and
    dimensions, then delegates to :func:`box_around_polygon_terrain` for
    terrain-aware line placement.  Works with any sensor that implements
    ``swath_width()`` and ``swath_offset_angles()`` — including
    ``LineScanner``, ``SidelookingRadar``, and ``LVIS``.

    ``LineScanner`` users who need altitude derived from a target pixel size
    should call :func:`altitude_msl_for_pixel_size` first.

    Args:
        instrument: Sensor with ``swath_width(altitude_agl)`` and
            ``swath_offset_angles()`` methods.
        altitude_msl: Flight altitude above mean sea level (Mode 2).
            Used as a reference altitude for box geometry in Mode 3.
        lat0: Latitude of the box center in decimal degrees.
        lon0: Longitude of the box center in decimal degrees.
        azimuth: Orientation of the box in degrees from true north.
        box_length: Along-track length of the box.
        box_width: Across-track width of the box.
        box_name: Name prefix for flight lines.
        start_numbering: Starting number for flight line naming.
        overlap: Percentage overlap between adjacent swaths (0–100).
        alternate_direction: Whether to alternate flight line directions.
        safe_altitude: Minimum clearance above terrain (Mode 2: global;
            Mode 3: per-line against local max terrain).
        polygon: Optional polygon to clip flight lines to.
        min_line_length: Minimum flight line length after clipping.
        target_agl: Desired altitude above ground level for Mode 3.
            When provided each flight line receives an individual
            ``altitude_msl`` computed as ``mean nadir terrain + target_agl``.

    Returns:
        A list of FlightLine objects with terrain-aware spacing.

    Raises:
        HyPlanValueError: If inputs fail validation or altitude clearance
            is insufficient.
    """
    _validate_inputs(
        altitude=altitude_msl,
        box_length=box_length,
        box_width=box_width,
        overlap=overlap,
        azimuth=azimuth,
        polygon=polygon,
    )
    azimuth = wrap_to_180(azimuth)  # type: ignore[assignment]

    rect = _rectangle_polygon(
        lat0, lon0, azimuth,
        box_length.m_as("meter"),
        box_width.m_as("meter"),
    )

    return box_around_polygon_terrain(
        instrument=instrument,
        altitude_msl=altitude_msl,
        polygon=rect,
        azimuth=azimuth,
        box_name=box_name,
        start_numbering=start_numbering,
        overlap=overlap,
        alternate_direction=alternate_direction,
        clip_to_polygon=polygon is not None,
        clip_polygon=polygon,
        safe_altitude=safe_altitude,
        min_line_length=min_line_length,
        target_agl=target_agl,
    )