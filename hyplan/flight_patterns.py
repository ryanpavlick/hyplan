"""Flight pattern generators for atmospheric sampling.

Each generator returns an ordered list of Waypoint objects with
segment_type="pattern" so that compute_flight_plan() labels the
connecting legs accordingly.
"""

from typing import List, Optional, Union

import numpy as np
import pymap3d.vincenty

from .units import ureg
from .exceptions import HyPlanValueError
from .geometry import wrap_to_180, wrap_to_360
from .waypoint import Waypoint
from .flight_line import FlightLine

__all__ = [
    "racetrack",
    "rosette",
    "polygon",
    "sawtooth",
    "spiral",
    "flight_lines_to_waypoint_path",
]


def _to_length_quantity(value, label="value"):
    """Convert a length value to a pint Quantity.

    Args:
        value: Float (interpreted as meters) or pint Quantity with length units.
        label: Parameter name for error messages.

    Returns:
        pint Quantity in meters.
    """
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), "meter")
    if hasattr(value, 'units') and value.check('[length]'):
        return value.to(ureg.meter)
    raise TypeError(f"{label} must be a float (meters) or a pint Quantity with length units")


def racetrack(
    center: tuple,
    heading: float,
    altitude: Union[float, "ureg.Quantity"],
    leg_length: Union[float, "ureg.Quantity"],
    n_legs: int = 1,
    offset: Union[float, "ureg.Quantity", list] = 0,
    altitudes: Optional[list] = None,
    stack_altitudes: Optional[list] = None,
) -> List[Waypoint]:
    """Generate parallel out-and-back legs.

    Handles racetracks, lawnmowers, bowling alleys, vertical walls,
    and stacked patterns through parameter variation.

    Args:
        center: (lat, lon) center point of the pattern.
        heading: Bearing of the legs in degrees from north.
        altitude: Altitude MSL for all legs (overridden by altitudes if given).
        leg_length: Length of each leg.
        n_legs: Number of parallel legs (default 1).
        offset: Crosstrack spacing between legs. Scalar = uniform spacing.
            List = per-leg offsets from center (length must equal n_legs).
        altitudes: Per-leg altitudes. Length must equal n_legs.
            Overrides altitude for each leg. Enables vertical walls.
        stack_altitudes: Repeat the entire pattern at each altitude in this list.
            Produces n_legs * len(stack_altitudes) total legs.

    Returns:
        List of Waypoint objects with segment_type="pattern".
    """
    center_lat, center_lon = center
    half_len_m = _to_length_quantity(leg_length, "leg_length").magnitude / 2.0
    if half_len_m <= 0:
        raise HyPlanValueError("leg_length must be positive")
    heading = float(heading)

    # Build per-leg crosstrack offsets (meters from center)
    if isinstance(offset, list):
        if len(offset) != n_legs:
            raise ValueError(f"offset list length ({len(offset)}) must equal n_legs ({n_legs})")
        offsets_m = [_to_length_quantity(o, "offset").magnitude for o in offset]
    else:
        spacing_m = _to_length_quantity(offset, "offset").magnitude
        # Center the legs: offsets are symmetric around 0
        offsets_m = [spacing_m * (i - (n_legs - 1) / 2.0) for i in range(n_legs)]

    # Build per-leg altitudes
    if stack_altitudes is not None:
        # Repeat pattern at each stack altitude
        stack_alts = [_to_length_quantity(a, "stack_altitudes") for a in stack_altitudes]
        all_offsets = []
        all_alts = []
        for sa in stack_alts:
            all_offsets.extend(offsets_m)
            if altitudes is not None:
                # per-leg altitudes override stack altitude
                all_alts.extend([_to_length_quantity(a, "altitudes") for a in altitudes])
            else:
                all_alts.extend([sa] * n_legs)
        offsets_m = all_offsets
        leg_alts = all_alts
    elif altitudes is not None:
        if len(altitudes) != n_legs:
            raise ValueError(f"altitudes length ({len(altitudes)}) must equal n_legs ({n_legs})")
        leg_alts = [_to_length_quantity(a, "altitudes") for a in altitudes]
    else:
        default_alt = _to_length_quantity(altitude, "altitude")
        leg_alts = [default_alt] * n_legs

    # Perpendicular direction for crosstrack offsets
    perp_az = heading + 90.0

    waypoints = []
    for i, (ct_offset, alt) in enumerate(zip(offsets_m, leg_alts)):
        # Compute the center of this leg (offset from pattern center)
        if ct_offset != 0:
            leg_center_lat, leg_center_lon = pymap3d.vincenty.vreckon(
                center_lat, center_lon, abs(ct_offset),
                perp_az if ct_offset >= 0 else perp_az - 180
            )
            leg_center_lon = wrap_to_180(leg_center_lon)
        else:
            leg_center_lat, leg_center_lon = center_lat, center_lon

        # Alternate direction: odd-indexed legs go in reverse
        if i % 2 == 0:
            fwd_az = heading
            rev_az = (heading + 180.0) % 360.0
        else:
            fwd_az = (heading + 180.0) % 360.0
            rev_az = heading

        # Start point: half_len behind center along fwd_az
        start_lat, start_lon = pymap3d.vincenty.vreckon(
            leg_center_lat, leg_center_lon, half_len_m, rev_az
        )
        start_lon = wrap_to_180(start_lon)

        # End point: half_len ahead of center along fwd_az
        end_lat, end_lon = pymap3d.vincenty.vreckon(
            leg_center_lat, leg_center_lon, half_len_m, fwd_az
        )
        end_lon = wrap_to_180(end_lon)

        waypoints.append(Waypoint(
            latitude=float(start_lat),
            longitude=float(start_lon),
            heading=wrap_to_360(fwd_az),
            altitude_msl=alt,
            name=f"Leg{i+1}_start",
            segment_type="pattern",
        ))
        waypoints.append(Waypoint(
            latitude=float(end_lat),
            longitude=float(end_lon),
            heading=wrap_to_360(fwd_az),
            altitude_msl=alt,
            name=f"Leg{i+1}_end",
            segment_type="pattern_turn",
        ))

    return waypoints


def rosette(
    center: tuple,
    heading: float,
    altitude: Union[float, "ureg.Quantity"],
    radius: Union[float, "ureg.Quantity"],
    n_lines: int = 3,
    angles: Optional[List[float]] = None,
) -> List[Waypoint]:
    """Generate radial lines through a center point.

    Creates a FlightLine centered on the point and rotates it for each
    line angle. Each line is a full diameter crossing through center.
    Produces 2 * n_lines waypoints.

    Args:
        center: (lat, lon) center point.
        heading: Bearing of the first line in degrees from north.
        altitude: Altitude MSL.
        radius: Half-length of each line (center to tip).
        n_lines: Number of lines (default 3). Lines are spaced at
            180/n_lines degree intervals.
        angles: Explicit angles for each line (degrees from north).
            Overrides n_lines and heading.

    Returns:
        List of Waypoint objects with segment_type="pattern".
    """
    center_lat, center_lon = center
    diameter = _to_length_quantity(radius, "radius") * 2
    if diameter.magnitude <= 0:
        raise HyPlanValueError("radius must be positive")
    alt = _to_length_quantity(altitude, "altitude")

    if angles is not None:
        line_angles = [float(a) for a in angles]
    else:
        line_angles = [heading + i * (180.0 / n_lines) for i in range(n_lines)]

    # Create the base flight line centered on the point along the first angle
    base_line = FlightLine.center_length_azimuth(
        lat=center_lat, lon=center_lon,
        length=diameter, az=line_angles[0],
        altitude_msl=alt,
    )

    waypoints = []
    for i, angle in enumerate(line_angles):
        # Rotate the base line to the desired angle
        rotation = angle - line_angles[0]
        fl = base_line.rotate_around_midpoint(rotation) if rotation != 0 else base_line

        # Use the FlightLine's waypoints directly, overriding name and segment_type
        wp1 = fl.waypoint1
        wp2 = fl.waypoint2
        waypoints.append(Waypoint(
            latitude=wp1.latitude, longitude=wp1.longitude,
            heading=wp1.heading, altitude_msl=wp1.altitude_msl,
            name=f"L{i+1}_start", segment_type="pattern",
        ))
        waypoints.append(Waypoint(
            latitude=wp2.latitude, longitude=wp2.longitude,
            heading=wp2.heading, altitude_msl=wp2.altitude_msl,
            name=f"L{i+1}_end", segment_type="pattern_turn",
        ))

    return waypoints


def polygon(
    center: tuple,
    heading: float,
    altitude: Union[float, "ureg.Quantity"],
    radius: Union[float, "ureg.Quantity"],
    n_sides: int = 4,
    aspect_ratio: float = 1.0,
    closed: bool = True,
) -> List[Waypoint]:
    """Generate a regular polygon perimeter (or stretched polygon).

    Replaces separate rectangle() and circle() generators.

    Args:
        center: (lat, lon) center point.
        heading: Bearing to the first vertex in degrees from north.
        altitude: Altitude MSL.
        radius: Circumscribed circle radius (center to vertex).
        n_sides: Number of sides (3=triangle, 4=square, 36=circle). Default 4.
        aspect_ratio: Stretch factor along the heading axis. 1.0 = regular
            polygon. 2.0 = twice as long as wide. Default 1.0.
        closed: If True, last waypoint repeats the first to close the loop.
            Default True.

    Returns:
        List of Waypoint objects with segment_type="pattern".
    """
    center_lat, center_lon = center
    radius_m = _to_length_quantity(radius, "radius").magnitude
    if radius_m <= 0:
        raise HyPlanValueError("radius must be positive")
    alt = _to_length_quantity(altitude, "altitude")
    heading_rad = np.radians(heading)

    # Compute vertex positions on the circumscribed ellipse
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    waypoints = []
    vertex_coords = []

    for angle in angles:
        # Unit circle position
        x = np.sin(angle)  # crosstrack (east-like in heading frame)
        y = np.cos(angle)  # along-track (north-like in heading frame)

        # Apply aspect ratio: stretch along heading axis
        y *= aspect_ratio

        # Convert to bearing and distance from center
        # Rotate from heading-frame to geographic bearing
        dx = x * np.cos(heading_rad) + y * np.sin(heading_rad)
        dy = -x * np.sin(heading_rad) + y * np.cos(heading_rad)

        dist = radius_m * np.sqrt(dx**2 + dy**2)
        bearing = np.degrees(np.arctan2(dx, dy))  # atan2(east, north) = bearing

        vlat, vlon = pymap3d.vincenty.vreckon(center_lat, center_lon, dist, bearing)
        vlon = wrap_to_180(vlon)
        vertex_coords.append((float(vlat), float(vlon)))

    # Compute tangent headings at each vertex
    n = len(vertex_coords)
    for i in range(n):
        lat_i, lon_i = vertex_coords[i]
        lat_next, lon_next = vertex_coords[(i + 1) % n]
        _, az = pymap3d.vincenty.vdist(lat_i, lon_i, lat_next, lon_next)
        tangent_heading = wrap_to_360(float(az))

        waypoints.append(Waypoint(
            latitude=lat_i,
            longitude=lon_i,
            heading=tangent_heading,
            altitude_msl=alt,
            name=f"V{i+1}",
            segment_type="pattern",
        ))

    if closed:
        # Repeat first vertex to close the loop
        waypoints.append(Waypoint(
            latitude=waypoints[0].latitude,
            longitude=waypoints[0].longitude,
            heading=waypoints[0].heading,
            altitude_msl=alt,
            name="V1",
            segment_type="pattern",
        ))

    return waypoints


def sawtooth(
    center: tuple,
    heading: float,
    altitude_min: Union[float, "ureg.Quantity"],
    altitude_max: Union[float, "ureg.Quantity"],
    leg_length: Union[float, "ureg.Quantity"],
    n_cycles: int = 1,
) -> List[Waypoint]:
    """Generate an oscillating altitude profile along a straight track.

    Args:
        center: (lat, lon) center point of the track.
        heading: Bearing of the track in degrees from north.
        altitude_min: Bottom of each oscillation.
        altitude_max: Top of each oscillation.
        leg_length: Total track length.
        n_cycles: Number of up-down cycles.

    Returns:
        List of Waypoint objects with segment_type="pattern".
    """
    center_lat, center_lon = center
    total_len_m = _to_length_quantity(leg_length, "leg_length").magnitude
    if total_len_m <= 0:
        raise HyPlanValueError("leg_length must be positive")
    alt_min = _to_length_quantity(altitude_min, "altitude_min")
    alt_max = _to_length_quantity(altitude_max, "altitude_max")
    heading = float(heading)

    n_points = 2 * n_cycles + 1
    half_len = total_len_m / 2.0

    # Start point: half_len behind center
    start_lat, start_lon = pymap3d.vincenty.vreckon(
        center_lat, center_lon, half_len, (heading + 180.0) % 360.0
    )
    start_lon = wrap_to_180(start_lon)

    # Spacing between waypoints along track
    spacing = total_len_m / (n_points - 1) if n_points > 1 else 0

    wp_heading = wrap_to_360(heading)
    waypoints = []

    for i in range(n_points):
        # Position along track
        dist_from_start = spacing * i
        if dist_from_start == 0:
            lat, lon = float(start_lat), float(start_lon)
        else:
            lat, lon = pymap3d.vincenty.vreckon(
                start_lat, start_lon, dist_from_start, heading
            )
            lon = wrap_to_180(lon)

        # Altitude alternates: starts at max, then min, max, min, ...
        alt = alt_max if i % 2 == 0 else alt_min

        waypoints.append(Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=wp_heading,
            altitude_msl=alt,
            name=f"ST{i+1}",
            segment_type="pattern_turn",
        ))

    return waypoints


def spiral(
    center: tuple,
    heading: float,
    altitude_start: Union[float, "ureg.Quantity"],
    altitude_end: Union[float, "ureg.Quantity"],
    radius: Union[float, "ureg.Quantity"],
    n_turns: float = 3.0,
    direction: str = "right",
    points_per_turn: int = 36,
) -> List[Waypoint]:
    """Generate a helical spiral pattern for vertical profiling.

    The aircraft flies a constant-radius circle while ascending or
    descending. Used for boundary layer profiling, aerosol vertical
    structure, and thermodynamic soundings.

    Args:
        center: (lat, lon) ground target point.
        heading: Bearing from center to the entry point (degrees from north).
            The aircraft enters tangent to the circle at this point.
        altitude_start: Altitude MSL at the start of the spiral.
        altitude_end: Altitude MSL at the end of the spiral.
        radius: Turn radius (distance from center to the flight path).
        n_turns: Number of complete revolutions (fractional OK). Default 3.
        direction: "right" (clockwise viewed from above) or "left"
            (counterclockwise). Default "right".
        points_per_turn: Number of waypoints per revolution. Default 36
            (one every 10 degrees).

    Returns:
        List of Waypoint objects with segment_type="pattern".

    Raises:
        HyPlanValueError: If n_turns <= 0 or points_per_turn < 3.
        ValueError: If direction is not "right" or "left".
    """
    if n_turns <= 0:
        raise HyPlanValueError("n_turns must be positive")
    if points_per_turn < 3:
        raise HyPlanValueError("points_per_turn must be at least 3")
    if direction not in ("right", "left"):
        raise ValueError(f"direction must be 'right' or 'left', got '{direction}'")

    center_lat, center_lon = center
    radius_m = _to_length_quantity(radius, "radius").magnitude
    if radius_m <= 0:
        raise HyPlanValueError("radius must be positive")
    alt_start = _to_length_quantity(altitude_start, "altitude_start")
    alt_end = _to_length_quantity(altitude_end, "altitude_end")

    # Total number of angular steps
    total_steps = int(n_turns * points_per_turn)
    if total_steps < 1:
        total_steps = 1

    # Angular increment per step (degrees)
    angle_step = 360.0 / points_per_turn
    if direction == "left":
        angle_step = -angle_step

    # Altitude interpolation (as pint Quantities)
    alt_start_m = alt_start.to(ureg.meter).magnitude
    alt_end_m = alt_end.to(ureg.meter).magnitude

    waypoints = []
    for i in range(total_steps + 1):
        # Bearing from center to this point on the circle
        bearing = heading + i * angle_step
        bearing = bearing % 360.0

        # Position on the circle
        lat, lon = pymap3d.vincenty.vreckon(
            center_lat, center_lon, radius_m, bearing
        )
        lon = wrap_to_180(lon)

        # Tangent heading (perpendicular to radial bearing)
        if direction == "right":
            tangent = wrap_to_360(bearing + 90.0)
        else:
            tangent = wrap_to_360(bearing - 90.0)

        # Altitude: linear interpolation
        frac = i / total_steps if total_steps > 0 else 0.0
        alt_m = alt_start_m + (alt_end_m - alt_start_m) * frac
        alt = ureg.Quantity(alt_m, "meter")

        waypoints.append(Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=tangent,
            altitude_msl=alt,
            name=f"SP{i+1}",
            segment_type="pattern",
        ))

    return waypoints


def flight_lines_to_waypoint_path(
    flight_lines: List[FlightLine],
    altitude: Union[float, "ureg.Quantity", None] = None,
) -> List[Waypoint]:
    """Convert a list of FlightLine objects into a connected waypoint path.

    Each flight line contributes its waypoint1 and waypoint2. No turn
    waypoints are inserted — compute_flight_plan() handles Dubins
    transitions between consecutive waypoints.

    Args:
        flight_lines: List of FlightLine objects (e.g. from box_around_polygon()).
        altitude: If provided, overrides the altitude on all waypoints.

    Returns:
        List of Waypoint objects with segment_type="pattern".
    """
    alt_override = _to_length_quantity(altitude, "altitude") if altitude is not None else None

    waypoints = []
    for i, fl in enumerate(flight_lines):
        for j, wp in enumerate([fl.waypoint1, fl.waypoint2]):
            alt = alt_override if alt_override is not None else wp.altitude_msl
            name = f"{fl.site_name or f'FL{i+1}'}_{['start', 'end'][j]}"
            seg_type = "pattern_turn" if j == 1 else "pattern"
            waypoints.append(Waypoint(
                latitude=wp.latitude, longitude=wp.longitude,
                heading=wp.heading, altitude_msl=alt,
                speed=wp.speed, name=name, segment_type=seg_type,
            ))

    return waypoints
