"""Flight pattern generators for atmospheric sampling.

Each generator returns a :class:`~hyplan.pattern.Pattern` object that
bundles the generated flight lines or waypoints with the parameters used
to produce them.  Line-based patterns (``racetrack``, ``rosette``) carry
``FlightLine`` objects.  Continuous-track patterns (``polygon``,
``sawtooth``, ``spiral``) carry ``Waypoint`` objects with
``segment_type="pattern"`` so ``compute_flight_plan()`` labels the
connecting legs accordingly.  ``compute_flight_plan`` accepts
:class:`Pattern` in its flight sequence and expands it inline.
"""

from typing import List, Optional, Union

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from .units import ureg
from .exceptions import HyPlanValueError, HyPlanTypeError
from .geometry import wrap_to_180, wrap_to_360
from .waypoint import Waypoint
from .flight_line import FlightLine
from .aircraft import Aircraft
from .pattern import Pattern
from .glint import GlintArc

__all__ = [
    "racetrack",
    "rosette",
    "polygon",
    "sawtooth",
    "spiral",
    "glint_arc",
    "flight_lines_to_waypoint_path",
    "coordinated_line",
]


def _to_length_quantity(value, label="value"):
    """Convert a length value to a pint Quantity."""
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), "meter")
    if hasattr(value, 'units') and value.check('[length]'):
        return value.to(ureg.meter)
    raise HyPlanTypeError(f"{label} must be a float (meters) or a pint Quantity with length units")


def _length_m(value, label="value") -> float:
    return _to_length_quantity(value, label).m_as(ureg.meter)


def _lines_dict(lines: List[FlightLine]) -> dict:
    """Key a list of FlightLines with sequential leg_N ids."""
    return {f"leg_{i+1}": fl for i, fl in enumerate(lines)}


def racetrack(
    center: tuple,
    heading: float,
    altitude: Union[float, "Quantity"],
    leg_length: Union[float, "Quantity"],
    n_legs: int = 1,
    offset: Union[float, "Quantity", list] = 0,
    altitudes: Optional[list] = None,
    stack_altitudes: Optional[list] = None,
    name: Optional[str] = None,
) -> Pattern:
    """Generate parallel out-and-back flight lines.

    Handles racetracks, lawnmowers, bowling alleys, vertical walls,
    and stacked patterns through parameter variation. Consecutive legs
    alternate direction so end-to-start transitions are short turns.

    Args:
        center: (lat, lon) center point of the pattern.
        heading: Bearing of the first leg in degrees from north.
        altitude: Altitude MSL for all legs (overridden by altitudes if given).
        leg_length: Length of each leg.
        n_legs: Number of parallel legs (default 1).
        offset: Crosstrack spacing between legs. Scalar = uniform spacing.
            List = per-leg offsets from center (length must equal n_legs).
        altitudes: Per-leg altitudes. Length must equal n_legs.
            Overrides altitude for each leg. Enables vertical walls.
        stack_altitudes: Repeat the entire pattern at each altitude in this list.
            Produces n_legs * len(stack_altitudes) total legs.
        name: Pattern display name (default "Racetrack").

    Returns:
        A line-based :class:`Pattern` (``kind="racetrack"``).
    """
    center_lat, center_lon = center
    leg_len_q = _to_length_quantity(leg_length, "leg_length")
    if leg_len_q.magnitude <= 0:
        raise HyPlanValueError("leg_length must be positive")
    heading = float(heading)

    # Per-leg crosstrack offsets (meters from pattern center)
    if isinstance(offset, list):
        if len(offset) != n_legs:
            raise HyPlanValueError(f"offset list length ({len(offset)}) must equal n_legs ({n_legs})")
        offsets_m = [_length_m(o, "offset") for o in offset]
    else:
        spacing_m = _length_m(offset, "offset")
        offsets_m = [spacing_m * (i - (n_legs - 1) / 2.0) for i in range(n_legs)]

    # Per-leg altitudes
    if stack_altitudes is not None:
        stack_alts = [_to_length_quantity(a, "stack_altitudes") for a in stack_altitudes]
        all_offsets = []
        all_alts = []
        for sa in stack_alts:
            all_offsets.extend(offsets_m)
            if altitudes is not None:
                all_alts.extend([_to_length_quantity(a, "altitudes") for a in altitudes])
            else:
                all_alts.extend([sa] * n_legs)
        offsets_m = all_offsets
        leg_alts = all_alts
    elif altitudes is not None:
        if len(altitudes) != n_legs:
            raise HyPlanValueError(f"altitudes length ({len(altitudes)}) must equal n_legs ({n_legs})")
        leg_alts = [_to_length_quantity(a, "altitudes") for a in altitudes]
    else:
        default_alt = _to_length_quantity(altitude, "altitude")
        leg_alts = [default_alt] * n_legs

    perp_az = heading + 90.0
    fwd_heading = heading % 360.0
    rev_heading = (heading + 180.0) % 360.0

    lines = []
    for i, (ct_offset, alt) in enumerate(zip(offsets_m, leg_alts)):
        if ct_offset != 0:
            leg_center_lat, leg_center_lon = pymap3d.vincenty.vreckon(
                center_lat, center_lon, abs(ct_offset),
                perp_az if ct_offset >= 0 else perp_az - 180,
            )
            leg_center_lon = wrap_to_180(leg_center_lon)
        else:
            leg_center_lat, leg_center_lon = center_lat, center_lon

        leg_az = fwd_heading if i % 2 == 0 else rev_heading

        lines.append(FlightLine.center_length_azimuth(
            lat=float(leg_center_lat),
            lon=float(leg_center_lon),
            length=leg_len_q,
            az=leg_az,
            altitude_msl=alt,
            site_name=f"Leg{i+1}",
        ))

    params = {
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "heading": heading,
        "altitude_msl_m": _length_m(altitude, "altitude"),
        "leg_length_m": leg_len_q.m_as(ureg.meter),
        "n_legs": int(n_legs),
        "offset_m": (
            [_length_m(o, "offset") for o in offset]
            if isinstance(offset, list) else _length_m(offset, "offset")
        ),
    }
    if altitudes is not None:
        params["altitudes_m"] = [_length_m(a, "altitudes") for a in altitudes]
    if stack_altitudes is not None:
        params["stack_altitudes_m"] = [_length_m(a, "stack_altitudes") for a in stack_altitudes]

    return Pattern(
        kind="racetrack",
        name=name or "Racetrack",
        params=params,
        lines=_lines_dict(lines),
    )


def rosette(
    center: tuple,
    heading: float,
    altitude: Union[float, "Quantity"],
    radius: Union[float, "Quantity"],
    n_lines: int = 3,
    angles: Optional[List[float]] = None,
    name: Optional[str] = None,
) -> Pattern:
    """Generate radial flight lines through a center point.

    Creates a FlightLine centered on the point along the first angle, then
    rotates it for each subsequent line angle. Each line is a full diameter
    crossing through center.

    Args:
        center: (lat, lon) center point.
        heading: Bearing of the first line in degrees from north.
        altitude: Altitude MSL.
        radius: Half-length of each line (center to tip).
        n_lines: Number of lines (default 3). Lines are spaced at
            180/n_lines degree intervals.
        angles: Explicit angles for each line (degrees from north).
            Overrides n_lines and heading.
        name: Pattern display name (default "Rosette").

    Returns:
        A line-based :class:`Pattern` (``kind="rosette"``).
    """
    center_lat, center_lon = center
    radius_q = _to_length_quantity(radius, "radius")
    diameter = radius_q * 2
    if diameter.magnitude <= 0:
        raise HyPlanValueError("radius must be positive")
    alt = _to_length_quantity(altitude, "altitude")

    if angles is not None:
        line_angles = [float(a) for a in angles]
    else:
        line_angles = [heading + i * (180.0 / n_lines) for i in range(n_lines)]

    base_line = FlightLine.center_length_azimuth(
        lat=center_lat, lon=center_lon,
        length=diameter, az=line_angles[0],
        altitude_msl=alt,
        site_name="L1",
    )

    lines = []
    for i, angle in enumerate(line_angles):
        rotation = angle - line_angles[0]
        if rotation == 0:
            fl = base_line
        else:
            fl = base_line.rotate_around_midpoint(rotation)
            fl.site_name = f"L{i+1}"
        lines.append(fl)

    params = {
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "heading": float(heading),
        "altitude_msl_m": alt.m_as(ureg.meter),
        "radius_m": radius_q.m_as(ureg.meter),
        "n_lines": int(n_lines),
    }
    if angles is not None:
        params["angles"] = [float(a) for a in angles]

    return Pattern(
        kind="rosette",
        name=name or "Rosette",
        params=params,
        lines=_lines_dict(lines),
    )


def polygon(
    center: tuple,
    heading: float,
    altitude: Union[float, "Quantity"],
    radius: Union[float, "Quantity"],
    n_sides: int = 4,
    aspect_ratio: float = 1.0,
    closed: bool = True,
    name: Optional[str] = None,
) -> Pattern:
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
        name: Pattern display name (default "Polygon").

    Returns:
        A waypoint-based :class:`Pattern` (``kind="polygon"``).
    """
    center_lat, center_lon = center
    radius_q = _to_length_quantity(radius, "radius")
    radius_m = radius_q.magnitude
    if radius_m <= 0:
        raise HyPlanValueError("radius must be positive")
    alt = _to_length_quantity(altitude, "altitude")
    heading_rad = np.radians(heading)

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    waypoints = []
    vertex_coords = []

    for angle in angles:
        x = np.sin(angle)
        y = np.cos(angle)
        y *= aspect_ratio

        dx = x * np.cos(heading_rad) + y * np.sin(heading_rad)
        dy = -x * np.sin(heading_rad) + y * np.cos(heading_rad)

        dist = radius_m * np.sqrt(dx**2 + dy**2)
        bearing = np.degrees(np.arctan2(dx, dy))

        vlat, vlon = pymap3d.vincenty.vreckon(center_lat, center_lon, dist, bearing)
        vlon = wrap_to_180(vlon)
        vertex_coords.append((float(vlat), float(vlon)))

    n = len(vertex_coords)
    for i in range(n):
        lat_i, lon_i = vertex_coords[i]
        lat_next, lon_next = vertex_coords[(i + 1) % n]
        _, az = pymap3d.vincenty.vdist(lat_i, lon_i, lat_next, lon_next)
        tangent_heading = wrap_to_360(float(az))

        waypoints.append(Waypoint(
            latitude=lat_i,
            longitude=lon_i,
            heading=tangent_heading,  # type: ignore[arg-type]
            altitude_msl=alt,
            name=f"V{i+1}",
            segment_type="pattern",
        ))

    if closed:
        waypoints.append(Waypoint(
            latitude=waypoints[0].latitude,
            longitude=waypoints[0].longitude,
            heading=waypoints[0].heading,  # type: ignore[arg-type]
            altitude_msl=alt,
            name="V1",
            segment_type="pattern",
        ))

    params = {
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "heading": float(heading),
        "altitude_msl_m": alt.m_as(ureg.meter),
        "radius_m": radius_q.m_as(ureg.meter),
        "n_sides": int(n_sides),
        "aspect_ratio": float(aspect_ratio),
        "closed": bool(closed),
    }
    return Pattern(
        kind="polygon",
        name=name or "Polygon",
        params=params,
        waypoints=waypoints,
    )


def sawtooth(
    center: tuple,
    heading: float,
    altitude_min: Union[float, "Quantity"],
    altitude_max: Union[float, "Quantity"],
    leg_length: Union[float, "Quantity"],
    n_cycles: int = 1,
    name: Optional[str] = None,
) -> Pattern:
    """Generate an oscillating altitude profile along a straight track.

    Args:
        center: (lat, lon) center point of the track.
        heading: Bearing of the track in degrees from north.
        altitude_min: Bottom of each oscillation.
        altitude_max: Top of each oscillation.
        leg_length: Total track length.
        n_cycles: Number of up-down cycles.
        name: Pattern display name (default "Sawtooth").

    Returns:
        A waypoint-based :class:`Pattern` (``kind="sawtooth"``).
    """
    center_lat, center_lon = center
    leg_len_q = _to_length_quantity(leg_length, "leg_length")
    total_len_m = leg_len_q.magnitude
    if total_len_m <= 0:
        raise HyPlanValueError("leg_length must be positive")
    alt_min = _to_length_quantity(altitude_min, "altitude_min")
    alt_max = _to_length_quantity(altitude_max, "altitude_max")
    heading = float(heading)

    n_points = 2 * n_cycles + 1
    half_len = total_len_m / 2.0

    start_lat, start_lon = pymap3d.vincenty.vreckon(
        center_lat, center_lon, half_len, (heading + 180.0) % 360.0
    )
    start_lon = wrap_to_180(start_lon)

    spacing = total_len_m / (n_points - 1) if n_points > 1 else 0

    wp_heading = wrap_to_360(heading)
    waypoints = []

    for i in range(n_points):
        dist_from_start = spacing * i
        if dist_from_start == 0:
            lat, lon = float(start_lat), float(start_lon)
        else:
            lat, lon = pymap3d.vincenty.vreckon(  # type: ignore[assignment]
                start_lat, start_lon, dist_from_start, heading
            )
            lon = wrap_to_180(lon)  # type: ignore[assignment]

        alt = alt_max if i % 2 == 0 else alt_min

        waypoints.append(Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=wp_heading,  # type: ignore[arg-type]
            altitude_msl=alt,
            name=f"ST{i+1}",
            segment_type="pattern_turn",
        ))

    params = {
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "heading": heading,
        "altitude_min_m": alt_min.m_as(ureg.meter),
        "altitude_max_m": alt_max.m_as(ureg.meter),
        "leg_length_m": leg_len_q.m_as(ureg.meter),
        "n_cycles": int(n_cycles),
    }
    return Pattern(
        kind="sawtooth",
        name=name or "Sawtooth",
        params=params,
        waypoints=waypoints,
    )


def spiral(
    center: tuple,
    heading: float,
    altitude_start: Union[float, "Quantity"],
    altitude_end: Union[float, "Quantity"],
    radius: Union[float, "Quantity"],
    n_turns: float = 3.0,
    direction: str = "right",
    points_per_turn: int = 36,
    name: Optional[str] = None,
) -> Pattern:
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
        name: Pattern display name (default "Spiral").

    Returns:
        A waypoint-based :class:`Pattern` (``kind="spiral"``).

    Raises:
        HyPlanValueError: If n_turns <= 0, points_per_turn < 3, or direction
            is not "right" or "left".
    """
    if n_turns <= 0:
        raise HyPlanValueError("n_turns must be positive")
    if points_per_turn < 3:
        raise HyPlanValueError("points_per_turn must be at least 3")
    if direction not in ("right", "left"):
        raise HyPlanValueError(f"direction must be 'right' or 'left', got '{direction}'")

    center_lat, center_lon = center
    radius_q = _to_length_quantity(radius, "radius")
    radius_m = radius_q.magnitude
    if radius_m <= 0:
        raise HyPlanValueError("radius must be positive")
    alt_start = _to_length_quantity(altitude_start, "altitude_start")
    alt_end = _to_length_quantity(altitude_end, "altitude_end")

    total_steps = int(n_turns * points_per_turn)
    if total_steps < 1:
        total_steps = 1

    angle_step = 360.0 / points_per_turn
    if direction == "left":
        angle_step = -angle_step

    alt_start_m = alt_start.m_as(ureg.meter)
    alt_end_m = alt_end.m_as(ureg.meter)

    waypoints = []
    for i in range(total_steps + 1):
        bearing = (heading + i * angle_step) % 360.0

        lat, lon = pymap3d.vincenty.vreckon(
            center_lat, center_lon, radius_m, bearing
        )
        lon = wrap_to_180(lon)

        if direction == "right":
            tangent = wrap_to_360(bearing + 90.0)
        else:
            tangent = wrap_to_360(bearing - 90.0)

        frac = i / total_steps if total_steps > 0 else 0.0
        alt_m = alt_start_m + (alt_end_m - alt_start_m) * frac
        alt = ureg.Quantity(alt_m, "meter")

        waypoints.append(Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=tangent,  # type: ignore[arg-type]
            altitude_msl=alt,
            name=f"SP{i+1}",
            segment_type="pattern",
        ))

    params = {
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "heading": float(heading),
        "altitude_start_m": alt_start_m,
        "altitude_end_m": alt_end_m,
        "radius_m": radius_q.m_as(ureg.meter),
        "n_turns": float(n_turns),
        "direction": direction,
        "points_per_turn": int(points_per_turn),
    }
    return Pattern(
        kind="spiral",
        name=name or "Spiral",
        params=params,
        waypoints=waypoints,
    )


def glint_arc(
    center: tuple,
    observation_datetime,
    altitude: Union[float, "Quantity"],
    speed: Union[float, "Quantity"],
    bank_angle: Optional[float] = None,
    bank_direction: str = "right",
    collection_length: Union[float, "Quantity", None] = None,
    densify_m: float = 200.0,
    name: Optional[str] = None,
) -> Pattern:
    """Generate a banked specular-glint arc as a waypoint pattern.

    Wraps :class:`~hyplan.glint.GlintArc` (Ayasse et al. 2022) so the arc
    can be carried inside a :class:`Pattern` and round-tripped through
    :meth:`Pattern.regenerate` and Campaign persistence.  At the arc
    midpoint the bank angle tilts the sensor to view the target with a
    glint angle of zero (perfect specular reflection); the rest of the
    arc keeps glint within a narrow band around it.

    Args:
        center: ``(target_lat, target_lon)`` of the surface point being
            observed.
        observation_datetime: UTC datetime for solar position. Same
            ``takeoff_time`` value used elsewhere in the planner.
        altitude: Aircraft altitude MSL.
        speed: Aircraft true airspeed used for turn-radius computation
            (typically ``aircraft.cruise_speed_at(altitude)``).
        bank_angle: Bank angle in degrees. ``None`` (default) auto-selects
            the solar zenith angle (valid only when SZA <= 60°).
        bank_direction: ``"right"`` (default) or ``"left"``.
        collection_length: Optional arc length to limit the collection
            (Quantity or float meters). ``None`` uses the full 180° arc.
        densify_m: Spacing of the densified arc waypoints, in meters.
        name: Pattern display name (default ``"Glint Arc"``).

    Returns:
        A waypoint-based :class:`Pattern` (``kind="glint_arc"``) whose
        waypoints trace the densified arc with per-segment headings.

    Raises:
        HyPlanValueError: If solar zenith is too small (<5°) or too large
            (>60° without an explicit ``bank_angle``), or if
            ``bank_direction`` is not "left"/"right".
    """
    target_lat, target_lon = center
    alt_q = _to_length_quantity(altitude, "altitude")
    if isinstance(speed, (int, float)):
        speed_q = ureg.Quantity(float(speed), ureg.meter / ureg.second)
    elif hasattr(speed, "units") and speed.check("[speed]"):
        speed_q = speed.to(ureg.meter / ureg.second)
    else:
        raise HyPlanTypeError("speed must be a float (m/s) or a pint Quantity with speed units")

    cl_q = None
    if collection_length is not None:
        cl_q = _to_length_quantity(collection_length, "collection_length")

    arc = GlintArc(
        target_lat=float(target_lat),
        target_lon=float(target_lon),
        observation_datetime=observation_datetime,
        altitude_msl=alt_q,
        speed=speed_q,
        bank_angle=bank_angle,
        bank_direction=bank_direction,
        collection_length=cl_q,
    )

    track = arc.track(precision=float(densify_m))
    coords = list(track.coords)  # [(lon, lat), ...]
    waypoints: List[Waypoint] = []
    n = len(coords)
    for i, (lon, lat) in enumerate(coords):
        if i < n - 1:
            lon_next, lat_next = coords[i + 1]
            _, az = pymap3d.vincenty.vdist(float(lat), float(lon), float(lat_next), float(lon_next))
            heading = wrap_to_360(float(az))
        else:
            heading = waypoints[-1].heading if waypoints else 0.0
        waypoints.append(Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=heading,  # type: ignore[arg-type]
            altitude_msl=alt_q,
            name=f"GA{i+1}",
            segment_type="pattern",
        ))

    obs_iso = observation_datetime.isoformat() if hasattr(observation_datetime, "isoformat") else str(observation_datetime)

    params = {
        "center_lat": float(target_lat),
        "center_lon": float(target_lon),
        "altitude_msl_m": alt_q.m_as(ureg.meter),
        "speed_mps": speed_q.m_as(ureg.meter / ureg.second),
        "observation_datetime": obs_iso,
        # bank_angle: store the *input* (None = auto from SZA) so that
        # regenerate() with a different observation_datetime still
        # auto-derives correctly.
        "bank_angle": (float(bank_angle) if bank_angle is not None else None),
        "bank_direction": bank_direction,
        "collection_length_m": (cl_q.m_as(ureg.meter) if cl_q is not None else None),
        "densify_m": float(densify_m),
        # Effective values from this generation, for display only:
        "effective_bank_angle": float(arc.bank_angle),
        "solar_azimuth": float(arc.solar_azimuth),
        "solar_zenith": float(arc.solar_zenith),
    }

    return Pattern(
        kind="glint_arc",
        name=name or "Glint Arc",
        params=params,
        waypoints=waypoints,
    )


def flight_lines_to_waypoint_path(
    flight_lines: List[FlightLine],
    altitude: Union[float, "Quantity", None] = None,
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
                heading=wp.heading, altitude_msl=alt,  # type: ignore[arg-type]
                speed=wp.speed, name=name, segment_type=seg_type,
            ))

    return waypoints


def coordinated_line(
    center: tuple,
    heading: float,
    primary_leg_length: Union[float, "Quantity"],
    primary_aircraft: Aircraft,
    secondary_aircraft: Aircraft,
    primary_altitude: Union[float, "Quantity"],
    secondary_altitude: Union[float, "Quantity"],
    ground_speed_ratio: Union[float, List[float], None] = None,
    primary_name: str = "P3",
    secondary_name: str = "ER2",
) -> dict:
    """Generate a coordinated dual-aircraft line pattern (five-point line).

    Two aircraft fly vertically stacked legs centered on a coordination
    point. The secondary (faster) aircraft's leg is extended symmetrically
    so both aircraft pass over the center point simultaneously.

    Based on the IMPACTS sampling strategy (Yorks et al. 2025, BAMS).

    Args:
        center: (lat, lon) coordination point where both aircraft overlap.
        heading: Bearing of the line in degrees from north.
        primary_leg_length: Leg length of the primary (slower) aircraft.
        primary_aircraft: Aircraft object for the primary (slower/sampling) platform.
        secondary_aircraft: Aircraft object for the secondary (faster/remote-sensing) platform.
        primary_altitude: Altitude MSL for the primary aircraft.
        secondary_altitude: Altitude MSL for the secondary aircraft.
        ground_speed_ratio: Ratio of secondary to primary ground speed.
            If None, computed from TAS at each altitude. Can be a list for
            multiple speed-ratio lines (e.g. [1.2, 1.45] for high/low P-3).
        primary_name: Name prefix for primary waypoints (default "P3").
        secondary_name: Name prefix for secondary waypoints (default "ER2").

    Returns:
        dict with keys:
            "primary": list of 2 Waypoints [start, end] for the primary aircraft.
            "secondary": list of 2 Waypoints [start, end] if single ratio,
                or list of such pairs if multiple ratios.
            "center": Waypoint at the coordination point.
            "ground_speed_ratio": the ratio(s) used.
    """
    center_lat, center_lon = center
    pri_len = _to_length_quantity(primary_leg_length, "primary_leg_length")
    pri_alt = _to_length_quantity(primary_altitude, "primary_altitude")
    sec_alt = _to_length_quantity(secondary_altitude, "secondary_altitude")
    heading = float(heading)
    fwd_az = wrap_to_360(heading)

    if ground_speed_ratio is None:
        sec_speed = secondary_aircraft.cruise_speed_at(sec_alt)
        pri_speed = primary_aircraft.cruise_speed_at(pri_alt)
        ratios = [float((sec_speed / pri_speed).magnitude)]
    elif isinstance(ground_speed_ratio, (int, float)):
        ratios = [float(ground_speed_ratio)]
    else:
        ratios = [float(r) for r in ground_speed_ratio]

    for r in ratios:
        if r <= 0:
            raise HyPlanValueError("ground_speed_ratio must be positive")

    pri_fl = FlightLine.center_length_azimuth(
        center_lat, center_lon, pri_len, heading,
        altitude_msl=pri_alt, site_name=primary_name,
    )
    primary_wps = [
        Waypoint(pri_fl.lat1, pri_fl.lon1, pri_fl.waypoint1.heading, pri_alt,  # type: ignore[arg-type]
                 name=f"{primary_name}_start", segment_type="pattern"),
        Waypoint(pri_fl.lat2, pri_fl.lon2, pri_fl.waypoint1.heading, pri_alt,  # type: ignore[arg-type]
                 name=f"{primary_name}_end", segment_type="pattern_turn"),
    ]

    center_wp = Waypoint(center_lat, center_lon, fwd_az, sec_alt,  # type: ignore[arg-type]
                         name="C1", segment_type="pattern")

    secondary_pairs = []
    for i, r in enumerate(ratios):
        sec_len = pri_len * r
        sec_fl = FlightLine.center_length_azimuth(
            center_lat, center_lon, sec_len, heading,
            altitude_msl=sec_alt, site_name=secondary_name,
        )
        suffix = f"_r{i+1}" if len(ratios) > 1 else ""
        secondary_pairs.append([
            Waypoint(sec_fl.lat1, sec_fl.lon1, sec_fl.waypoint1.heading, sec_alt,  # type: ignore[arg-type]
                     name=f"{secondary_name}_start{suffix}", segment_type="pattern"),
            Waypoint(sec_fl.lat2, sec_fl.lon2, sec_fl.waypoint1.heading, sec_alt,  # type: ignore[arg-type]
                     name=f"{secondary_name}_end{suffix}", segment_type="pattern_turn"),
        ])

    return {
        "primary": primary_wps,
        "secondary": secondary_pairs[0] if len(ratios) == 1 else secondary_pairs,
        "center": center_wp,
        "ground_speed_ratio": ratios[0] if len(ratios) == 1 else ratios,
    }
