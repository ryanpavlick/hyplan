"""Segment record builders and flight-phase classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from pint import Quantity
import pymap3d.vincenty

from ..units import ureg
from ..aircraft import Aircraft
from ..airports import Airport
from ..waypoint import Waypoint
from ..flight_line import FlightLine
from ..winds.utils import _resolve_wind_factor

if TYPE_CHECKING:
    from ..winds import WindField


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing in degrees from point 1 to point 2."""
    _, az = pymap3d.vincenty.vdist(lat1, lon1, lat2, lon2)
    return float(az)


def _direct_segment_record(
    start_wp: Waypoint,
    end_wp: Waypoint,
    aircraft: Aircraft,
    segment_type: str,
    wind_speed: Optional[Quantity] = None,
    wind_direction: Optional[float] = None,
    wind_source: Optional["WindField"] = None,
    segment_time=None,
) -> dict:
    """Create a direct great-circle segment between two pattern waypoints.

    Used for densely-spaced pattern waypoints (spiral, polygon, etc.) where
    a Dubins path would distort the intended geometry.
    """
    from shapely.geometry import LineString as _LineString

    geom = _LineString([
        (start_wp.longitude, start_wp.latitude),
        (end_wp.longitude, end_wp.latitude),
    ])

    # Distance via Vincenty (metres)
    dist_m, _ = pymap3d.vincenty.vdist(
        start_wp.latitude, start_wp.longitude,
        end_wp.latitude, end_wp.longitude,
    )
    dist_m = float(dist_m)
    dist_nm = ureg.Quantity(dist_m, "meter").m_as(ureg.nautical_mile)

    # Average altitude for speed lookup
    alt_start = start_wp.altitude_msl or ureg.Quantity(0, "foot")
    alt_end = end_wp.altitude_msl or ureg.Quantity(0, "foot")
    avg_alt = (alt_start + alt_end) / 2.0
    speed = start_wp.speed if start_wp.speed is not None else aircraft.cruise_speed_at(avg_alt)
    heading = _bearing_between(
        start_wp.latitude, start_wp.longitude, end_wp.latitude, end_wp.longitude,
    )
    factor = _resolve_wind_factor(
        speed, heading,
        start_wp.latitude, start_wp.longitude, avg_alt, segment_time,
        wind_source, wind_speed, wind_direction,
    )
    time_min = (ureg.Quantity(dist_m, "meter") / speed).m_as(ureg.minute) * factor

    return {
        "geometry": geom,
        "start_lat": start_wp.latitude,
        "start_lon": start_wp.longitude,
        "end_lat": end_wp.latitude,
        "end_lon": end_wp.longitude,
        "start_altitude": alt_start.m_as(ureg.foot),
        "end_altitude": alt_end.m_as(ureg.foot),
        "start_heading": start_wp.heading,
        "end_heading": end_wp.heading,
        "time_to_segment": time_min,
        "segment_type": segment_type,
        "segment_name": f"{start_wp.name or 'WP'} to {end_wp.name or 'WP'}",
        "distance": dist_nm,
    }


_GRAVITY_MS2 = 9.80665


def loiter_orbit_geometry(
    waypoint: Waypoint,
    aircraft: Aircraft,
    n_points: int = 72,
):
    """Closed ground-track polygon for a Waypoint loiter (right-hand orbit).

    Computes the cruise turn radius from the aircraft's bank-by-phase
    cruise angle and cruise speed at the waypoint's altitude:

        r = v² / (g · tan(φ_cruise))

    The orbit center is placed perpendicular to the waypoint's heading
    on the right side (FAA-standard right-hand hold), so the waypoint
    sits on the orbit. The returned ring traces a single full revolution
    sampled at ``n_points`` evenly spaced bearings.

    Args:
        waypoint: Loiter waypoint. ``altitude_msl`` and ``heading`` are
            both required.
        aircraft: Aircraft model (uses ``cruise_speed_at`` and
            ``turn_model.bank_by_phase.cruise_deg``).
        n_points: Sample resolution around the orbit (default 72 → 5° spacing).

    Returns:
        :class:`shapely.geometry.LineString` of (lon, lat) coordinates,
        closed (first point == last point), in WGS84.
    """
    from shapely.geometry import LineString
    from shapely.ops import transform
    from ..geometry import get_utm_transforms

    if waypoint.altitude_msl is None:
        raise ValueError(
            "loiter_orbit_geometry requires waypoint.altitude_msl to be set."
        )

    speed_mps = aircraft.cruise_speed_at(waypoint.altitude_msl).m_as("meter/second")
    bank_rad = np.radians(aircraft.turn_model.bank_by_phase.cruise_deg)
    radius_m = (speed_mps ** 2) / (_GRAVITY_MS2 * np.tan(bank_rad))

    # Project the waypoint to UTM and offset perpendicular-right of the heading
    # to find the orbit center.
    from shapely.geometry import Point
    wp_pt = Point(waypoint.longitude, waypoint.latitude)
    to_utm, from_utm = get_utm_transforms([wp_pt])
    wp_utm = transform(to_utm, wp_pt)
    perp_az_rad = np.radians(waypoint.heading + 90.0)
    cx = wp_utm.x + radius_m * np.sin(perp_az_rad)
    cy = wp_utm.y + radius_m * np.cos(perp_az_rad)

    # Closed orbit ring: first sample lies at the waypoint itself.
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    start_angle = np.arctan2(wp_utm.y - cy, wp_utm.x - cx)
    # Right-hand hold goes clockwise viewed from above → angle decreases.
    sample_angles = start_angle - angles
    xs = cx + radius_m * np.cos(sample_angles)
    ys = cy + radius_m * np.sin(sample_angles)

    pts_utm = [Point(x, y) for x, y in zip(xs, ys)]
    pts_wgs = [transform(from_utm, p) for p in pts_utm]
    return LineString([(p.x, p.y) for p in pts_wgs])


def create_flight_line_record(flight_line: FlightLine, aircraft: Aircraft) -> dict:
    """
    Create a flight line record dictionary for inclusion in a flight plan DataFrame.

    Args:
        flight_line (FlightLine): The flight line to convert.
        aircraft (Aircraft): Aircraft used to compute segment timing.

    Returns:
        dict: Record with geometry, endpoints, altitudes (feet MSL), headings,
            time (minutes), segment type, and distance (nautical miles).
    """
    return {
        "geometry": flight_line.geometry,
        "start_lat": flight_line.lat1,
        "start_lon": flight_line.lon1,
        "end_lat": flight_line.lat2,
        "end_lon": flight_line.lon2,
        "start_altitude": flight_line.altitude_msl.m_as(ureg.foot),
        "end_altitude": flight_line.altitude_msl.m_as(ureg.foot),
        "start_heading": flight_line.waypoint1.heading,
        "end_heading": flight_line.waypoint2.heading,
        "time_to_segment": (flight_line.length / aircraft.cruise_speed_at(flight_line.altitude_msl)).m_as(ureg.minute),
        "segment_type": "flight_line",
        "segment_name": flight_line.site_name,
        "distance": flight_line.length.m_as(ureg.nautical_mile)
    }


def process_flight_phase(
    start: Union[Airport, Waypoint],
    end: Union[Airport, Waypoint],
    phase_info: dict,
    segment_name: str,
    override_segment_type: str | None = None,
) -> List[dict]:
    """Process a flight phase using the detailed ``phase_info``.

    For each sub-phase in ``phase_info["phases"]``, this function determines
    the segment type based on the altitude change:

    - If ascending, the phase is labeled ``"takeoff"`` when ``segment_name``
      is ``"Departure"``, otherwise ``"climb"``.
    - If descending, the phase is labeled ``"approach"`` when ``segment_name``
      is ``"Arrival"``, otherwise ``"descent"``.
    - If no altitude change, the phase is labeled ``"transit"``.

    If ``override_segment_type`` is provided (e.g. ``"pattern"`` or
    ``"sampling"`` from a flight-pattern waypoint), it replaces the default
    ``"transit"`` label for level segments while climb/descent labels are
    preserved.

    Returns a list of record dictionaries for inclusion in the flight plan.
    """
    records = []
    dubins_path = phase_info["dubins_path"]
    full_geom = dubins_path.geometry
    total_geom_length = full_geom.length  # in geometry units (degrees)

    # Split geometry proportionally by time (which aligns with the plot x-axis).
    # Phase distances can exceed the Dubins path length (e.g. IFR approach extends
    # beyond the horizontal track), so time is a more reliable splitting key.
    phase_items = list(phase_info["phases"].items())
    phase_times = []
    for phase, details in phase_items:
        dt = (details["end_time"] - details["start_time"]).m_as(ureg.minute)
        phase_times.append(dt)

    # Phases with explicit "geometry" entries (e.g., the terminal approach
    # segment from Aircraft.time_to_return) use that geometry verbatim and
    # don't share the Dubins path.  The Dubins-slicing math has to normalize
    # against the *non-explicit* phase total only — otherwise the Dubins-
    # backed phases get truncated and the descent ends short of its real
    # endpoint.
    dubins_total_time = sum(
        dt for dt, (phase, details) in zip(phase_times, phase_items)
        if details.get("geometry") is None
    )
    can_split = dubins_total_time > 0

    cumulative_frac = 0.0
    for i, (phase, details) in enumerate(phase_items):
        # Determine the segment type based on altitude information.
        # Use a 1-foot tolerance for floating-point noise from the Dubins solver.
        alt_diff_ft = (details["end_altitude"] - details["start_altitude"]).m_as(ureg.foot)
        if phase == "approach":
            # Phase-name key wins: an explicit "approach" phase from
            # Aircraft.time_to_return is always the terminal arrival
            # segment regardless of the outer segment_name (which is
            # "Return" in the standard engine.py call path, not
            # "Arrival").
            seg_type = "approach"
        elif alt_diff_ft > 1.0:
            seg_type = "takeoff" if segment_name == "Departure" else "climb"
        elif alt_diff_ft < -1.0:
            seg_type = "approach" if segment_name == "Arrival" else "descent"
        else:
            seg_type = override_segment_type or "transit"

        start_heading = details.get("start_heading", getattr(start, "heading", None))
        end_heading   = details.get("end_heading", getattr(end, "heading", None))

        if "distance" in details:
            phase_distance_nm = details["distance"].m_as(ureg.nautical_mile)
        else:
            phase_distance_nm = None

        # If the caller attached an explicit geometry (e.g., the
        # terminal approach segment from Aircraft.time_to_return), use
        # it verbatim — no Dubins-slicing for this phase.  Otherwise
        # split the Dubins path proportionally by phase time within the
        # Dubins-only total.
        if details.get("geometry") is not None:
            phase_geom = details["geometry"]
        elif can_split and phase_times[i] > 0:
            from shapely.geometry import LineString as _LineString
            frac_start = cumulative_frac
            frac_end = min(1.0, cumulative_frac + phase_times[i] / dubins_total_time)

            start_dist = frac_start * total_geom_length
            end_dist = frac_end * total_geom_length

            # Extract sub-linestring using interpolation
            n_sample = max(2, int((frac_end - frac_start) * len(full_geom.coords)))
            dists = np.linspace(start_dist, end_dist, n_sample)
            points = [full_geom.interpolate(d) for d in dists]
            phase_geom = _LineString(points)

            cumulative_frac = frac_end
        else:
            phase_geom = full_geom

        # Get actual start/end coords from the sub-geometry
        geom_coords = list(phase_geom.coords)
        start_lon_g, start_lat_g = geom_coords[0][0], geom_coords[0][1]
        end_lon_g, end_lat_g = geom_coords[-1][0], geom_coords[-1][1]

        records.append({
            "geometry": phase_geom,
            "start_lat": start_lat_g,
            "start_lon": start_lon_g,
            "end_lat": end_lat_g,
            "end_lon": end_lon_g,
            "start_altitude": details["start_altitude"].m_as(ureg.foot),
            "end_altitude": details["end_altitude"].m_as(ureg.foot),
            "start_heading": start_heading,
            "end_heading": end_heading,
            "time_to_segment": (details["end_time"] - details["start_time"]).m_as(ureg.minute),
            "segment_type": seg_type,
            "segment_name": segment_name,
            "distance": phase_distance_nm
        })

    return records
