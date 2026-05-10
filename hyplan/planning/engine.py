"""Flight plan computation and segment classification.

Assembles a sequence of flight lines and waypoints into a complete mission
plan with takeoff, transit, data-collection, and landing phases.
:func:`compute_flight_plan` builds each inter-segment path via the
aircraft's :meth:`Aircraft._hybrid_path` (2D Dubins horizontally +
integrated ``climb_profile`` / ``descent_profile`` vertically),
classifies each phase (takeoff, climb, transit, descent, approach,
flight_line), and returns a :class:`~geopandas.GeoDataFrame` with timing,
distance, altitude, and geometry for every segment.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from pint import Quantity

from ..units import ureg
from ..aircraft import Aircraft
from ..airports import Airport
from ..waypoint import Waypoint, is_waypoint
from ..flight_line import FlightLine
from ..pattern import Pattern
from ..geometry import process_linestring
from ..exceptions import HyPlanValueError
from ..winds.utils import (
    _resolve_track_hold_solution,
    _resolve_wind_uv,
)
from .segments import _direct_segment_record, process_flight_phase

if TYPE_CHECKING:
    from ..aircraft import ClimbPlan
    from ..winds import WindField

__all__ = [
    "compute_flight_plan",
    "flag_below_min_safe_speed",
]


def expand_sequence(
    flight_sequence: list[FlightLine | Waypoint | Pattern],
) -> list[FlightLine | Waypoint]:
    """Expand any :class:`Pattern` objects into their underlying flight
    lines and waypoints.

    Patterns become a flat list of their constituent elements (via
    :meth:`Pattern.elements`); :class:`FlightLine` and
    :class:`Waypoint` entries pass through unchanged.  Returns a new
    list; the input is not mutated.

    This is the same expansion :func:`compute_flight_plan` performs
    internally before timing each segment, exposed as a top-level
    utility so callers can preview the unwrapped sequence (useful for
    validation, plotting, or feeding into a different planner).
    """
    expanded: list[FlightLine | Waypoint] = []
    for seg in flight_sequence:
        if isinstance(seg, Pattern):
            expanded.extend(seg.elements())
        else:
            expanded.append(seg)
    return expanded


def compute_flight_plan(
    aircraft: Aircraft,
    flight_sequence: list[FlightLine | Waypoint | Pattern],
    takeoff_airport: Airport | None = None,
    return_airport: Airport | None = None,
    start_offset: float = 5,
    end_offset: float = 1,
    wind_speed: Quantity | None = None,
    wind_direction: float | None = None,
    wind_source: WindField | None = None,
    takeoff_time: datetime.datetime | None = None,
    climb_plan: ClimbPlan | str | None = "auto",
    wind_sampling: str = "cruise_midpoint",
    n_samples: int = 20,
) -> gpd.GeoDataFrame:
    """
    Compute a flight plan with segment classifications.

    Segment types are determined as follows:
      - "takeoff" for the very first ascending phase,
      - "climb" for any subsequent ascending phase,
      - "transit" for level flight,
      - "descent" for descending flight,
      - "flight_line" for dedicated flight line segments,
      - "approach" for the final descending phase into the return airport.

    Args:
        aircraft: Aircraft performance model used for timing.
        flight_sequence: Ordered list of flight lines and/or waypoints.
        takeoff_airport: Optional departure airport (prepends a takeoff phase).
        return_airport: Optional arrival airport (appends an approach phase).
        start_offset: Pre-extension of each flight line (nautical miles).
        end_offset: Post-extension of each flight line (nautical miles).
        wind_speed: Optional constant wind speed as a ``pint.Quantity``
            (e.g. ``30 * ureg.knot``). When supplied together with
            ``wind_direction``, the wind is decomposed into a
            ``(u_east, v_north)`` vector and applied two ways:

            1. Horizontal Dubins arcs in the takeoff, inter-line
               transit, descent, and return phases use trochoidal
               geometry (see
               :class:`~hyplan.dubins3d._TrochoidDubins2D`), so turn
               arcs are wind-drift-corrected and segment timing
               reflects the actual ground track.
            2. Each flight line is solved with a wind-corrected
               heading (crab angle) and the resulting ground speed
               from full kinematics, so both headwind/tailwind *and*
               crosswind components affect segment time.

            Defaults to no wind (still-air geometry and timing).
        wind_direction: Direction the wind is blowing *from*, in degrees
            true (meteorological convention: 0° = wind from north, 90° =
            from east). Required when ``wind_speed`` is set. Ignored when
            ``wind_speed`` is None or zero.
        wind_source: A :class:`~hyplan.winds.WindField` providing
            per-segment wind.  Takes precedence over ``wind_speed`` /
            ``wind_direction``.  Cannot be combined with those parameters.
        takeoff_time: UTC datetime of takeoff.  Required when
            ``wind_source`` is a gridded wind field (MERRA-2, GMAO) so
            that each segment can be queried at the correct time.
        climb_plan: Climb-out hold plan for the takeoff phase.  Three forms:

            * ``"auto"`` (default) — use the aircraft's
              ``typical_climb_out.explicit_climb_plan`` if defined,
              otherwise no holds.  Reproduces empirical-typical
              wall-clock TOC for aircraft whose factory populates
              the policy (e.g. NASA_ER2 ships with a ~12-min FL356
              hold representing the median weight-band `.delay`
              orbit observed across 138 IWG1 sorties).
            * ``None`` — no holds; pure active-climb integration.
              Caller is opting out of any typical-mission absorption.
            * :class:`~hyplan.aircraft.ClimbPlan` — caller-supplied
              pauses, used as-is.  Each pause renders as a
              ``"loiter"`` segment in the output dataframe (zero
              forward distance, hold duration only), with the climb
              split into one row per inter-pause segment.
        wind_sampling: How wind is sampled per phase when
            ``wind_source`` is provided.

            * ``"cruise_midpoint"`` (default) — single ``(u, v)`` per
              leg sampled at the leg midpoint at cruise altitude;
              passed to ``Aircraft._hybrid_path`` as ``wind=(u, v)``
              and used for both 2D Dubins (trochoidal) and the
              vertical phases.  Backward-compatible with v1.5.
            * ``"phase_midpoint"`` — climb and descent each get their
              own wind sample at the phase-mid altitude near the
              phase-mid distance, projected onto the bearing.  Cruise
              still gets a single midpoint sample at cruise altitude.
              Captures vertical wind shear (jet streams) that
              ``"cruise_midpoint"`` misses.

            Ignored when ``wind_source`` is None (no wind / scalar
            wind / wind_speed+wind_direction paths).
        n_samples: Number of points sampled along each Dubins phase
            sub-linestring (climb / cruise / transit / descent).
            Defaults to 20.  Increase for smoother-looking turns in
            plotted output (e.g. ``n_samples=80`` removes visible
            polygonalization on tight Dubins arcs); has no effect on
            timing, which is computed analytically.
    """
    if wind_sampling not in ("cruise_midpoint", "phase_midpoint"):
        raise HyPlanValueError(
            f"wind_sampling must be 'cruise_midpoint' or 'phase_midpoint', "
            f"got {wind_sampling!r}."
        )
    # Resolve the "auto" sentinel by reading the aircraft's
    # typical_climb_out policy.  No-op when the aircraft hasn't
    # populated the policy.
    if isinstance(climb_plan, str):
        if climb_plan != "auto":
            raise HyPlanValueError(
                f"climb_plan must be a ClimbPlan, None, or \"auto\"; "
                f"got {climb_plan!r}."
            )
        climb_plan = (
            aircraft.typical_climb_out.explicit_climb_plan
            if aircraft.typical_climb_out is not None
            else None
        )
    # Validate wind parameter combinations
    if wind_source is not None and wind_speed is not None:
        raise HyPlanValueError(
            "Cannot specify both wind_source and wind_speed/wind_direction. "
            "Use one or the other."
        )
    if wind_speed is not None and wind_speed.magnitude != 0 and wind_direction is None:
        raise HyPlanValueError(
            "wind_direction is required when wind_speed is non-zero"
        )
    # Gridded wind fields need takeoff_time; simple fields do not
    if wind_source is not None and takeoff_time is None:
        from ..winds.gridded import _GriddedWindField
        if isinstance(wind_source, _GriddedWindField):
            raise HyPlanValueError(
                "takeoff_time is required when using a gridded wind field "
                "(MERRA2WindField, GMAOWindField, GFSWindField)."
            )

    # Cumulative elapsed time for wind queries
    cumulative_minutes = 0.0

    def _current_time() -> datetime.datetime | None:
        if takeoff_time is None:
            return None
        return takeoff_time + datetime.timedelta(minutes=cumulative_minutes)

    def _phase_wind_kwargs(
        mid_lat: float, mid_lon: float, alt: Quantity,
    ) -> dict[str, Any]:
        """Pick the wind kwargs to forward to time_to_* methods.

        For ``wind_sampling="phase_midpoint"`` (and a wind_source
        provider supplied), pass ``wind_source`` + ``t_anchor`` so
        ``_hybrid_path`` does its own per-phase sampling.  Otherwise
        resolve to a single ``(u, v)`` at the leg midpoint and pass as
        ``wind=`` (the v1.5 behavior).
        """
        if wind_sampling == "phase_midpoint" and wind_source is not None:
            return {
                "wind_source": wind_source,
                "t_anchor": _current_time(),
            }
        uv = _resolve_wind_uv(
            mid_lat, mid_lon, alt, _current_time(),
            wind_source, wind_speed, wind_direction,
        )
        return {"wind": uv}
    # Expand Patterns into their underlying flight lines or waypoints.
    expanded = expand_sequence(flight_sequence)

    # Apply offsets to flight lines, if applicable.  Use a fresh
    # variable name (not `flight_sequence`) so the post-expansion
    # narrower type sticks.
    flight_seq: list[FlightLine | Waypoint] = [
        seg.offset_along(ureg.Quantity(-start_offset, "nautical_mile"),
                           ureg.Quantity(end_offset, "nautical_mile"))
        if isinstance(seg, FlightLine) else seg
        for seg in expanded
    ]

    records = []

    # Process takeoff phase if a takeoff airport is provided.
    if takeoff_airport:
        first_target = flight_seq[0]
        if isinstance(first_target, FlightLine):
            first_target = first_target.waypoint1
        mid_lat = (takeoff_airport.latitude + first_target.latitude) / 2
        mid_lon = (takeoff_airport.longitude + first_target.longitude) / 2
        takeoff_info = aircraft.time_to_takeoff(
            takeoff_airport, first_target,
            climb_plan=climb_plan,
            n_samples=n_samples,
            **_phase_wind_kwargs(
                mid_lat, mid_lon, first_target.altitude_msl,
            ),
        )
        takeoff_records = process_flight_phase(
            takeoff_airport, first_target, takeoff_info, "Departure",
        )
        for r in takeoff_records:
            cumulative_minutes += r["time_to_segment"]
        records.extend(takeoff_records)

    # Process connecting/cruise phases between flight segments.
    for i, segment in enumerate(flight_seq):
        # Process FlightLine segments separately.
        if isinstance(segment, FlightLine):
            fl_record = _build_flight_line_record(
                aircraft, segment, _current_time(),
                wind_source, wind_speed, wind_direction,
            )
            records.append(fl_record)
            cumulative_minutes += fl_record["time_to_segment"]

        # Insert loiter segment if the current waypoint has a delay.
        if is_waypoint(segment) and segment.delay is not None and segment.delay.magnitude > 0:
            loiter_record = _build_loiter_record(aircraft, segment)
            records.append(loiter_record)
            cumulative_minutes += loiter_record["time_to_segment"]

        # Process the connecting phase between the current and next segment.
        if i + 1 < len(flight_seq):
            end = flight_seq[i + 1]
            start_wp = segment.waypoint2 if isinstance(segment, FlightLine) else segment
            end_wp = end.waypoint1 if isinstance(end, FlightLine) else end

            # For intra-pattern waypoints (e.g. spiral, polygon), connect
            # with a direct segment to preserve the original pattern
            # geometry.  Transitions marked "pattern_turn" fall through to
            # Dubins so the aircraft gets a realistic turn.
            departing_is_pattern = (
                is_waypoint(segment) and is_waypoint(end)
                and segment.segment_type == "pattern"
                and end.segment_type in ("pattern", "pattern_turn")
            )
            if departing_is_pattern:
                rec = _direct_segment_record(
                    start_wp, end_wp, aircraft, segment.segment_type,  # type: ignore[union-attr,arg-type]  # is_waypoint not a TypeGuard
                    wind_speed=wind_speed, wind_direction=wind_direction,
                    wind_source=wind_source, segment_time=_current_time(),
                )
                cumulative_minutes += rec["time_to_segment"]
                records.append(rec)
                continue

            # Use per-waypoint speed override if set on the departing waypoint.
            speed_override = None
            if is_waypoint(segment) and segment.speed is not None:
                speed_override = segment.speed

            mid_lat = (start_wp.latitude + end_wp.latitude) / 2
            mid_lon = (start_wp.longitude + end_wp.longitude) / 2
            cruise_info = aircraft.time_to_cruise(
                start_wp, end_wp,
                true_air_speed=speed_override,
                n_samples=n_samples,
                **_phase_wind_kwargs(
                    mid_lat, mid_lon, end_wp.altitude_msl,
                ),
            )
            if cruise_info["total_time"].m_as(ureg.minute) > 0:
                phase_name = (
                    "Departure" if (i == 0 and takeoff_airport is None) else
                    f"{getattr(segment, 'site_name', getattr(segment, 'name', 'Unknown'))} to "
                    f"{getattr(end, 'site_name', getattr(end, 'name', 'Unknown'))}"
                )
                # Use waypoint segment_type if set (e.g. "pattern", "sampling"),
                # otherwise process_flight_phase determines the type from altitude.
                wp_seg_type = None
                if is_waypoint(segment):
                    wp_seg_type = segment.segment_type
                cruise_records = process_flight_phase(
                    start_wp, end_wp, cruise_info, phase_name,
                    override_segment_type=wp_seg_type,
                )
                for r in cruise_records:
                    cumulative_minutes += r["time_to_segment"]
                records.extend(cruise_records)

    # Process the approach phase if a return airport is provided.
    if return_airport:
        last_target = flight_seq[-1]
        if isinstance(last_target, FlightLine):
            last_target = last_target.waypoint2
        mid_lat = (last_target.latitude + return_airport.latitude) / 2
        mid_lon = (last_target.longitude + return_airport.longitude) / 2
        return_info = aircraft.time_to_return(
            last_target, return_airport,
            n_samples=n_samples,
            **_phase_wind_kwargs(
                mid_lat, mid_lon, last_target.altitude_msl,
            ),
        )
        if return_info["total_time"].m_as(ureg.minute) > 0:
            return_records = process_flight_phase(
                last_target, return_airport, return_info, "Return",
            )
            for r in return_records:
                cumulative_minutes += r["time_to_segment"]
            records.extend(return_records)

    # Create and return the GeoDataFrame.
    df = pd.DataFrame(records)
    flight_plan_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs="EPSG:4326")
    return flight_plan_gdf


# ---------------------------------------------------------------------------
# Per-segment record builders for compute_flight_plan
# ---------------------------------------------------------------------------


def _build_flight_line_record(
    aircraft: Aircraft,
    segment: FlightLine,
    current_time: datetime.datetime | None,
    wind_source: WindField | None,
    wind_speed: Quantity | None,
    wind_direction: float | None,
) -> dict[str, Any]:
    """Build the GeoDataFrame record for a single FlightLine segment.

    Solves the crab-aware track-hold problem at the line midpoint
    using the supplied wind field, then returns a record with the
    line geometry plus crab/groundspeed metadata.
    """
    track_geometry = segment.track()
    latitudes, longitudes, _, distances = process_linestring(track_geometry)
    if len(distances) == 0:
        raise HyPlanValueError(
            f"Flight line {segment.site_name} produced an empty track"
        )
    segment_distance = distances[-1]

    fl_tas = aircraft.cruise_speed_at(segment.altitude_msl)
    track_deg = segment.waypoint1.heading  # forward azimuth = desired track

    mid_idx = len(latitudes) // 2
    sol = _resolve_track_hold_solution(
        fl_tas, track_deg,
        latitudes[mid_idx], longitudes[mid_idx],
        segment.altitude_msl, current_time,
        wind_source, wind_speed, wind_direction,
    )

    time_to_segment = (
        ureg.Quantity(segment_distance, "meter") / sol["groundspeed"]
    ).m_as(ureg.minute)

    return {
        "geometry": track_geometry,
        "start_lat": latitudes[0],
        "start_lon": longitudes[0],
        "end_lat": latitudes[-1],
        "end_lon": longitudes[-1],
        "start_altitude": segment.altitude_msl.m_as(ureg.foot),
        "end_altitude": segment.altitude_msl.m_as(ureg.foot),
        "segment_type": "flight_line",
        "segment_name": segment.site_name,
        "distance": ureg.Quantity(segment_distance, "meter").m_as(ureg.nautical_mile),
        "time_to_segment": time_to_segment,
        "start_heading": sol["heading_deg"],
        "end_heading": sol["heading_deg"],
        "planned_track": track_deg,
        "wind_corrected_heading": sol["heading_deg"],
        "crab_angle_deg": sol["crab_angle_deg"],
        "groundspeed_kts": sol["groundspeed"].m_as(ureg.knot),
        "tailwind_kts": sol["alongtrack_wind"].m_as(ureg.knot),
        "crosswind_kts": sol["crosstrack_wind"].m_as(ureg.knot),
    }


def _build_loiter_record(
    aircraft: Aircraft,
    segment: Waypoint,
) -> dict[str, Any]:
    """Build the loiter (hold-orbit) record for a Waypoint with non-zero delay.

    When altitude is known we render an actual hold-orbit ground track
    using the aircraft's cruise speed and turn radius — distance is
    the real ground covered during the loiter, not the orbit
    circumference.  With no altitude we fall back to a Point and zero
    distance for callers that pass minimal Waypoints.
    """
    assert segment.delay is not None
    loiter_time = segment.delay.m_as(ureg.minute)

    if segment.altitude_msl is not None:
        from .segments import loiter_orbit_geometry
        loiter_geom = loiter_orbit_geometry(segment, aircraft)
        speed_mps = aircraft.cruise_speed_at(segment.altitude_msl).m_as("meter/second")
        distance_m = speed_mps * segment.delay.m_as(ureg.second)
        distance_nm = ureg.Quantity(distance_m, "meter").m_as(ureg.nautical_mile)
    else:
        from shapely.geometry import Point as _Point
        loiter_geom = _Point(segment.longitude, segment.latitude)
        distance_nm = 0.0

    return {
        "geometry": loiter_geom,
        "start_lat": segment.latitude,
        "start_lon": segment.longitude,
        "end_lat": segment.latitude,
        "end_lon": segment.longitude,
        "start_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,
        "end_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,
        "segment_type": "loiter",
        "segment_name": segment.name,
        "distance": distance_nm,
        "time_to_segment": loiter_time,
        "start_heading": segment.heading,
        "end_heading": segment.heading,
    }


def flag_below_min_safe_speed(
    plan: gpd.GeoDataFrame,
    aircraft: Aircraft,
    *,
    margin: float = 1.3,
) -> gpd.GeoDataFrame:
    """Return rows from *plan* whose effective TAS is below the stall margin.

    For each row, the planned TAS is taken from the appropriate aircraft
    schedule at the segment's altitude:

    * ``flight_line`` and ``loiter`` -> :meth:`Aircraft.cruise_speed_at`
    * ``climb`` (any segment marked as climb) -> :meth:`Aircraft.climb_speed_at`
    * ``descent`` -> :meth:`Aircraft.descent_speed_at`
    * any other segment is skipped (no TAS-vs-altitude check applicable).

    The altitude used is the segment's ``end_altitude`` (or
    ``start_altitude`` if ``end_altitude`` is missing).  The minimum
    safe TAS is :meth:`Aircraft.min_safe_speed_at` with the supplied
    *margin*.

    Returns a slice of *plan* containing only the rows where planned
    TAS < min_safe_speed, plus two added columns: ``planned_tas_kts``
    and ``min_safe_tas_kts``.  Empty GeoDataFrame if everything is
    fine.

    Raises:
        HyPlanValueError: If ``aircraft.stall_speed_cas`` is None.
    """
    if aircraft.stall_speed_cas is None:
        raise HyPlanValueError(
            f"{aircraft.aircraft_type} has no stall_speed_cas calibrated; "
            f"cannot evaluate min-safe-speed margin."
        )

    schedule_for = {
        "flight_line": aircraft.cruise_speed_at,
        "loiter":      aircraft.cruise_speed_at,
        "climb":       aircraft.climb_speed_at,
        "descent":     aircraft.descent_speed_at,
    }

    rows = []
    for idx, row in plan.iterrows():
        seg_type = row.get("segment_type")
        speed_at = schedule_for.get(seg_type)
        if speed_at is None:
            continue
        alt_ft = row.get("end_altitude") or row.get("start_altitude")
        if alt_ft is None:
            continue
        alt_q = ureg.Quantity(float(alt_ft), "feet")
        planned_tas = speed_at(alt_q).to(ureg.knot).magnitude
        min_safe = aircraft.min_safe_speed_at(alt_q, margin=margin).to(ureg.knot).magnitude
        if planned_tas < min_safe:
            rows.append((idx, planned_tas, min_safe))

    if not rows:
        empty = plan.iloc[0:0].copy()
        empty["planned_tas_kts"] = []
        empty["min_safe_tas_kts"] = []
        return empty

    indices = [r[0] for r in rows]
    out = plan.loc[indices].copy()
    out["planned_tas_kts"] = [r[1] for r in rows]
    out["min_safe_tas_kts"] = [r[2] for r in rows]
    return out
