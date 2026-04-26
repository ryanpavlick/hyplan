"""Flight plan computation and segment classification.

Assembles a sequence of flight lines and waypoints into a complete mission
plan with takeoff, transit, data-collection, and landing phases.
:func:`compute_flight_plan` connects segments using 3-D Dubins paths,
classifies each phase (takeoff, climb, transit, descent, approach,
flight_line), and returns a :class:`~geopandas.GeoDataFrame` with timing,
distance, altitude, and geometry for every segment.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, List, Optional, Union

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
    from ..winds import WindField

__all__ = [
    "compute_flight_plan",
]


def compute_flight_plan(
    aircraft: Aircraft,
    flight_sequence: List[Union[FlightLine, Waypoint, Pattern]],
    takeoff_airport: Optional[Airport] = None,
    return_airport: Optional[Airport] = None,
    start_offset: float = 5,
    end_offset: float = 1,
    wind_speed: Optional[Quantity] = None,
    wind_direction: Optional[float] = None,
    wind_source: Optional["WindField"] = None,
    takeoff_time: Optional[datetime.datetime] = None,
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
            ``wind_direction``, every segment time is adjusted by the
            headwind/tailwind component along its heading:
            ``time = distance / (TAS − wind · cos(wind_from − heading))``.
            Crosswind effects on ground speed are ignored. Defaults to no
            wind, preserving the pre-v1.1 behavior exactly.
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
    """
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
        from ..winds import _GriddedWindField
        if isinstance(wind_source, _GriddedWindField):
            raise HyPlanValueError(
                "takeoff_time is required when using a gridded wind field "
                "(MERRA2WindField, GMAOWindField, GFSWindField)."
            )

    # Cumulative elapsed time for wind queries
    cumulative_minutes = 0.0

    def _current_time() -> Optional[datetime.datetime]:
        if takeoff_time is None:
            return None
        return takeoff_time + datetime.timedelta(minutes=cumulative_minutes)
    # Expand Patterns into their underlying flight lines or waypoints.
    expanded: list = []
    for seg in flight_sequence:
        if isinstance(seg, Pattern):
            expanded.extend(seg.elements())
        else:
            expanded.append(seg)
    flight_sequence = expanded

    # Apply offsets to flight lines, if applicable.
    flight_sequence = [
        seg.offset_along(ureg.Quantity(-start_offset, "nautical_mile"),
                           ureg.Quantity(end_offset, "nautical_mile"))
        if isinstance(seg, FlightLine) else seg
        for seg in flight_sequence
    ]

    records = []

    # Process takeoff phase if a takeoff airport is provided.
    if takeoff_airport:
        first_target = flight_sequence[0]
        if isinstance(first_target, FlightLine):
            first_target = first_target.waypoint1
        mid_lat = (takeoff_airport.latitude + first_target.latitude) / 2
        mid_lon = (takeoff_airport.longitude + first_target.longitude) / 2
        takeoff_wind_uv = _resolve_wind_uv(
            mid_lat, mid_lon,
            first_target.altitude_msl, _current_time(),  # type: ignore[arg-type]
            wind_source, wind_speed, wind_direction,
        )
        takeoff_info = aircraft.time_to_takeoff(
            takeoff_airport, first_target, wind=takeoff_wind_uv,
        )
        takeoff_records = process_flight_phase(
            takeoff_airport, first_target, takeoff_info, "Departure",
        )
        for r in takeoff_records:
            cumulative_minutes += r["time_to_segment"]
        records.extend(takeoff_records)

    # Process connecting/cruise phases between flight segments.
    for i, segment in enumerate(flight_sequence):
        # Process FlightLine segments separately.
        if isinstance(segment, FlightLine):
            track_geometry = segment.track()
            latitudes, longitudes, _, distances = process_linestring(track_geometry)
            if len(distances) == 0:
                raise HyPlanValueError(
                    f"Flight line {segment.site_name} produced an empty track"
                )
            segment_distance = distances[-1]

            # Crab-aware timing: solve for heading and groundspeed
            # given the desired track and wind at the line midpoint.
            fl_tas = aircraft.cruise_speed_at(segment.altitude_msl)
            track_deg = segment.waypoint1.heading  # forward azimuth = desired track

            mid_idx = len(latitudes) // 2
            mid_lat = latitudes[mid_idx]
            mid_lon = longitudes[mid_idx]

            sol = _resolve_track_hold_solution(
                fl_tas, track_deg,  # type: ignore[arg-type]
                mid_lat, mid_lon,  # type: ignore[arg-type]
                segment.altitude_msl, _current_time(),
                wind_source, wind_speed, wind_direction,
            )

            time_to_segment = (
                ureg.Quantity(segment_distance, "meter") / sol["groundspeed"]
            ).m_as(ureg.minute)

            start_heading = sol["heading_deg"]
            end_heading = sol["heading_deg"]

            records.append({
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
                "start_heading": start_heading,
                "end_heading": end_heading,
                "planned_track": track_deg,
                "wind_corrected_heading": sol["heading_deg"],
                "crab_angle_deg": sol["crab_angle_deg"],
                "groundspeed_kts": sol["groundspeed"].m_as(ureg.knot),
                "tailwind_kts": sol["alongtrack_wind"].m_as(ureg.knot),
                "crosswind_kts": sol["crosstrack_wind"].m_as(ureg.knot),
            })
            cumulative_minutes += time_to_segment

        # Insert loiter segment if the current waypoint has a delay.
        if is_waypoint(segment) and segment.delay is not None and segment.delay.magnitude > 0:  # type: ignore[union-attr]
            from shapely.geometry import Point as _Point
            loiter_time = segment.delay.m_as(ureg.minute)  # type: ignore[union-attr]
            records.append({
                "geometry": _Point(segment.longitude, segment.latitude),  # type: ignore[union-attr]
                "start_lat": segment.latitude,  # type: ignore[union-attr]
                "start_lon": segment.longitude,  # type: ignore[union-attr]
                "end_lat": segment.latitude,  # type: ignore[union-attr]
                "end_lon": segment.longitude,  # type: ignore[union-attr]
                "start_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,  # type: ignore[union-attr]
                "end_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,  # type: ignore[union-attr]
                "segment_type": "loiter",
                "segment_name": segment.name,  # type: ignore[union-attr]
                "distance": 0.0,
                "time_to_segment": loiter_time,
                "start_heading": segment.heading,  # type: ignore[union-attr]
                "end_heading": segment.heading  # type: ignore[union-attr]
            })
            cumulative_minutes += loiter_time

        # Process the connecting phase between the current and next segment.
        if i + 1 < len(flight_sequence):
            end = flight_sequence[i + 1]
            start_wp = segment.waypoint2 if isinstance(segment, FlightLine) else segment
            end_wp = end.waypoint1 if isinstance(end, FlightLine) else end

            # For intra-pattern waypoints (e.g. spiral, polygon), connect
            # with a direct segment to preserve the original pattern
            # geometry.  Transitions marked "pattern_turn" fall through to
            # Dubins so the aircraft gets a realistic turn.
            departing_is_pattern = (
                is_waypoint(segment) and is_waypoint(end)
                and segment.segment_type == "pattern"  # type: ignore[union-attr]
                and end.segment_type in ("pattern", "pattern_turn")  # type: ignore[union-attr,operator]
            )
            if departing_is_pattern:
                rec = _direct_segment_record(
                    start_wp, end_wp, aircraft, segment.segment_type,  # type: ignore[union-attr,arg-type]
                    wind_speed=wind_speed, wind_direction=wind_direction,
                    wind_source=wind_source, segment_time=_current_time(),
                )
                cumulative_minutes += rec["time_to_segment"]
                records.append(rec)
                continue

            # Use per-waypoint speed override if set on the departing waypoint.
            speed_override = None
            if is_waypoint(segment) and segment.speed is not None:  # type: ignore[union-attr]
                speed_override = segment.speed  # type: ignore[union-attr]

            mid_lat = (start_wp.latitude + end_wp.latitude) / 2
            mid_lon = (start_wp.longitude + end_wp.longitude) / 2
            cruise_wind_uv = _resolve_wind_uv(
                mid_lat, mid_lon,
                end_wp.altitude_msl, _current_time(),  # type: ignore[arg-type]
                wind_source, wind_speed, wind_direction,
            )
            cruise_info = aircraft.time_to_cruise(
                start_wp, end_wp,
                true_air_speed=speed_override,
                wind=cruise_wind_uv,
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
                    wp_seg_type = segment.segment_type  # type: ignore[union-attr]
                cruise_records = process_flight_phase(
                    start_wp, end_wp, cruise_info, phase_name,
                    override_segment_type=wp_seg_type,  # type: ignore[arg-type]
                )
                for r in cruise_records:
                    cumulative_minutes += r["time_to_segment"]
                records.extend(cruise_records)

    # Process the approach phase if a return airport is provided.
    if return_airport:
        last_target = flight_sequence[-1]
        if isinstance(last_target, FlightLine):
            last_target = last_target.waypoint2
        mid_lat = (last_target.latitude + return_airport.latitude) / 2
        mid_lon = (last_target.longitude + return_airport.longitude) / 2
        return_wind_uv = _resolve_wind_uv(
            mid_lat, mid_lon,
            last_target.altitude_msl, _current_time(),  # type: ignore[arg-type]
            wind_source, wind_speed, wind_direction,
        )
        return_info = aircraft.time_to_return(
            last_target, return_airport, wind=return_wind_uv,
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
