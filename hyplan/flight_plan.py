"""Flight plan computation and segment classification.

Assembles a sequence of flight lines and waypoints into a complete mission
plan with takeoff, transit, data-collection, and landing phases.
:func:`compute_flight_plan` connects segments using 3-D Dubins paths,
classifies each phase (takeoff, climb, transit, descent, approach,
flight_line), and returns a :class:`~geopandas.GeoDataFrame` with timing,
distance, altitude, and geometry for every segment.
"""

from typing import List, Optional, Union

import numpy as np
import geopandas as gpd
import pandas as pd
from pint import Quantity
import pymap3d.vincenty
from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .waypoint import Waypoint, is_waypoint
from .flight_line import FlightLine
from .geometry import process_linestring
from .exceptions import HyPlanValueError

__all__ = [
    "compute_flight_plan",
    "create_flight_line_record",
    "process_flight_phase",
]


def _wind_factor(
    tas: Quantity,
    heading_deg: float,
    wind_speed: Optional[Quantity],
    wind_from_deg: Optional[float],
) -> float:
    """Multiplicative wind-correction factor for a segment's no-wind time.

    Given the true airspeed and a constant wind vector (speed + direction
    *from which* the wind is blowing, per meteorological convention), returns
    ``TAS / ground_speed`` where

        ground_speed = TAS − wind_speed · cos(wind_from_deg − heading_deg)

    Multiply a no-wind ``distance / TAS`` time by this factor to obtain the
    wind-corrected time. Returns 1.0 when no wind is supplied. Crosswind
    effects on ground speed are ignored (valid small-angle approximation
    that matches standard pre-flight planning practice).

    Raises:
        HyPlanValueError: If the headwind exceeds TAS, yielding a
            non-positive ground speed (unflyable).
    """
    if wind_speed is None or wind_speed.magnitude == 0 or wind_from_deg is None:
        return 1.0
    if heading_deg is None:
        return 1.0
    rel_angle_rad = np.radians(wind_from_deg - heading_deg)
    headwind = wind_speed * float(np.cos(rel_angle_rad))
    ground_speed = tas - headwind
    if ground_speed.m_as(ureg.knot) <= 0:
        raise HyPlanValueError(
            f"Headwind {headwind.to(ureg.knot):.1f} exceeds TAS "
            f"{tas.to(ureg.knot):.1f} on heading {heading_deg:.0f}°; unflyable."
        )
    return float((tas / ground_speed).to_base_units().magnitude)


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
) -> dict:
    """Create a direct great-circle segment between two pattern waypoints.

    Used for densely-spaced pattern waypoints (spiral, rosette, etc.) where
    a Dubins path would distort the intended geometry.
    """
    from shapely.geometry import LineString as _LineString
    import pymap3d.vincenty

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
    factor = _wind_factor(speed, heading, wind_speed, wind_direction)
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


def compute_flight_plan(
    aircraft: Aircraft,
    flight_sequence: List[Union[FlightLine, Waypoint]],
    takeoff_airport: Optional[Airport] = None,
    return_airport: Optional[Airport] = None,
    start_offset: float = 5,
    end_offset: float = 1,
    wind_speed: Optional[Quantity] = None,
    wind_direction: Optional[float] = None,
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
    """
    if wind_speed is not None and wind_speed.magnitude != 0 and wind_direction is None:
        raise HyPlanValueError(
            "wind_direction is required when wind_speed is non-zero"
        )
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
        takeoff_info = aircraft.time_to_takeoff(takeoff_airport, first_target)
        takeoff_bearing = _bearing_between(
            takeoff_airport.latitude, takeoff_airport.longitude,
            first_target.latitude, first_target.longitude,
        )
        takeoff_tas = aircraft.cruise_speed_at(first_target.altitude_msl)
        takeoff_factor = _wind_factor(
            takeoff_tas, takeoff_bearing, wind_speed, wind_direction,
        )
        takeoff_records = process_flight_phase(
            takeoff_airport, first_target, takeoff_info, "Departure",
        )
        for r in takeoff_records:
            r["time_to_segment"] *= takeoff_factor
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

            # Calculate time_to_segment using the computed segment_distance.
            fl_tas = aircraft.cruise_speed_at(segment.altitude_msl)
            fl_time_no_wind = (
                ureg.Quantity(segment_distance, 'meter') / fl_tas
            ).m_as(ureg.minute)
            fl_factor = _wind_factor(
                fl_tas, segment.waypoint1.heading, wind_speed, wind_direction,
            )
            time_to_segment = fl_time_no_wind * fl_factor

            # Use the FlightLine's own heading properties.
            start_heading = segment.waypoint1.heading  # From FlightLine.az12
            end_heading = segment.waypoint2.heading    # From FlightLine.az21 (adjusted)

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
                "distance": ureg.Quantity(segment_distance, 'meter').m_as(ureg.nautical_mile),
                "time_to_segment": time_to_segment,
                "start_heading": start_heading,
                "end_heading": end_heading
            })

        # Insert loiter segment if the current waypoint has a delay.
        if is_waypoint(segment) and segment.delay is not None and segment.delay.magnitude > 0:
            from shapely.geometry import Point as _Point
            records.append({
                "geometry": _Point(segment.longitude, segment.latitude),
                "start_lat": segment.latitude,
                "start_lon": segment.longitude,
                "end_lat": segment.latitude,
                "end_lon": segment.longitude,
                "start_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,
                "end_altitude": segment.altitude_msl.m_as(ureg.foot) if segment.altitude_msl else None,
                "segment_type": "loiter",
                "segment_name": segment.name,
                "distance": 0.0,
                "time_to_segment": segment.delay.m_as(ureg.minute),
                "start_heading": segment.heading,
                "end_heading": segment.heading
            })

        # Process the connecting phase between the current and next segment.
        if i + 1 < len(flight_sequence):
            end = flight_sequence[i + 1]
            start_wp = segment.waypoint2 if isinstance(segment, FlightLine) else segment
            end_wp = end.waypoint1 if isinstance(end, FlightLine) else end

            # For intra-pattern waypoints (e.g. spiral, polygon, within a
            # racetrack leg), connect with a direct segment to preserve the
            # original pattern geometry.  Inter-leg transitions (marked
            # "pattern_turn") fall through to Dubins so the aircraft gets a
            # realistic turn between legs.
            departing_is_pattern = (
                is_waypoint(segment) and is_waypoint(end)
                and segment.segment_type == "pattern"
                and end.segment_type in ("pattern", "pattern_turn")
            )
            if departing_is_pattern:
                records.append(_direct_segment_record(
                    start_wp, end_wp, aircraft, segment.segment_type,
                    wind_speed=wind_speed, wind_direction=wind_direction,
                ))
                continue

            # Use per-waypoint speed override if set on the departing waypoint.
            speed_override = None
            if is_waypoint(segment) and segment.speed is not None:
                speed_override = segment.speed

            cruise_info = aircraft.time_to_cruise(start_wp, end_wp, true_air_speed=speed_override)
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
                # Wind correction: scale every phase time by the factor
                # computed from the dominant leg bearing. Using a single
                # leg bearing (start→end) rather than per-phase headings
                # keeps the approximation simple and matches how pilots
                # pre-plan wind on straight cruise legs.
                cruise_bearing = _bearing_between(
                    start_wp.latitude, start_wp.longitude,
                    end_wp.latitude, end_wp.longitude,
                )
                cruise_tas = speed_override or aircraft.cruise_speed_at(
                    end_wp.altitude_msl
                )
                cruise_factor = _wind_factor(
                    cruise_tas, cruise_bearing, wind_speed, wind_direction,
                )
                for r in cruise_records:
                    r["time_to_segment"] *= cruise_factor
                records.extend(cruise_records)

    # Process the approach phase if a return airport is provided.
    if return_airport:
        last_target = flight_sequence[-1]
        if isinstance(last_target, FlightLine):
            last_target = last_target.waypoint2
        return_info = aircraft.time_to_return(last_target, return_airport)
        if return_info["total_time"].m_as(ureg.minute) > 0:
            return_records = process_flight_phase(
                last_target, return_airport, return_info, "Return",
            )
            return_bearing = _bearing_between(
                last_target.latitude, last_target.longitude,
                return_airport.latitude, return_airport.longitude,
            )
            return_tas = aircraft.cruise_speed_at(last_target.altitude_msl)
            return_factor = _wind_factor(
                return_tas, return_bearing, wind_speed, wind_direction,
            )
            for r in return_records:
                r["time_to_segment"] *= return_factor
            records.extend(return_records)

    # Create and return the GeoDataFrame.
    df = pd.DataFrame(records)
    flight_plan_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs="EPSG:4326")
    return flight_plan_gdf


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
    override_segment_type: str = None,
) -> List[dict]:
    """
    Process a flight phase using the detailed phase_info.

    For each sub-phase in phase_info["phases"], this function determines the segment type
    based on the altitude change:
      - If ascending, the phase is labeled "takeoff" when segment_name is "Departure",
        otherwise "climb".
      - If descending, the phase is labeled "approach" when segment_name is "Arrival",
        otherwise "descent".
      - If no altitude change, the phase is labeled "transit".

    If ``override_segment_type`` is provided (e.g. "pattern" or "sampling" from a
    flight-pattern waypoint), it replaces the default "transit" label for level
    segments while climb/descent labels are preserved.

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

    total_time = sum(phase_times)
    can_split = total_time > 0

    cumulative_frac = 0.0
    for i, (phase, details) in enumerate(phase_items):
        # Determine the segment type based on altitude information.
        if details["start_altitude"] < details["end_altitude"]:
            seg_type = "takeoff" if segment_name == "Departure" else "climb"
        elif details["start_altitude"] > details["end_altitude"]:
            seg_type = "approach" if segment_name == "Arrival" else "descent"
        else:
            seg_type = override_segment_type or "transit"

        start_heading = details.get("start_heading", getattr(start, "heading", None))
        end_heading   = details.get("end_heading", getattr(end, "heading", None))

        if "distance" in details:
            phase_distance_nm = details["distance"].m_as(ureg.nautical_mile)
        else:
            phase_distance_nm = None

        # Split geometry along the path by cumulative time fraction
        if can_split and phase_times[i] > 0:
            from shapely.geometry import LineString as _LineString
            frac_start = cumulative_frac
            frac_end = min(1.0, cumulative_frac + phase_times[i] / total_time)

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


# Backward-compatible re-exports — these functions now live in hyplan.plotting
