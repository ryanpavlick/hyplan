from typing import List, Optional, Union

import numpy as np
import geopandas as gpd
import pandas as pd
from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .waypoint import Waypoint, is_waypoint
from .flight_line import FlightLine
from .geometry import process_linestring

__all__ = [
    "compute_flight_plan",
    "create_flight_line_record",
    "process_flight_phase",
]


def _direct_segment_record(
    start_wp: Waypoint,
    end_wp: Waypoint,
    aircraft: Aircraft,
    segment_type: str,
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
    dist_nm = ureg.Quantity(dist_m, "meter").to(ureg.nautical_mile).magnitude

    # Average altitude for speed lookup
    alt_start = start_wp.altitude_msl or ureg.Quantity(0, "foot")
    alt_end = end_wp.altitude_msl or ureg.Quantity(0, "foot")
    avg_alt = (alt_start + alt_end) / 2.0
    speed = start_wp.speed if start_wp.speed is not None else aircraft.cruise_speed_at(avg_alt)
    time_min = (ureg.Quantity(dist_m, "meter") / speed).to(ureg.minute).magnitude

    return {
        "geometry": geom,
        "start_lat": start_wp.latitude,
        "start_lon": start_wp.longitude,
        "end_lat": end_wp.latitude,
        "end_lon": end_wp.longitude,
        "start_altitude": alt_start.to(ureg.foot).magnitude,
        "end_altitude": alt_end.to(ureg.foot).magnitude,
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
    """
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
        records.extend(process_flight_phase(takeoff_airport, first_target, takeoff_info, "Departure"))

    # Process connecting/cruise phases between flight segments.
    for i, segment in enumerate(flight_sequence):
        # Process FlightLine segments separately.
        if isinstance(segment, FlightLine):
            track_geometry = segment.track()
            latitudes, longitudes, _, distances = process_linestring(track_geometry)
            segment_distance = distances[-1] if len(distances) > 0 else 0

            # Calculate time_to_segment using the computed segment_distance.
            time_to_segment = (ureg.Quantity(segment_distance, 'meter') / aircraft.cruise_speed_at(segment.altitude_msl)).to(ureg.minute).magnitude

            # Use the FlightLine's own heading properties.
            start_heading = segment.waypoint1.heading  # From FlightLine.az12
            end_heading = segment.waypoint2.heading    # From FlightLine.az21 (adjusted)

            records.append({
                "geometry": track_geometry,
                "start_lat": latitudes[0],
                "start_lon": longitudes[0],
                "end_lat": latitudes[-1],
                "end_lon": longitudes[-1],
                "start_altitude": segment.altitude_msl.to(ureg.foot).magnitude,
                "end_altitude": segment.altitude_msl.to(ureg.foot).magnitude,
                "segment_type": "flight_line",
                "segment_name": segment.site_name,
                "distance": ureg.Quantity(segment_distance, 'meter').to(ureg.nautical_mile).magnitude,
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
                "start_altitude": segment.altitude_msl.to(ureg.foot).magnitude if segment.altitude_msl else None,
                "end_altitude": segment.altitude_msl.to(ureg.foot).magnitude if segment.altitude_msl else None,
                "segment_type": "loiter",
                "segment_name": segment.name,
                "distance": 0.0,
                "time_to_segment": segment.delay.to(ureg.minute).magnitude,
                "start_heading": segment.heading,
                "end_heading": segment.heading
            })

        # Process the connecting phase between the current and next segment.
        if i + 1 < len(flight_sequence):
            end = flight_sequence[i + 1]
            start_wp = segment.waypoint2 if isinstance(segment, FlightLine) else segment
            end_wp = end.waypoint1 if isinstance(end, FlightLine) else end

            # For consecutive pattern waypoints (e.g. spiral, rosette),
            # connect with a direct segment instead of a Dubins path so the
            # original pattern geometry is preserved.
            both_pattern = (
                is_waypoint(segment) and is_waypoint(end)
                and segment.segment_type == "pattern"
                and end.segment_type == "pattern"
            )
            if both_pattern:
                records.append(_direct_segment_record(
                    start_wp, end_wp, aircraft, segment.segment_type,
                ))
                continue

            # Use per-waypoint speed override if set on the departing waypoint.
            speed_override = None
            if is_waypoint(segment) and segment.speed is not None:
                speed_override = segment.speed

            cruise_info = aircraft.time_to_cruise(start_wp, end_wp, true_air_speed=speed_override)
            if cruise_info["total_time"].to(ureg.minute).magnitude > 0:
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
                records.extend(process_flight_phase(
                    start_wp, end_wp, cruise_info, phase_name,
                    override_segment_type=wp_seg_type,
                ))

    # Process the approach phase if a return airport is provided.
    if return_airport:
        last_target = flight_sequence[-1]
        if isinstance(last_target, FlightLine):
            last_target = last_target.waypoint2
        return_info = aircraft.time_to_return(last_target, return_airport)
        if return_info["total_time"].to(ureg.minute).magnitude > 0:
            records.extend(process_flight_phase(last_target, return_airport, return_info, "Return"))

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
        "start_altitude": flight_line.altitude_msl.to(ureg.foot).magnitude,
        "end_altitude": flight_line.altitude_msl.to(ureg.foot).magnitude,
        "start_heading": flight_line.waypoint1.heading,
        "end_heading": flight_line.waypoint2.heading,
        "time_to_segment": (flight_line.length / aircraft.cruise_speed_at(flight_line.altitude_msl)).to(ureg.minute).magnitude,
        "segment_type": "flight_line",
        "segment_name": flight_line.site_name,
        "distance": flight_line.length.to(ureg.nautical_mile).magnitude
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
        dt = (details["end_time"] - details["start_time"]).to(ureg.minute).magnitude
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
            phase_distance_nm = details["distance"].to(ureg.nautical_mile).magnitude
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
            "start_altitude": details["start_altitude"].to(ureg.foot).magnitude,
            "end_altitude": details["end_altitude"].to(ureg.foot).magnitude,
            "start_heading": start_heading,
            "end_heading": end_heading,
            "time_to_segment": (details["end_time"] - details["start_time"]).to(ureg.minute).magnitude,
            "segment_type": seg_type,
            "segment_name": segment_name,
            "distance": phase_distance_nm
        })

    return records


# Backward-compatible re-exports — these functions now live in hyplan.plotting
from .plotting import plot_flight_plan, terrain_profile_along_track, plot_altitude_trajectory
