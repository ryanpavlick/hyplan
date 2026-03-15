import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .dubins_path import Waypoint
from .flight_line import FlightLine
from .geometry import process_linestring


def compute_flight_plan(
    aircraft: Aircraft,
    flight_sequence: list,
    takeoff_airport: Airport = None,
    return_airport: Airport = None,
    start_offset=5,
    end_offset=1
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

        # Process the connecting phase between the current and next segment.
        if i + 1 < len(flight_sequence):
            end = flight_sequence[i + 1]
            start_wp = segment.waypoint2 if isinstance(segment, FlightLine) else segment
            end_wp = end.waypoint1 if isinstance(end, FlightLine) else end

            cruise_info = aircraft.time_to_cruise(start_wp, end_wp)
            if cruise_info["total_time"].to(ureg.minute).magnitude > 0:
                phase_name = (
                    "Departure" if (i == 0 and takeoff_airport is None) else
                    f"{getattr(segment, 'site_name', getattr(segment, 'name', 'Unknown'))} to "
                    f"{getattr(end, 'site_name', getattr(end, 'name', 'Unknown'))}"
                )
                records.extend(process_flight_phase(start_wp, end_wp, cruise_info, phase_name))

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


def create_flight_line_record(flight_line, aircraft):
    """
    Create a flight line record dictionary for inclusion in a flight plan DataFrame.

    Args:
        flight_line (FlightLine): The flight line to convert.
        aircraft (Aircraft): Aircraft used to compute segment timing.

    Returns:
        dict: Record with geometry, endpoints, altitudes (feet), headings,
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


def process_flight_phase(start, end, phase_info, segment_name):
    """
    Process a flight phase using the detailed phase_info.

    For each sub-phase in phase_info["phases"], this function determines the segment type
    based on the altitude change:
      - If ascending, the phase is labeled "takeoff" when segment_name is "Departure",
        otherwise "climb".
      - If descending, the phase is labeled "approach" when segment_name is "Arrival",
        otherwise "descent".
      - If no altitude change, the phase is labeled "transit".
    
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
            seg_type = "transit"

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


def plot_flight_plan(flight_plan_gdf, takeoff_airport, return_airport, flight_sequence):
    """
    Plot the computed flight plan on a 2D map with airports, waypoints, and flight lines.

    Args:
        flight_plan_gdf (GeoDataFrame): Flight plan from compute_flight_plan().
        takeoff_airport (Airport): Departure airport (plotted as red star).
        return_airport (Airport): Arrival airport (plotted as blue star).
        flight_sequence (list): Sequence of FlightLine and Waypoint objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    flight_plan_gdf.plot(ax=ax, column="segment_type", legend=True, cmap="viridis")

    # Plot takeoff and return airports.
    ax.scatter(takeoff_airport.longitude, takeoff_airport.latitude, color='red', marker='*', s=200, label='Takeoff Airport')
    ax.scatter(return_airport.longitude, return_airport.latitude, color='blue', marker='*', s=200, label='Return Airport')

    # Plot waypoints and flight lines from the flight sequence.
    for item in flight_sequence:
        if isinstance(item, Waypoint):
            ax.scatter(item.longitude, item.latitude, color='green', marker='o', s=100, label=item.name)
        elif isinstance(item, FlightLine):
            x, y = zip(*item.geometry.coords)
            ax.plot(x, y, color='black', linestyle='dashed', linewidth=2, label=item.site_name)

    ax.set_title("Flight Plan")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()


def terrain_profile_along_track(flight_plan_gdf, dem_file=None):
    """
    Sample terrain elevation along the flight plan track.

    Extracts lat/lon points from each segment geometry, queries DEM
    elevation, and returns arrays of cumulative time and terrain height.

    Args:
        flight_plan_gdf (GeoDataFrame): Flight plan from compute_flight_plan().
        dem_file (str, optional): Path to DEM file. If None, one is auto-downloaded.

    Returns:
        tuple: (times, elevations) where times is cumulative minutes and
            elevations is terrain height in feet MSL, both as numpy arrays.
    """
    from .terrain import get_elevations, generate_demfile

    all_lats, all_lons, all_times = [], [], []
    cumulative_time = 0.0

    for _, row in flight_plan_gdf.iterrows():
        geom = row["geometry"]
        seg_time = row["time_to_segment"]

        if geom is None or geom.is_empty or seg_time == 0:
            cumulative_time += seg_time
            continue

        coords = np.array(geom.coords)
        lons = coords[:, 0]
        lats = coords[:, 1]
        n_pts = len(lats)

        # Distribute time linearly along the segment's geometry points
        seg_times = cumulative_time + np.linspace(0, seg_time, n_pts)

        all_lats.append(lats)
        all_lons.append(lons)
        all_times.append(seg_times)
        cumulative_time += seg_time

    if not all_lats:
        return np.array([]), np.array([])

    all_lats = np.concatenate(all_lats)
    all_lons = np.concatenate(all_lons)
    all_times = np.concatenate(all_times)

    if dem_file is None:
        dem_file = generate_demfile(all_lats, all_lons)

    elevations_m = get_elevations(all_lats, all_lons, dem_file)
    elevations_ft = elevations_m * 3.28084

    return all_times, elevations_ft


def plot_altitude_trajectory(flight_plan_gdf, aircraft=None, dem_file=None, show_terrain=True):
    """
    Plot altitude vs. time trajectory with optional terrain profile.

    If an Aircraft is provided, climb/takeoff segments are drawn with the
    realistic curved profile (ROC decreases with altitude). Otherwise all
    segments are drawn as straight lines.

    Args:
        flight_plan_gdf (GeoDataFrame): Flight plan from compute_flight_plan().
        aircraft (Aircraft, optional): Aircraft used for the flight plan.
        dem_file (str, optional): Path to DEM file for terrain. If None, auto-downloaded.
        show_terrain (bool): If True, overlay terrain elevation beneath the flight path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot terrain profile first (underneath)
    if show_terrain:
        try:
            terrain_times, terrain_elevations = terrain_profile_along_track(
                flight_plan_gdf, dem_file=dem_file
            )
            if len(terrain_times) > 0:
                ax.fill_between(
                    terrain_times, 0, terrain_elevations,
                    color="saddlebrown", alpha=0.3, label="Terrain"
                )
                ax.plot(terrain_times, terrain_elevations, color="saddlebrown",
                        linewidth=0.8, alpha=0.6)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not load terrain profile: {e}")

    # Plot aircraft altitude segments
    cumulative_time = 0
    for _, row in flight_plan_gdf.iterrows():
        seg_type = row["segment_type"]
        t_seg = row["time_to_segment"]
        h_start = row["start_altitude"]
        h_end = row["end_altitude"]

        # Use curved profile for climb/takeoff if aircraft is available
        if aircraft is not None and seg_type in ("climb", "takeoff") and h_end > h_start:
            profile_t, profile_h = aircraft.climb_altitude_profile(
                h_start * ureg.feet, h_end * ureg.feet
            )
            # Scale profile time to match the segment time (accounts for horizontal travel)
            if profile_t[-1] > 0:
                profile_t = profile_t * (t_seg / profile_t[-1])
            ax.plot(
                cumulative_time + profile_t, profile_h,
                marker="o", markevery=[0, -1],
                label=row["segment_name"]
            )
        else:
            ax.plot(
                [cumulative_time, cumulative_time + t_seg],
                [h_start, h_end],
                marker="o",
                label=row["segment_name"]
            )
        cumulative_time += t_seg

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Altitude (feet MSL)")
    ax.set_title("Altitude vs. Time Trajectory")
    ax.legend()
    ax.grid(True)
    plt.show()
