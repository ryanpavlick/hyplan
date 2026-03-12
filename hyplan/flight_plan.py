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
            time_to_segment = (ureg.Quantity(segment_distance, 'meter') / aircraft.cruise_speed).to(ureg.minute).magnitude

            # Use the FlightLine's own heading properties.
            start_heading = segment.waypoint1.heading  # From FlightLine.az12
            end_heading = segment.waypoint2.heading    # From FlightLine.az21 (adjusted)

            records.append({
                "geometry": track_geometry,
                "start_lat": latitudes[0],
                "start_lon": longitudes[0],
                "end_lat": latitudes[-1],
                "end_lon": longitudes[-1],
                "start_altitude": segment.altitude.to(ureg.foot).magnitude,
                "end_altitude": segment.altitude.to(ureg.foot).magnitude,
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
    Helper function to create a flight line record for the DataFrame.
    """
    return {
        "geometry": flight_line.geometry,
        "start_lat": flight_line.lat1,
        "start_lon": flight_line.lon1,
        "end_lat": flight_line.lat2,
        "end_lon": flight_line.lon2,
        "start_altitude": flight_line.altitude.to(ureg.foot).magnitude,
        "end_altitude": flight_line.altitude.to(ureg.foot).magnitude,
        "start_heading": flight_line.waypoint1.heading,
        "end_heading": flight_line.waypoint2.heading,
        "time_to_segment": (flight_line.length / aircraft.cruise_speed).to(ureg.minute).magnitude,
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

    for phase, details in phase_info["phases"].items():
        # Determine the segment type based on altitude information.
        if details["start_altitude"] < details["end_altitude"]:
            seg_type = "takeoff" if segment_name == "Departure" else "climb"
        elif details["start_altitude"] > details["end_altitude"]:
            seg_type = "approach" if segment_name == "Arrival" else "descent"
        else:
            seg_type = "transit"

        # Use the heading provided in details if available; otherwise, fall back.
        start_heading = details.get("start_heading", getattr(start, "heading", None))
        end_heading   = details.get("end_heading", getattr(end, "heading", None))

        # Use the phase's distance if provided; otherwise, default to dubins_path.length.
        if "distance" in details:
            phase_distance_nm = details["distance"].to(ureg.nautical_mile).magnitude
        else:
            phase_distance_nm = None
            # phase_distance_nm = dubins_path.length.to(ureg.nautical_mile).magnitude

        # TODO: Split dubins_path.geometry into per-phase sub-segments using
        # phase distances so each sub-phase has its own distinct geometry.
        records.append({
            "geometry": dubins_path.geometry,
            "start_lat": start.latitude,
            "start_lon": start.longitude,
            "end_lat": end.latitude,
            "end_lon": end.longitude,
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
    Plot the computed flight plan along with airports, waypoints, and flight lines.
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


def plot_altitude_trajectory(flight_plan_gdf):
    """
    Plot altitude vs. time trajectory.
    """
    plt.figure(figsize=(10, 5))
    cumulative_time = 0
    for _, row in flight_plan_gdf.iterrows():
        plt.plot(
            [cumulative_time, cumulative_time + row["time_to_segment"]],
            [row["start_altitude"], row["end_altitude"]],
            marker="o",
            label=row["segment_name"]
        )
        cumulative_time += row["time_to_segment"]

    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (feet MSL)")
    plt.title("Altitude vs. Time Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()
