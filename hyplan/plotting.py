import folium
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from hyplan.aircraft import Aircraft
from hyplan.airports import Airport
from hyplan.flight_line import FlightLine
from hyplan.dubins_path import Waypoint
from hyplan.units import ureg

__all__ = [
    "map_flight_lines",
    "plot_flight_plan",
    "terrain_profile_along_track",
    "plot_altitude_trajectory",
]


def map_flight_lines(
    flight_lines: List[FlightLine],
    center: Tuple[float, float] = None,
    zoom_start: int = 6,
    line_color: str = "blue",
    line_weight: int = 3
) -> folium.Map:
    """
    Create an interactive folium map displaying a list of FlightLine objects.
    
    Args:
        flight_lines (List[FlightLine]): List of FlightLine objects to display.
        center (tuple, optional): A tuple (latitude, longitude) to center the map.
                                  If None, the center is computed as the average of the start points.
        zoom_start (int, optional): Initial zoom level for the map (default is 6).
        line_color (str, optional): Color for the flight lines (default is "blue").
        line_weight (int, optional): Thickness of the flight lines (default is 3).
    
    Returns:
        folium.Map: A folium Map object with the flight lines added.
    """
    # Compute the center from the FlightLine start points if not provided.
    if center is None:
        lats = [fl.lat1 for fl in flight_lines]
        lons = [fl.lon1 for fl in flight_lines]
        center = (np.mean(lats), np.mean(lons))
    
    # Create the folium map centered at the computed center.
    m = folium.Map(location=center, zoom_start=zoom_start)
    
    # Add each FlightLine to the map.
    for fl in flight_lines:
        # Extract coordinates from the FlightLine geometry.
        # Shapely's LineString returns (lon, lat) coordinates; folium expects (lat, lon).
        coords = [(lat, lon) for lon, lat in list(fl.geometry.coords)]
        
        # Create a popup HTML string with some properties.
        popup_html = (f"<span style=\"font-family: 'Courier New', monospace;\">"
            f"<b>{fl.site_name}</b><br>"
            f"Investigator: {fl.investigator}<br>"
            f"Site Description: {fl.site_description}<br>"
            f"Altitude MSL: {fl.altitude_msl:.2f}<br>"
            f"Length: {fl.length:.2f}<br>"
            f"Azimuth: {fl.az12:.2f}<br>"
            f"Start: {fl.lat1:.4f}, {fl.lon1:.4f}<br>"
            f"End: {fl.lat2:.4f}, {fl.lon2:.4f}<br>"
            f"</span>"
        )

        iframe = folium.IFrame(popup_html)
        popup = folium.Popup(iframe,
                     min_width=300,
                     max_width=500)
    
        
        # Add the polyline for the flight line.
        folium.PolyLine(
            locations=coords,
            color=line_color,
            weight=line_weight,
            popup=popup,
            tooltip=fl.site_name
        ).add_to(m)


    return m


def plot_flight_plan(flight_plan_gdf: gpd.GeoDataFrame, takeoff_airport: Airport, return_airport: Airport, flight_sequence: list) -> None:
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


def terrain_profile_along_track(flight_plan_gdf: gpd.GeoDataFrame, dem_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_altitude_trajectory(flight_plan_gdf: gpd.GeoDataFrame, aircraft: Optional[Aircraft] = None, dem_file: Optional[str] = None, show_terrain: bool = True) -> None:
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
