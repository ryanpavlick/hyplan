import folium
import numpy as np
from typing import List, Tuple
from hyplan.flight_line import FlightLine
from hyplan.units import ureg


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
