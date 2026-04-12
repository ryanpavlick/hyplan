"""Visualization utilities for flight plans, flight lines, and airspace.

Provides Folium interactive maps (:func:`map_flight_lines`,
:func:`map_airspace`), Matplotlib cartopy maps
(:func:`plot_airspace_map`), altitude profiles
(:func:`plot_flight_plan`, :func:`plot_altitude_trajectory`,
:func:`plot_vertical_profile`), conflict matrices
(:func:`plot_conflict_matrix`), oceanic track maps
(:func:`plot_oceanic_tracks`), and terrain profile extraction
(:func:`terrain_profile_along_track`).
"""

import folium
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from typing import List, Optional, Tuple
from hyplan.aircraft import Aircraft
from hyplan.airports import Airport
from hyplan.flight_line import FlightLine
from hyplan.waypoint import is_waypoint
from hyplan.units import ureg

__all__ = [
    "map_flight_lines",
    "plot_flight_plan",
    "terrain_profile_along_track",
    "plot_altitude_trajectory",
    "plot_airspace_map",
    "plot_oceanic_tracks",
    "plot_vertical_profile",
    "plot_conflict_matrix",
    "map_airspace",
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
        if is_waypoint(item):
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


# ---------------------------------------------------------------------------
# Airspace visualization
# ---------------------------------------------------------------------------

# Color map for airspace classes
_AIRSPACE_COLORS = {
    "RESTRICTED": ("red", 0.25),
    "PROHIBITED": ("darkred", 0.30),
    "DANGER": ("orange", 0.25),
    "SFRA": ("darkorange", 0.30),
    "TFR": ("magenta", 0.25),
    "B": ("blue", 0.20),
    "C": ("purple", 0.20),
    "D": ("teal", 0.20),
    "E": ("steelblue", 0.15),
    "MODE_C": ("lightsteelblue", 0.10),
    "OTHER": ("gray", 0.15),
}


def _plot_airspace_polygon(ax, airspace, color, alpha, transform, hatch=None):
    """Plot a single airspace polygon on a cartopy axis."""
    polys = (
        list(airspace.geometry.geoms)
        if airspace.geometry.geom_type == "MultiPolygon"
        else [airspace.geometry]
    )
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha, color=color, hatch=hatch,
                transform=transform)
        ax.plot(x, y, "--", color=color, linewidth=0.8, alpha=0.6,
                transform=transform)


def plot_airspace_map(
    airspaces,
    flight_lines=None,
    conflicts=None,
    near_misses=None,
    inactive_airspaces=None,
    title="Airspace Map",
    figsize=(14, 10),
    show_labels=True,
    buffer_m=None,
    extent=None,
):
    """Plot airspaces, flight lines, conflicts, and near-misses on a cartopy map.

    Args:
        airspaces: List of Airspace objects to display.
        flight_lines: Optional list of flight line objects.
        conflicts: Optional list of AirspaceConflict objects.
        near_misses: Optional list of AirspaceConflict with severity="NEAR_MISS".
        inactive_airspaces: Optional list of schedule-filtered airspaces to
            show grayed out.
        title: Plot title string.
        figsize: Figure size tuple.
        show_labels: If True, label airspaces with name and altitude.
        buffer_m: If set, draw a buffer halo around near-miss airspaces.

    Returns:
        Matplotlib Figure and Axes.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patches as mpatches

    transform = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": transform})

    # Basemap
    ax.add_feature(cfeature.LAND, facecolor="#f0efe6")
    ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAKES, facecolor="#d6eaf8", edgecolor="gray", linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")

    # Track seen classes for legend
    seen_classes = set()
    conflicts = conflicts or []
    near_misses = near_misses or []

    # Helper: check if a point is within the visible extent
    def _in_extent(cx, cy):
        if extent is None:
            return True
        return extent[0] <= cx <= extent[1] and extent[2] <= cy <= extent[3]

    # Plot inactive airspaces (grayed out) if provided
    if inactive_airspaces:
        for a in inactive_airspaces:
            cx, cy = a.geometry.centroid.coords[0]
            if not _in_extent(cx, cy):
                continue
            _plot_airspace_polygon(ax, a, "lightgray", 0.15, transform, hatch="//")
            if show_labels:
                ax.text(cx, cy, f"{a.name}\n(inactive)", ha="center", va="center",
                        fontsize=6, color="gray", style="italic", transform=transform,
                        path_effects=[
                            _pe.withStroke(linewidth=2, foreground="white"),
                        ])

    # Plot active airspace polygons
    for a in airspaces:
        cx, cy = a.geometry.centroid.coords[0]

        # Choose style based on source/class
        if a.airspace_class == "TFR" or a.source == "faa_tfr":
            color, alpha = "magenta", 0.20
            hatch = "xx"
            seen_classes.add("TFR")
        elif a.airspace_class == "SFRA":
            color, alpha = "darkorange", 0.25
            hatch = "//"
            seen_classes.add("SFRA")
        else:
            color, alpha = _AIRSPACE_COLORS.get(a.airspace_class, ("gray", 0.15))
            hatch = None
            seen_classes.add(a.airspace_class)

        _plot_airspace_polygon(ax, a, color, alpha, transform, hatch=hatch)

        if show_labels and _in_extent(cx, cy):
            label = a.name
            if len(label) > 30:
                label = label[:27] + "..."
            alt_label = f"{a.floor_ft:,.0f}\u2013{a.ceiling_ft:,.0f} ft"
            ax.text(cx, cy, f"{label}\n{alt_label}", ha="center", va="center",
                    fontsize=7, color=color, weight="bold", transform=transform,
                    clip_on=True,
                    path_effects=[
                        _pe.withStroke(linewidth=2.5, foreground="white"),
                    ])

    # Near-miss buffer halos
    if buffer_m and near_misses:
        buf_deg = buffer_m / 111_000.0
        nm_airspace_ids = {id(nm.airspace) for nm in near_misses}
        for a in airspaces:
            if id(a) in nm_airspace_ids:
                buffered = a.geometry.buffer(buf_deg)
                x, y = buffered.exterior.xy
                ax.fill(x, y, alpha=0.08, color="gold", transform=transform)
                ax.plot(x, y, ":", color="gold", linewidth=1.5, alpha=0.5,
                        transform=transform)

    # Flight lines
    if flight_lines:
        conflicting_indices = {c.flight_line_index for c in conflicts}
        nm_indices = {nm.flight_line_index for nm in near_misses}

        for i, fl in enumerate(flight_lines):
            lons, lats = zip(*fl.geometry.coords)
            if i in conflicting_indices:
                color, style, lw = "red", "-", 2.5
            elif i in nm_indices:
                color, style, lw = "#DAA520", "--", 2
            else:
                color, style, lw = "green", "-", 2
            ax.plot(lons, lats, style, color=color, linewidth=lw, markersize=4,
                    marker="o", transform=transform)
            ax.annotate(fl.site_name, xy=(lons[0], lats[0]), fontsize=6,
                        xytext=(5, 5), textcoords="offset points",
                        transform=transform)

    # Conflict intersections
    for c in conflicts:
        geom = c.horizontal_intersection
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            if hasattr(part, "coords") and len(part.coords) >= 2:
                x, y = zip(*part.coords)
                ax.plot(x, y, color="red", linewidth=5, alpha=0.4,
                        transform=transform)

    # Entry/exit markers
    for c in conflicts:
        if c.entry_point:
            ax.plot(c.entry_point[0], c.entry_point[1], "D", color="red",
                    markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                    transform=transform, zorder=10)
            ax.annotate("E", xy=c.entry_point, fontsize=5, color="darkred",
                        weight="bold", xytext=(-8, 6), textcoords="offset points",
                        transform=transform)
        if c.exit_point and c.exit_point != c.entry_point:
            ax.plot(c.exit_point[0], c.exit_point[1], "D", color="orange",
                    markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                    transform=transform, zorder=10)
            ax.annotate("X", xy=c.exit_point, fontsize=5, color="darkorange",
                        weight="bold", xytext=(-8, 6), textcoords="offset points",
                        transform=transform)

    # Near-miss distance annotations
    for nm in near_misses:
        if nm.distance_to_boundary_m is not None and flight_lines:
            fl = flight_lines[nm.flight_line_index]
            mid = fl.geometry.interpolate(0.5, normalized=True)
            ax.annotate(
                f"{nm.distance_to_boundary_m:,.0f} m",
                xy=(mid.x, mid.y), fontsize=6, color="#B8860B",
                weight="bold", xytext=(8, -8), textcoords="offset points",
                transform=transform,
                arrowprops=dict(arrowstyle="->", color="#DAA520", lw=0.8),
            )

    # Legend
    legend_items = []
    if conflicts:
        legend_items.append(mpatches.Patch(color="red", alpha=0.4, label="Conflict"))
    if near_misses:
        legend_items.append(mpatches.Patch(
            facecolor="gold", edgecolor="#DAA520", alpha=0.3,
            linestyle="--", label="Near-miss buffer",
        ))
    if flight_lines:
        legend_items.append(mpatches.Patch(color="green", alpha=0.5, label="Clear line"))
    for cls in sorted(seen_classes):
        color, _ = _AIRSPACE_COLORS.get(cls, ("gray", 0.15))
        legend_items.append(mpatches.Patch(color=color, alpha=0.3, label=f"{cls}"))
    if inactive_airspaces:
        legend_items.append(mpatches.Patch(
            facecolor="lightgray", alpha=0.3, hatch="//", label="Inactive (schedule)",
        ))
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", fontsize=7, framealpha=0.9)

    # Extent: explicit, flight-line-based, or all features
    if extent is not None:
        ax.set_extent(extent, crs=transform)
    else:
        all_lons, all_lats = [], []
        # Prefer flight lines for extent if available
        if flight_lines:
            for fl in flight_lines:
                lons, lats = zip(*fl.geometry.coords)
                all_lons.extend(lons)
                all_lats.extend(lats)
        if not all_lons:
            # Fall back to airspace polygons
            for a in airspaces:
                polys = list(a.geometry.geoms) if a.geometry.geom_type == "MultiPolygon" else [a.geometry]
                for p in polys:
                    x, y = p.exterior.xy
                    all_lons.extend(x)
                    all_lats.extend(y)
            if inactive_airspaces:
                for a in inactive_airspaces:
                    polys = list(a.geometry.geoms) if a.geometry.geom_type == "MultiPolygon" else [a.geometry]
                    for p in polys:
                        x, y = p.exterior.xy
                        all_lons.extend(x)
                        all_lats.extend(y)
        if all_lons:
            buf = 0.15
            ax.set_extent([min(all_lons) - buf, max(all_lons) + buf,
                           min(all_lats) - buf, max(all_lats) + buf], crs=transform)

    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig, ax


def plot_conflict_matrix(
    flight_lines,
    airspaces,
    title="Conflict Matrix",
    figsize=None,
):
    """Plot a flight-line vs. airspace conflict matrix.

    Cells are colored: RED = full conflict, ORANGE = horizontal-only
    (altitude clear), GRAY = no intersection.

    Args:
        flight_lines: List of flight line objects.
        airspaces: List of Airspace objects.
        title: Plot title.
        figsize: Optional figure size; auto-sized if None.

    Returns:
        Matplotlib Figure and Axes.
    """
    from shapely import STRtree
    import matplotlib.patches as mpatches

    n_fl = len(flight_lines)
    n_as = len(airspaces)
    if figsize is None:
        figsize = (max(6, n_as * 0.8 + 2), max(4, n_fl * 0.6 + 2))

    as_geoms = [a.geometry for a in airspaces]
    tree = STRtree(as_geoms)

    fig, ax = plt.subplots(figsize=figsize)

    for fl_idx, fl in enumerate(flight_lines):
        fl_geom = fl.geometry
        fl_alt_ft = fl.altitude_msl.m_as(ureg.foot)
        candidates = set(tree.query(fl_geom, predicate="intersects"))

        for as_idx, airspace in enumerate(airspaces):
            if as_idx in candidates:
                overlap_floor = max(fl_alt_ft, airspace.floor_ft)
                overlap_ceil = min(fl_alt_ft, airspace.ceiling_ft)
                if overlap_floor <= overlap_ceil:
                    color = "#d32f2f"  # red — full conflict
                else:
                    color = "#ff9800"  # orange — horiz only
            else:
                color = "#e0e0e0"  # gray

            rect = plt.Rectangle((as_idx, fl_idx), 1, 1, facecolor=color,
                                 edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(as_idx + 0.5, fl_idx + 0.5,
                    f"{fl_alt_ft:,.0f}", ha="center", va="center",
                    fontsize=7, color="white" if color != "#e0e0e0" else "gray")

    ax.set_xlim(0, n_as)
    ax.set_ylim(0, n_fl)
    ax.set_xticks([i + 0.5 for i in range(n_as)])
    ax.set_xticklabels([a.name[:15] for a in airspaces], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(n_fl)])
    ax.set_yticklabels([fl.site_name for fl in flight_lines], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12)

    legend_items = [
        mpatches.Patch(color="#d32f2f", label="Conflict (horiz + vert)"),
        mpatches.Patch(color="#ff9800", label="Horiz only (alt clear)"),
        mpatches.Patch(color="#e0e0e0", label="No intersection"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7, bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    return fig, ax


def plot_vertical_profile(
    flight_line,
    airspaces,
    dem_file=None,
    title=None,
    figsize=(12, 5),
):
    """Plot an altitude cross-section along a flight line showing airspace bands.

    X-axis is distance along the route (NM), Y-axis is altitude (ft MSL).
    Airspace floors and ceilings are shown as colored bands.

    Args:
        flight_line: A single flight line object.
        airspaces: List of Airspace objects to check.
        dem_file: Optional DEM GeoTIFF path for terrain profile.
        title: Plot title; auto-generated if None.
        figsize: Figure size tuple.

    Returns:
        Matplotlib Figure and Axes.
    """
    import math
    from shapely.geometry import box as box_geom

    fig, ax = plt.subplots(figsize=figsize)

    fl_geom = flight_line.geometry
    fl_alt_ft = flight_line.altitude_msl.m_as(ureg.foot)

    # Compute total distance in NM
    coords = list(fl_geom.coords)
    total_dist_nm = 0
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        total_dist_nm += 2 * math.asin(math.sqrt(a)) * 3440.065

    # Terrain profile
    if dem_file:
        try:
            from .terrain import get_elevations
            n_pts = max(50, int(total_dist_nm * 2))
            fracs = np.linspace(0, 1, n_pts)
            pts = [fl_geom.interpolate(f, normalized=True) for f in fracs]
            lats = np.array([p.y for p in pts])
            lons = np.array([p.x for p in pts])
            elev_m = get_elevations(lats, lons, dem_file)
            elev_ft = elev_m * 3.28084
            dist_nm = fracs * total_dist_nm
            ax.fill_between(dist_nm, 0, elev_ft, color="saddlebrown", alpha=0.3, label="Terrain")
            ax.plot(dist_nm, elev_ft, color="saddlebrown", linewidth=0.8, alpha=0.6)
        except Exception:
            pass

    # Airspace bands — color by class, not just severity
    _profile_colors = {
        "RESTRICTED": ("#d32f2f", 0.25),
        "PROHIBITED": ("#b71c1c", 0.30),
        "WARNING_AREA": ("#e65100", 0.20),
        "SFRA": ("#ff6f00", 0.25),
        "TFR": ("#9c27b0", 0.20),
        "B": ("#1565c0", 0.20),
        "C": ("#7b1fa2", 0.18),
        "D": ("#00695c", 0.18),
        "E": ("#78909c", 0.10),
        "DANGER": ("#e65100", 0.20),
        "OTHER": ("#90a4ae", 0.12),
    }
    from .airspace import classify_severity
    for a in airspaces:
        if not fl_geom.intersects(a.geometry):
            continue

        intersection = fl_geom.intersection(a.geometry)
        if intersection.is_empty:
            continue

        # Compute distance range of intersection
        start_frac = fl_geom.project(intersection.interpolate(0, normalized=True), normalized=True)
        end_frac = fl_geom.project(intersection.interpolate(1, normalized=True), normalized=True)
        d_start = start_frac * total_dist_nm
        d_end = end_frac * total_dist_nm

        sev = classify_severity(a.airspace_type)
        color, alpha = _profile_colors.get(a.airspace_class, ("#90a4ae", 0.12))
        hatch = "///" if getattr(a, "floor_reference", "MSL") == "SFC" else None

        ax.fill_between(
            [d_start, d_end], a.floor_ft, a.ceiling_ft,
            color=color, alpha=alpha, hatch=hatch,
            label=f"{a.name[:25]} ({a.airspace_class})",
        )
        # Floor/ceiling edge lines
        ax.plot([d_start, d_end], [a.floor_ft, a.floor_ft], "-",
                color=color, linewidth=1.2, alpha=0.7)
        ax.plot([d_start, d_end], [a.ceiling_ft, a.ceiling_ft], "-",
                color=color, linewidth=1.2, alpha=0.7)
        # Vertical edges
        ax.plot([d_start, d_start], [a.floor_ft, a.ceiling_ft], "-",
                color=color, linewidth=0.8, alpha=0.5)
        ax.plot([d_end, d_end], [a.floor_ft, a.ceiling_ft], "-",
                color=color, linewidth=0.8, alpha=0.5)
        # Label inside the band
        mid_d = (d_start + d_end) / 2
        mid_alt = (a.floor_ft + a.ceiling_ft) / 2
        label_name = a.name if len(a.name) <= 20 else a.name[:17] + "..."
        ax.text(mid_d, mid_alt, label_name, ha="center", va="center",
                fontsize=6, color=color, alpha=0.8, weight="bold",
                path_effects=[_pe.withStroke(linewidth=2, foreground="white")])

    # Flight altitude line
    ax.axhline(y=fl_alt_ft, color="blue", linewidth=2.5,
               label=f"Flight alt: {fl_alt_ft:,.0f} ft", zorder=5)

    # Entry/exit markers from conflicts
    from shapely.geometry import Point
    from .airspace import check_airspace_conflicts
    conflicts = check_airspace_conflicts([flight_line], airspaces)
    for c in conflicts:
        if c.entry_point:
            entry_frac = fl_geom.project(Point(c.entry_point), normalized=True)
            ax.axvline(x=entry_frac * total_dist_nm, color="red", linestyle="--",
                       linewidth=1, alpha=0.6, label="Entry")
        if c.exit_point:
            exit_frac = fl_geom.project(Point(c.exit_point), normalized=True)
            ax.axvline(x=exit_frac * total_dist_nm, color="orange", linestyle="--",
                       linewidth=1, alpha=0.6, label="Exit")

    ax.set_xlabel("Distance along route (NM)")
    ax.set_ylabel("Altitude (ft MSL)")
    ax.set_title(title or f"Vertical Profile — {flight_line.site_name}", fontsize=12)
    ax.set_xlim(0, total_dist_nm)
    ax.set_ylim(bottom=0)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_oceanic_tracks(
    tracks,
    flight_lines=None,
    title="Oceanic Tracks",
    figsize=(16, 8),
    projection=None,
):
    """Plot NAT or PACOT oceanic tracks on a wide-projection cartopy map.

    Tracks are colored by direction: green=eastbound, blue=westbound.
    Waypoints are labeled along each track.

    Args:
        tracks: List of OceanicTrack objects.
        flight_lines: Optional list of flight lines to overlay.
        title: Plot title.
        figsize: Figure size tuple.
        projection: Cartopy projection; defaults to PlateCarree.

    Returns:
        Matplotlib Figure and Axes.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if projection is None:
        projection = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    ax.add_feature(cfeature.LAND, facecolor="#f0efe6")
    ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3)

    for t in tracks:
        if t.geometry is None or t.geometry.is_empty:
            continue

        # Color by direction
        if t.east_levels:
            color, style = "#2e7d32", "-"
            direction = "E"
        elif t.west_levels:
            color, style = "#1565c0", "-"
            direction = "W"
        else:
            color, style = "gray", "--"
            direction = "?"

        lons, lats = zip(*[(w[0], w[1]) for w in t.waypoints])
        ax.plot(lons, lats, style, color=color, linewidth=1.8, alpha=0.8,
                transform=transform)

        # Track label at midpoint
        mid_idx = len(t.waypoints) // 2
        mid_lon, mid_lat, _ = t.waypoints[mid_idx]
        levels = t.east_levels or t.west_levels or []
        level_str = f" FL{levels[0]}" if levels else ""
        ax.text(mid_lon, mid_lat + 0.5, f"{t.ident}{level_str}",
                fontsize=7, color=color, weight="bold", ha="center",
                transform=transform)

        # Waypoint markers
        for lon, lat, ident in t.waypoints:
            ax.plot(lon, lat, ".", color=color, markersize=4, transform=transform)
            if ident:
                ax.text(lon, lat - 0.8, ident, fontsize=4, color=color,
                        ha="center", alpha=0.7, transform=transform)

    # Overlay flight lines if provided
    if flight_lines:
        for fl in flight_lines:
            lons, lats = zip(*fl.geometry.coords)
            ax.plot(lons, lats, "-o", color="red", linewidth=2, markersize=4,
                    transform=transform)

    # Subtitle with validity
    if tracks:
        valid_from = tracks[0].valid_from[:16] if tracks[0].valid_from else "?"
        valid_to = tracks[0].valid_to[:16] if tracks[0].valid_to else "?"
        subtitle = f"Valid: {valid_from} \u2013 {valid_to}"
    else:
        subtitle = ""

    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    plt.tight_layout()
    return fig, ax


def map_airspace(
    airspaces,
    flight_lines=None,
    conflicts=None,
    near_misses=None,
    center=None,
    zoom_start=8,
) -> folium.Map:
    """Create an interactive Folium map with airspace overlays.

    Airspace polygons are displayed as GeoJSON with hover tooltips showing
    name, class, altitude, schedule, and severity. Flight lines are
    color-coded by conflict status. Layer groups allow toggling by type.

    Args:
        airspaces: List of Airspace objects.
        flight_lines: Optional list of flight line objects.
        conflicts: Optional list of AirspaceConflict objects.
        near_misses: Optional list of near-miss AirspaceConflict objects.
        center: (lat, lon) map center; auto-computed if None.
        zoom_start: Initial zoom level.

    Returns:
        folium.Map with all layers.
    """
    from shapely.geometry import mapping
    from .airspace import classify_severity

    conflicts = conflicts or []
    near_misses = near_misses or []

    # Auto-center
    if center is None:
        all_lats, all_lons = [], []
        for a in airspaces:
            c = a.geometry.centroid
            all_lats.append(c.y)
            all_lons.append(c.x)
        if flight_lines:
            for fl in flight_lines:
                for lon, lat in fl.geometry.coords:
                    all_lats.append(lat)
                    all_lons.append(lon)
        center = (np.mean(all_lats), np.mean(all_lons)) if all_lats else (0, 0)

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Group airspaces by type for layer control
    _color_map = {
        "RESTRICTED": "red", "PROHIBITED": "darkred", "DANGER": "orange",
        "SFRA": "darkorange", "TFR": "purple", "B": "blue", "C": "purple",
        "D": "teal", "E": "steelblue", "OTHER": "gray",
    }

    layer_groups = {}
    for a in airspaces:
        group_name = a.airspace_class if a.airspace_class in _color_map else "OTHER"
        if group_name not in layer_groups:
            layer_groups[group_name] = folium.FeatureGroup(name=group_name)

        color = _color_map.get(group_name, "gray")
        severity = classify_severity(a.airspace_type)

        tooltip_text = (
            f"<b>{a.name}</b><br>"
            f"Class: {a.airspace_class}<br>"
            f"Alt: {a.floor_ft:,.0f} \u2013 {a.ceiling_ft:,.0f} ft {a.floor_reference}<br>"
            f"Severity: {severity}<br>"
        )
        if a.schedule:
            tooltip_text += f"Schedule: {a.schedule}<br>"
        if a.effective_start:
            tooltip_text += f"Effective: {a.effective_start}<br>"

        geojson = mapping(a.geometry)
        folium.GeoJson(
            geojson,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c,
                "weight": 1.5, "fillOpacity": 0.2,
            },
            tooltip=folium.Tooltip(tooltip_text),
        ).add_to(layer_groups[group_name])

        # Permanent label at centroid
        centroid = a.geometry.centroid
        label_name = a.name if len(a.name) <= 20 else a.name[:17] + "..."
        folium.Marker(
            location=(centroid.y, centroid.x),
            icon=folium.DivIcon(html=(
                f'<div style="font-size:9px;color:{color};font-weight:bold;'
                f'text-shadow:1px 1px 1px white,-1px -1px 1px white,'
                f'1px -1px 1px white,-1px 1px 1px white;'
                f'white-space:nowrap">{label_name}</div>'
            )),
        ).add_to(layer_groups[group_name])

    for group in layer_groups.values():
        group.add_to(m)

    # Flight lines
    if flight_lines:
        fl_group = folium.FeatureGroup(name="Flight Lines")
        conflicting_indices = {c.flight_line_index for c in conflicts}
        nm_indices = {nm.flight_line_index for nm in near_misses}

        for i, fl in enumerate(flight_lines):
            coords = [(lat, lon) for lon, lat in fl.geometry.coords]
            if i in conflicting_indices:
                color, dash = "red", None
            elif i in nm_indices:
                color, dash = "gold", "8 4"
            else:
                color, dash = "green", None

            folium.PolyLine(
                coords, color=color, weight=3,
                dash_array=dash,
                tooltip=fl.site_name,
            ).add_to(fl_group)

            # Permanent label at midpoint
            mid = fl.geometry.interpolate(0.5, normalized=True)
            folium.Marker(
                location=(mid.y, mid.x),
                icon=folium.DivIcon(html=(
                    f'<div style="font-size:10px;color:{color};font-weight:bold;'
                    f'text-shadow:1px 1px 1px white,-1px -1px 1px white,'
                    f'1px -1px 1px white,-1px 1px 1px white;'
                    f'white-space:nowrap">{fl.site_name}</div>'
                )),
            ).add_to(fl_group)
        fl_group.add_to(m)

    # Entry/exit markers
    if conflicts:
        marker_group = folium.FeatureGroup(name="Entry/Exit Points")
        for c in conflicts:
            if c.entry_point:
                folium.CircleMarker(
                    location=(c.entry_point[1], c.entry_point[0]),
                    radius=5, color="red", fill=True, fill_color="red",
                    tooltip=f"Entry: {c.airspace.name[:30]}",
                ).add_to(marker_group)
            if c.exit_point and c.exit_point != c.entry_point:
                folium.CircleMarker(
                    location=(c.exit_point[1], c.exit_point[0]),
                    radius=5, color="orange", fill=True, fill_color="orange",
                    tooltip=f"Exit: {c.airspace.name[:30]}",
                ).add_to(marker_group)
        marker_group.add_to(m)

    # Near-miss annotations
    if near_misses and flight_lines:
        nm_group = folium.FeatureGroup(name="Near-Misses")
        for nm in near_misses:
            fl = flight_lines[nm.flight_line_index]
            mid = fl.geometry.interpolate(0.5, normalized=True)
            dist_str = f"{nm.distance_to_boundary_m:,.0f} m" if nm.distance_to_boundary_m else "?"
            folium.Marker(
                location=(mid.y, mid.x),
                icon=folium.DivIcon(html=f'<div style="font-size:10px;color:#B8860B;font-weight:bold">{dist_str}</div>'),
                tooltip=f"Near-miss: {nm.airspace.name[:30]} ({dist_str})",
            ).add_to(nm_group)
        nm_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
