import logging
import os
import time
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import simplekml
from shapely.geometry import Point, LineString, Polygon
from sunposition import sunpos
import pymap3d.vincenty

from .terrain import get_cache_root
from .download import download_file
from .geometry import wrap_to_180

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SatelliteInfo:
    """Descriptor for an Earth observation satellite."""
    name: str
    norad_id: int
    swath_width_km: float
    celestrak_group: str = "earth-observation"
    max_sza: float = 90.0
    aliases: List[str] = field(default_factory=list)


SATELLITE_REGISTRY: Dict[str, SatelliteInfo] = {
    "PACE": SatelliteInfo("PACE", 58927, swath_width_km=2663, max_sza=75.0),
    "Landsat-8": SatelliteInfo("Landsat-8", 39084, swath_width_km=185, celestrak_group="resource", max_sza=75.0),
    "Landsat-9": SatelliteInfo("Landsat-9", 49260, swath_width_km=185, celestrak_group="resource", max_sza=75.0),
    "Sentinel-2A": SatelliteInfo("Sentinel-2A", 40697, swath_width_km=290, max_sza=75.0),
    "Sentinel-2B": SatelliteInfo("Sentinel-2B", 42063, swath_width_km=290, max_sza=75.0),
    "Sentinel-3A": SatelliteInfo("Sentinel-3A", 41335, swath_width_km=1270, max_sza=75.0),
    "Sentinel-3B": SatelliteInfo("Sentinel-3B", 43437, swath_width_km=1270, max_sza=75.0),
    "JPSS-1": SatelliteInfo("JPSS-1 (NOAA-20)", 43013, swath_width_km=3000, celestrak_group="noaa", max_sza=90.0),
    "JPSS-2": SatelliteInfo("JPSS-2 (NOAA-21)", 54234, swath_width_km=3000, celestrak_group="noaa", max_sza=90.0),
    "Aqua": SatelliteInfo("Aqua", 27424, swath_width_km=2330, max_sza=90.0),
    "Terra": SatelliteInfo("Terra", 25994, swath_width_km=2330, max_sza=90.0),
    "ICESat-2": SatelliteInfo("ICESat-2", 43613, swath_width_km=6.5, max_sza=180.0),
    "CALIPSO": SatelliteInfo("CALIPSO", 29108, swath_width_km=0.1, max_sza=180.0),
    "CloudSat": SatelliteInfo("CloudSat", 29107, swath_width_km=1.4, max_sza=180.0),
    "EarthCARE": SatelliteInfo("EarthCARE", 60313, swath_width_km=150, max_sza=180.0),
}


def get_satellite(name: str) -> SatelliteInfo:
    """Look up a satellite by name or alias.

    Args:
        name: Satellite name (key in SATELLITE_REGISTRY) or an alias.

    Returns:
        SatelliteInfo for the requested satellite.

    Raises:
        ValueError: If the satellite is not found.
    """
    if name in SATELLITE_REGISTRY:
        return SATELLITE_REGISTRY[name]
    for sat in SATELLITE_REGISTRY.values():
        if name in sat.aliases or name == sat.name:
            return sat
    raise ValueError(
        f"Unknown satellite: {name}. "
        f"Available: {list(SATELLITE_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# TLE caching
# ---------------------------------------------------------------------------

def _get_tle_cache_dir() -> str:
    """Return the TLE cache subdirectory under HyPlan's cache root."""
    cache_dir = os.path.join(get_cache_root(), "tle_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _tle_cache_path(norad_id: int) -> str:
    """Return the file path for a cached TLE file."""
    return os.path.join(_get_tle_cache_dir(), f"{norad_id}.tle")


def _is_tle_stale(cache_path: str, max_age_hours: float = 24.0) -> bool:
    """Check if a cached TLE file is older than max_age_hours."""
    if not os.path.exists(cache_path):
        return True
    age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
    return age_hours > max_age_hours


def fetch_tle(
    satellite: Union[str, SatelliteInfo],
    max_age_hours: float = 24.0,
):
    """Fetch the TLE for a satellite, using cache when available.

    Downloads from CelesTrak if the cache is missing or stale.

    Args:
        satellite: Satellite name or SatelliteInfo object.
        max_age_hours: Maximum age of cached TLE before re-fetching.

    Returns:
        skyfield.sgp4lib.EarthSatellite: Loaded satellite object.

    Raises:
        RuntimeError: If the TLE cannot be fetched or parsed.
    """
    from skyfield.api import load as sf_load

    if isinstance(satellite, str):
        satellite = get_satellite(satellite)

    cache_path = _tle_cache_path(satellite.norad_id)
    url = (
        f"https://celestrak.org/NORAD/elements/gp.php"
        f"?CATNR={satellite.norad_id}&FORMAT=TLE"
    )

    if _is_tle_stale(cache_path, max_age_hours):
        logger.info(f"Fetching TLE for {satellite.name} (NORAD {satellite.norad_id})")
        download_file(cache_path, url, replace=True)

    # Parse the TLE file
    with open(cache_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 2:
        raise RuntimeError(
            f"TLE file for {satellite.name} has fewer than 2 lines: {cache_path}"
        )

    # Skyfield expects name + two TLE lines
    ts = sf_load.timescale()
    if len(lines) == 3:
        # name, line1, line2
        from skyfield.api import EarthSatellite
        sat = EarthSatellite(lines[1], lines[2], name=lines[0], ts=ts)
    else:
        # line1, line2 only
        from skyfield.api import EarthSatellite
        sat = EarthSatellite(lines[0], lines[1], name=satellite.name, ts=ts)

    # Warn if TLE epoch is far from now
    epoch_jd = sat.epoch.tt
    now_jd = ts.now().tt
    days_from_epoch = abs(now_jd - epoch_jd)
    if days_from_epoch > 7:
        logger.warning(
            f"TLE for {satellite.name} is {days_from_epoch:.1f} days from epoch. "
            f"Predictions may be inaccurate beyond ~7 days."
        )

    return sat


def clear_tle_cache(confirm: bool = True) -> None:
    """Clear the TLE cache directory.

    Args:
        confirm: If True, prompt the user for confirmation before clearing.
    """
    tle_dir = os.path.join(get_cache_root(), "tle_cache")
    if not os.path.exists(tle_dir):
        logger.info(f"TLE cache directory {tle_dir} does not exist.")
        return

    if confirm:
        user_input = input(
            f"Are you sure you want to delete all files in {tle_dir}? (yes/no): "
        ).strip().lower()
        if user_input not in ("yes", "y"):
            logger.info("TLE cache clear operation canceled by the user.")
            return

    shutil.rmtree(tle_dir)
    logger.info(f"TLE cache directory {tle_dir} cleared successfully.")


# ---------------------------------------------------------------------------
# Ground track computation
# ---------------------------------------------------------------------------

def compute_ground_track(
    satellite: Union[str, SatelliteInfo],
    start_time: datetime,
    end_time: datetime,
    time_step_s: float = 30.0,
    max_tle_age_hours: float = 24.0,
) -> gpd.GeoDataFrame:
    """Compute the satellite ground track over a time window.

    Args:
        satellite: Satellite name or SatelliteInfo object.
        start_time: Start of the time window (UTC datetime).
        end_time: End of the time window (UTC datetime).
        time_step_s: Time step for propagation in seconds.
        max_tle_age_hours: Maximum TLE cache age.

    Returns:
        GeoDataFrame with columns: satellite_name, norad_id, timestamp,
        latitude, longitude, altitude_km, solar_zenith. Geometry is Point.
    """
    from skyfield.api import load as sf_load

    if isinstance(satellite, str):
        satellite = get_satellite(satellite)

    sat_obj = fetch_tle(satellite, max_age_hours=max_tle_age_hours)
    ts = sf_load.timescale()

    # Build time array
    total_seconds = (end_time - start_time).total_seconds()
    n_steps = int(total_seconds / time_step_s) + 1
    dt_offsets = [start_time + timedelta(seconds=i * time_step_s) for i in range(n_steps)]

    from skyfield.api import utc as sf_utc
    t_array = ts.from_datetimes([dt.replace(tzinfo=sf_utc) for dt in dt_offsets])

    # Propagate
    geocentric = sat_obj.at(t_array)
    subpoint = geocentric.subpoint()

    lats = subpoint.latitude.degrees
    lons = wrap_to_180(subpoint.longitude.degrees)
    alt_km = subpoint.elevation.km

    # Solar zenith at each sub-satellite point
    timestamps = np.array(dt_offsets)
    solar_az, solar_zen, _, _, _ = sunpos(
        dt=timestamps,
        latitude=lats,
        longitude=lons,
        elevation=0.0,
        radians=False,
    )

    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]

    gdf = gpd.GeoDataFrame(
        {
            "satellite_name": satellite.name,
            "norad_id": satellite.norad_id,
            "timestamp": timestamps,
            "latitude": lats,
            "longitude": lons,
            "altitude_km": alt_km,
            "solar_zenith": solar_zen,
        },
        geometry=geometry,
        crs="EPSG:4326",
    )

    return gdf


# ---------------------------------------------------------------------------
# Swath footprint
# ---------------------------------------------------------------------------

def _compute_headings(lats, lons):
    """Compute forward azimuths between consecutive ground track points."""
    headings = np.zeros(len(lats))
    for i in range(len(lats) - 1):
        _, az = pymap3d.vincenty.vdist(lats[i], lons[i], lats[i + 1], lons[i + 1])
        headings[i] = az
    headings[-1] = headings[-2] if len(headings) > 1 else 0.0
    return headings


def _segment_passes(lats, timestamps, time_step_s):
    """Split a ground track into individual passes.

    A new pass starts when there is a time gap > 2 * time_step_s or when the
    latitude direction reverses (crossing a pole).

    Returns:
        List of (start_idx, end_idx) tuples.
    """
    if len(lats) < 2:
        return [(0, len(lats))]

    breaks = [0]
    for i in range(1, len(lats)):
        td = timestamps[i] - timestamps[i - 1]
        # Handle both datetime and numpy datetime64
        if hasattr(td, 'total_seconds'):
            dt = td.total_seconds()
        else:
            dt = td / np.timedelta64(1, 's')
        if dt > 2 * time_step_s:
            breaks.append(i)
    breaks.append(len(lats))

    passes = []
    for i in range(len(breaks) - 1):
        start, end = breaks[i], breaks[i + 1]
        if end - start >= 2:
            passes.append((start, end))

    return passes


def compute_swath_footprint(
    ground_track_gdf: gpd.GeoDataFrame,
    swath_width_km: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Generate swath footprint polygons from a ground track GeoDataFrame.

    For each pass segment, constructs a polygon by offsetting perpendicular to
    the track direction by half the swath width on each side.

    Args:
        ground_track_gdf: GeoDataFrame from compute_ground_track().
        swath_width_km: Override swath width in km. If None, looks up the
            satellite's swath_width_km from SATELLITE_REGISTRY.

    Returns:
        GeoDataFrame with one row per pass segment. Columns: satellite_name,
        norad_id, pass_start, pass_end, ascending, geometry (Polygon).
    """
    if ground_track_gdf.empty:
        return gpd.GeoDataFrame(
            columns=["satellite_name", "norad_id", "pass_start", "pass_end",
                     "ascending", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    sat_name = ground_track_gdf["satellite_name"].iloc[0]
    norad_id = ground_track_gdf["norad_id"].iloc[0]

    if swath_width_km is None:
        sat_info = get_satellite(sat_name)
        swath_width_km = sat_info.swath_width_km

    half_swath_m = swath_width_km * 1000.0 / 2.0

    lats = ground_track_gdf["latitude"].values
    lons = ground_track_gdf["longitude"].values
    timestamps = ground_track_gdf["timestamp"].values

    time_step_s = 30.0
    if len(timestamps) > 1:
        dt0 = (pd.Timestamp(timestamps[1]) - pd.Timestamp(timestamps[0])).total_seconds()
        time_step_s = max(dt0, 1.0)

    headings = _compute_headings(lats, lons)
    passes = _segment_passes(lats, timestamps, time_step_s)

    rows = []
    for start, end in passes:
        p_lats = lats[start:end]
        p_lons = lons[start:end]
        p_headings = headings[start:end]
        p_times = timestamps[start:end]

        # Perpendicular azimuths
        az_port = (p_headings - 90.0) % 360.0
        az_starboard = (p_headings + 90.0) % 360.0

        # Offset points
        port_lats, port_lons = pymap3d.vincenty.vreckon(
            p_lats, p_lons, half_swath_m, az_port
        )
        star_lats, star_lons = pymap3d.vincenty.vreckon(
            p_lats, p_lons, half_swath_m, az_starboard
        )

        port_lons = wrap_to_180(port_lons)
        star_lons = wrap_to_180(star_lons)

        # Build polygon: port side forward, starboard side reversed
        poly_lons = np.concatenate([port_lons, star_lons[::-1]])
        poly_lats = np.concatenate([port_lats, star_lats[::-1]])

        # Check for antimeridian crossing (large longitude jump)
        lon_diffs = np.abs(np.diff(poly_lons))
        if np.any(lon_diffs > 180):
            logger.warning(
                f"Swath polygon for {sat_name} crosses the antimeridian. "
                f"Polygon may be invalid in EPSG:4326."
            )

        polygon = Polygon(zip(poly_lons, poly_lats))
        ascending = p_lats[-1] > p_lats[0]

        rows.append({
            "satellite_name": sat_name,
            "norad_id": norad_id,
            "pass_start": pd.Timestamp(p_times[0]),
            "pass_end": pd.Timestamp(p_times[-1]),
            "ascending": ascending,
            "geometry": polygon,
        })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Overpass finding
# ---------------------------------------------------------------------------

def find_overpasses(
    satellite: Union[str, SatelliteInfo],
    region: Union[Polygon, gpd.GeoDataFrame],
    start_time: datetime,
    end_time: datetime,
    time_step_s: float = 10.0,
    max_sza: Optional[float] = None,
    include_swath: bool = True,
    max_tle_age_hours: float = 24.0,
) -> gpd.GeoDataFrame:
    """Find satellite passes over a geographic region within a time window.

    Args:
        satellite: Satellite name or SatelliteInfo object.
        region: Shapely Polygon or GeoDataFrame defining the area of interest.
        start_time: Start of search window (UTC datetime).
        end_time: End of search window (UTC datetime).
        time_step_s: Time step in seconds for propagation.
        max_sza: Maximum solar zenith angle for usable passes. If None, uses
            the satellite's default max_sza.
        include_swath: If True, geometry is swath Polygon. If False, geometry
            is ground track LineString.
        max_tle_age_hours: Maximum TLE cache age.

    Returns:
        GeoDataFrame with one row per overpass. Columns: satellite_name,
        norad_id, pass_start, pass_end, ascending, ground_track,
        solar_zenith_at_center, is_usable.
    """
    if isinstance(satellite, str):
        satellite = get_satellite(satellite)

    if max_sza is None:
        max_sza = satellite.max_sza

    # Extract region polygon
    if isinstance(region, gpd.GeoDataFrame):
        region_poly = region.geometry.unary_union.convex_hull
    else:
        region_poly = region

    # Buffer the region by half the swath width for candidate detection
    # Use a rough degree approximation (1 deg ~ 111 km)
    buffer_deg = (satellite.swath_width_km / 2.0) / 111.0
    search_region = region_poly.buffer(buffer_deg)

    # Coarse scan: propagate at 10x the requested time step
    coarse_step = min(time_step_s * 10, 60.0)
    coarse_track = compute_ground_track(
        satellite, start_time, end_time,
        time_step_s=coarse_step,
        max_tle_age_hours=max_tle_age_hours,
    )

    if coarse_track.empty:
        return _empty_overpass_gdf()

    # Find coarse points within the buffered region
    within_mask = coarse_track.geometry.within(search_region)
    if not within_mask.any():
        return _empty_overpass_gdf()

    # Identify time windows around candidate points
    candidate_times = coarse_track.loc[within_mask, "timestamp"].values
    windows = _merge_time_windows(candidate_times, margin_s=coarse_step * 2)

    # Fine scan each window
    fine_tracks = []
    for win_start, win_end in windows:
        ft = compute_ground_track(
            satellite, win_start, win_end,
            time_step_s=time_step_s,
            max_tle_age_hours=max_tle_age_hours,
        )
        fine_tracks.append(ft)

    if not fine_tracks:
        return _empty_overpass_gdf()

    fine_track = pd.concat(fine_tracks, ignore_index=True)
    fine_track = gpd.GeoDataFrame(fine_track, geometry="geometry", crs="EPSG:4326")

    # Filter to points within the search region
    within_fine = fine_track.geometry.within(search_region)
    if not within_fine.any():
        return _empty_overpass_gdf()

    fine_track_in = fine_track.loc[within_fine].copy()

    # Segment into individual passes
    lats = fine_track_in["latitude"].values
    timestamps = fine_track_in["timestamp"].values
    passes = _segment_passes(lats, timestamps, time_step_s)

    rows = []
    for start, end in passes:
        seg = fine_track_in.iloc[start:end]
        if len(seg) < 2:
            continue

        p_lats = seg["latitude"].values
        p_lons = seg["longitude"].values
        p_times = seg["timestamp"].values
        p_sza = seg["solar_zenith"].values

        # Center point solar zenith
        mid_idx = len(p_sza) // 2
        sza_center = p_sza[mid_idx]
        is_usable = sza_center < max_sza

        ascending = p_lats[-1] > p_lats[0]
        ground_track_line = LineString(zip(p_lons, p_lats))

        if include_swath:
            # Build swath polygon for this pass segment
            seg_gdf = seg.copy().reset_index(drop=True)
            swath_gdf = compute_swath_footprint(seg_gdf, satellite.swath_width_km)
            if not swath_gdf.empty:
                geom = swath_gdf.geometry.iloc[0]
            else:
                geom = ground_track_line
        else:
            geom = ground_track_line

        rows.append({
            "satellite_name": satellite.name,
            "norad_id": satellite.norad_id,
            "pass_start": pd.Timestamp(p_times[0]),
            "pass_end": pd.Timestamp(p_times[-1]),
            "pass_duration_s": (pd.Timestamp(p_times[-1]) - pd.Timestamp(p_times[0])).total_seconds(),
            "ascending": ascending,
            "ground_track": ground_track_line,
            "solar_zenith_at_center": sza_center,
            "is_usable": is_usable,
            "geometry": geom,
        })

    if not rows:
        return _empty_overpass_gdf()

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def find_all_overpasses(
    satellites: Optional[List[Union[str, SatelliteInfo]]] = None,
    region: Union[Polygon, gpd.GeoDataFrame] = None,
    start_time: datetime = None,
    end_time: datetime = None,
    max_sza: Optional[float] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Find overpasses for multiple satellites and concatenate results.

    Args:
        satellites: List of satellite names or SatelliteInfo objects. If None,
            uses all entries in SATELLITE_REGISTRY.
        region: Geographic region of interest.
        start_time: Start of search window (UTC).
        end_time: End of search window (UTC).
        max_sza: Override max SZA for all satellites. If None, uses each
            satellite's default.
        **kwargs: Additional keyword arguments passed to find_overpasses().

    Returns:
        GeoDataFrame of combined overpass results, sorted by pass_start.
    """
    if satellites is None:
        satellites = list(SATELLITE_REGISTRY.keys())

    results = []
    for sat in satellites:
        try:
            gdf = find_overpasses(
                sat, region, start_time, end_time,
                max_sza=max_sza, **kwargs,
            )
            if not gdf.empty:
                results.append(gdf)
        except Exception as e:
            sat_name = sat if isinstance(sat, str) else sat.name
            logger.warning(f"Failed to compute overpasses for {sat_name}: {e}")

    if not results:
        return _empty_overpass_gdf()

    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    combined = combined.sort_values("pass_start").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Flight plan overlap
# ---------------------------------------------------------------------------

def compute_overpass_overlap(
    flight_plan_gdf: gpd.GeoDataFrame,
    overpasses_gdf: gpd.GeoDataFrame,
    flight_time_utc: datetime,
    max_time_offset_min: float = 60.0,
) -> gpd.GeoDataFrame:
    """Compute spatial overlap between flight plan segments and satellite overpasses.

    Args:
        flight_plan_gdf: GeoDataFrame from flight_plan.compute_flight_plan().
        overpasses_gdf: GeoDataFrame from find_overpasses().
        flight_time_utc: UTC datetime when the flight starts (used to compute
            absolute times for flight plan segments).
        max_time_offset_min: Maximum time offset (minutes) to consider as
            a valid overlap.

    Returns:
        GeoDataFrame with overlap results. Columns: satellite_name,
        segment_name, time_offset_min, overlap_area_km2, geometry.
    """
    from .geometry import get_utm_transforms

    if overpasses_gdf.empty or flight_plan_gdf.empty:
        return gpd.GeoDataFrame(
            columns=["satellite_name", "segment_name", "time_offset_min",
                     "overlap_area_km2", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    rows = []
    for _, overpass in overpasses_gdf.iterrows():
        overpass_geom = overpass["geometry"]
        pass_mid = overpass["pass_start"] + (overpass["pass_end"] - overpass["pass_start"]) / 2

        for _, segment in flight_plan_gdf.iterrows():
            seg_geom = segment["geometry"]
            if seg_geom is None or seg_geom.is_empty:
                continue

            # Compute time offset
            if "time_to_segment" in segment and pd.notna(segment.get("time_to_segment")):
                seg_time = flight_time_utc + timedelta(
                    hours=float(segment["time_to_segment"])
                )
                time_offset = abs((pass_mid - seg_time).total_seconds()) / 60.0
            else:
                time_offset = float("nan")

            if not np.isnan(time_offset) and time_offset > max_time_offset_min:
                continue

            # Spatial intersection
            if not overpass_geom.intersects(seg_geom):
                continue

            intersection = overpass_geom.intersection(seg_geom)
            if intersection.is_empty:
                continue

            # Compute area in km^2 using UTM
            centroid = intersection.centroid
            to_utm, from_utm = get_utm_transforms(centroid.x, centroid.y)
            intersection_utm = intersection  # simplified: use degree-based area
            try:
                from shapely.ops import transform
                intersection_utm = transform(to_utm, intersection)
                area_km2 = intersection_utm.area / 1e6
            except Exception:
                area_km2 = float("nan")

            seg_name = segment.get("segment_name", "")

            rows.append({
                "satellite_name": overpass["satellite_name"],
                "segment_name": seg_name,
                "time_offset_min": time_offset,
                "overlap_area_km2": area_km2,
                "geometry": intersection,
            })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["satellite_name", "segment_name", "time_offset_min",
                     "overlap_area_km2", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# KML export
# ---------------------------------------------------------------------------

def overpasses_to_kml(
    overpasses_gdf: gpd.GeoDataFrame,
    kml_filename: str,
) -> None:
    """Export overpass results to a KML file.

    Each pass becomes a KML Placemark with the satellite name and time
    in the description.

    Args:
        overpasses_gdf: GeoDataFrame from find_overpasses().
        kml_filename: Output KML file path.
    """
    kml = simplekml.Kml()

    colors = [
        simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green,
        simplekml.Color.yellow, simplekml.Color.cyan, simplekml.Color.magenta,
        simplekml.Color.orange, simplekml.Color.white,
    ]

    sat_names = overpasses_gdf["satellite_name"].unique()
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(sat_names)}

    for _, row in overpasses_gdf.iterrows():
        geom = row["geometry"]
        sat_name = row["satellite_name"]
        t_start = row["pass_start"]
        t_end = row["pass_end"]
        sza = row.get("solar_zenith_at_center", float("nan"))
        usable = row.get("is_usable", True)

        description = (
            f"Satellite: {sat_name}\n"
            f"Pass start: {t_start}\n"
            f"Pass end: {t_end}\n"
            f"SZA at center: {sza:.1f}°\n"
            f"Usable: {usable}"
        )

        name = f"{sat_name} {t_start}"
        color = color_map.get(sat_name, simplekml.Color.white)

        if geom.geom_type == "Polygon":
            coords = [(lon, lat) for lon, lat in geom.exterior.coords]
            pol = kml.newpolygon(name=name, outerboundaryis=coords)
            pol.description = description
            pol.style.polystyle.color = simplekml.Color.changealpha("77", color)
            pol.style.linestyle.color = color
            pol.style.linestyle.width = 2
        elif geom.geom_type == "LineString":
            coords = [(lon, lat) for lon, lat in geom.coords]
            ls = kml.newlinestring(name=name, coords=coords)
            ls.description = description
            ls.style.linestyle.color = color
            ls.style.linestyle.width = 2

    kml.save(kml_filename)
    logger.info(f"Overpasses exported to KML: {kml_filename}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_overpass_gdf() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with the overpass schema."""
    return gpd.GeoDataFrame(
        columns=["satellite_name", "norad_id", "pass_start", "pass_end",
                 "pass_duration_s", "ascending", "ground_track",
                 "solar_zenith_at_center", "is_usable", "geometry"],
        geometry="geometry", crs="EPSG:4326",
    )


def _merge_time_windows(timestamps, margin_s=120.0):
    """Merge nearby timestamps into contiguous time windows.

    Args:
        timestamps: Array of numpy datetime64 or datetime objects.
        margin_s: Merge windows closer than this many seconds.

    Returns:
        List of (start_datetime, end_datetime) tuples.
    """
    if len(timestamps) == 0:
        return []

    ts_sorted = np.sort(timestamps)
    windows = []
    win_start = pd.Timestamp(ts_sorted[0]) - timedelta(seconds=margin_s)
    win_end = pd.Timestamp(ts_sorted[0]) + timedelta(seconds=margin_s)

    for t in ts_sorted[1:]:
        t_pd = pd.Timestamp(t)
        if t_pd - win_end < timedelta(seconds=margin_s):
            win_end = t_pd + timedelta(seconds=margin_s)
        else:
            windows.append((win_start.to_pydatetime(), win_end.to_pydatetime()))
            win_start = t_pd - timedelta(seconds=margin_s)
            win_end = t_pd + timedelta(seconds=margin_s)

    windows.append((win_start.to_pydatetime(), win_end.to_pydatetime()))
    return windows
