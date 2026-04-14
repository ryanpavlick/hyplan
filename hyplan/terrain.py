"""Terrain analysis using Copernicus DEM data.

Downloads, caches, and queries 30-meter Copernicus GLO-30 DEM tiles
from AWS. Provides bulk elevation lookup, DEM tile merging via rasterio,
and a vectorized ray-terrain intersection algorithm for computing
off-nadir ground intersection points.

References
----------
Data source: Copernicus DEM GLO-30, European Space Agency, distributed
via AWS Open Data (s3://copernicus-dem-30m).
"""

import logging
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from shapely.geometry import box
from rtree import index
import rasterio
import rasterio.merge
from typing import List, Tuple
import pymap3d.los
import pymap3d.aer

from .download import download_file
from .exceptions import HyPlanRuntimeError, HyPlanValueError
from .geometry import process_linestring

logger = logging.getLogger(__name__)

__all__ = [
    "get_cache_root",
    "clear_cache",
    "clear_localdem_cache",
    "build_tile_index",
    "download_dem_files",
    "merge_tiles",
    "generate_demfile",
    "get_elevations",
    "get_min_max_elevations",
    "ray_terrain_intersection",
    "terrain_elevation_along_track",
    "terrain_aspect_azimuth",
    "surface_normal_at",
]

# Minimum cos(tilt) magnitude below which ray-terrain intersection is undefined
_COS_TILT_MIN = 1e-6


def get_cache_root(custom_path: str = None) -> str:
    """Get the root directory for caching files."""
    return custom_path or os.environ.get("HYPLAN_CACHE_ROOT", f"{tempfile.gettempdir()}/hyplan")

def clear_cache() -> None:
    """Clears the entire cache directory after confirming it is safe to do so."""
    cache_dir = get_cache_root()
    if not cache_dir.startswith(tempfile.gettempdir()):
        raise HyPlanValueError(f"Refusing to clear unsafe cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"Cache directory {cache_dir} cleared.")
    else:
        logger.info(f"Cache directory {cache_dir} does not exist.")

def clear_localdem_cache(confirm: bool = True) -> None:
    """
    Clears the local DEM cache directory.

    This removes all files in the 'localdem' subdirectory of the cache root,
    which stores downloaded DEM tiles.

    Args:
        confirm (bool): If True, prompt the user for confirmation before clearing the cache.
    """
    localdem_dir = os.path.join(get_cache_root(), "localdem")

    if not os.path.exists(localdem_dir):
        logger.info(f"Local DEM cache directory {localdem_dir} does not exist.")
        return

    cache_root = get_cache_root()
    if not os.path.commonpath([localdem_dir, cache_root]) == cache_root:
        raise HyPlanValueError(f"Refusing to clear unsafe directory: {localdem_dir}")

    if confirm:
        user_input = input(f"Are you sure you want to delete all files in {localdem_dir}? (yes/no): ").strip().lower()
        if user_input not in ("yes", "y"):
            logger.info("Local DEM cache clear operation canceled by the user.")
            return

    try:
        shutil.rmtree(localdem_dir)
        logger.info(f"Local DEM cache directory {localdem_dir} cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to clear local DEM cache: {e}")
        raise


def build_tile_index(tile_list_file: str) -> Tuple[index.Index, List[Tuple[str, box]]]:
    """
    Build an R-tree spatial index for DEM tiles from a tile list file.

    Each line in the file is a tile name encoding lat/lon in the filename.
    Tiles are parsed into 1x1 degree bounding boxes and indexed spatially.

    Args:
        tile_list_file (str): Path to the text file listing available DEM tiles.

    Returns:
        Tuple[index.Index, List[Tuple[str, box]]]: A tuple of (rtree_index,
            tile_bboxes) where tile_bboxes is a list of (tile_name, bounding_box) pairs.
    """
    idx = index.Index()
    tile_bboxes = []

    with open(tile_list_file) as file:
        for i, line in enumerate(file):
            tile = line.strip()
            try:
                lat, _, lon = tile.replace("_COG", "").split("_")[3:6]
                lon = -1 * float(lon[1:]) if "W" in lon else float(lon[1:])
                lat = -1 * float(lat[1:]) if "S" in lat else float(lat[1:])
                bbox = box(lon, lat, lon + 1, lat + 1)
                idx.insert(i, bbox.bounds)
                tile_bboxes.append((tile, bbox))
            except Exception as e:
                logger.warning(f"Skipping invalid tile entry: {tile} ({e})")

    return idx, tile_bboxes

def download_dem_files(lon_min: float, lat_min: float, lon_max: float, lat_max: float, aws_dir: str) -> List[str]:
    """
    Download DEM tile files covering a geographic bounding box.

    Tiles are downloaded from the specified AWS directory and cached locally.
    Already-downloaded tiles are reused from the cache.

    Args:
        lon_min (float): Western longitude bound (degrees).
        lat_min (float): Southern latitude bound (degrees).
        lon_max (float): Eastern longitude bound (degrees).
        lat_max (float): Northern latitude bound (degrees).
        aws_dir (str): Base URL of the AWS-hosted DEM tile directory.

    Returns:
        List[str]: List of local file paths to the downloaded DEM tiles.
    """
    localdem_dir = os.path.join(get_cache_root(), "localdem")
    os.makedirs(localdem_dir, exist_ok=True)

    tile_list_file = os.path.join(localdem_dir, "tileList.txt")
    if not os.path.exists(tile_list_file):
        download_file(tile_list_file, f"{aws_dir}tileList.txt")

    idx, tile_bboxes = build_tile_index(tile_list_file)
    query_bbox = box(lon_min, lat_min, lon_max, lat_max)
    matching_tiles = [tile_bboxes[i][0] for i in idx.intersection(query_bbox.bounds)]

    if not matching_tiles:
        logger.info("No overlapping DEM tiles found.")
        return []

    downloaded_files = []
    with ThreadPoolExecutor() as executor:
        futures = {}
        for tile in matching_tiles:
            tile_url = f"{aws_dir}{tile}/{tile}.tif"
            tile_file = os.path.join(localdem_dir, f"{tile}.tif")
            if not os.path.exists(tile_file):
                logger.info(f"Submitting download for tile: {tile_url}")
                futures[executor.submit(download_file, tile_file, tile_url)] = tile_file
            else:
                downloaded_files.append(tile_file)

        for future, tile_file in futures.items():
            try:
                future.result()
                downloaded_files.append(tile_file)
            except Exception as e:
                logger.error(f"Error downloading tile {tile_file}: {e}")

    # Validate tiles — remove corrupt files so they re-download next time
    valid_files = []
    for tile_file in downloaded_files:
        try:
            with rasterio.open(tile_file) as src:
                src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            valid_files.append(tile_file)
        except Exception:
            logger.warning(
                f"Corrupt DEM tile detected, removing: {tile_file}"
            )
            os.remove(tile_file)

    return valid_files


def merge_tiles(output_filename: str, tile_file_list: List[str]) -> None:
    """
    Merge multiple DEM tile files into a single GeoTIFF using rasterio.

    Args:
        output_filename (str): Path for the merged output GeoTIFF file.
        tile_file_list (List[str]): List of file paths to DEM tiles to merge.

    Raises:
        ValueError: If tile_file_list is empty or contains invalid paths.
        RuntimeError: If merge operation fails.
    """
    if not tile_file_list:
        raise HyPlanValueError("No tiles provided for merging.")

    invalid_tiles = [tile for tile in tile_file_list if not tile or not os.path.exists(tile)]
    if invalid_tiles:
        raise HyPlanValueError(f"Invalid or missing raster files: {invalid_tiles}")

    try:
        logger.info(f"Merging {len(tile_file_list)} tiles into {output_filename}")
        datasets = [rasterio.open(p) for p in tile_file_list]
        merged, merged_transform = rasterio.merge.merge(datasets)
        for ds in datasets:
            ds.close()
        # Write merged result
        profile = {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "count": 1,
            "dtype": merged.dtype,
            "crs": "EPSG:4326",
            "transform": merged_transform,
        }
        with rasterio.open(output_filename, "w", **profile) as dst:
            dst.write(merged)
        logger.info(f"Successfully merged tiles into {output_filename}")
    except Exception as e:
        logger.error(f"Failed to merge tiles: {e}")
        raise HyPlanRuntimeError(f"Tile merging failed: {e}")


def generate_demfile(latitude: np.ndarray, longitude: np.ndarray, aws_dir: str = "https://copernicus-dem-30m.s3.amazonaws.com/") -> str:
    """Generate a DEM file covering the specified latitude and longitude extents."""
    dem_cache_dir = os.path.join(get_cache_root(), "dem_cache")
    os.makedirs(dem_cache_dir, exist_ok=True)

    lon_min, lon_max = np.min(longitude) - 0.1, np.max(longitude) + 0.1
    lat_min, lat_max = np.min(latitude) - 0.1, np.max(latitude) + 0.1

    # Use rounded float values to create unique cache keys for different extents
    cache_filename = os.path.join(
        dem_cache_dir,
        f"{lat_min:.2f}_{lon_min:.2f}_{lat_max:.2f}_{lon_max:.2f}.tif"
    )
    if os.path.exists(cache_filename):
        logger.info(f"Using cached DEM file: {cache_filename}")
        return cache_filename

    tile_files = download_dem_files(lon_min, lat_min, lon_max, lat_max, aws_dir)
    if not tile_files:
        raise FileNotFoundError("No DEM tiles available for the specified area.")

    merge_tiles(cache_filename, tile_files)
    return cache_filename

@lru_cache(maxsize=8)
def _load_dem(dem_file: str, mtime: float):
    """Load a DEM raster + geotransform once and cache it.

    Keyed on (path, mtime) so an updated file invalidates the cache.
    Bounded to 8 entries to avoid unbounded memory growth across many DEMs.
    Returns ``(raster, geotransform, raster_min, raster_max)``.
    """
    try:
        with rasterio.open(dem_file) as src:
            raster = src.read(1)
            t = src.transform
            # Convert rasterio Affine to GDAL-style geotransform tuple
            # so downstream code (get_elevations, surface_normal_at) is unchanged.
            geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)
    except Exception as e:
        raise HyPlanRuntimeError(f"Could not open DEM file: {dem_file}") from e
    raster.setflags(write=False)  # protect cached array from in-place mutation
    raster_min = float(np.nanmin(raster))
    raster_max = float(np.nanmax(raster))
    return raster, geotransform, raster_min, raster_max


def _get_dem_cached(dem_file: str):
    """Public wrapper that handles mtime lookup so callers don't have to."""
    try:
        mtime = os.path.getmtime(dem_file)
    except OSError as e:
        raise HyPlanRuntimeError(f"Could not stat DEM file: {dem_file}") from e
    return _load_dem(dem_file, mtime)


def get_elevations(lats: np.ndarray, lons: np.ndarray, dem_file: str) -> np.ndarray:
    """
    Extract elevation values for given latitudes and longitudes from a DEM file.

    Reads the entire raster band once and indexes it in bulk, rather than
    querying pixel-by-pixel. The raster is cached per ``(path, mtime)`` so
    repeated calls against the same DEM (the common case in flight planning)
    pay the read cost only once.
    """
    raster, geotransform, _, _ = _get_dem_cached(dem_file)

    # np.round() rounds to nearest pixel center before int conversion; no truncation occurs.
    xs = np.round((lons - geotransform[0]) / geotransform[1]).astype(int)
    ys = np.round((lats - geotransform[3]) / geotransform[5]).astype(int)

    out_of_bounds = (xs < 0) | (xs >= raster.shape[1]) | (ys < 0) | (ys >= raster.shape[0])
    if np.any(out_of_bounds):
        n_oob = int(out_of_bounds.sum())
        logger.warning(
            f"{n_oob} query point(s) fall outside the DEM extent. "
            "Edge pixel elevations will be used for these points."
        )

    xs = np.clip(xs, 0, raster.shape[1] - 1)
    ys = np.clip(ys, 0, raster.shape[0] - 1)

    return raster[ys, xs]


def get_min_max_elevations(dem_file: str) -> Tuple[float, float]:
    """
    Get the minimum and maximum elevation values from a DEM file.

    Args:
        dem_file (str): Path to the DEM file.

    Returns:
        Tuple[float, float]: (min_elevation, max_elevation) in the DEM file.
    """
    _, _, raster_min, raster_max = _get_dem_cached(dem_file)
    return raster_min, raster_max


def terrain_elevation_along_track(flight_line, dem_file: str,
                                   precision: float = 100.0) -> dict:
    """Min, mean, and max terrain elevation (m MSL) along a flight line's nadir track.

    Samples the DEM at evenly-spaced points along the flight line and returns
    summary statistics useful for Mode 3 per-line altitude planning.

    Args:
        flight_line: A FlightLine object with a ``track(precision)`` method.
        dem_file: Path to a DEM GeoTIFF covering the flight line.
        precision: Along-track sampling interval in meters. Default 100 m.

    Returns:
        Dict with keys ``"min"``, ``"mean"``, and ``"max"`` (all in meters MSL).
    """
    lats, lons, *_ = process_linestring(flight_line.track(precision=precision))
    elevations = get_elevations(lats, lons, dem_file).astype(float)
    return {
        "min": float(np.nanmin(elevations)),
        "mean": float(np.nanmean(elevations)),
        "max": float(np.nanmax(elevations)),
    }


def terrain_aspect_azimuth(polygon, dem_file: str = None) -> float:
    """Dominant terrain gradient direction (degrees from north) for a polygon.

    Computes the dominant downslope azimuth from the DEM gradient over the
    polygon area.  To minimise altitude variation along each flight line
    (as recommended by Zhao et al. 2021 for Mode 3), orient flight lines
    *perpendicular* to the returned azimuth::

        flight_azimuth = (terrain_aspect_azimuth(polygon) + 90) % 360

    Args:
        polygon: Shapely Polygon defining the survey area (WGS84 lon/lat).
        dem_file: Path to a DEM GeoTIFF.  If ``None``, one is downloaded and
            cached from the Copernicus GLO-30 archive.

    Returns:
        Dominant downslope azimuth in degrees clockwise from true north
        (range 0–360).
    """
    if dem_file is None:
        coords = np.array(polygon.exterior.coords)
        lons_poly = coords[:, 0]
        lats_poly = coords[:, 1]
        dem_file = generate_demfile(lats_poly, lons_poly)

    raster, _, _, _ = _get_dem_cached(dem_file)
    elevations = raster.astype(float)

    # np.gradient returns (d/d_row, d/d_col).  In a north-up GeoTIFF rows
    # increase southward, so the north component is the *negative* row gradient.
    dy, dx = np.gradient(elevations)
    north_component = -dy   # positive = elevation increases going north
    east_component = dx     # positive = elevation increases going east

    # Aspect: direction of steepest ascent, clockwise from north.
    aspect = np.degrees(np.arctan2(east_component, north_component))
    downslope = (aspect + 180.0) % 360.0  # reverse to get downslope direction

    # Circular mean, ignoring flat pixels (lowest quartile of gradient magnitude)
    magnitude = np.sqrt(north_component ** 2 + east_component ** 2)
    threshold = np.percentile(magnitude, 25)
    significant = magnitude > threshold

    downslope_rad = np.radians(downslope[significant])
    mean_sin = float(np.nanmean(np.sin(downslope_rad)))
    mean_cos = float(np.nanmean(np.cos(downslope_rad)))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


def ray_terrain_intersection(
    lat0: np.ndarray,
    lon0: np.ndarray,
    h0: float,
    az: np.ndarray,
    tilt: np.ndarray,
    precision: float = 10.0,
    dem_file: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch computation of ray-terrain intersections using a DEM for multiple observer positions.
    Vectorized to handle multiple observers efficiently.

    Args:
        lat0 (np.ndarray): Array of observer latitudes (degrees).
        lon0 (np.ndarray): Array of observer longitudes (degrees).
        h0 (float): Altitude of the observer above the ellipsoid (meters).
        az (np.ndarray): Azimuth angle of the ray (degrees).
        tilt (np.ndarray): Tilt angle of the ray below horizontal (degrees, 0–90).
        precision (float): Precision of the slant range sampling (meters).
        dem_file (str): Path to the DEM file. If None, one will be generated.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (intersection_lats, intersection_lons,
            intersection_alts). Observers with no terrain intersection have NaN in all
            three arrays.

    Raises:
        ValueError: If tilt or azimuth values are out of range, or if tilt is too close
            to horizontal (±90°) where slant-range geometry is undefined.
    """
    lat0 = np.atleast_1d(lat0)
    lon0 = np.atleast_1d(lon0)
    az = np.atleast_1d(az)
    tilt = np.atleast_1d(tilt)
    n_obs = len(lat0)

    if np.any((tilt < -90) | (tilt > 90)):
        raise HyPlanValueError("Tilt angles must be between -90 and 90 degrees.")
    if np.any((az < 0) | (az > 360)):
        raise HyPlanValueError("Azimuth angle must be between 0 and 360 degrees.")

    cos_tilt = np.cos(np.radians(tilt))
    if np.any(np.abs(cos_tilt) < _COS_TILT_MIN):
        raise HyPlanValueError(
            "One or more tilt angles are too close to ±90° (horizontal). "
            "Ray-terrain intersection is undefined for near-horizontal rays."
        )

    # Compute slant range to ellipsoid for each observer.
    lat_ell, lon_ell, rng_ell = pymap3d.los.lookAtSpheroid(lat0, lon0, h0, az, tilt)
    rng_ell = np.atleast_1d(rng_ell)

    # For steep depression angles (near-nadir), lookAtSpheroid returns
    # extremely large ranges because the ray grazes along the ellipsoid
    # surface.  Cap the range at a flat-earth estimate with a generous
    # safety factor to keep the search window reasonable.
    rng_flat = h0 / cos_tilt
    rng_capped = np.minimum(rng_ell, rng_flat * 2.0)

    auto_dem = dem_file is None

    if auto_dem:
        # Include both observer and ellipsoid-hit positions so the DEM
        # covers the full ray-march extent.
        dem_lats = np.concatenate([lat0, np.atleast_1d(lat_ell)])
        dem_lons = np.concatenate([lon0, np.atleast_1d(lon_ell)])
        dem_file = generate_demfile(dem_lats, dem_lons)

    min_elev, max_elev = get_min_max_elevations(dem_file)
    max_elev = min(h0, max_elev)
    if np.any(min_elev > h0):
        raise HyPlanValueError("Observer altitude is below the minimum terrain elevation.")

    # Per-observer slant range bounds using capped range
    upper_bound = np.ceil((rng_capped - (min_elev / cos_tilt)) / precision) * precision
    lower_bound = np.maximum(
        0.0,
        np.floor((rng_capped - (max_elev / cos_tilt)) / precision) * precision,
    )

    # Global range array spanning the worst-case window across all observers.
    # A per-observer validity mask (below) ensures each observer's intersection
    # is only detected within its own valid slant-range window, preventing
    # cross-observer false positives.
    rs = np.arange(lower_bound.min(), upper_bound.max() + precision, precision)

    tilt_el = tilt - 90.0  # Convert depression angle to elevation angle for aer2geodetic

    lats, lons, alts = pymap3d.aer.aer2geodetic(
        az[np.newaxis, :], tilt_el[np.newaxis, :], rs[:, np.newaxis],
        lat0[np.newaxis, :], lon0[np.newaxis, :], h0
    )

    # If the DEM was auto-downloaded, check whether the ray-march grid
    # extends beyond it and re-download a wider DEM if necessary.
    if auto_dem:
        grid_lat_min, grid_lat_max = float(lats.min()), float(lats.max())
        grid_lon_min, grid_lon_max = float(lons.min()), float(lons.max())
        dem_lat_min = float(dem_lats.min()) - 0.1
        dem_lon_min = float(dem_lons.min()) - 0.1
        dem_lat_max = float(dem_lats.max()) + 0.1
        dem_lon_max = float(dem_lons.max()) + 0.1

        if (grid_lat_min < dem_lat_min or grid_lat_max > dem_lat_max or
                grid_lon_min < dem_lon_min or grid_lon_max > dem_lon_max):
            wider_lats = np.array([grid_lat_min, grid_lat_max])
            wider_lons = np.array([grid_lon_min, grid_lon_max])
            dem_file = generate_demfile(wider_lats, wider_lons)

    lats_flat = lats.ravel()
    lons_flat = lons.ravel()

    dem_elevations = get_elevations(lats_flat, lons_flat, dem_file).reshape(lats.shape)

    # Per-observer validity mask: only accept intersections within [lower_bound[i], upper_bound[i]].
    # Without this, a range step that is geometrically valid for one observer could be
    # incorrectly attributed to another observer whose window doesn't cover that range.
    valid_range = (rs[:, np.newaxis] >= lower_bound[np.newaxis, :]) & \
                  (rs[:, np.newaxis] <= upper_bound[np.newaxis, :])

    mask = (dem_elevations > alts) & valid_range

    # Detect observers with no terrain intersection and return NaN for them
    has_intersection = mask.any(axis=0)
    safe_idx = np.where(has_intersection, np.argmax(mask, axis=0), 0)
    col_idx = np.arange(n_obs)

    intersection_lats = np.where(has_intersection, lats[safe_idx, col_idx], np.nan)
    intersection_lons = np.where(has_intersection, lons[safe_idx, col_idx], np.nan)
    intersection_alts = np.where(has_intersection, dem_elevations[safe_idx, col_idx], np.nan)

    n_miss = int((~has_intersection).sum())
    if n_miss:
        logger.warning(f"{n_miss} observer(s) had no terrain intersection within the valid slant-range window.")

    return intersection_lats, intersection_lons, intersection_alts


# ---------------------------------------------------------------------------
# Surface normal computation
# ---------------------------------------------------------------------------

# Approximate meters per degree of latitude (WGS84 mean)
_M_PER_DEG_LAT = 111_320.0


def surface_normal_at(
    lats: np.ndarray,
    lons: np.ndarray,
    dem_file: str,
) -> np.ndarray:
    """Compute outward surface normal unit vectors in the ENU frame.

    Uses central-difference gradients from the 3x3 DEM neighbourhood
    around each query point, converted to metric slope.

    Args:
        lats: Latitude(s) in decimal degrees.
        lons: Longitude(s) in decimal degrees.
        dem_file: Path to a GeoTIFF DEM file.

    Returns:
        ``(N, 3)`` array of ``[east, north, up]`` unit normal components.
        On flat terrain the normal is ``[0, 0, 1]``.
    """
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(lons, dtype=float))

    raster, gt, _, _ = _get_dem_cached(dem_file)
    rows, cols = raster.shape

    # Pixel coordinates (float for sub-pixel, int for indexing)
    px_x = (lons - gt[0]) / gt[1]
    px_y = (lats - gt[3]) / gt[5]
    ix = np.round(px_x).astype(int)
    iy = np.round(px_y).astype(int)

    # Clamp to valid range, leaving a 1-pixel border for central differences
    ix = np.clip(ix, 1, cols - 2)
    iy = np.clip(iy, 1, rows - 2)

    # Central differences: dz/dx (east-west) and dz/dy (north-south) in pixels
    # raster[row, col]: row increases southward, col increases eastward
    z_w = raster[iy, ix - 1].astype(float)
    z_e = raster[iy, ix + 1].astype(float)
    z_n = raster[iy - 1, ix].astype(float)
    z_s = raster[iy + 1, ix].astype(float)

    dz_dx_px = (z_e - z_w) / 2.0  # positive = rising eastward
    dz_dy_px = (z_n - z_s) / 2.0  # positive = rising northward
    # Note: z_n uses iy-1 because row index decreases northward in the raster

    # Convert pixel gradients to metric gradients
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(lats))
    pixel_width_m = abs(gt[1]) * m_per_deg_lon  # meters per pixel, east-west
    pixel_height_m = abs(gt[5]) * _M_PER_DEG_LAT  # meters per pixel, north-south

    dz_dx = dz_dx_px / pixel_width_m  # dz/dx in m/m (dimensionless slope)
    dz_dy = dz_dy_px / pixel_height_m  # dz/dy in m/m

    # Surface normal: n = normalize([-dz/dx, -dz/dy, 1])
    n = len(lats)
    normals = np.empty((n, 3), dtype=float)
    normals[:, 0] = -dz_dx  # east
    normals[:, 1] = -dz_dy  # north
    normals[:, 2] = 1.0     # up

    # Normalize to unit vectors
    magnitudes = np.sqrt(np.sum(normals ** 2, axis=1, keepdims=True))
    normals /= magnitudes

    return normals
