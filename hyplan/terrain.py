import logging
import math
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from shapely.geometry import box
from rtree import index
from osgeo import gdal
from typing import List, Tuple
import pymap3d.los
import pymap3d.aer

from .download import download_file

logger = logging.getLogger(__name__)

# Minimum cos(tilt) magnitude below which ray-terrain intersection is undefined
_COS_TILT_MIN = 1e-6


def get_cache_root(custom_path: str = None) -> str:
    """Get the root directory for caching files."""
    return custom_path or os.environ.get("HYPLAN_CACHE_ROOT", f"{tempfile.gettempdir()}/hyplan")

def clear_cache():
    """Clears the entire cache directory after confirming it is safe to do so."""
    cache_dir = get_cache_root()
    if not cache_dir.startswith(tempfile.gettempdir()):
        raise ValueError(f"Refusing to clear unsafe cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"Cache directory {cache_dir} cleared.")
    else:
        logger.info(f"Cache directory {cache_dir} does not exist.")

def clear_localdem_cache(confirm: bool = True):
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
        raise ValueError(f"Refusing to clear unsafe directory: {localdem_dir}")

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
    """Build an R-tree spatial index for DEM tiles."""
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

    return downloaded_files


def merge_tiles(output_filename, tile_file_list):
    if not tile_file_list:
        raise ValueError("No tiles provided for merging.")

    invalid_tiles = [tile for tile in tile_file_list if not tile or not os.path.exists(tile)]
    if invalid_tiles:
        raise ValueError(f"Invalid or missing raster files: {invalid_tiles}")

    try:
        logger.info(f"Merging {len(tile_file_list)} tiles into {output_filename}")
        gdal.Warp(
            destNameOrDestDS=output_filename,
            srcDSOrSrcDSTab=tile_file_list,
            format="GTiff",
        )
        logger.info(f"Successfully merged tiles into {output_filename}")
    except Exception as e:
        logger.error(f"Failed to merge tiles: {e}")
        raise RuntimeError(f"Tile merging failed: {e}")


def generate_demfile(latitude: np.ndarray, longitude: np.ndarray, aws_dir: str = "https://copernicus-dem-30m.s3.amazonaws.com/") -> str:
    """Generate a DEM file covering the specified latitude and longitude extents."""
    dem_cache_dir = os.path.join(get_cache_root(), "dem_cache")
    os.makedirs(dem_cache_dir, exist_ok=True)

    lon_min, lon_max = np.min(longitude) - 0.1, np.max(longitude) + 0.1
    lat_min, lat_max = np.min(latitude) - 0.1, np.max(latitude) + 0.1

    # Use math.floor to correctly handle negative coordinates (int() truncates toward zero)
    cache_filename = os.path.join(
        dem_cache_dir,
        f"{math.floor(lat_min)}_{math.floor(lon_min)}_{math.floor(lat_max)}_{math.floor(lon_max)}.tif"
    )
    if os.path.exists(cache_filename):
        logger.info(f"Using cached DEM file: {cache_filename}")
        return cache_filename

    tile_files = download_dem_files(lon_min, lat_min, lon_max, lat_max, aws_dir)
    if not tile_files:
        raise FileNotFoundError("No DEM tiles available for the specified area.")

    merge_tiles(cache_filename, tile_files)
    return cache_filename

def get_elevations(lats: np.ndarray, lons: np.ndarray, dem_file: str) -> np.ndarray:
    """
    Extract elevation values for given latitudes and longitudes from a DEM file.

    Reads the entire raster band once and indexes it in bulk, rather than
    querying pixel-by-pixel, for efficient batch lookups.
    """
    dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    if not dataset:
        raise RuntimeError(f"Could not open DEM file: {dem_file}")

    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    if not band:
        raise RuntimeError(f"DEM file does not contain valid raster data: {dem_file}")

    raster = band.ReadAsArray()
    dataset = None  # Close the dataset

    xs = ((lons - geotransform[0]) / geotransform[1]).astype(int)
    ys = ((lats - geotransform[3]) / geotransform[5]).astype(int)

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
    dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    if not dataset:
        raise RuntimeError(f"Could not open DEM file: {dem_file}")

    band = dataset.GetRasterBand(1)
    if not band:
        raise RuntimeError(f"DEM file does not contain valid raster data: {dem_file}")

    min_val, max_val = band.ComputeRasterMinMax()
    dataset = None  # Close the dataset

    return min_val, max_val

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
        raise ValueError("Tilt angles must be between -90 and 90 degrees.")
    if np.any((az < 0) | (az > 360)):
        raise ValueError("Azimuth angle must be between 0 and 360 degrees.")

    cos_tilt = np.cos(np.radians(tilt))
    if np.any(np.abs(cos_tilt) < _COS_TILT_MIN):
        raise ValueError(
            "One or more tilt angles are too close to ±90° (horizontal). "
            "Ray-terrain intersection is undefined for near-horizontal rays."
        )

    # Compute slant range to ellipsoid for each observer
    lat_ell, lon_ell, rng_ell = pymap3d.los.lookAtSpheroid(lat0, lon0, h0, az, tilt)

    if dem_file is None:
        dem_file = generate_demfile(lat_ell, lon_ell)

    min_elev, max_elev = get_min_max_elevations(dem_file)
    max_elev = min(h0, max_elev)
    if np.any(min_elev > h0):
        raise ValueError("Observer altitude is below the minimum terrain elevation.")

    # Per-observer slant range bounds
    upper_bound = np.ceil((rng_ell - (min_elev / cos_tilt)) / precision) * precision
    lower_bound = np.floor((rng_ell - (max_elev / cos_tilt)) / precision) * precision

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
