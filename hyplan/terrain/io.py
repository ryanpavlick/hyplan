"""DEM download, merge, cache management, and raster loading."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import rasterio
import rasterio.merge
from rtree import index
from shapely.geometry import box

from ..download import download_file
from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from ._demgrid import DEMGrid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_cache_root(custom_path: str | None = None) -> str:
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


# ---------------------------------------------------------------------------
# Tile index and download
# ---------------------------------------------------------------------------

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
                lat_str, _, lon_str = tile.replace("_COG", "").split("_")[3:6]
                lon_f = -1 * float(lon_str[1:]) if "W" in lon_str else float(lon_str[1:])
                lat_f = -1 * float(lat_str[1:]) if "S" in lat_str else float(lat_str[1:])
                bbox = box(lon_f, lat_f, lon_f + 1, lat_f + 1)
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


# ---------------------------------------------------------------------------
# DEM loading and caching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_dem(dem_file: str, mtime: float) -> DEMGrid:
    """Load a DEM raster once and cache it as a :class:`DEMGrid`.

    Keyed on (path, mtime) so an updated file invalidates the cache.
    Bounded to 8 entries to avoid unbounded memory growth across many DEMs.
    """
    try:
        with rasterio.open(dem_file) as src:
            raster = src.read(1)
            t = src.transform
            geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)
            bounds = (src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top)
            nodata = src.nodata
    except Exception as e:
        raise HyPlanRuntimeError(f"Could not open DEM file: {dem_file}") from e
    raster.setflags(write=False)  # protect cached array from in-place mutation
    return DEMGrid(
        array=raster,
        geotransform=geotransform,
        bounds=bounds,
        nodata=nodata,
    )


def load_dem(dem_file: str) -> DEMGrid:
    """Load a DEM file and return a cached :class:`DEMGrid`.

    Handles mtime lookup so callers don't have to.
    """
    try:
        mtime = os.path.getmtime(dem_file)
    except OSError as e:
        raise HyPlanRuntimeError(f"Could not stat DEM file: {dem_file}") from e
    return _load_dem(dem_file, mtime)
