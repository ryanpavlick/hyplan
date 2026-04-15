"""Ray-terrain intersection and surface normal computation."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pymap3d.aer
import pymap3d.los

from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from .elevation import get_elevations, get_min_max_elevations
from .io import generate_demfile, load_dem

logger = logging.getLogger(__name__)

# Minimum cos(tilt) magnitude below which ray-terrain intersection is undefined
_COS_TILT_MIN = 1e-6

# Approximate meters per degree of latitude (WGS84 mean)
_M_PER_DEG_LAT = 111_320.0


def ray_terrain_intersection(
    lat0: np.ndarray,
    lon0: np.ndarray,
    h0: float,
    az: np.ndarray,
    tilt: np.ndarray,
    precision: float = 10.0,
    dem_file: str | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch computation of ray-terrain intersections using a DEM for multiple observer positions.
    Vectorized to handle multiple observers efficiently.

    Args:
        lat0 (np.ndarray): Array of observer latitudes (degrees).
        lon0 (np.ndarray): Array of observer longitudes (degrees).
        h0 (float): Altitude of the observer above the ellipsoid (meters).
        az (np.ndarray): Azimuth angle of the ray (degrees).
        tilt (np.ndarray): Tilt angle of the ray below horizontal (degrees, 0-90).
        precision (float): Precision of the slant range sampling (meters).
        dem_file (str): Path to the DEM file. If None, one will be generated.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (intersection_lats, intersection_lons,
            intersection_alts). Observers with no terrain intersection have NaN in all
            three arrays.

    Raises:
        ValueError: If tilt or azimuth values are out of range, or if tilt is too close
            to horizontal (+-90 deg) where slant-range geometry is undefined.
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
            "One or more tilt angles are too close to +-90 deg (horizontal). "
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

    assert dem_file is not None  # guaranteed by auto_dem branch above
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

    dem = load_dem(dem_file)
    raster = dem.array
    gt = dem.geotransform
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

    return normals  # type: ignore[no-any-return]
