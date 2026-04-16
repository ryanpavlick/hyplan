"""Vegetation phenology data from MODIS products via NASA EarthData.

Retrieves NDVI/EVI (MOD13A1/MYD13A1), LAI/FPAR (MOD15A2H), and phenological
transition dates (MCD12Q2) using ``earthaccess`` for search/download and
``rasterio`` for raster processing.  **Requires NASA Earthdata credentials.**
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb

from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from . import _qa

if TYPE_CHECKING:  # pragma: no cover
    import rasterio
    import xarray as xr
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Product configuration
# ---------------------------------------------------------------------------

_PRODUCT_CONFIG = {
    "ndvi": {
        "short_name": "MOD13A1",
        "short_name_aqua": "MYD13A1",
        "subdataset": "500m 16 days NDVI",
        "qa_subdataset": "500m 16 days pixel reliability",
        "scale_factor": 0.0001,
        "valid_range": (-2000, 10000),
        "qa_func": _qa.apply_vi_qa_mask,
    },
    "evi": {
        "short_name": "MOD13A1",
        "short_name_aqua": "MYD13A1",
        "subdataset": "500m 16 days EVI",
        "qa_subdataset": "500m 16 days pixel reliability",
        "scale_factor": 0.0001,
        "valid_range": (-2000, 10000),
        "qa_func": _qa.apply_vi_qa_mask,
    },
    "lai": {
        "short_name": "MOD15A2H",
        "subdataset": "Lai_500m",
        "qa_subdataset": "FparLai_QC",
        "scale_factor": 0.1,
        "valid_range": (0, 100),
        "qa_func": _qa.apply_lai_qa_mask,
    },
    "fpar": {
        "short_name": "MOD15A2H",
        "subdataset": "Fpar_500m",
        "qa_subdataset": "FparLai_QC",
        "scale_factor": 0.01,
        "valid_range": (0, 100),
        "qa_func": _qa.apply_lai_qa_mask,
    },
    "phenology": {
        "short_name": "MCD12Q2",
        "subdatasets": {
            "greenup_doy": "Greenup.Num_Modes_01",
            "midgreenup_doy": "MidGreenup.Num_Modes_01",
            "peak_doy": "Peak.Num_Modes_01",
            "maturity_doy": "Maturity.Num_Modes_01",
            "midgreendown_doy": "MidGreendown.Num_Modes_01",
            "senescence_doy": "Senescence.Num_Modes_01",
            "dormancy_doy": "Dormancy.Num_Modes_01",
        },
        "qa_subdataset": "QA_Detailed.Num_Modes_01",
    },
}

_VALID_PRODUCTS = set(_PRODUCT_CONFIG.keys())


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _require_rasterio():
    """Import and return rasterio, raising a clear error if missing."""
    try:
        import rasterio as _rio

        return _rio
    except ImportError:
        raise HyPlanRuntimeError(
            "rasterio is required for phenology analysis. "
            "Install with: pip install rasterio"
        )


def _require_xarray():
    """Import and return xarray, raising a clear error if missing."""
    try:
        import xarray as _xr

        return _xr
    except ImportError:
        raise HyPlanRuntimeError(
            "xarray is required for spatial phenology analysis. "
            "Install with: pip install hyplan[phenology]"
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _drop_z(geom: "BaseGeometry") -> "BaseGeometry":
    """Strip Z coordinates from a Shapely geometry, returning 2D."""
    return wkb.loads(wkb.dumps(geom, output_dimension=2))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(short_name: str) -> str:
    """Return (and create) the cache directory for a MODIS product."""
    from ..terrain.io import get_cache_root

    cache_dir = os.path.join(get_cache_root(), "phenology", short_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# ---------------------------------------------------------------------------
# earthaccess search & download
# ---------------------------------------------------------------------------

def _search_granules(
    short_name: str,
    bounding_box: tuple[float, float, float, float],
    date_start: str,
    date_stop: str,
) -> list:
    """Search NASA CMR for MODIS granules.

    Parameters
    ----------
    short_name : str
        MODIS collection short name (e.g. ``"MOD13A1"``).
    bounding_box : tuple
        ``(west, south, east, north)`` in WGS84 degrees.
    date_start, date_stop : str
        ISO date strings ``"YYYY-MM-DD"``.

    Returns
    -------
    list
        earthaccess granule result objects.
    """
    from .._auth import _require_earthaccess

    earthaccess = _require_earthaccess()

    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bounding_box,
        temporal=(date_start, date_stop),
    )
    logger.info(
        "CMR search for %s (%s to %s): %d granules",
        short_name, date_start, date_stop, len(results),
    )
    return results  # type: ignore[no-any-return]


def _download_granules(granules: list, cache_dir: str) -> list[str]:
    """Download granules to *cache_dir*, skipping cached files.

    Returns list of local file paths (HDF).
    """
    from .._auth import _require_earthaccess

    earthaccess = _require_earthaccess()

    if not granules:
        return []

    paths = earthaccess.download(granules, local_path=cache_dir)
    # earthaccess.download returns a list of Path objects
    return [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Raster processing helpers
# ---------------------------------------------------------------------------

def _find_subdataset(hdf_path: str, name_fragment: str) -> str | None:
    """Find the full subdataset URI matching *name_fragment*."""
    rio = _require_rasterio()

    with rio.open(hdf_path) as src:
        for sds_name, _sds_desc in src.subdatasets:
            if name_fragment in sds_name:
                return sds_name  # type: ignore[no-any-return]
    return None


def _list_subdatasets(hdf_path: str) -> list[tuple[str, str]]:
    """Return list of (subdataset_uri, description) for an HDF file."""
    rio = _require_rasterio()

    with rio.open(hdf_path) as src:
        return list(zip(src.subdatasets, src.subdatasets))


def _read_hdf4_subdataset(hdf_path: str, subdataset_name: str) -> tuple[np.ndarray, dict]:
    """Read a subdataset from a MODIS HDF4-EOS file using pyhdf.

    Returns the data array and metadata dict with grid parameters.
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise HyPlanRuntimeError(
            "pyhdf is required to read MODIS HDF4 files. "
            "Install with: pip install pyhdf"
        )

    hdf = SD(hdf_path, SDC.READ)

    # Find matching subdataset
    datasets = hdf.datasets()
    match = None
    for ds_name in datasets:
        if subdataset_name in ds_name:
            match = ds_name
            break
    if match is None:
        hdf.end()
        raise HyPlanRuntimeError(
            f"Subdataset '{subdataset_name}' not found in {hdf_path}. "
            f"Available: {list(datasets.keys())}"
        )

    sds = hdf.select(match)
    data = sds.get()
    attrs = sds.attributes()
    sds.endaccess()

    # Get grid metadata from global attributes
    grid_meta = {}
    global_attrs = hdf.attributes()
    for attr_name in ("StructMetadata.0", "CoreMetadata.0"):
        if attr_name in global_attrs:
            grid_meta[attr_name] = global_attrs[attr_name]

    hdf.end()
    return data, grid_meta


def _parse_modis_grid_bounds(grid_meta: dict) -> tuple[float, float, float, float]:
    """Extract MODIS sinusoidal grid bounds from StructMetadata.0.

    Returns (ulx, uly, lrx, lry) in sinusoidal meters.
    """
    import re
    struct = grid_meta.get("StructMetadata.0", "")

    ul_match = re.search(r"UpperLeftPointMtrs=\(([^,]+),([^)]+)\)", struct)
    lr_match = re.search(r"LowerRightMtrs=\(([^,]+),([^)]+)\)", struct)

    if ul_match and lr_match:
        ulx = float(ul_match.group(1))
        uly = float(ul_match.group(2))
        lrx = float(lr_match.group(1))
        lry = float(lr_match.group(2))
        return ulx, uly, lrx, lry

    raise HyPlanRuntimeError(
        "Could not parse MODIS grid bounds from StructMetadata.0"
    )


def _read_and_clip_subdataset(
    hdf_path: str,
    subdataset_name: str,
    polygon_geom: "BaseGeometry",
) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Read a subdataset, reproject to WGS84, clip to polygon.

    Uses pyhdf to read HDF4-EOS files (GDAL HDF4 driver not required),
    then rasterio for reprojection and clipping.

    Returns the clipped data array and its transform.
    """
    rio = _require_rasterio()
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    # Read data from HDF4 using pyhdf
    data, grid_meta = _read_hdf4_subdataset(hdf_path, subdataset_name)
    ulx, uly, lrx, lry = _parse_modis_grid_bounds(grid_meta)

    nrows, ncols = data.shape
    src_crs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
        "+R=6371007.181 +units=m +no_defs"
    )
    src_transform = from_bounds(ulx, lry, lrx, uly, ncols, nrows)

    # Reproject to WGS84
    dst_crs = CRS.from_epsg(4326)
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, ncols, nrows,
        left=ulx, bottom=lry, right=lrx, top=uly,
    )

    reprojected = np.empty((height, width), dtype=data.dtype)
    reproject(
        source=data,
        destination=reprojected,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # Clip to polygon using in-memory GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": reprojected.dtype,
        "crs": dst_crs,
        "transform": transform,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem_ds:
            mem_ds.write(reprojected, 1)
            clipped, clipped_transform = rio_mask(
                mem_ds,
                [polygon_geom.__geo_interface__],
                crop=True,
                nodata=0,
            )

    return clipped[0], clipped_transform


def _parse_modis_date(hdf_path: str) -> datetime:
    """Extract acquisition date from a MODIS HDF filename.

    MODIS filenames contain ``AYYYYDDD`` (e.g. ``A2020017``).
    """
    basename = os.path.basename(hdf_path)
    match = re.search(r"A(\d{4})(\d{3})", basename)
    if match:
        year = int(match.group(1))
        doy = int(match.group(2))
        return datetime.strptime(f"{year}{doy:03d}", "%Y%j")
    raise HyPlanValueError(
        f"Cannot parse MODIS date from filename: {basename}"
    )


# ---------------------------------------------------------------------------
# Per-granule extraction
# ---------------------------------------------------------------------------

def _extract_vi_from_granule(
    hdf_path: str,
    polygon_geom: "BaseGeometry",
    config: dict,
    spatial_mode: str,
) -> dict | None:
    """Extract vegetation index values from a single granule.

    Returns a dict of column values, or None if no valid pixels.
    """
    data, _ = _read_and_clip_subdataset(
        hdf_path, config["subdataset"], polygon_geom,
    )
    qa, _ = _read_and_clip_subdataset(
        hdf_path, config["qa_subdataset"], polygon_geom,
    )

    # Apply QA mask
    masked = config["qa_func"](data, qa)

    # Apply valid range
    if "valid_range" in config:
        lo, hi = config["valid_range"]
        masked = np.ma.masked_outside(masked, lo, hi)

    if masked.count() == 0:
        return None

    dt = _parse_modis_date(hdf_path)

    if spatial_mode == "mean":
        return {
            "date": dt,
            "year": dt.year,
            "day_of_year": dt.timetuple().tm_yday,
            "value": float(np.ma.mean(masked)) * config["scale_factor"],
        }
    else:  # pixel_stats
        scaled = masked.astype(np.float64) * config["scale_factor"]
        return {
            "date": dt,
            "year": dt.year,
            "day_of_year": dt.timetuple().tm_yday,
            "value_mean": float(np.ma.mean(scaled)),
            "value_std": float(np.ma.std(scaled)),
            "value_min": float(np.ma.min(scaled)),
            "value_max": float(np.ma.max(scaled)),
            "pixel_count": int(scaled.count()),
        }


def _extract_phenology_from_granule(
    hdf_path: str,
    polygon_geom: "BaseGeometry",
    config: dict,
) -> dict | None:
    """Extract phenology transition DOYs from a single MCD12Q2 granule.

    Returns a dict of column values, or None if no valid pixels.
    """
    subdatasets = config["subdatasets"]

    # Read QA band
    qa, _ = _read_and_clip_subdataset(
        hdf_path, config["qa_subdataset"], polygon_geom,
    )

    # Read each stage and apply QA
    stage_data = {}
    for stage_name, sds_name in subdatasets.items():
        data, _ = _read_and_clip_subdataset(hdf_path, sds_name, polygon_geom)
        stage_data[stage_name] = data

    masked_stages = _qa.apply_phenology_qa_mask(stage_data, qa)

    # Parse year from filename
    dt = _parse_modis_date(hdf_path)
    result = {"year": dt.year}

    any_valid = False
    for stage_name, masked in masked_stages.items():
        # Convert epoch days to DOY
        doys = _qa.convert_mcd12q2_dates(masked.data)
        # Apply the QA mask to DOY values
        doy_masked: np.ma.MaskedArray = np.ma.masked_array(doys, mask=masked.mask | np.isnan(doys))

        if doy_masked.count() > 0:
            result[stage_name] = float(np.ma.mean(doy_masked))  # type: ignore[assignment]
            any_valid = True
        else:
            result[stage_name] = np.nan  # type: ignore[assignment]

    return result if any_valid else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_phenology(
    polygon_file: str,
    product: str = "ndvi",
    year_start: int = 2003,
    year_stop: int = 2022,
    satellite: str = "terra",
    spatial_mode: str = "mean",
    source: str = "appeears",
) -> pd.DataFrame:
    """Fetch historical vegetation phenology data for flight planning.

    Two data sources are available:

    - **AppEEARS** (``source="appeears"``, default): Submits a point
      sample request to NASA's AppEEARS service, which extracts time
      series server-side and returns only the values.  Fast (minutes)
      but samples at polygon centroids rather than full spatial averaging.

    - **Granule download** (``source="granules"``): Downloads full MODIS
      HDF4 granules via ``earthaccess``, clips to each polygon, and
      computes spatial statistics locally.  Slower (downloads GBs of
      data) but provides full spatial coverage including ``pixel_stats``
      mode.

    Parameters
    ----------
    polygon_file : str
        Path to a GeoJSON or shapefile with a ``Name`` column.
    product : str
        One of ``"ndvi"``, ``"evi"``, ``"lai"``, ``"fpar"``,
        ``"phenology"``.  Default ``"ndvi"``.
    year_start : int
        First year to include (default 2003).
    year_stop : int
        Last year to include, inclusive (default 2022).
    satellite : str
        ``"terra"``, ``"aqua"``, or ``"combined"`` (Terra + Aqua
        merged).  Only applies to ndvi/evi.  Default ``"terra"``.
    spatial_mode : str
        ``"mean"`` for a single spatial mean per polygon per timestep,
        or ``"pixel_stats"`` for mean/std/min/max/count.
        Default ``"mean"``.  Ignored for ``product="phenology"``.
        ``"pixel_stats"`` requires ``source="granules"``.
    source : str
        ``"appeears"`` (default) for fast server-side extraction, or
        ``"granules"`` for full granule download with local processing.

    Returns
    -------
    pd.DataFrame
        Schema depends on *product*:

        * **ndvi/evi/lai/fpar** (spatial_mode="mean"):
          ``polygon_id``, ``date``, ``year``, ``day_of_year``, ``value``
        * **ndvi/evi/lai/fpar** (spatial_mode="pixel_stats"):
          ``polygon_id``, ``date``, ``year``, ``day_of_year``,
          ``value_mean``, ``value_std``, ``value_min``, ``value_max``,
          ``pixel_count``
        * **phenology**:
          ``polygon_id``, ``year``, ``greenup_doy``,
          ``midgreenup_doy``, ``peak_doy``, ``maturity_doy``,
          ``midgreendown_doy``, ``senescence_doy``, ``dormancy_doy``

    Raises
    ------
    HyPlanValueError
        If the polygon file lacks a ``Name`` column, *product* is not
        recognized, or *satellite* is invalid.
    HyPlanRuntimeError
        If required packages are missing or authentication fails.
    """
    if product not in _VALID_PRODUCTS:
        raise HyPlanValueError(
            f"Unknown phenology product: {product!r}. "
            f"Choose from: {sorted(_VALID_PRODUCTS)}"
        )

    if satellite not in ("terra", "aqua", "combined"):
        raise HyPlanValueError(
            f"Unknown satellite: {satellite!r}. "
            f"Use 'terra', 'aqua', or 'combined'."
        )

    if product in ("lai", "fpar", "phenology") and satellite != "terra":
        raise HyPlanValueError(
            f"satellite parameter is only supported for ndvi/evi products. "
            f"{product!r} is Terra-only."
        )

    if spatial_mode not in ("mean", "pixel_stats"):
        raise HyPlanValueError(
            f"Unknown spatial_mode: {spatial_mode!r}. "
            f"Use 'mean' or 'pixel_stats'."
        )

    if source not in ("appeears", "granules"):
        raise HyPlanValueError(
            f"Unknown source: {source!r}. Use 'appeears' or 'granules'."
        )

    if spatial_mode == "pixel_stats" and source == "appeears":
        raise HyPlanValueError(
            "spatial_mode='pixel_stats' requires source='granules'. "
            "AppEEARS returns point samples, not pixel-level statistics."
        )

    if product == "phenology" and source == "appeears":
        raise HyPlanValueError(
            "product='phenology' (MCD12Q2 transitions) is not yet "
            "supported via AppEEARS. Use source='granules'."
        )

    # ── AppEEARS path (fast, server-side extraction) ──
    if source == "appeears":
        from ._appeears import fetch_appeears_timeseries

        gdf = gpd.read_file(polygon_file)
        if "Name" not in gdf.columns:
            raise HyPlanValueError(
                "Polygon file must contain a 'Name' column."
            )

        # Map satellite choice to AppEEARS product key
        appeears_product = product
        if satellite == "aqua" and product in ("ndvi", "evi"):
            appeears_product = f"{product}_aqua"

        coords = []
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            coords.append({
                "id": row["Name"],
                "latitude": c.y,
                "longitude": c.x,
            })

        df = fetch_appeears_timeseries(
            coordinates=coords,
            product=appeears_product,
            year_start=year_start,
            year_stop=year_stop,
        )

        if satellite == "combined" and product in ("ndvi", "evi"):
            # Also fetch Aqua and merge
            df_aqua = fetch_appeears_timeseries(
                coordinates=coords,
                product=f"{product}_aqua",
                year_start=year_start,
                year_stop=year_stop,
            )
            df = pd.concat([df, df_aqua], ignore_index=True)
            df = (
                df.groupby(["polygon_id", "year", "day_of_year"], as_index=False)
                .agg(date=("date", "first"), value=("value", "mean"))
            )

        return df.sort_values(
            ["polygon_id", "year", "day_of_year"]
        ).reset_index(drop=True)

    # ── Granule download path (slow, full spatial processing) ──
    # Load and validate polygons
    gdf = gpd.read_file(polygon_file)
    if "Name" not in gdf.columns:
        raise HyPlanValueError(
            "Polygon file must contain a 'Name' column identifying each polygon."
        )

    config = _PRODUCT_CONFIG[product]
    date_start = f"{year_start}-01-01"
    date_stop = f"{year_stop}-12-31"

    # Determine which short_names to query
    if product in ("ndvi", "evi"):
        if satellite == "terra":
            short_names = [config["short_name"]]  # type: ignore[index]
        elif satellite == "aqua":
            short_names = [config["short_name_aqua"]]  # type: ignore[index]
        else:  # combined
            short_names = [config["short_name"], config["short_name_aqua"]]  # type: ignore[index]
    else:
        short_names = [config["short_name"]]  # type: ignore[index]

    # Authenticate
    from .._auth import _earthdata_login

    _earthdata_login()

    all_rows = []

    for _, row in gdf.iterrows():
        polygon_name = row["Name"]
        geom = _drop_z(row.geometry)
        bbox = geom.bounds  # (minx, miny, maxx, maxy)
        # earthaccess expects (west, south, east, north)
        bounding_box = (bbox[0], bbox[1], bbox[2], bbox[3])

        for short_name in short_names:
            cache_dir = _get_cache_dir(short_name)

            granules = _search_granules(
                short_name, bounding_box, date_start, date_stop,
            )
            if not granules:
                logger.warning(
                    "No granules found for %s, polygon %s",
                    short_name, polygon_name,
                )
                continue

            hdf_paths = _download_granules(granules, cache_dir)

            for hdf_path in hdf_paths:
                try:
                    if product == "phenology":
                        result = _extract_phenology_from_granule(
                            hdf_path, geom, config,  # type: ignore[arg-type]
                        )
                    else:
                        result = _extract_vi_from_granule(
                            hdf_path, geom, config, spatial_mode,  # type: ignore[arg-type]
                        )
                except Exception:
                    logger.warning(
                        "Failed to process %s for polygon %s",
                        hdf_path, polygon_name,
                        exc_info=True,
                    )
                    continue

                if result is not None:
                    result["polygon_id"] = polygon_name
                    all_rows.append(result)

    if not all_rows:
        logger.warning("No valid data extracted for any polygon.")
        if product == "phenology":
            return pd.DataFrame(
                columns=["polygon_id", "year"] + list(
                    _PRODUCT_CONFIG["phenology"]["subdatasets"].keys()  # type: ignore[index]
                )
            )
        elif spatial_mode == "mean":
            return pd.DataFrame(
                columns=["polygon_id", "date", "year", "day_of_year", "value"]
            )
        else:
            return pd.DataFrame(
                columns=[
                    "polygon_id", "date", "year", "day_of_year",
                    "value_mean", "value_std", "value_min", "value_max",
                    "pixel_count",
                ]
            )

    df = pd.DataFrame(all_rows)

    # For combined satellite, average duplicate (polygon_id, year, doy) entries
    if product != "phenology" and satellite == "combined":
        if spatial_mode == "mean":
            group_cols = ["polygon_id", "year", "day_of_year"]
            df = df.groupby(group_cols, as_index=False).agg(
                date=("date", "first"),
                value=("value", "mean"),
            )
        else:
            group_cols = ["polygon_id", "year", "day_of_year"]
            df = df.groupby(group_cols, as_index=False).agg(
                date=("date", "first"),
                value_mean=("value_mean", "mean"),
                value_std=("value_std", "mean"),
                value_min=("value_min", "min"),
                value_max=("value_max", "max"),
                pixel_count=("pixel_count", "sum"),
            )

    return df.sort_values(
        ["polygon_id", "year", "day_of_year"] if "day_of_year" in df.columns
        else ["polygon_id", "year"]
    ).reset_index(drop=True)


def fetch_phenology_spatial(
    polygon_file: str,
    product: str = "ndvi",
    year_start: int = 2003,
    year_stop: int = 2022,
    satellite: str = "terra",
) -> dict[str, "xr.DataArray"]:
    """Compute per-pixel time-averaged vegetation index for each polygon.

    Returns a dictionary mapping polygon name to an xarray DataArray
    with dimensions ``(latitude, longitude)``.

    Parameters
    ----------
    polygon_file : str
        Path to a GeoJSON or shapefile with a ``Name`` column.
    product : str
        ``"ndvi"`` or ``"evi"``.  Default ``"ndvi"``.
    year_start : int
        First year (default 2003).
    year_stop : int
        Last year (default 2022).
    satellite : str
        ``"terra"``, ``"aqua"``, or ``"combined"``.

    Returns
    -------
    dict[str, xarray.DataArray]
        Polygon name to DataArray of mean vegetation index values.
    """
    xr = _require_xarray()

    if product not in ("ndvi", "evi"):
        raise HyPlanValueError(
            f"fetch_phenology_spatial only supports 'ndvi' or 'evi', "
            f"got {product!r}."
        )

    gdf = gpd.read_file(polygon_file)
    if "Name" not in gdf.columns:
        raise HyPlanValueError(
            "Polygon file must contain a 'Name' column identifying each polygon."
        )

    config = _PRODUCT_CONFIG[product]
    date_start = f"{year_start}-01-01"
    date_stop = f"{year_stop}-12-31"

    if satellite == "terra":
        short_names = [config["short_name"]]  # type: ignore[index]
    elif satellite == "aqua":
        short_names = [config["short_name_aqua"]]  # type: ignore[index]
    else:
        short_names = [config["short_name"], config["short_name_aqua"]]  # type: ignore[index]

    from .._auth import _earthdata_login

    _earthdata_login()

    result = {}

    for _, row in gdf.iterrows():
        polygon_name = row["Name"]
        geom = _drop_z(row.geometry)
        bbox = geom.bounds
        bounding_box = (bbox[0], bbox[1], bbox[2], bbox[3])

        all_arrays = []

        for short_name in short_names:
            cache_dir = _get_cache_dir(short_name)
            granules = _search_granules(
                short_name, bounding_box, date_start, date_stop,
            )
            hdf_paths = _download_granules(granules, cache_dir)

            for hdf_path in hdf_paths:
                try:
                    data, transform = _read_and_clip_subdataset(
                        hdf_path, config["subdataset"], geom,  # type: ignore[index]
                    )
                    qa, _ = _read_and_clip_subdataset(
                        hdf_path, config["qa_subdataset"], geom,  # type: ignore[index]
                    )

                    masked = config["qa_func"](data, qa)  # type: ignore[index]
                    if "valid_range" in config:  # type: ignore[operator]
                        lo, hi = config["valid_range"]  # type: ignore[index]
                        masked = np.ma.masked_outside(masked, lo, hi)

                    scaled = masked.astype(np.float64) * config["scale_factor"]  # type: ignore[index]
                    all_arrays.append(scaled)
                except Exception:
                    logger.warning(
                        "Failed to process %s for polygon %s",
                        hdf_path, polygon_name,
                        exc_info=True,
                    )
                    continue

        if all_arrays:
            # Stack and compute temporal mean, ignoring masked values
            stacked = np.ma.stack(all_arrays, axis=0)
            mean_arr = np.ma.mean(stacked, axis=0).filled(np.nan)

            # Build coordinate arrays from transform
            from rasterio.transform import xy

            rows_idx, cols_idx = np.arange(mean_arr.shape[0]), np.arange(mean_arr.shape[1])
            lats = np.array([xy(transform, r, 0)[1] for r in rows_idx])
            lons = np.array([xy(transform, 0, c)[0] for c in cols_idx])

            result[polygon_name] = xr.DataArray(
                mean_arr,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                name=product,
            )

    return result
