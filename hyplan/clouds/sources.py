"""Cloud fraction data sources (GEE/MODIS and Open-Meteo/ERA5 historical).

GEE functions query MODIS Terra/Aqua surface reflectance imagery at 1 km
resolution over user-supplied polygons.  **Requires GEE authentication.**

Open-Meteo functions query the ERA5 reanalysis archive at 0.25 deg resolution
using each polygon's centroid.  **No authentication required.**
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict

import geopandas as gpd
import pandas as pd
from shapely import wkb

from ..exceptions import HyPlanRuntimeError, HyPlanValueError

if TYPE_CHECKING:  # pragma: no cover
    import ee
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Earth Engine lazy initialization
# ---------------------------------------------------------------------------

_ee_initialized = False
_ee = None  # Populated by _get_ee()


def _get_ee():
    """Return the ``ee`` module, importing and initializing on first call.

    Raises:
        HyPlanRuntimeError: If ``earthengine-api`` is not installed or
            initialization fails.
    """
    global _ee_initialized, _ee
    if not _ee_initialized:
        try:
            import ee as _ee_mod
        except ImportError:
            raise HyPlanRuntimeError(
                "earthengine-api is required for cloud analysis. "
                "Install it with: pip install hyplan[clouds]"
            )
        try:
            _ee_mod.Initialize()
        except Exception as e:
            raise HyPlanRuntimeError(
                "Earth Engine initialization failed. "
                "Run ee.Authenticate() first."
            ) from e
        _ee = _ee_mod
        _ee_initialized = True
    return _ee


# ---------------------------------------------------------------------------
# GEE / MODIS helpers
# ---------------------------------------------------------------------------

def _drop_z(geom: "BaseGeometry") -> "BaseGeometry":
    """Strip Z coordinates from a Shapely geometry, returning a 2D geometry."""
    return wkb.loads(wkb.dumps(geom, output_dimension=2))


_VALID_SATELLITES = {"both", "terra", "aqua"}

_SATELLITE_COLLECTIONS = {
    "terra": "MODIS/061/MOD09GA",
    "aqua": "MODIS/061/MYD09GA",
}


def get_binary_cloud(image: "ee.Image") -> "ee.Image":
    """Generate a binary cloud mask for a MODIS image.

    The MOD09GA/MYD09GA state_1km band encodes cloud state in bits 0-1:
      00 = clear, 01 = cloudy, 10 = mixed, 11 = not set.
    Any non-zero value (bits 0-1 != 00) is treated as cloudy.

    Args:
        image: An Earth Engine image with a ``state_1km`` QA band.

    Returns:
        Binary cloud mask (1 for cloudy/mixed, 0 for clear).
    """
    _get_ee()
    qa = image.select("state_1km")
    clouds = qa.bitwiseAnd(3).gt(0)
    date_char = image.date().format('yyyy-MM-dd')
    result = clouds.set("date_char", date_char)
    result = result.set("satellite", image.get("satellite"))
    return result  # type: ignore[no-any-return]


def calculate_cloud_fraction(image: "ee.Image", polygon_geometry: "ee.Geometry") -> "ee.Feature":
    """Calculate the cloud fraction over a polygon for a MODIS image.

    Args:
        image: An Earth Engine MODIS image.
        polygon_geometry: A polygon geometry for the region of interest.

    Returns:
        An ``ee.Feature`` with date and cloud fraction properties.
    """
    ee = _get_ee()
    reduction = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_geometry,
        scale=1000
    )
    cloud_fraction = reduction.get('state_1km')
    return ee.Feature(None, {  # type: ignore[no-any-return]
        'date_char': image.get('date_char'),
        'cloud_fraction': cloud_fraction,
        'satellite': image.get('satellite'),
    })


def create_date_ranges(day_start: int, day_stop: int, year_start: int, year_stop: int) -> list:
    """Create date ranges for filtering Earth Engine image collections.

    Supports year-boundary crossings (e.g., day_start=335, day_stop=60 for a
    December-to-February campaign). When day_start > day_stop, each year-pair
    produces two date ranges.

    Args:
        day_start: Start day of the year (1-365).
        day_stop: End day of the year (1-365).
        year_start: Start year for the ranges.
        year_stop: End year for the ranges.

    Returns:
        List of (start_date, end_date) tuples suitable for ``filterDate``.
    """
    date_ranges = []

    if day_start <= day_stop:
        for year in range(year_start, year_stop + 1):
            date_ranges.append((f"{year}-{day_start:03}", f"{year}-{day_stop + 1:03}"))
    else:
        for year in range(year_start, year_stop + 1):
            date_ranges.append((f"{year}-{day_start:03}", f"{year + 1}-001"))
            date_ranges.append((f"{year + 1}-001", f"{year + 1}-{day_stop + 1:03}"))

    return date_ranges


def create_cloud_data_array_with_limit(
    polygon_file: str,
    year_start: int,
    year_stop: int,
    day_start: int,
    day_stop: int,
    limit: int = 5000,
    satellite: str = "both",
    split_satellite: bool = False,
) -> pd.DataFrame:
    """Fetch MODIS cloud data for polygons via Google Earth Engine.

    Queries MODIS Terra and/or Aqua surface reflectance imagery and computes
    daily cloud fractions over each polygon at 1 km resolution.

    Args:
        polygon_file: Path to a GeoJSON or shapefile with a ``Name`` column.
        year_start: Start year.
        year_stop: End year.
        day_start: Start day-of-year.
        day_stop: End day-of-year.
        limit: Max images per date range (default 5000).
        satellite: ``"both"``, ``"terra"``, or ``"aqua"``.
        split_satellite: If ``True``, include a ``satellite`` column.

    Returns:
        DataFrame with columns ``polygon_id``, ``year``, ``day_of_year``,
        ``cloud_fraction`` (plus ``satellite`` if *split_satellite*).
    """
    if satellite not in _VALID_SATELLITES:
        raise HyPlanValueError(
            f"satellite must be one of {_VALID_SATELLITES}, got {satellite!r}"
        )

    ee = _get_ee()
    try:
        gdf = gpd.read_file(polygon_file)
        if gdf.empty:
            raise HyPlanValueError("Polygon file is empty or invalid.")
    except Exception as e:
        raise HyPlanRuntimeError(f"Failed to load polygon file: {polygon_file}") from e

    if 'Name' not in gdf.columns:
        raise HyPlanValueError(f"Polygon file must contain a 'Name' column. Found columns: {list(gdf.columns)}")

    gdf = gdf[['Name', 'geometry']].copy()
    gdf['geometry'] = gdf['geometry'].apply(_drop_z)

    results = []
    date_ranges = create_date_ranges(day_start, day_stop, year_start, year_stop)

    sats_to_query = (
        list(_SATELLITE_COLLECTIONS.keys())
        if satellite == "both"
        else [satellite]
    )

    try:
        cloud_data = ee.ImageCollection([])
        for start, stop in date_ranges:
            for sat_name in sats_to_query:
                col = (
                    ee.ImageCollection(_SATELLITE_COLLECTIONS[sat_name])
                    .filterDate(start, stop)
                    .limit(limit)
                    .map(lambda img, _s=sat_name: img.set("satellite", _s))
                )
                cloud_data = cloud_data.merge(col)
        cloud_data = cloud_data.map(get_binary_cloud)
    except Exception as e:
        raise HyPlanRuntimeError("Error occurred while processing MODIS data.") from e

    for _, row in gdf.iterrows():
        polygon_name = row['Name']
        polygon_geometry = ee.Geometry(row['geometry'].__geo_interface__)

        mapped_results = cloud_data.map(lambda image: calculate_cloud_fraction(image, polygon_geometry))
        feature_list = ee.FeatureCollection(mapped_results).limit(limit).getInfo()['features']

        for feature in feature_list:
            properties = feature['properties']
            date_char = properties.get('date_char')
            if date_char is None:
                continue
            cloud_fraction = properties.get('cloud_fraction')
            if cloud_fraction is not None:
                year, month, day = [int(x) for x in date_char.split('-')]
                day_of_year = pd.Timestamp(year=year, month=month, day=day).dayofyear
                rec = {
                    'year': year,
                    'day_of_year': day_of_year,
                    'polygon_id': polygon_name,
                    'cloud_fraction': cloud_fraction,
                }
                if split_satellite:
                    rec['satellite'] = properties.get('satellite', 'unknown')
                results.append(rec)

    results_df = pd.DataFrame(results)
    group_cols = ['polygon_id', 'year', 'day_of_year']
    if split_satellite:
        group_cols.append('satellite')
    aggregated_df = results_df.groupby(group_cols).mean().reset_index()
    return aggregated_df


# ---------------------------------------------------------------------------
# Open-Meteo ERA5 historical cloud fraction (centroid-based, no auth)
# ---------------------------------------------------------------------------

_OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoCloudFraction:
    """Fetch daily cloud fraction from the Open-Meteo ERA5 archive.

    Uses the `Open-Meteo Historical Weather API
    <https://open-meteo.com/en/docs/historical-weather-api>`_ to retrieve
    daily mean total cloud cover from ERA5 reanalysis at 0.25 deg (~25 km)
    resolution.  **No authentication required.**

    For each polygon the centroid is used as the query point.  One HTTP
    request per polygon covers the entire date range (all years at once),
    so even 20-year x 5-polygon queries complete in seconds.

    Args:
        url: Override the Open-Meteo archive endpoint (for testing).
    """

    def __init__(self, url: str | None = None):
        self._url = url or _OPENMETEO_ARCHIVE_URL

    def fetch(
        self,
        polygons: gpd.GeoDataFrame,
        year_start: int,
        year_stop: int,
        day_start: int,
        day_stop: int,
    ) -> pd.DataFrame:
        """Fetch daily cloud fraction for each polygon.

        Args:
            polygons: GeoDataFrame with a ``Name`` column and polygon
                geometries (WGS 84).
            year_start: First year to include.
            year_stop: Last year to include (inclusive).
            day_start: Start day-of-year (1-365).
            day_stop: End day-of-year (1-365).  If ``day_start > day_stop``
                the range crosses a year boundary (e.g. Dec to Feb).

        Returns:
            DataFrame with columns ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction`` (0.0-1.0).
        """
        import requests as _requests  # type: ignore[import-untyped]

        if "Name" not in polygons.columns:
            raise HyPlanValueError(
                f"Polygon GeoDataFrame must have a 'Name' column. "
                f"Found: {list(polygons.columns)}"
            )

        crosses_year = day_start > day_stop

        start_date = f"{year_start}-01-01"
        if crosses_year:
            end_date = f"{year_stop + 1}-12-31"
        else:
            end_date = f"{year_stop}-12-31"

        rows: list[dict] = []

        for _, row in polygons.iterrows():
            name = row["Name"]
            centroid = row["geometry"].centroid
            lat, lon = centroid.y, centroid.x

            logger.info("Fetching Open-Meteo cloud cover for %s (%.2f, %.2f)", name, lat, lon)

            resp = _requests.get(
                self._url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": "cloud_cover_mean",
                    "timezone": "UTC",
                },
                timeout=60,
            )
            if resp.status_code != 200:
                raise HyPlanRuntimeError(
                    f"Open-Meteo request failed (HTTP {resp.status_code}): "
                    f"{resp.text[:200]}"
                )

            data = resp.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            cloud_pct = daily.get("cloud_cover_mean", [])

            for date_str, pct in zip(dates, cloud_pct):
                if pct is None:
                    continue
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                year = dt.year
                doy = dt.timetuple().tm_yday

                if year < year_start or year > (year_stop + 1 if crosses_year else year_stop):
                    continue

                if crosses_year:
                    in_window = doy >= day_start or doy <= day_stop
                else:
                    in_window = day_start <= doy <= day_stop

                if not in_window:
                    continue

                if crosses_year and doy <= day_stop:
                    season_year = year - 1
                else:
                    season_year = year

                if season_year < year_start or season_year > year_stop:
                    continue

                rows.append({
                    "polygon_id": name,
                    "year": season_year,
                    "day_of_year": doy,
                    "cloud_fraction": pct / 100.0,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return df.groupby(["polygon_id", "year", "day_of_year"]).mean().reset_index()


def fetch_cloud_fraction(
    polygon_file: str,
    year_start: int,
    year_stop: int,
    day_start: int,
    day_stop: int,
    source: str = "openmeteo",
    **kwargs,
) -> pd.DataFrame:
    """Fetch historical cloud fraction data for flight planning.

    Factory function that selects the appropriate cloud data source and
    returns a DataFrame compatible with :func:`simulate_visits` and
    :func:`plot_yearly_cloud_fraction_heatmaps_with_visits`.

    Args:
        polygon_file: Path to a GeoJSON or shapefile with a ``Name`` column.
        year_start: First year to include.
        year_stop: Last year to include (inclusive).
        day_start: Start day-of-year (1-365).
        day_stop: End day-of-year (1-365).
        source: ``"openmeteo"`` (ERA5, 0.25 deg, no auth) or
            ``"gee"`` (MODIS, 1 km, requires GEE auth).
        **kwargs: Passed to the source constructor.

    Returns:
        DataFrame with columns ``polygon_id``, ``year``,
        ``day_of_year``, ``cloud_fraction``.
    """
    gdf = gpd.read_file(polygon_file)

    if source == "openmeteo":
        sat = kwargs.pop("satellite", "both")
        kwargs.pop("split_satellite", None)
        if sat != "both":
            raise HyPlanValueError(
                "Morning/afternoon (satellite) discrimination is only "
                "available with source='gee'."
            )
        return OpenMeteoCloudFraction(**kwargs).fetch(
            gdf, year_start, year_stop, day_start, day_stop,
        )
    elif source == "gee":
        return create_cloud_data_array_with_limit(
            polygon_file, year_start, year_stop, day_start, day_stop,
            **kwargs,
        )
    else:
        raise HyPlanValueError(
            f"Unknown cloud fraction source: {source!r}. "
            f"Use 'openmeteo' or 'gee'."
        )


# ---------------------------------------------------------------------------
# Spatial cloud fraction maps (GEE, per-pixel)
# ---------------------------------------------------------------------------

def fetch_cloud_fraction_spatial(
    polygon_file: str,
    year_start: int,
    year_stop: int,
    day_start: int,
    day_stop: int,
    scale: int = 1000,
    satellite: str = "both",
) -> "dict[str, object]":
    """Compute a per-pixel mean cloud fraction map for each polygon.

    Uses Google Earth Engine to produce a time-averaged cloud fraction
    raster at the native MODIS resolution within each polygon's bounding
    box.  Requires GEE authentication.

    Args:
        polygon_file: Path to a GeoJSON or shapefile with a ``Name`` column.
        year_start: First year to include.
        year_stop: Last year to include (inclusive).
        day_start: Start day-of-year (1-365).
        day_stop: End day-of-year (1-365).
        scale: Output resolution in metres (default 1000 = MODIS native).
        satellite: ``"both"``, ``"terra"``, or ``"aqua"``.

    Returns:
        Dictionary mapping polygon name to ``xarray.DataArray`` with
        dimensions ``(latitude, longitude)`` and values 0.0-1.0.
    """
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        raise HyPlanRuntimeError(
            "xarray and numpy are required for spatial cloud maps. "
            "Install with: pip install xarray numpy"
        )

    if satellite not in _VALID_SATELLITES:
        raise HyPlanValueError(
            f"satellite must be one of {_VALID_SATELLITES}, got {satellite!r}"
        )

    ee = _get_ee()

    gdf = gpd.read_file(polygon_file)
    if "Name" not in gdf.columns:
        raise HyPlanValueError(
            f"Polygon file must have a 'Name' column. Found: {list(gdf.columns)}"
        )
    gdf["geometry"] = gdf["geometry"].apply(_drop_z)

    date_ranges = create_date_ranges(day_start, day_stop, year_start, year_stop)

    sats_to_query = (
        list(_SATELLITE_COLLECTIONS.keys())
        if satellite == "both"
        else [satellite]
    )

    cloud_data = ee.ImageCollection([])
    for start, stop in date_ranges:
        for sat_name in sats_to_query:
            col = (
                ee.ImageCollection(_SATELLITE_COLLECTIONS[sat_name])
                .filterDate(start, stop)
            )
            cloud_data = cloud_data.merge(col)
    cloud_data = cloud_data.map(get_binary_cloud)
    mean_image = cloud_data.mean()

    result: dict[str, object] = {}

    for _, row in gdf.iterrows():
        name = row["Name"]
        geom = ee.Geometry(row["geometry"].__geo_interface__)
        bounds = row["geometry"].bounds

        logger.info("Downloading spatial cloud map for %s", name)

        try:
            url = mean_image.getDownloadURL({
                "region": geom,
                "scale": scale,
                "format": "NPY",
            })
        except Exception as exc:
            raise HyPlanRuntimeError(
                f"GEE download URL generation failed for {name}"
            ) from exc

        import requests as _requests  # type: ignore[import-untyped]
        import io
        resp = _requests.get(url, timeout=120)
        if resp.status_code != 200:
            raise HyPlanRuntimeError(
                f"GEE download failed for {name} (HTTP {resp.status_code})"
            )

        arr = np.load(io.BytesIO(resp.content), allow_pickle=True)
        if arr.dtype.names:
            data = arr["state_1km"].astype(float)
        else:
            data = arr.astype(float)

        nrows, ncols = data.shape
        minx, miny, maxx, maxy = bounds
        lons = np.linspace(minx, maxx, ncols)
        lats = np.linspace(maxy, miny, nrows)

        da = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
            name="cloud_fraction_mean",
            attrs={"units": "fraction", "long_name": "Mean cloud fraction"},
        )
        result[name] = da

    return result
