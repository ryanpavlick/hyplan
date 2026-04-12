"""
HyPlan Clouds

Overview:
Optical remote sensing of the Earth's surface often requires clear skies. Deploying airborne remote sensing
instruments can be costly, with daily costs for aircraft, labor, and per diem travel expenses for aircraft
and instrument teams. This script addresses the question:
"Statistically, how many days is it likely to take to acquire clear-sky observations for a given set of flight boxes?".

The script operates under several simplifying assumptions:
- Each flight box is "flyable" in a single day given clear skies from a single base of operations.
- Instantaneous MODIS Terra/Aqua overpasses are representative of clear-sky conditions throughout the flight day.
- Other environmental state parameters (e.g., tides, wind speeds) do not influence go/no-go decisions for flights.

Key Features:
1. **Cloud Data Processing**:
    - Reads geospatial polygon data (GeoJSON) representing flight areas for airborne optical remote sensing.
    - Fetches MODIS cloud fraction data from Google Earth Engine for specified years and day ranges.
    - Aggregates daily cloud fraction data for each polygon.

2. **Flight Simulation**:
    - Simulates daily flight schedules to visit polygons based on a maximum cloud fraction threshold.
    - Enforces constraints such as maximum consecutive flight days and optional weekend exclusions.

3. **Visualization**:
    - Generates heatmaps of cloud conditions, visit days, and rest days for each simulated year.
    - Produces cumulative distribution function (CDF) plots to estimate the likelihood of completing visits.

Campaigns that cross a year boundary (e.g. December to February) are supported:
set day_start > day_stop (e.g. day_start=335, day_stop=60).

References
----------
Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale
geospatial analysis for everyone. *Remote Sensing of Environment*, 202,
18-27. doi:10.1016/j.rse.2017.06.031

Data source: MODIS Terra (MOD09GA) and Aqua (MYD09GA) surface reflectance
daily L2G products from NASA LP DAAC, accessed via Google Earth Engine.
"""

# Core Libraries
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, TYPE_CHECKING

# Geospatial Libraries
import geopandas as gpd
from shapely import wkb

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .exceptions import HyPlanRuntimeError, HyPlanValueError

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import ee  # Google Earth Engine, optional runtime dep
    from shapely.geometry.base import BaseGeometry

__all__ = [
    "get_binary_cloud", "calculate_cloud_fraction", "create_date_ranges",
    "create_cloud_data_array_with_limit", "simulate_visits",
    "plot_yearly_cloud_fraction_heatmaps_with_visits",
    "OpenMeteoCloudFraction", "fetch_cloud_fraction",
    "summarize_cloud_fraction_by_doy", "plot_doy_cloud_fraction",
    "OpenMeteoCloudForecast", "fetch_cloud_forecast",
    "fetch_cloud_fraction_spatial", "plot_cloud_fraction_spatial",
]

logger = logging.getLogger(__name__)

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


def _drop_z(geom: "BaseGeometry") -> "BaseGeometry":
    """
    Strip Z coordinates from a Shapely geometry, returning a 2D geometry.

    Args:
        geom: A Shapely geometry, potentially with Z coordinates.

    Returns:
        A 2D Shapely geometry with Z values removed.
    """
    return wkb.loads(wkb.dumps(geom, output_dimension=2))


def get_binary_cloud(image: "ee.Image") -> "ee.Image":
    """
    Generates a binary cloud mask for a given MODIS image.

    The MOD09GA/MYD09GA state_1km band encodes cloud state in bits 0-1:
      00 = clear, 01 = cloudy, 10 = mixed, 11 = not set.
    Any non-zero value (bits 0-1 != 00) is treated as cloudy.

    Parameters:
        image (ee.Image): An Earth Engine image with a "state_1km" quality assessment band.

    Returns:
        ee.Image: Binary cloud mask (1 for cloudy/mixed, 0 for clear) with an added "date_char" property.
    """
    _get_ee()
    qa = image.select("state_1km")
    clouds = qa.bitwiseAnd(3).gt(0)
    date_char = image.date().format('yyyy-MM-dd')
    result = clouds.set("date_char", date_char)
    # Propagate satellite tag when present (for split_satellite mode)
    result = result.set("satellite", image.get("satellite"))
    return result

def calculate_cloud_fraction(image: "ee.Image", polygon_geometry: "ee.Geometry") -> "ee.Feature":
    """
    Calculates the cloud fraction over a given polygon for a MODIS image.

    Parameters:
        image (ee.Image): An Earth Engine MODIS image.
        polygon_geometry (ee.Geometry): A polygon geometry representing the region of interest.

    Returns:
        ee.Feature: A feature containing the date and calculated cloud fraction for the polygon.
    """
    ee = _get_ee()
    reduction = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_geometry,
        scale=1000
    )
    cloud_fraction = reduction.get('state_1km')
    return ee.Feature(None, {
        'date_char': image.get('date_char'),
        'cloud_fraction': cloud_fraction,
        'satellite': image.get('satellite'),
    })

def create_date_ranges(day_start: int, day_stop: int, year_start: int, year_stop: int) -> list:
    """
    Creates date ranges for filtering Earth Engine image collections.

    Supports year-boundary crossings (e.g., day_start=335, day_stop=60 for a
    December-to-February campaign). When day_start > day_stop, each year-pair
    produces two date ranges: one from day_start to Dec 31 and one from Jan 1
    to day_stop in the following year.

    Parameters:
        day_start (int): Start day of the year (1-365).
        day_stop (int): End day of the year (1-365).
        year_start (int): Start year for the ranges.
        year_stop (int): End year for the ranges.

    Returns:
        list of tuples: A list of date range tuples (start_date, end_date) in
            YYYY-DDD or YYYY-MM-DD format suitable for Earth Engine filterDate.
    """
    date_ranges = []

    if day_start <= day_stop:
        # Normal case: campaign within a single calendar year
        for year in range(year_start, year_stop + 1):
            date_ranges.append((f"{year}-{day_start:03}", f"{year}-{day_stop + 1:03}"))
    else:
        # Year-boundary crossing: day_start (e.g., 335) > day_stop (e.g., 60)
        # Each "season" spans from day_start of year N to day_stop of year N+1
        for year in range(year_start, year_stop + 1):
            # Part 1: day_start to Dec 31 of current year
            date_ranges.append((f"{year}-{day_start:03}", f"{year + 1}-001"))
            # Part 2: Jan 1 to day_stop of following year
            date_ranges.append((f"{year + 1}-001", f"{year + 1}-{day_stop + 1:03}"))

    return date_ranges

_VALID_SATELLITES = {"both", "terra", "aqua"}

_SATELLITE_COLLECTIONS = {
    "terra": "MODIS/061/MOD09GA",
    "aqua": "MODIS/061/MYD09GA",
}


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
    """
    Processes MODIS cloud data for polygons and calculates daily cloud fractions.

    The polygon file must contain a 'Name' column identifying each polygon.

    Parameters:
        polygon_file (str): Path to a GeoJSON or shapefile containing polygons with a 'Name' column.
        year_start (int): Start year for data processing.
        year_stop (int): End year for data processing.
        day_start (int): Start day of the year for data processing.
        day_stop (int): End day of the year for data processing.
        limit (int, optional): Maximum number of images to process per date range. Default is 5000.
        satellite (str): Which MODIS satellite to use: ``"both"`` (default),
            ``"terra"`` (morning, ~10:30 local), or ``"aqua"`` (afternoon,
            ~13:30 local).
        split_satellite (bool): If ``True``, include a ``satellite`` column
            in the output so Terra and Aqua observations are kept separate.
            When ``True`` with ``satellite="both"``, the same
            ``(polygon_id, year, day_of_year)`` may appear twice.  Filter or
            aggregate before passing to :func:`simulate_visits`.

    Returns:
        pd.DataFrame: A DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'
            (plus 'satellite' if *split_satellite* is True).
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
# Open-Meteo cloud fraction (no auth, ERA5 0.25°)
# ---------------------------------------------------------------------------

_OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoCloudFraction:
    """Fetch daily cloud fraction from the Open-Meteo ERA5 archive.

    Uses the `Open-Meteo Historical Weather API
    <https://open-meteo.com/en/docs/historical-weather-api>`_ to retrieve
    daily mean total cloud cover from ERA5 reanalysis at 0.25° (~25 km)
    resolution.  **No authentication required.**

    For each polygon the centroid is used as the query point.  One HTTP
    request per polygon covers the entire date range (all years at once),
    so even 20-year × 5-polygon queries complete in seconds.

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
            day_start: Start day-of-year (1–365).
            day_stop: End day-of-year (1–365).  If ``day_start > day_stop``
                the range crosses a year boundary (e.g. Dec → Feb).

        Returns:
            DataFrame with columns ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction`` (0.0–1.0).
        """
        import requests as _requests

        if "Name" not in polygons.columns:
            raise HyPlanValueError(
                f"Polygon GeoDataFrame must have a 'Name' column. "
                f"Found: {list(polygons.columns)}"
            )

        crosses_year = day_start > day_stop

        # Build the overall date window.  Open-Meteo accepts ISO dates.
        start_date = f"{year_start}-01-01"
        if crosses_year:
            # Need data through day_stop of year_stop + 1
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

                # Filter to requested year/day-of-year window
                if year < year_start or year > (year_stop + 1 if crosses_year else year_stop):
                    continue

                if crosses_year:
                    # Accept days in [day_start, 365] or [1, day_stop]
                    in_window = doy >= day_start or doy <= day_stop
                else:
                    in_window = day_start <= doy <= day_stop

                if not in_window:
                    continue

                # For cross-year campaigns, assign the "season year" as the
                # year containing day_start (matching simulate_visits logic).
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

        # Average duplicates (shouldn't happen, but for safety)
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
        day_start: Start day-of-year (1–365).
        day_stop: End day-of-year (1–365).
        source: Data source to use:

            ``"openmeteo"``
                ERA5 reanalysis via Open-Meteo (0.25°, no auth).
            ``"gee"``
                MODIS via Google Earth Engine (1 km, requires GEE auth).

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
# Cloud cover *forecasts* (near-term, not historical)
# ---------------------------------------------------------------------------

_OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoCloudForecast:
    """Fetch cloud cover forecasts from the Open-Meteo Forecast API.

    Provides up to 16 days of forecast cloud cover at any global location.
    **No authentication required.**

    For each polygon the centroid is used as the query point.

    Args:
        url: Override the Open-Meteo forecast endpoint (for testing).
    """

    _MAX_FORECAST_DAYS = 16

    def __init__(self, url: str | None = None):
        self._url = url or _OPENMETEO_FORECAST_URL

    def fetch(
        self,
        polygons: gpd.GeoDataFrame,
        forecast_days: int = 7,
        hourly: bool = False,
        models: "list[str] | None" = None,
    ) -> pd.DataFrame:
        """Fetch cloud cover forecast for each polygon.

        Args:
            polygons: GeoDataFrame with a ``Name`` column and polygon
                geometries (WGS 84).
            forecast_days: Number of days to forecast (1–16).
            hourly: If ``True``, return hourly data with layer breakdown
                (low / mid / high cloud cover).  Otherwise return daily means.
            models: Optional list of NWP model identifiers to use (e.g.
                ``["ecmwf_ifs025"]``).  ``None`` uses Open-Meteo's automatic
                best-match selection.

        Returns:
            DataFrame with columns:

            * **daily** (default): ``polygon_id``, ``date``,
              ``cloud_fraction`` (0.0–1.0).
            * **hourly**: ``polygon_id``, ``date``, ``hour``,
              ``cloud_fraction``, ``cloud_fraction_low``,
              ``cloud_fraction_mid``, ``cloud_fraction_high``.
        """
        import requests as _requests

        if "Name" not in polygons.columns:
            raise HyPlanValueError(
                f"Polygon GeoDataFrame must have a 'Name' column. "
                f"Found: {list(polygons.columns)}"
            )

        if not 1 <= forecast_days <= self._MAX_FORECAST_DAYS:
            raise HyPlanValueError(
                f"forecast_days must be between 1 and {self._MAX_FORECAST_DAYS}, "
                f"got {forecast_days}."
            )

        rows: list[dict] = []

        for _, row in polygons.iterrows():
            name = row["Name"]
            centroid = row["geometry"].centroid
            lat, lon = centroid.y, centroid.x

            logger.info(
                "Fetching Open-Meteo cloud forecast for %s (%.2f, %.2f)",
                name, lat, lon,
            )

            params: dict = {
                "latitude": lat,
                "longitude": lon,
                "forecast_days": forecast_days,
                "timezone": "UTC",
            }
            if hourly:
                params["hourly"] = (
                    "cloud_cover,cloud_cover_low,"
                    "cloud_cover_mid,cloud_cover_high"
                )
            else:
                params["daily"] = "cloud_cover_mean"

            if models:
                params["models"] = ",".join(models)

            resp = _requests.get(self._url, params=params, timeout=60)
            if resp.status_code != 200:
                raise HyPlanRuntimeError(
                    f"Open-Meteo forecast request failed "
                    f"(HTTP {resp.status_code}): {resp.text[:200]}"
                )

            data = resp.json()

            if hourly:
                h = data.get("hourly", {})
                times = h.get("time", [])
                cc = h.get("cloud_cover", [])
                cc_low = h.get("cloud_cover_low", [])
                cc_mid = h.get("cloud_cover_mid", [])
                cc_high = h.get("cloud_cover_high", [])
                for t, c, cl, cm, ch in zip(times, cc, cc_low, cc_mid, cc_high):
                    if c is None:
                        continue
                    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                    rows.append({
                        "polygon_id": name,
                        "date": dt.date(),
                        "hour": dt.hour,
                        "cloud_fraction": c / 100.0,
                        "cloud_fraction_low": (cl or 0) / 100.0,
                        "cloud_fraction_mid": (cm or 0) / 100.0,
                        "cloud_fraction_high": (ch or 0) / 100.0,
                    })
            else:
                d = data.get("daily", {})
                dates = d.get("time", [])
                cloud_pct = d.get("cloud_cover_mean", [])
                for date_str, pct in zip(dates, cloud_pct):
                    if pct is None:
                        continue
                    rows.append({
                        "polygon_id": name,
                        "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
                        "cloud_fraction": pct / 100.0,
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_cloud_forecast(
    polygon_file: str,
    source: str = "openmeteo",
    forecast_days: int = 7,
    hourly: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fetch cloud cover forecasts for flight planning.

    Args:
        polygon_file: Path to a GeoJSON or shapefile with a ``Name`` column.
        source: Forecast source.  Currently only ``"openmeteo"`` is supported.
        forecast_days: Number of days to forecast (1–16).
        hourly: If ``True``, return hourly resolution with cloud-layer
            breakdown.
        **kwargs: Passed to the source constructor (e.g. ``models``).

    Returns:
        DataFrame — see :class:`OpenMeteoCloudForecast` for column details.
    """
    gdf = gpd.read_file(polygon_file)

    if source == "openmeteo":
        models = kwargs.pop("models", None)
        return OpenMeteoCloudForecast(**kwargs).fetch(
            gdf, forecast_days=forecast_days, hourly=hourly, models=models,
        )
    else:
        raise HyPlanValueError(
            f"Unknown cloud forecast source: {source!r}. "
            f"Currently only 'openmeteo' is supported."
        )


def summarize_cloud_fraction_by_doy(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Compute a "typical year" cloud fraction summary per polygon.

    Averages cloud fraction across all years for each
    ``(polygon_id, day_of_year)`` pair, producing a single seasonal profile
    per polygon.

    Args:
        df: Cloud fraction DataFrame with columns ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction`` (as returned by
            :func:`fetch_cloud_fraction`).
        window: Optional rolling-mean window size (centered) applied to the
            per-polygon DOY mean.  ``None`` disables smoothing.

    Returns:
        DataFrame with columns ``polygon_id``, ``day_of_year``,
        ``cloud_fraction_mean``, ``cloud_fraction_std``,
        ``cloud_fraction_count``.
    """
    required = {"polygon_id", "year", "day_of_year", "cloud_fraction"}
    if not required.issubset(df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required}. "
            f"Found: {set(df.columns)}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=["polygon_id", "day_of_year",
                     "cloud_fraction_mean", "cloud_fraction_std",
                     "cloud_fraction_count"]
        )

    summary = (
        df.groupby(["polygon_id", "day_of_year"])["cloud_fraction"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "cloud_fraction_mean",
            "std": "cloud_fraction_std",
            "count": "cloud_fraction_count",
        })
        .reset_index()
    )

    if window is not None:
        summary["cloud_fraction_mean"] = (
            summary.groupby("polygon_id")["cloud_fraction_mean"]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
        )

    return summary


def plot_doy_cloud_fraction(
    summary_df: pd.DataFrame,
    ax: "plt.Axes | None" = None,
    show_std: bool = True,
    **kwargs,
) -> "plt.Axes":
    """Line plot of DOY cloud fraction for each polygon.

    Args:
        summary_df: Output of :func:`summarize_cloud_fraction_by_doy`.
        ax: Matplotlib Axes to plot on.  Created if ``None``.
        show_std: If ``True``, draw a shaded ±1 std-dev band.
        **kwargs: Passed to ``ax.plot()``.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, grp in summary_df.groupby("polygon_id"):
        grp = grp.sort_values("day_of_year")
        ax.plot(grp["day_of_year"], grp["cloud_fraction_mean"],
                label=name, **kwargs)
        if show_std and "cloud_fraction_std" in grp.columns:
            ax.fill_between(
                grp["day_of_year"],
                grp["cloud_fraction_mean"] - grp["cloud_fraction_std"],
                grp["cloud_fraction_mean"] + grp["cloud_fraction_std"],
                alpha=0.2,
            )

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Cloud Fraction")
    ax.legend()
    return ax


def simulate_visits(
    df: pd.DataFrame,
    day_start: int,
    day_stop: int,
    year_start: int,
    year_stop: int,
    cloud_fraction_threshold: float = 0.10,
    rest_day_threshold: int = 6,
    exclude_weekends: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
    """
    Simulate visits to polygons based on cloud fraction thresholds, ensuring no more than one visit per day.
    Adds rest days after a set number of consecutive visits and resets counters on weekends or when no polygons meet the threshold.

    On each visitable day, the alphabetically first unvisited polygon that meets the cloud threshold is chosen.
    Rest days count toward total_days but no polygon is visited.

    Parameters:
        df (pd.DataFrame): Cloud fraction data with columns: 'polygon_id', 'year', 'day_of_year', 'cloud_fraction'.
        day_start (int): Start day of the year for simulation.
        day_stop (int): End day of the year for simulation.
        year_start (int): Start year for simulation.
        year_stop (int): End year for simulation.
        cloud_fraction_threshold (float): Maximum allowable cloud fraction for a visit.
        rest_day_threshold (int): Maximum number of consecutive visits before a rest day is required.
        exclude_weekends (bool): If True, skip weekends and reset the counter for rest days.
        debug (bool): If True, enable detailed logging for debugging.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
            - DataFrame summarizing total days simulated per year.
            - Dictionary of visit days for each polygon, organized by year.
            - Dictionary of rest days for each year.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build the day sequence. When day_start > day_stop the campaign
    # crosses a year boundary (e.g. Dec 1 through Feb 28).
    crosses_year = day_start > day_stop

    visit_days = []
    visit_tracker = {}
    rest_days = {}

    for year in range(year_start, year_stop + 1):
        visited_polygons = set()
        remaining_polygons = set(df['polygon_id'].unique())
        visit_tracker[year] = {}
        rest_days[year] = []
        total_days = 0
        consecutive_visits = 0

        if crosses_year:
            # e.g. day 335..365 in *year*, then day 1..day_stop in *year+1*
            last_day_of_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
            day_sequence = list(range(day_start, last_day_of_year + 1)) + list(range(1, day_stop + 1))
        else:
            day_sequence = list(range(day_start, day_stop + 1))

        for seq_idx, current_day_of_year in enumerate(day_sequence):
            # Determine which calendar year this day falls in
            if crosses_year and current_day_of_year < day_start:
                current_year = year + 1
            else:
                current_year = year

            total_days += 1
            current_date = datetime(current_year, 1, 1) + timedelta(days=current_day_of_year - 1)

            if exclude_weekends and current_date.weekday() >= 5:
                logger.debug(f"Skipping weekend on day {current_day_of_year} of year {current_year}")
                consecutive_visits = 0
                continue

            daily_df = df[(df['year'] == current_year) & (df['day_of_year'] == current_day_of_year)]
            daily_df = daily_df[~daily_df['polygon_id'].isin(visited_polygons)]
            visitable_polygons = daily_df[daily_df['cloud_fraction'] <= cloud_fraction_threshold]

            if not visitable_polygons.empty:
                if consecutive_visits < rest_day_threshold:
                    polygon_to_visit = visitable_polygons.sort_values(by='polygon_id').iloc[0]
                    polygon_id = polygon_to_visit['polygon_id']

                    visited_polygons.add(polygon_id)
                    remaining_polygons.discard(polygon_id)

                    if polygon_id not in visit_tracker[year]:
                        visit_tracker[year][polygon_id] = []
                    visit_tracker[year][polygon_id].append(current_day_of_year)

                    logger.debug(f"Visiting polygon {polygon_id} on day {current_day_of_year} of year {current_year}")
                    consecutive_visits += 1
                else:
                    rest_days[year].append(current_day_of_year)
                    logger.info(f"Rest day added on day {current_day_of_year} of year {current_year}")
                    consecutive_visits = 0
            else:
                logger.debug(f"No visitable polygons on day {current_day_of_year} of year {current_year}")
                consecutive_visits = 0

            if not remaining_polygons:
                logger.info(f"All polygons visited for year {year}.")
                break

        visit_days.append({'year': year, 'days': total_days})

    return pd.DataFrame(visit_days), visit_tracker, rest_days


# ---------------------------------------------------------------------------
# Spatial cloud fraction maps (per-pixel, not polygon-mean)
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
        day_start: Start day-of-year (1–365).
        day_stop: End day-of-year (1–365).
        scale: Output resolution in metres (default 1000 = MODIS native).
        satellite: ``"both"``, ``"terra"``, or ``"aqua"``.

    Returns:
        Dictionary mapping polygon name → ``xarray.DataArray`` with
        dimensions ``(latitude, longitude)`` and values 0.0–1.0.

    Raises:
        HyPlanRuntimeError: If GEE initialisation fails or download errors.
        HyPlanValueError: If the polygon file is invalid.
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

    # Build merged collection and compute temporal mean server-side
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
        bounds = row["geometry"].bounds  # (minx, miny, maxx, maxy)

        logger.info("Downloading spatial cloud map for %s", name)

        # Use getDownloadURL for GeoTIFF export (avoids sampleRectangle limits)
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

        import requests as _requests
        resp = _requests.get(url, timeout=120)
        if resp.status_code != 200:
            raise HyPlanRuntimeError(
                f"GEE download failed for {name} (HTTP {resp.status_code})"
            )

        # GEE NPY format returns a structured numpy array
        import io
        arr = np.load(io.BytesIO(resp.content), allow_pickle=True)
        # Extract the cloud band (state_1km)
        if arr.dtype.names:
            data = arr["state_1km"].astype(float)
        else:
            data = arr.astype(float)

        # Build lat/lon coordinates from bounds and array shape
        nrows, ncols = data.shape
        minx, miny, maxx, maxy = bounds
        lons = np.linspace(minx, maxx, ncols)
        lats = np.linspace(maxy, miny, nrows)  # top to bottom

        da = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
            name="cloud_fraction_mean",
            attrs={"units": "fraction", "long_name": "Mean cloud fraction"},
        )
        result[name] = da

    return result


def plot_cloud_fraction_spatial(
    spatial_data: "dict[str, object]",
    polygon_file: str | None = None,
    ncols: int = 2,
) -> "plt.Figure":
    """Plot per-pixel cloud fraction maps.

    Args:
        spatial_data: Dictionary mapping polygon name → ``xarray.DataArray``,
            as returned by :func:`fetch_cloud_fraction_spatial`.
        polygon_file: Optional path to polygon file for boundary overlay.
        ncols: Number of subplot columns.

    Returns:
        The matplotlib Figure.
    """
    import numpy as np

    n = len(spatial_data)
    if n == 0:
        raise HyPlanValueError("spatial_data is empty — nothing to plot.")

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    overlay_gdf = None
    if polygon_file is not None:
        overlay_gdf = gpd.read_file(polygon_file)

    for idx, (name, da) in enumerate(spatial_data.items()):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.pcolormesh(
            da.coords["longitude"], da.coords["latitude"], da.values,
            cmap="viridis_r", vmin=0, vmax=1,
        )
        if overlay_gdf is not None:
            row = overlay_gdf[overlay_gdf["Name"] == name]
            if not row.empty:
                row.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Cloud Fraction")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_yearly_cloud_fraction_heatmaps_with_visits(
    cloud_data_df: pd.DataFrame, visit_tracker: Dict[int, Dict[str, list]], rest_days: Dict[int, list],
    cloud_fraction_threshold: float = 0.10, exclude_weekends: bool = False,
    day_start: int = 1, day_stop: int = 365
) -> None:
    """
    Generates heatmaps of cloud fraction for each year, including visit markers and rest day highlights.

    Parameters:
        cloud_data_df (pd.DataFrame): DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'.
        visit_tracker (dict): A dictionary of visit days for each polygon, organized by year.
        rest_days (dict): A dictionary of rest days for each year.
        cloud_fraction_threshold (float): Threshold to classify cloud fraction as clear (white) or cloudy (black).
        exclude_weekends (bool): If True, weekends are highlighted and skipped in the heatmap.
        day_start (int): Start day of the year to include in the heatmap.
        day_stop (int): End day of the year to include in the heatmap.

    Returns:
        None: Displays heatmaps for each year with clear/cloudy days, visit days, and rest day markers.
    """
    required_columns = {'polygon_id', 'year', 'day_of_year', 'cloud_fraction'}
    if not required_columns.issubset(cloud_data_df.columns):
        raise HyPlanValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Define a custom colormap: lightgrey (no data), white (clear), black (cloudy), grey (visited), purple (weekend), orange (rest days)
    cmap = mcolors.ListedColormap(['lightgrey', 'white', 'black', 'grey', 'purple', 'orange'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    unique_years = cloud_data_df['year'].unique()
    for year in sorted(unique_years):
        year_data = cloud_data_df[(cloud_data_df['year'] == year) &
                                  (cloud_data_df['day_of_year'] >= day_start) &
                                  (cloud_data_df['day_of_year'] <= day_stop)]
        heatmap_data = year_data.pivot(index='polygon_id', columns='day_of_year', values='cloud_fraction')
        heatmap_data = heatmap_data.reindex(columns=range(day_start, day_stop + 1), fill_value=float('nan'))

        binary_data = (heatmap_data > cloud_fraction_threshold).astype(int)
        binary_data[heatmap_data.isna()] = -1
        status_data = binary_data.copy()

        stars_x = []
        stars_y = []
        rest_days_set = set(rest_days.get(year, [])) if rest_days else set()

        for i, polygon_id in enumerate(status_data.index):
            if polygon_id in visit_tracker.get(year, {}):
                visit_days_list = sorted(visit_tracker[year][polygon_id])
                for visit_day in visit_days_list:
                    if day_start <= visit_day <= day_stop:
                        stars_x.append(visit_day - day_start + 0.5)
                        stars_y.append(i + 0.5)

                        for day in range(visit_day + 1, day_stop + 1):
                            if exclude_weekends:
                                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                                if weekday < 5:
                                    status_data.loc[polygon_id, day] = 2
                            else:
                                status_data.loc[polygon_id, day] = 2

        for rest_day in rest_days_set:
            if day_start <= rest_day <= day_stop:
                status_data.iloc[:, rest_day - day_start] = 4

        if exclude_weekends:
            for day in range(day_start, day_stop + 1):
                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                if weekday >= 5:
                    status_data.loc[:, day] = 3

        try:
            import seaborn as sns
        except ImportError:
            raise HyPlanRuntimeError(
                "seaborn is required for cloud heatmaps. "
                "Install it with: pip install hyplan[clouds]"
            )
        plt.figure(figsize=(16, 8))
        sns.heatmap(status_data, cmap=cmap, norm=norm, cbar=False,
                    linewidths=0.5, linecolor='gray', square=True)
        plt.scatter(stars_x, stars_y, color='red', marker='*', s=150, label='Visit Day')
        plt.title(f'Cloud Fraction Heatmap with Visits for Year {year}')
        plt.xlabel('Day of Year')
        plt.ylabel('Polygon ID')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
