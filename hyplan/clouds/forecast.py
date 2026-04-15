"""Cloud cover forecast fetchers (Open-Meteo Forecast API)."""

from __future__ import annotations

import logging
from datetime import datetime

import geopandas as gpd
import pandas as pd

from ..exceptions import HyPlanRuntimeError, HyPlanValueError

logger = logging.getLogger(__name__)

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
            forecast_days: Number of days to forecast (1-16).
            hourly: If ``True``, return hourly data with layer breakdown
                (low / mid / high cloud cover).  Otherwise return daily means.
            models: Optional list of NWP model identifiers to use (e.g.
                ``["ecmwf_ifs025"]``).  ``None`` uses Open-Meteo's automatic
                best-match selection.

        Returns:
            DataFrame with columns:

            * **daily** (default): ``polygon_id``, ``date``,
              ``cloud_fraction`` (0.0-1.0).
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
        forecast_days: Number of days to forecast (1-16).
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
