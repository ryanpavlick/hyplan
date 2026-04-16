"""AppEEARS API client for MODIS vegetation index time series.

Submits point sample requests to NASA's AppEEARS service, polls for
completion, and returns results as a pandas DataFrame.  Much faster
than downloading full granules since extraction happens server-side.

Requires NASA Earthdata credentials (EARTHDATA_USERNAME + EARTHDATA_PASSWORD
environment variables, or ~/.netrc).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ..exceptions import HyPlanRuntimeError

logger = logging.getLogger(__name__)

_BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"

# Product/layer mapping
_APPEEARS_LAYERS = {
    "ndvi": {"product": "MOD13A1.061", "layer": "_500m_16_days_NDVI"},
    "evi": {"product": "MOD13A1.061", "layer": "_500m_16_days_EVI"},
    "lai": {"product": "MOD15A2H.061", "layer": "Lai_500m"},
    "fpar": {"product": "MOD15A2H.061", "layer": "Fpar_500m"},
    "ndvi_aqua": {"product": "MYD13A1.061", "layer": "_500m_16_days_NDVI"},
    "evi_aqua": {"product": "MYD13A1.061", "layer": "_500m_16_days_EVI"},
}


def _get_credentials() -> tuple[str, str]:
    """Get EarthData username and password from environment or .netrc."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password

    # Try .netrc
    import netrc
    try:
        info = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if info:
            return info[0], info[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    raise HyPlanRuntimeError(
        "NASA Earthdata credentials required for AppEEARS. Set "
        "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables, "
        "or configure ~/.netrc for urs.earthdata.nasa.gov."
    )


def _login() -> str:
    """Authenticate with AppEEARS and return a Bearer token."""
    username, password = _get_credentials()
    resp = requests.post(
        f"{_BASE_URL}/login",
        auth=(username, password),
        headers={"Content-Length": "0"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise HyPlanRuntimeError(
            f"AppEEARS login failed (HTTP {resp.status_code}): {resp.text}"
        )
    return resp.json()["token"]


def fetch_appeears_timeseries(
    coordinates: list[dict],
    product: str = "ndvi",
    year_start: int = 2010,
    year_stop: int = 2022,
    poll_interval: int = 15,
    max_wait: int = 600,
) -> pd.DataFrame:
    """Fetch vegetation index time series from AppEEARS.

    Parameters
    ----------
    coordinates : list of dict
        Each dict has ``id``, ``latitude``, ``longitude``.
    product : str
        One of ``"ndvi"``, ``"evi"``, ``"lai"``, ``"fpar"``.
    year_start, year_stop : int
        Date range (inclusive).
    poll_interval : int
        Seconds between status checks.
    max_wait : int
        Maximum seconds to wait for task completion.

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_id``, ``date``, ``year``, ``day_of_year``, ``value``.
    """
    if product not in _APPEEARS_LAYERS:
        raise HyPlanRuntimeError(
            f"Unknown product '{product}'. Choose from: {list(_APPEEARS_LAYERS)}"
        )

    layer_info = _APPEEARS_LAYERS[product]
    token = _login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Submit task
    task_payload = {
        "task_type": "point",
        "task_name": f"hyplan_{product}_{year_start}_{year_stop}",
        "params": {
            "dates": [{
                "startDate": f"01-01-{year_start}",
                "endDate": f"12-31-{year_stop}",
                "recurring": False,
            }],
            "layers": [{
                "product": layer_info["product"],
                "layer": layer_info["layer"],
            }],
            "coordinates": [
                {
                    "id": c["id"],
                    "latitude": c["latitude"],
                    "longitude": c["longitude"],
                    "category": c.get("id", "site"),
                }
                for c in coordinates
            ],
        },
    }

    logger.info("Submitting AppEEARS task for %s (%d-%d)...", product, year_start, year_stop)
    resp = requests.post(f"{_BASE_URL}/task", json=task_payload, headers=headers, timeout=30)
    if resp.status_code not in (200, 202):
        raise HyPlanRuntimeError(
            f"AppEEARS task submission failed (HTTP {resp.status_code}): {resp.text}"
        )
    task_id = resp.json()["task_id"]
    logger.info("Task submitted: %s", task_id)

    # Poll for completion
    elapsed = 0
    while elapsed < max_wait:
        resp = requests.get(f"{_BASE_URL}/status/{task_id}", headers=headers, timeout=30)
        status = resp.json().get("status", "unknown")
        logger.info("Task %s status: %s (%.0fs elapsed)", task_id, status, elapsed)

        if status == "done":
            break
        if status in ("error", "failed"):
            raise HyPlanRuntimeError(f"AppEEARS task failed: {resp.json()}")

        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise HyPlanRuntimeError(
            f"AppEEARS task {task_id} did not complete within {max_wait}s. "
            f"Check status at https://appeears.earthdatacloud.nasa.gov/task/{task_id}"
        )

    # Download results
    resp = requests.get(f"{_BASE_URL}/bundle/{task_id}", headers=headers, timeout=30)
    bundle = resp.json()

    # Find the CSV results file
    csv_files = [f for f in bundle.get("files", []) if f["file_name"].endswith(".csv")]
    if not csv_files:
        raise HyPlanRuntimeError(
            f"No CSV results found in AppEEARS bundle for task {task_id}"
        )

    # Download the CSV
    csv_file = csv_files[0]
    file_id = csv_file["file_id"]
    resp = requests.get(
        f"{_BASE_URL}/bundle/{task_id}/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )

    # Parse CSV
    from io import StringIO
    raw_df = pd.read_csv(StringIO(resp.text))
    logger.info("AppEEARS CSV: %d rows, columns: %s", len(raw_df), list(raw_df.columns)[:5])

    # Find the value column — AppEEARS prefixes with product ID
    # e.g. "MOD13A1_061__500m_16_days_NDVI"
    value_col = None
    for col in raw_df.columns:
        if layer_info["layer"] in col:
            value_col = col
            break
    if value_col is None:
        raise HyPlanRuntimeError(
            f"Could not find layer '{layer_info['layer']}' in AppEEARS columns: "
            f"{list(raw_df.columns)}"
        )

    # AppEEARS returns pre-scaled values (NDVI as 0.0-1.0, not raw int16)
    raw_df["_date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["_date", value_col])

    # Filter invalid values
    raw_df = raw_df[(raw_df[value_col] >= -0.2) & (raw_df[value_col] <= 1.5)]

    df = pd.DataFrame({
        "polygon_id": raw_df["ID"],
        "date": raw_df["_date"],
        "year": raw_df["_date"].dt.year,
        "day_of_year": raw_df["_date"].dt.dayofyear,
        "value": raw_df[value_col].astype(float),
    })

    logger.info("AppEEARS returned %d valid observations", len(df))
    return df.sort_values(["polygon_id", "year", "day_of_year"]).reset_index(drop=True)
