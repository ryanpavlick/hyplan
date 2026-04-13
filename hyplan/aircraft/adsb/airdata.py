"""Air-relative reconstruction from ADS-B ground track and wind field.

Subtracts wind from ground velocity to obtain true airspeed (TAS) and
air-relative heading.  This is HyPlan's key differentiator — using its
own :class:`~hyplan.winds.WindField` objects (MERRA-2, GFS, etc.) for
physically accurate wind correction rather than treating groundspeed as
an approximation of TAS.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from ...units import ureg

if TYPE_CHECKING:
    from ...winds import WindField

logger = logging.getLogger(__name__)

# Knots ↔ m/s conversion factor
_KT_TO_MPS = 0.514444


def reconstruct_airdata(
    phased_df: pd.DataFrame,
    wind_field: WindField,
) -> pd.DataFrame:
    """Reconstruct air-relative velocities from ground track and wind.

    For each observation, queries the wind vector at ``(lat, lon,
    altitude, time)`` and applies the standard wind triangle to compute
    TAS and air-relative heading.

    Args:
        phased_df: DataFrame from :func:`~.phases.label_phases` with
            columns ``timestamp``, ``latitude``, ``longitude``,
            ``altitude``, ``groundspeed``, ``track``, ``vertical_rate``,
            ``phase``.
        wind_field: Any :class:`~hyplan.winds.WindField` instance.

    Returns:
        Copy of *phased_df* with added columns:

        - ``wind_u_mps``: Eastward wind component (m/s).
        - ``wind_v_mps``: Northward wind component (m/s).
        - ``tas_kt``: True airspeed (knots).
        - ``heading_true_deg``: Air-relative heading (degrees true).
        - ``wind_speed_kt``: Wind speed magnitude (knots).
        - ``wind_from_deg``: Wind direction (meteorological, degrees).
    """
    from ...winds import StillAirField

    df = phased_df.copy()
    n = len(df)

    # Fast path: still air — TAS equals groundspeed
    if isinstance(wind_field, StillAirField):
        df["wind_u_mps"] = 0.0
        df["wind_v_mps"] = 0.0
        df["tas_kt"] = df["groundspeed"]
        df["heading_true_deg"] = df["track"]
        df["wind_speed_kt"] = 0.0
        df["wind_from_deg"] = 0.0
        return df

    # --- Vectorised ground velocity ---
    gs_mps = df["groundspeed"].values * _KT_TO_MPS
    track_rad = np.radians(df["track"].values)
    v_gnd_east = gs_mps * np.sin(track_rad)
    v_gnd_north = gs_mps * np.cos(track_rad)

    # --- Wind lookup (point-by-point) ---
    wind_u = np.zeros(n)
    wind_v = np.zeros(n)

    lats = df["latitude"].values
    lons = df["longitude"].values
    alts = df["altitude"].values
    timestamps = df["timestamp"].values

    for i in range(n):
        alt_qty = float(alts[i]) * ureg.feet
        ts = _to_datetime(timestamps[i])
        u, v = wind_field.wind_at(lats[i], lons[i], alt_qty, ts)
        wind_u[i] = u.m_as(ureg.meter / ureg.second)
        wind_v[i] = v.m_as(ureg.meter / ureg.second)

    # --- Wind triangle ---
    v_air_east = v_gnd_east - wind_u
    v_air_north = v_gnd_north - wind_v
    tas_mps = np.sqrt(v_air_east ** 2 + v_air_north ** 2)
    heading_rad = np.arctan2(v_air_east, v_air_north)
    heading_deg = np.degrees(heading_rad) % 360.0

    # Wind speed and direction (meteorological)
    ws_mps = np.sqrt(wind_u ** 2 + wind_v ** 2)
    wind_from_rad = np.arctan2(-wind_u, -wind_v)
    wind_from_deg = np.degrees(wind_from_rad) % 360.0

    df["wind_u_mps"] = wind_u
    df["wind_v_mps"] = wind_v
    df["tas_kt"] = tas_mps / _KT_TO_MPS
    df["heading_true_deg"] = heading_deg
    df["wind_speed_kt"] = ws_mps / _KT_TO_MPS
    df["wind_from_deg"] = wind_from_deg

    logger.info(
        "Reconstructed airdata: %d points, mean wind %.1f kt",
        n,
        float(np.mean(ws_mps / _KT_TO_MPS)),
    )
    return df


def resolve_wind_field(
    wind_source: Union[str, WindField, None],
    phased_dfs: List[pd.DataFrame],
    margin_deg: float = 2.0,
    margin_hours: float = 2.0,
) -> WindField:
    """Create a :class:`WindField` for a set of trajectory DataFrames.

    Mirrors the :func:`~hyplan.winds.wind_field_from_plan` pattern but
    computes the bounding box from trajectory data rather than a flight
    sequence.

    Args:
        wind_source: ``"merra2"``, ``"gfs"``, ``"gmao"``,
            ``"still_air"``, a :class:`WindField` instance, or *None*
            (defaults to ``"still_air"``).
        phased_dfs: List of DataFrames (from :func:`label_phases`) used
            to compute the spatial/temporal bounding box for gridded
            wind fields.
        margin_deg: Spatial margin in degrees around the bounding box.
        margin_hours: Temporal margin in hours around the time range.

    Returns:
        A :class:`WindField` instance with data pre-fetched (for gridded
        sources).
    """
    from ...winds import StillAirField

    if wind_source is None or wind_source == "still_air":
        return StillAirField()

    if not isinstance(wind_source, str):
        # Already a WindField instance
        return wind_source

    # Compute bounding box from trajectory data
    all_lats, all_lons, all_alts, all_times = [], [], [], []
    for df in phased_dfs:
        all_lats.append(df["latitude"].values)
        all_lons.append(df["longitude"].values)
        all_alts.append(df["altitude"].values)
        all_times.append(df["timestamp"].values)

    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    alts = np.concatenate(all_alts)
    times = np.concatenate(all_times)

    lat_min, lat_max = float(lats.min()) - margin_deg, float(lats.max()) + margin_deg
    lon_min, lon_max = float(lons.min()) - margin_deg, float(lons.max()) + margin_deg
    alt_max_ft = float(alts.max())

    time_min = pd.Timestamp(times.min()).to_pydatetime()
    time_max = pd.Timestamp(times.max()).to_pydatetime()
    time_min -= datetime.timedelta(hours=margin_hours)
    time_max += datetime.timedelta(hours=margin_hours)

    kwargs = dict(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        time_start=time_min,
        time_end=time_max,
    )

    if wind_source == "merra2":
        from ...winds import MERRA2WindField

        return MERRA2WindField(**kwargs)
    elif wind_source == "gmao":
        from ...winds import GMAOWindField

        return GMAOWindField(**kwargs)
    elif wind_source == "gfs":
        from ...winds import GFSWindField

        return GFSWindField(**kwargs)
    else:
        raise ValueError(
            f"Unknown wind source: {wind_source!r}. "
            f"Use 'still_air', 'merra2', 'gmao', 'gfs', or a WindField instance."
        )


def _to_datetime(ts) -> datetime.datetime:
    """Convert a numpy/pandas timestamp to a stdlib datetime."""
    if isinstance(ts, datetime.datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()
