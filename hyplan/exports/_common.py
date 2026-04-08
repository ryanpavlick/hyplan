"""Shared helpers for hyplan.exports.

Waypoint extraction, name generation, and small numeric / solar helpers
used by every export format. Kept private (single underscore prefix)
because the API surface lives in the per-format modules.
"""

import datetime
import math
import warnings
from typing import List

import geopandas as gpd
import pandas as pd

from ..units import ureg

__all__ = [
    "extract_waypoints",
    "generate_wp_names",
    "_safe_float",
    "_utc_fraction",
    "_compute_sza",
    "_compute_solar_azimuth",
]


def extract_waypoints(plan: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract a waypoint table from a flight plan GeoDataFrame.

    Each row in the flight plan represents a segment.  This function pulls
    out one waypoint per segment boundary (start of each segment + end of
    the last segment).

    Returns:
        DataFrame with columns: ``wp``, ``lat``, ``lon``, ``alt_m``,
        ``alt_kft``, ``heading``, ``speed_mps``, ``speed_kt``,
        ``dist_km``, ``dist_nm``, ``cum_dist_km``, ``cum_dist_nm``,
        ``leg_time_min``, ``cum_time_min``, ``segment_type``,
        ``segment_name``.
    """
    rows = []
    cum_dist_km = 0.0
    cum_dist_nm = 0.0
    cum_time = 0.0

    for i, row in plan.iterrows():
        alt_ft = _safe_float(row.get("start_altitude"), default=0.0, field="start_altitude")
        alt_m = (alt_ft * ureg.foot).m_as(ureg.meter)
        alt_kft = alt_ft / 1000.0

        dist_nm = row.get("distance", 0.0) or 0.0
        dist_km = (dist_nm * ureg.nautical_mile).m_as(ureg.kilometer)
        leg_time = row.get("time_to_segment", 0.0) or 0.0

        # Derive speed from distance and time
        if leg_time > 0 and dist_nm > 0:
            speed_kt = dist_nm / (leg_time / 60.0)
            speed_mps = (speed_kt * ureg.knot).m_as("m/s")
        else:
            speed_kt = 0.0
            speed_mps = 0.0

        rows.append({
            "wp": i,
            "lat": row["start_lat"],
            "lon": row["start_lon"],
            "alt_m": alt_m,
            "alt_ft": alt_ft,
            "alt_kft": alt_kft,
            "heading": _safe_float(row.get("start_heading")),
            "speed_mps": speed_mps,
            "speed_kt": speed_kt,
            "dist_km": dist_km,
            "dist_nm": dist_nm,
            "cum_dist_km": cum_dist_km,
            "cum_dist_nm": cum_dist_nm,
            "leg_time_min": leg_time,
            "cum_time_min": cum_time,
            "segment_type": row.get("segment_type", ""),
            "segment_name": row.get("segment_name", ""),
        })

        cum_dist_km += dist_km
        cum_dist_nm += dist_nm
        cum_time += leg_time

    # Append the final waypoint (end of last segment)
    if len(plan) > 0:
        last = plan.iloc[-1]
        alt_ft = _safe_float(last.get("end_altitude"), default=0.0, field="end_altitude")
        rows.append({
            "wp": len(plan),
            "lat": last["end_lat"],
            "lon": last["end_lon"],
            "alt_m": alt_ft * 0.3048,
            "alt_ft": alt_ft,
            "alt_kft": alt_ft / 1000.0,
            "heading": _safe_float(last.get("end_heading")),
            "speed_mps": 0.0,
            "speed_kt": 0.0,
            "dist_km": 0.0,
            "dist_nm": 0.0,
            "cum_dist_km": cum_dist_km,
            "cum_dist_nm": cum_dist_nm,
            "leg_time_min": 0.0,
            "cum_time_min": cum_time,
            "segment_type": "",
            "segment_name": last.get("segment_name", ""),
        })

    return pd.DataFrame(rows)


def generate_wp_names(n: int, prefix: str = "H",
                      date: datetime.date = None) -> List[str]:
    """Generate MovingLines-compatible 5-char waypoint names.

    Pattern: ``{prefix}{day:02d}{wp:02d}`` — e.g. ``'H2101'`` for prefix
    'H', day 21, waypoint 1.

    Args:
        n: Number of names to generate.
        prefix: Single-character prefix (default 'H' for hyplan).
        date: Date to extract day-of-month (default today).

    Returns:
        List of 5-character waypoint name strings.
    """
    if date is None:
        date = datetime.date.today()
    day = date.day
    return [f"{prefix[0]}{day:02d}{i:02d}" for i in range(n)]


def _safe_float(val, default: float = 0.0, field: str = "") -> float:
    """Convert a value to float, replacing None/NaN with *default*.

    Emits a warning when substituting a default so that missing data
    (especially altitude) does not silently produce zeros in pilot-facing
    exports.
    """
    if val is None:
        if field:
            warnings.warn(
                f"Missing value for '{field}' replaced with {default}",
                stacklevel=3,
            )
        return default
    try:
        f = float(val)
        if math.isnan(f):
            if field:
                warnings.warn(
                    f"NaN value for '{field}' replaced with {default}",
                    stacklevel=3,
                )
            return default
        return f
    except (TypeError, ValueError):
        return default


def _utc_fraction(minutes_from_midnight: float) -> float:
    """Convert minutes from midnight to Excel time fraction (0-1)."""
    return minutes_from_midnight / 1440.0


def _compute_sza(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute solar zenith angle, returning -9999 if sun module unavailable."""
    try:
        from ..sun import sunpos
        ts = pd.DatetimeIndex([dt], tz='UTC')
        _, zenith, *_ = sunpos(ts, lat, lon, elevation=0)
        return float(zenith[0])
    except Exception:
        return -9999.0


def _compute_solar_azimuth(lat: float, lon: float,
                           dt: datetime.datetime) -> float:
    """Compute solar azimuth, returning -9999 if unavailable."""
    try:
        from ..sun import sunpos
        ts = pd.DatetimeIndex([dt], tz='UTC')
        azimuth, *_ = sunpos(ts, lat, lon, elevation=0)
        return float(azimuth[0])
    except Exception:
        return -9999.0
