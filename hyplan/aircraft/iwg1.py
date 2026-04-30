"""NASA IWG1 in-situ flight log loader.

IWG1 (Inter-agency Working Group 1) is NASA's standard interchange
format for airborne in-situ aircraft data: timestamp, position,
multiple altitude sources, true airspeed, measured winds, attitude,
and atmospheric state at ~5 s cadence.

This module reads a per-sortie ``.txt`` IWG1 file and returns a
normalized :class:`pandas.DataFrame` with HyPlan-conventional column
names and units (timestamps tz-naive UTC, altitudes in ft, speeds in
kt, vertical rate in fpm, etc.) — directly usable by the same
phase-labeling and schedule-fitting pipeline that the ADS-B path uses.

Compared with ADS-B, IWG1 includes directly measured TAS, wind, and
attitude, so no reconstruction step is needed and the calibration
becomes a more direct read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..exceptions import HyPlanValueError

# Conversion factors (exact).
_M_PER_FT = 0.3048
_KT_PER_MPS = 1.9438444924406046
_FPM_PER_MPS = 60.0 / _M_PER_FT  # 196.85...

# IWG1 columns we care about, mapped to (output_name, unit_converter).
# Source units: ``GPS MSL Altitude``/``WGS84 Altitude`` in m, ``Pressure
# Altitude`` and ``Radar Altitude`` in ft, speeds in m/s, angles in deg,
# pressures in hPa, temperatures in °C.  Lat/lon are deg.
_COLUMN_MAP = {
    "Latitude": ("latitude", lambda x: x),
    "Longitude": ("longitude", lambda x: x),
    "Pressure Altitude": ("altitude", lambda x: x),  # ft, kept as-is
    "GPS MSL Altitude": ("altitude_gps_ft", lambda x: x / _M_PER_FT),
    "Radar Altitude": ("altitude_radar_ft", lambda x: x),
    "Ground Speed": ("groundspeed", lambda x: x * _KT_PER_MPS),
    "True Airspeed": ("tas_kt", lambda x: x * _KT_PER_MPS),
    "Indicated Airspeed": ("ias_kt", lambda x: x * _KT_PER_MPS),
    "Mach Number": ("mach", lambda x: x),
    "True Heading": ("true_heading", lambda x: x % 360.0),
    "Track": ("track", lambda x: x % 360.0),
    "Pitch": ("pitch_deg", lambda x: x),
    "Roll": ("roll_deg", lambda x: x),
    "Angle of Attack": ("aoa_deg", lambda x: x),
    "Wind Speed": ("wind_speed_kt", lambda x: x * _KT_PER_MPS),
    "Wind Direction": ("wind_direction_deg", lambda x: x % 360.0),
    "Ambient Temp": ("ambient_temp_c", lambda x: x),
    "Static Press": ("static_pressure_hpa", lambda x: x),
}


def load_iwg1(path: Union[str, Path]) -> pd.DataFrame:
    """Load one IWG1 ``.txt`` file into a normalized DataFrame.

    Args:
        path: Local path to an IWG1 CSV file.  Header row names the
            columns (lat/lon/altitude/speed/etc.); each subsequent line
            starts with the ``IWG1`` literal followed by a timestamp
            and the data record.

    Returns:
        DataFrame sorted by timestamp with these columns:

        * ``timestamp`` — tz-naive UTC ``pd.Timestamp``.
        * ``latitude``, ``longitude`` — degrees.
        * ``altitude`` — feet, MSL pressure altitude (the canonical
          HyPlan altitude column for downstream phase labeling).
        * ``altitude_gps_ft`` — feet, GPS MSL altitude.
        * ``altitude_radar_ft`` — feet, radar altitude (sparse).
        * ``groundspeed``, ``tas_kt``, ``ias_kt`` — knots.
        * ``mach`` — dimensionless.
        * ``vertical_rate`` — feet/min, derived from altitude
          time-derivative (the IWG1 ``Vertical Velocity`` column is
          always empty in observed files).
        * ``true_heading``, ``track`` — degrees, [0, 360).
        * ``pitch_deg``, ``roll_deg``, ``aoa_deg`` — degrees.
        * ``wind_speed_kt``, ``wind_direction_deg`` — knots, degrees
          (meteorological "from" convention).
        * ``ambient_temp_c`` — °C.
        * ``static_pressure_hpa`` — hPa.

    Raises:
        HyPlanValueError: If the file is empty or missing the
            ``TimeStamp`` column.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"IWG1 file not found: {p}")

    raw = pd.read_csv(p)
    raw.columns = raw.columns.str.strip()
    if raw.empty:
        raise HyPlanValueError(f"IWG1 file is empty: {p}")
    if "TimeStamp" not in raw.columns:
        raise HyPlanValueError(
            f"IWG1 file {p} missing required 'TimeStamp' column."
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(raw["TimeStamp"], utc=True).dt.tz_localize(None)

    for src, (dst, conv) in _COLUMN_MAP.items():
        if src in raw.columns:
            out[dst] = conv(pd.to_numeric(raw[src], errors="coerce"))
        else:
            out[dst] = np.nan

    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Derive vertical_rate (fpm) from the altitude time-derivative —
    # the IWG1 Vertical Velocity column is always empty in observed
    # files. Use a centered finite difference smoothed by a 3-sample
    # rolling median to suppress per-fix altitude jitter.
    if len(out) >= 2:
        alt_ft = out["altitude"].to_numpy(dtype=float)
        t_s = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            d_alt = np.gradient(alt_ft, t_s)  # ft/s
        vs_fpm = d_alt * 60.0
        # Rolling median to suppress 1-fix altitude jitter (IWG1 pressure
        # altitude reports to 2.5 ft resolution, which gives ~30 fpm noise
        # at the 5 s cadence).
        out["vertical_rate"] = (
            pd.Series(vs_fpm).rolling(window=3, center=True, min_periods=1).median().to_numpy()
        )
    else:
        out["vertical_rate"] = np.nan

    return out


def trim_ground_taxi(
    df: pd.DataFrame,
    *,
    groundspeed_threshold_kt: float = 25.0,
) -> pd.DataFrame:
    """Trim pre-takeoff / post-landing taxi from a sortie DataFrame.

    Defines the **airborne window** as the contiguous range from the
    first to the last fix where ``groundspeed > groundspeed_threshold_kt``.
    Anything slower is treated as taxi or standing (including the
    aircraft parked at an elevated ramp, where altitude alone would be
    misleading — many airports have the ramp tens of feet above the
    runway threshold).

    Default 25 kt cleanly separates ER-2 taxi (typically ≤ 15 kt
    observed) from takeoff and landing rollouts (≥ 30 kt) while still
    capturing the very slow end of rollout and any unusual taxi
    excursions.

    Args:
        df: Output of :func:`load_iwg1`.
        groundspeed_threshold_kt: Ground speed above which a fix is
            considered airborne.

    Returns:
        Sliced DataFrame from first-airborne to last-airborne fix
        (inclusive). The index is reset.

    If the trace contains no airborne fixes (e.g., a ground-test
    record or a missing ``groundspeed`` column), the returned DataFrame
    is empty.
    """
    if df.empty or "groundspeed" not in df.columns:
        return df.iloc[0:0].reset_index(drop=True)

    gs = df["groundspeed"]
    airborne = gs > groundspeed_threshold_kt
    if not airborne.any():
        return df.iloc[0:0].reset_index(drop=True)
    first = int(airborne.values.argmax())
    last = int(len(airborne) - 1 - airborne.values[::-1].argmax())
    return df.iloc[first : last + 1].reset_index(drop=True)
