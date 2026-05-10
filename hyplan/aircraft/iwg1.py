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
from typing import Any, Callable

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
_COLUMN_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
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


def load_iwg1(path: str | Path) -> pd.DataFrame:
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

    # NaN-out implausible GPS positions.  Some IWG1 deliveries contain
    # isolated rogue rows with lat / lon noise spikes — sometimes
    # values clearly outside Earth (lat=-214) and sometimes "valid"
    # values that are nevertheless far from the rest of the trace
    # (e.g., a single row jumping to the South Pacific in the middle
    # of a US east-coast sortie).  These are typically 1-2 fixes per
    # sortie out of ~18k and don't move bin-median calibrations, but
    # they break ground-tracks plots and could distort haversine
    # distance computations.  Two filters:
    #
    #   (a) absolute bounds: |lat| > 90 or |lon| > 180 are physically
    #       impossible.
    #   (b) isolated spikes: a row whose position differs by > 1
    #       degree from both immediate neighbors.  At 1-Hz cadence
    #       this corresponds to a >60 nmi jump in one second, which
    #       no airframe can do.
    #
    # In both cases the lat / lon columns are NaN'd; the rest of the
    # row's air-data (TAS / altitude / etc.) is preserved since it
    # comes from independent sensors.
    bad_pos = (out["latitude"].abs() > 90) | (out["longitude"].abs() > 180)
    if len(out) >= 3:
        lat = out["latitude"]
        lon = out["longitude"]
        JUMP_DEG = 1.0
        lat_jump = (lat.diff().abs() > JUMP_DEG) & (lat.diff(-1).abs() > JUMP_DEG)
        lon_jump = (lon.diff().abs() > JUMP_DEG) & (lon.diff(-1).abs() > JUMP_DEG)
        bad_pos = bad_pos | lat_jump.fillna(False) | lon_jump.fillna(False)
    if bad_pos.any():
        out.loc[bad_pos, ["latitude", "longitude"]] = np.nan

    # NaN-out implausible altitude and TAS values.  Some IWG1
    # deliveries use sentinel values (e.g., Pressure Altitude=-34055,
    # TAS=-1338.9) for "no data" rows; 2013-era N806NA ER-2 sorties
    # also have short runs of alt=0 / TAS=4060 kt mid-cruise (1-3 fixes
    # at a time).  Two filters per column:
    #
    #   (a) absolute bounds: alt < -2000 ft or > 100000 ft is
    #       physically impossible for any HyPlan aircraft; TAS < 0 or
    #       > 2000 kt is impossible.
    #   (b) rolling-median outlier: an 11-row centered median is the
    #       baseline; any row deviating by > 1000 ft (alt) or > 100
    #       kt (TAS) from it is NaN'd.  Catches 1-3 row drop-out
    #       runs that the simple "differs from both neighbors" test
    #       misses.
    bad_alt = (out["altitude"] < -2000) | (out["altitude"] > 100000)
    if len(out) >= 11:
        alt = out["altitude"]
        alt_med = alt.rolling(11, center=True, min_periods=5).median()
        bad_alt = bad_alt | ((alt - alt_med).abs() > 1000.0)
    bad_alt = bad_alt.fillna(False)
    if bad_alt.any():
        out.loc[bad_alt, "altitude"] = np.nan

    bad_tas = (out["tas_kt"] < 0) | (out["tas_kt"] > 2000)
    if len(out) >= 11:
        tas = out["tas_kt"]
        tas_med = tas.rolling(11, center=True, min_periods=5).median()
        bad_tas = bad_tas | ((tas - tas_med).abs() > 100.0)
    bad_tas = bad_tas.fillna(False)
    if bad_tas.any():
        out.loc[bad_tas, "tas_kt"] = np.nan

    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Drop near-duplicate fixes (consecutive fixes within 100 ms) before
    # computing vertical_rate.  Some IWG1 files contain duplicated
    # records ~12 ms apart; np.gradient over those tiny dt values
    # produces wildly spurious VS spikes (>10000 fpm) that sneak past
    # the 3-sample rolling median and contaminate phase classification.
    if len(out) >= 2:
        dt_ms = out["timestamp"].diff().dt.total_seconds() * 1000.0
        keep = (dt_ms.isna()) | (dt_ms >= 100.0)
        out = out[keep].reset_index(drop=True)

    # Derive vertical_rate (fpm) using a long-baseline finite
    # difference: VS at fix i = (alt[t + 90s] - alt[t - 90s]) / 180s.
    # The IWG1 Vertical Velocity column is always empty in observed
    # files, so we have to derive it.  A 3-minute baseline averages
    # out the autopilot ±100 ft tracking oscillations at cruise
    # (period ~30-60 sec) that a per-fix gradient captures as
    # transient ±500 fpm spikes — those misclassify sustained
    # cruise as climb/descent.  A longer window (5 min) recovers
    # more cruise time on sorties with genuine step-cruise drift
    # but at the cost of smearing TOC/TOD transitions by 90 vs
    # 150 sec — 3 min is the sweet spot.
    if len(out) >= 2:
        alt_ft = out["altitude"].to_numpy(dtype=float)
        t_s = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        half = 90.0  # +/- 90 sec baseline -> 180-sec (3 min) window
        lo_idx = np.searchsorted(t_s, t_s - half, side="left")
        hi_idx = np.searchsorted(t_s, t_s + half, side="right") - 1
        # Clip to valid range so vectorized indexing works.
        lo_idx = np.clip(lo_idx, 0, len(t_s) - 1)
        hi_idx = np.clip(hi_idx, 0, len(t_s) - 1)
        dt = t_s[hi_idx] - t_s[lo_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            vs_fpm = np.where(
                dt > 0,
                (alt_ft[hi_idx] - alt_ft[lo_idx]) / dt * 60.0,
                np.nan,
            )
        out["vertical_rate"] = vs_fpm
    else:
        out["vertical_rate"] = np.nan

    return out


def split_iwg1_alltracks(
    src: str | Path,
    dest_dir: str | Path,
    *,
    tail_label: str,
    gap_threshold_hr: float = 6.0,
) -> list[Path]:
    """Split a concatenated multi-sortie IWG1 CSV into per-sortie files.

    The IWG1 "all-tracks" delivery format is a single CSV with one
    ``HEADER,...`` row followed by ``IWG1,timestamp,...`` data rows
    spanning many sorties in chronological order.  This function
    detects sortie boundaries by gaps in the timestamp sequence and
    writes one file per sortie under ``dest_dir``, each prefixed with
    the original HEADER row so the result is directly loadable by
    :func:`load_iwg1`.

    Args:
        src: Path to the all-tracks CSV.
        dest_dir: Output directory; created if missing.
        tail_label: Filename prefix for per-sortie files
            (``{tail_label}_{YYYY-MM-DD}.txt``).  The date suffix is the
            takeoff date of the sortie in UTC.
        gap_threshold_hr: Minimum inter-row time gap, in hours, that
            counts as a sortie boundary.  Default 6 hr cleanly separates
            ferry / repositioning legs flown the same day from genuine
            next-day departures while ignoring brief log dropouts
            within a single sortie.

    Returns:
        List of written paths, in chronological order.

    Existing files in ``dest_dir`` with matching names are overwritten.
    """
    src = Path(src)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    text = src.read_text()
    if not text:
        raise HyPlanValueError(f"IWG1 all-tracks file is empty: {src}")

    # Some all-tracks deliveries contain records glued together without
    # a separating newline ("...,XIWG1,2025-07-21T..."), so we can't
    # rely on f.readlines() alone — restore line breaks before each
    # data-record marker first.
    import re
    text = re.sub(r"(?<!\n)IWG1,", "\nIWG1,", text)
    lines = text.splitlines(keepends=True)

    # Some deliveries omit the HEADER row entirely and start straight
    # with IWG1 data rows.  Detect and inject the canonical IWG1 header
    # so downstream load_iwg1 can read column names.
    if lines[0].lstrip().startswith("HEADER,"):
        header = lines[0]
        if not header.endswith("\n"):
            header = header + "\n"
        data_start = 1
    else:
        header = (
            "HEADER,TimeStamp,Latitude,Longitude,GPS MSL Altitude,"
            "WGS84 Altitude,Pressure Altitude,Radar Altitude,Ground Speed,"
            "True Airspeed,Indicated Airspeed,Mach Number,Vertical Velocity,"
            "True Heading,Track,Drift,Pitch,Roll,Side Slip,Angle of Attack,"
            "Ambient Temp,Dew Point,Total Air Temp,Static Press,"
            "Dynamic Press,Cabin Press,Wind Speed,Wind Direction,"
            "Vertical Wind Speed,Solar Zenith Angle,Sun Elevation Aircraft,"
            "Sun Azimuth Ground,Sun Azimuth Aircraft\n"
        )
        data_start = 0
    data_lines = [ln for ln in lines[data_start:] if ln.strip()]
    # Ensure each data line ends with a newline so the per-sortie
    # output files round-trip cleanly through pd.read_csv.
    data_lines = [ln if ln.endswith("\n") else ln + "\n" for ln in data_lines]
    if not data_lines:
        raise HyPlanValueError(
            f"IWG1 all-tracks file has no data rows: {src}"
        )

    # Parse just the timestamp column (index 1, after the IWG1 literal)
    # without dragging the full CSV through pandas — saves memory and
    # lets us preserve original row formatting on write-out.
    ts_strs = [ln.split(",", 2)[1] for ln in data_lines]
    timestamps = pd.to_datetime(ts_strs, utc=True, errors="raise")
    timestamps = timestamps.tz_localize(None)

    # Sort lines + timestamps together so out-of-order rows fall into
    # the correct sortie even if the source CSV isn't strictly sorted.
    order = np.argsort(timestamps.values, kind="mergesort")
    timestamps = timestamps[order]
    data_lines = [data_lines[i] for i in order]

    gap_threshold_s = gap_threshold_hr * 3600.0
    gap_s = np.diff(timestamps.values).astype("timedelta64[s]").astype(float)
    boundaries = np.where(gap_s >= gap_threshold_s)[0] + 1
    starts = [0, *boundaries.tolist()]
    ends = [*boundaries.tolist(), len(data_lines)]

    written: list[Path] = []
    for s, e in zip(starts, ends):
        date = timestamps[s].strftime("%Y-%m-%d")
        out_path = dest / f"{tail_label}_{date}.txt"
        with out_path.open("w") as f:
            f.write(header)
            f.writelines(data_lines[s:e])
        written.append(out_path)
    return written


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
