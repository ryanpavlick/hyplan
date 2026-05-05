"""NASA ICARTT (.ict) airborne in-situ data loader.

ICARTT is the standard NASA Earth Science file format for airborne
in-situ measurements (FFI 1001 — fixed format, single independent
variable).  Many campaigns publish per-sortie .ict files containing
the same air-data fields IWG1 carries (TAS, IAS, Mach, altitude,
attitude, position, wind), just with different column names and a
text-header preamble describing the schema.

This loader produces a DataFrame matching :func:`load_iwg1`'s contract
so calibration notebooks can ingest ICARTT data the same way they
ingest IWG1 .txt files.  Column-name patterns are mapped flexibly
since each campaign uses different conventions (e.g.,
``True_Air_Speed``, ``TrueAirSpd``, ``TAS_ms-1``).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..exceptions import HyPlanValueError

_M_PER_FT = 0.3048
_KT_PER_MPS = 1.9438444924406046

# Patterns mapping ICARTT column names to canonical IWG1-style fields.
# The first match (case-insensitive substring) wins.  ``unit_hint`` is
# one of {"ft", "m", "kt", "mps", "fpm", "deg", "mach", "C", "hPa"};
# the actual column-name unit string (when present in the ICT header)
# overrides the hint.
_COLUMN_PATTERNS = [
    # (canonical_name, [name_patterns], unit_hint)
    # NCAR EOL HIAPER LRT files use short symbolic names (GLAT, TASX,
    # ATX, etc.); patterns are listed alongside the longer NASA Ames
    # / ICARTT conventions.
    ("latitude",         [r"\blatitude\b",      r"\blat[_\s-]*deg\b", r"\bgpslat\b",
                          r"^glat$"],                                                   "deg"),
    ("longitude",        [r"\blongitude\b",     r"\blon[_\s-]*deg\b", r"\bgpslon\b",
                          r"^glon$"],                                                   "deg"),
    ("altitude",         [r"^pressure[_\s-]*altitude", r"^press[_\s-]*alt",
                          r"^paltf?$"],                                                 "ft"),
    ("altitude_gps_ft",  [r"\bgps[_\s-]*alt", r"^gps[_\s-]*altitude\b", r"^ggalt$"],    "m"),
    ("altitude_radar_ft",[r"\bradar[_\s-]*altitude", r"\bradar[_\s-]*alt"],             "ft"),
    ("groundspeed",      [r"\bground[_\s-]*speed\b", r"^gsf$"],                         "kt"),
    ("tas_kt",           [r"\btrue[_\s-]*air[_\s-]*speed\b", r"\btrueairspd\b",
                          r"\btas[_\s-]*ms\b", r"\btas\b", r"^tasx?$"],                 "kt"),
    ("ias_kt",           [r"\bindicated[_\s-]*air[_\s-]*speed\b", r"\bias\b",
                          r"^iasx?$"],                                                  "kt"),
    ("mach",             [r"\bmach[_\s-]*number\b", r"\bmach\b", r"^machx?$"],          "mach"),
    ("vertical_velocity",[r"\bvertical[_\s-]*speed\b", r"\bvert[_\s-]*wind",
                          r"^vspd$"],                                                   "fpm"),
    ("true_heading",     [r"\btrue[_\s-]*heading\b", r"^heading\b", r"\bhdg[_\s-]*deg\b",
                          r"^thdg$"],                                                   "deg"),
    ("track",            [r"\btrack[_\s-]*angle\b", r"\btrack\b", r"^tkat$"],           "deg"),
    ("pitch_deg",        [r"\bpitch[_\s-]*angle\b", r"\bpitch[_\s-]*deg\b", r"^pitch\b"],"deg"),
    ("roll_deg",         [r"\broll[_\s-]*angle\b", r"\broll[_\s-]*deg\b", r"^roll\b"],  "deg"),
    ("aoa_deg",          [r"\bangle[_\s-]*of[_\s-]*attack\b", r"\baoa\b", r"^attack$"], "deg"),
    ("wind_speed_kt",    [r"\bwind[_\s-]*speed\b", r"\bwspd[_\s-]*ms\b", r"^wsc?$"],    "kt"),
    ("wind_direction_deg",[r"\bwind[_\s-]*direction\b", r"\bwdir[_\s-]*deg\b",
                           r"^wdc?$"],                                                  "deg"),
    ("ambient_temp_c",   [r"\bstatic[_\s-]*air[_\s-]*temp", r"\bambtemp\b",
                          r"\btstat[_\s-]*degc\b", r"\bambient[_\s-]*temp\b",
                          r"^atx?$"],                                                   "C"),
    ("static_pressure_hpa",[r"\bstatic[_\s-]*press", r"\bpstat[_\s-]*mb\b",
                            r"\bstaticprs\b", r"^psxc$"],                               "hPa"),
]


def _convert_to_canonical(name: str, unit_str: str, x: pd.Series) -> pd.Series:
    """Convert numeric column to the canonical unit, given ICT unit string."""
    u = (unit_str or "").lower().strip()
    if name in ("altitude", "altitude_radar_ft"):
        if "m" in u and "ft" not in u:  # meters
            return x / _M_PER_FT
        return x  # assume ft
    if name == "altitude_gps_ft":
        # GPS_Altitude is normally in meters in IWG1 spec; ICARTT
        # campaigns vary.  If unit string says ft, keep as-is.
        if "ft" in u:
            return x
        return x / _M_PER_FT
    if name in ("groundspeed", "tas_kt", "ias_kt", "wind_speed_kt"):
        if "m/s" in u or "ms-1" in u or "ms_1" in u or u in ("ms", "mps"):
            return x * _KT_PER_MPS
        return x  # assume kt (most ICARTT files use knots for these)
    if name == "vertical_velocity":
        if "m/s" in u or "ms-1" in u:
            return x * 60.0 / _M_PER_FT
        return x  # assume fpm
    if name == "true_heading" or name == "track":
        return x % 360.0
    return x


def load_icartt(path: Union[str, Path]) -> pd.DataFrame:
    """Load one ICARTT ``.ict`` file into a DataFrame matching the
    :func:`load_iwg1` schema.

    The ICARTT FFI 1001 header structure:

    * Line 1: ``n_header_lines, FFI`` (e.g. ``68, 1001``)
    * Line 7: file UTC date (YYYY, MM, DD, ...)
    * Line 9: independent variable name (typically seconds-past-midnight)
    * Line 10: number of dependent variables (NV)
    * Lines 13..12+NV: dependent variable name + units (one per line)
    * Line ``n_header_lines``: column-header CSV (independent var
      followed by NV dependent vars)
    * Lines after: data rows.

    Returns a DataFrame with the standard ``timestamp``, ``latitude``,
    ``longitude``, ``altitude``, ``tas_kt``, ``vertical_rate``,
    ``roll_deg``, etc. columns that downstream calibration notebooks
    expect.  Missing values (sentinel ``-9999``) become NaN.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ICARTT file not found: {p}")

    with p.open("r") as f:
        lines = f.readlines()
    if not lines:
        raise HyPlanValueError(f"ICARTT file is empty: {p}")

    # Line 1: header_count, FFI
    n_header = int(lines[0].split(",")[0].strip())
    # Line 7: UTC start date
    date_parts = [int(x) for x in re.findall(r"\d+", lines[6])][:3]
    if len(date_parts) != 3:
        raise HyPlanValueError(f"ICARTT line 7 not a date: {lines[6].strip()!r}")
    base_date = datetime(date_parts[0], date_parts[1], date_parts[2])
    # Line 10: number of dependent variables
    n_vars = int(lines[9].strip())
    # Lines 13..12+n_vars: dependent variable name+unit
    var_lines = lines[12:12 + n_vars]
    # Build (name, unit) tuples
    dep_vars = []
    for ln in var_lines:
        parts = [s.strip() for s in ln.split(",")]
        name = parts[0]
        unit = parts[1] if len(parts) > 1 else ""
        dep_vars.append((name, unit))

    # Header line is line `n_header` (1-indexed) — actually it's index n_header-1.
    header_line = lines[n_header - 1]
    columns = [s.strip() for s in header_line.split(",")]

    # Read the data rows.  Use pandas with skiprows = n_header.
    raw = pd.read_csv(
        p, skiprows=n_header, header=None, names=columns,
        on_bad_lines="skip", low_memory=False,
    )
    if raw.empty:
        raise HyPlanValueError(f"ICARTT file has no data rows: {p}")

    # Replace sentinel values with NaN.  Per-column missing markers are
    # in line 12 (NV values), but using a global sentinel set covers
    # the typical conventions.
    SENTINELS = {-9999, -99999, -7777, -8888}
    raw = raw.replace(list(SENTINELS), np.nan)

    out = pd.DataFrame()
    # Independent variable: seconds past midnight on base_date.
    indep_col = columns[0]
    seconds = pd.to_numeric(raw[indep_col], errors="coerce")
    out["timestamp"] = base_date + pd.to_timedelta(seconds, unit="s")

    # Map each canonical column.  Multiple ICT columns can match the
    # same canonical name (e.g., "Altitude" vs "GPS_Altitude"); take
    # the first match.
    used_columns = {indep_col}
    name_to_unit = {n: u for n, u in dep_vars}
    for canonical, patterns, unit_hint in _COLUMN_PATTERNS:
        if canonical in out.columns:
            continue
        for col in columns:
            if col in used_columns:
                continue
            ucol = col.lower()
            for pat in patterns:
                if re.search(pat, ucol, re.IGNORECASE):
                    unit = name_to_unit.get(col, unit_hint)
                    series = pd.to_numeric(raw[col], errors="coerce")
                    out[canonical] = _convert_to_canonical(canonical, unit, series)
                    used_columns.add(col)
                    break
            if canonical in out.columns:
                break

    # Backfill missing-but-required columns with NaN so downstream
    # code that always reads them (load_iwg1 contract) doesn't KeyError.
    for canonical, _, _ in _COLUMN_PATTERNS:
        if canonical not in out.columns:
            out[canonical] = np.nan

    # Sort.  Don't dedupe like load_iwg1 does — ICARTT files are
    # often sub-second sampled (e.g., 50 ms / 20 Hz),
    # which the IWG1 100 ms threshold would discard.
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # If Pressure Altitude is unavailable for the whole file, fall
    # back to GPS altitude.  Many ICARTT campaigns publish only GPS
    # altitude; for calibration purposes either works (within QNH /
    # geoid uncertainty, ±100-200 ft at most altitudes).
    if out["altitude"].notna().sum() == 0 and out["altitude_gps_ft"].notna().sum() > 0:
        out["altitude"] = out["altitude_gps_ft"]

    # Apply the same out-of-range / spike filters as load_iwg1.
    bad_pos = (out["latitude"].abs() > 90) | (out["longitude"].abs() > 180)
    if len(out) >= 3:
        for col in ("latitude", "longitude"):
            v = out[col]
            JUMP_DEG = 1.0
            jump = (v.diff().abs() > JUMP_DEG) & (v.diff(-1).abs() > JUMP_DEG)
            bad_pos = bad_pos | jump.fillna(False)
    if bad_pos.any():
        out.loc[bad_pos, ["latitude", "longitude"]] = np.nan

    bad_alt = (out["altitude"] < -2000) | (out["altitude"] > 100000)
    if len(out) >= 11:
        med = out["altitude"].rolling(11, center=True, min_periods=5).median()
        bad_alt = bad_alt | ((out["altitude"] - med).abs() > 1000.0)
    if bad_alt.fillna(False).any():
        out.loc[bad_alt.fillna(False), "altitude"] = np.nan

    bad_tas = (out["tas_kt"] < 0) | (out["tas_kt"] > 2000)
    if len(out) >= 11:
        tmed = out["tas_kt"].rolling(11, center=True, min_periods=5).median()
        bad_tas = bad_tas | ((out["tas_kt"] - tmed).abs() > 100.0)
    if bad_tas.fillna(False).any():
        out.loc[bad_tas.fillna(False), "tas_kt"] = np.nan

    # Derive vertical_rate using the same long-baseline 180s finite
    # difference as load_iwg1.
    if len(out) >= 2:
        alt_ft = out["altitude"].to_numpy(dtype=float)
        t_s = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        half = 90.0
        lo_idx = np.searchsorted(t_s, t_s - half, side="left")
        hi_idx = np.searchsorted(t_s, t_s + half, side="right") - 1
        lo_idx = np.clip(lo_idx, 0, len(t_s) - 1)
        hi_idx = np.clip(hi_idx, 0, len(t_s) - 1)
        dt = t_s[hi_idx] - t_s[lo_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            vs = np.where(dt > 0, (alt_ft[hi_idx] - alt_ft[lo_idx]) / dt * 60.0, np.nan)
        out["vertical_rate"] = vs
    else:
        out["vertical_rate"] = np.nan

    return out


def detect_platform(path: Union[str, Path]) -> Optional[str]:
    """Return the ``PLATFORM:`` metadata line value from an ICARTT
    file, or None if not present.  Useful for identifying which
    aircraft a given file represents (some campaign folders contain
    files from multiple airframes)."""
    p = Path(path)
    with p.open("r") as f:
        # PLATFORM: usually appears in the post-header free-text block
        # within the first ~120 lines.  Don't read more than that.
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            if "PLATFORM:" in line:
                return line.split("PLATFORM:", 1)[1].strip()
    return None
