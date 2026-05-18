"""Loaders for NOAA HRD/AOML 1-second flight-level files.

Two related formats sharing the same per-storm directory:

* **G-IV-SP ARWO** (``YYYYMMDDU<#>.01.txt``).  Header is 5 lines
  including ``Tail Number : 5302/5303``.  Whitespace-delimited columns
  ``GMT Time, ..., PA, ..., ROLL, ..., TAS, THD, TRK, ..., WDir, WSpd``.
  PA / GPSA in feet, TAS / GS / WSpd in knots, ROLL / PITCH in degrees.
  See ARWO Version 19.x file format spec.

* **P-3 1-sec text** (``YYYYMMDDH<#>.1sec.txt`` / ``...I<#>...``).
  Two-line preamble (storm name + flight ID, then blank) followed by
  a column header and a units row.  Columns include
  ``TIME, Lat, Lon, Head, Track, GnSpd, TAS, GeoAl, Press, WndDr,
  WndSp, Tempr, Dewpt, D Val, RdAlt, MixR, VtWnd, SfcPr, ThetaE``.
  GnSpd / TAS / WndSp in **m/s**, GeoAl / RdAlt / D Val in **meters**.

Both loaders return a canonical DataFrame with ``timestamp,
altitude (ft), tas_kt, vertical_rate (fpm), roll_deg`` (where
available), suitable for the ``_common.py`` calibration recipe.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import M_PER_S_TO_KT, M_TO_FT, smooth_diff

# ---------------------------------------------------------------------------
# G-IV-SP ARWO loader (.01.txt)
# ---------------------------------------------------------------------------

_GIV_HEADER_LINES = 5  # Export line + Version + File Version + Tail + blank




def _parse_giv_date(first_line: str) -> date | None:
    """Pull the MM/DD/YYYY out of the ARWO export header line."""
    m = re.search(r"on\s+(\d{2})/(\d{2})/(\d{4})", first_line)
    if m:
        mo, dd, yy = map(int, m.groups())
        try:
            return date(yy, mo, dd)
        except ValueError:
            return None
    return None


def load_giv_arwo(path: Path) -> pd.DataFrame | None:
    """Load one G-IV-SP ARWO file, return canonical DataFrame."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    if len(lines) < _GIV_HEADER_LINES + 2:
        return None
    flight_date = _parse_giv_date(lines[0])
    if flight_date is None:
        return None
    # Column header is line 5 (0-indexed); data starts at line 6.
    header = lines[_GIV_HEADER_LINES].split()
    # The header has duplicates ("V V" → "V", "V"; "WD WS" appear twice).
    # We only care about a fixed subset of columns by name; keep raw
    # whitespace tokens and index by first-occurrence.
    name_to_idx: dict[str, int] = {}
    for i, name in enumerate(header):
        name_to_idx.setdefault(name, i)

    needed = {
        "GMT": name_to_idx.get("GMT"),
        "PA": name_to_idx.get("PA"),
        "TAS": name_to_idx.get("TAS"),
        "GS": name_to_idx.get("GS"),
        "ROLL": name_to_idx.get("ROLL"),
        "PITCH": name_to_idx.get("PITCH"),
        "THD": name_to_idx.get("THD"),
        "LAT": name_to_idx.get("LAT"),
        "LON": name_to_idx.get("LON"),
    }
    # GMT is "GMT Time" (two-word header) — combine.
    gmt_idx = needed["GMT"]
    if gmt_idx is None:
        return None

    rows: list[list[str]] = []
    for ln in lines[_GIV_HEADER_LINES + 1:]:
        toks = ln.split()
        if len(toks) < 30:
            continue
        rows.append(toks)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Columns: token 0 is "HH:MM:SS" (joins "GMT Time" header position 0).
    # All other tokens are aligned to header indices [1..N].
    def col_at(idx: int | None) -> pd.Series | None:
        return None if idx is None or idx >= df.shape[1] else df[idx]

    times = col_at(0)
    if times is None:
        return None

    # Parse HH:MM:SS, advance day at midnight rollover.
    parsed = pd.to_datetime(times, format="%H:%M:%S", errors="coerce")
    if parsed.isna().all():
        return None
    base = pd.Timestamp(flight_date)
    # Wall clock + base date; bump day where time wraps backwards.
    secs = parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second
    diff = secs.diff().fillna(0)
    day_offset = (diff < -1800).cumsum()  # rollover detected on >30-min jump
    timestamps = base + pd.to_timedelta(secs, unit="s") + pd.to_timedelta(day_offset, unit="D")

    out = pd.DataFrame({"timestamp": timestamps})
    # Most ARWO field positions are header_idx - 0 in the data row (since
    # "GMT Time" occupies index 0 + 1 in header but the "Time" subword is
    # absorbed and the data line has just one token there).  Use header
    # indexing minus 1 to align the trailing fields.
    def num(idx: int | None, scale: float = 1.0) -> pd.Series:
        if idx is None or idx - 1 >= df.shape[1]:
            return pd.Series([np.nan] * len(df))
        # Header has "GMT Time" as 2 tokens but data has 1 token for time
        # → all subsequent header-tokens are offset by -1 in the data row.
        return pd.to_numeric(df[idx - 1], errors="coerce") * scale

    out["altitude"] = num(needed["PA"])  # PA already in feet
    out["tas_kt"] = num(needed["TAS"])
    out["groundspeed"] = num(needed["GS"])
    out["roll_deg"] = num(needed["ROLL"])
    out["pitch_deg"] = num(needed["PITCH"])
    out["heading_deg"] = num(needed["THD"])

    out = out.dropna(subset=["altitude", "tas_kt"]).reset_index(drop=True)
    if out.empty:
        return None
    t_s = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    out["vertical_rate"] = smooth_diff(t_s, out["altitude"].to_numpy(dtype=float)) * 60.0
    out = out.dropna(subset=["vertical_rate"]).reset_index(drop=True)
    # Drop ground fixes via heuristic (no WOW; PA<200 ft + TAS<50 kt)
    if not out.empty:
        out = out[~((out["altitude"] < 200) & (out["tas_kt"] < 50))].reset_index(drop=True)
    return out if not out.empty else None


# ---------------------------------------------------------------------------
# P-3 1-sec text loader (.1sec.txt)
# ---------------------------------------------------------------------------

def _parse_p3_date(filename: str) -> date | None:
    """Extract YYYYMMDD from the filename."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", filename)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def load_p3_1sec(path: Path) -> pd.DataFrame | None:
    """Load one P-3 1-sec ARWO file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    if len(lines) < 5:
        return None
    flight_date = _parse_p3_date(path.name)
    if flight_date is None:
        return None
    # Two-line preamble (storm + ID, then blank), column header on line 2,
    # units row on line 3, data starts on line 4.  The units row contains
    # tokens like "HHMMSS", "Deg N", "m/s", "m" — confirm and skip it.
    header_tokens = lines[2].split()
    # The header has a few multi-word column names ("D Val") that split
    # into separate tokens; data rows treat them as a single column.
    # Collapse known multi-word names back together.
    header: list[str] = []
    skip_next = False
    for i, tok in enumerate(header_tokens):
        if skip_next:
            skip_next = False
            continue
        if tok == "D" and i + 1 < len(header_tokens) and header_tokens[i + 1] == "Val":
            header.append("DVal")
            skip_next = True
        else:
            header.append(tok)
    units_row = lines[3].split()
    rows: list[list[str]] = []
    for ln in lines[4:]:
        toks = ln.split()
        if len(toks) < 5:
            continue
        # Skip the units row if its tokens reappear (some files have a
        # repeated units row).
        if toks[0] == "HHMMSS":
            continue
        rows.append(toks)
    if not rows:
        return None

    name_to_idx: dict[str, int] = {}
    for i, name in enumerate(header):
        name_to_idx.setdefault(name, i)

    df = pd.DataFrame(rows)

    # TIME column at idx 0, format HHMMSS as integer
    times_raw = df[0]
    base = pd.Timestamp(flight_date)
    def _parse_hhmmss(s: str) -> pd.Timedelta | float:
        s = s.strip()
        if not s.isdigit() or len(s) != 6:
            return np.nan
        h, m, sec = int(s[0:2]), int(s[2:4]), int(s[4:6])
        return pd.Timedelta(hours=h, minutes=m, seconds=sec)
    deltas = times_raw.apply(_parse_hhmmss)
    valid = ~deltas.isna()
    if not valid.any():
        return None
    df = df[valid].reset_index(drop=True)
    deltas = deltas[valid].reset_index(drop=True)
    timestamps = pd.Series([base] * len(deltas)) + pd.to_timedelta(deltas)
    # Day rollover: monotonic correction
    secs = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    secs_diff = np.diff(secs, prepend=secs[0])
    day_jumps = np.cumsum((secs_diff < -1800).astype(int))
    timestamps = timestamps + pd.to_timedelta(day_jumps, unit="D")

    def num(name: str, scale: float = 1.0) -> pd.Series:
        idx = name_to_idx.get(name)
        if idx is None or idx >= df.shape[1]:
            return pd.Series([np.nan] * len(df))
        return pd.to_numeric(df[idx], errors="coerce") * scale

    out = pd.DataFrame({"timestamp": timestamps})
    out["altitude"] = num("GeoAl") * M_TO_FT       # m → ft
    out["tas_kt"] = num("TAS") * M_PER_S_TO_KT     # m/s → kt
    out["groundspeed"] = num("GnSpd") * M_PER_S_TO_KT
    out["heading_deg"] = num("Head")
    if "RdAlt" in name_to_idx:
        out["radar_alt_ft"] = num("RdAlt") * M_TO_FT
    # Lon column header is "Lon" (Deg W convention from header — usually
    # already negative West, but sample shows positive values; flip sign
    # if values are uniformly positive over a Caribbean storm).
    out["lat"] = num("Lat")
    lon_raw = num("Lon")
    # If header units row says "Deg W" and median lon is positive, flip sign.
    if "Lon" in name_to_idx and lon_raw.notna().any() and lon_raw.median() > 0:
        idx = name_to_idx["Lon"]
        if idx < len(units_row) and "W" in units_row[idx]:
            lon_raw = -lon_raw
    out["lon"] = lon_raw

    out = out.dropna(subset=["altitude", "tas_kt"]).reset_index(drop=True)
    if out.empty:
        return None
    # Sentinel filter: drop fixes with non-physical values (negative or
    # absurd altitudes / TAS).  HRD files include a handful of garbage
    # placeholder rows pre-takeoff with values like -1.0 m / -0.1 m/s.
    out = out[
        (out["altitude"] > -500)
        & (out["altitude"] < 60_000)
        & (out["tas_kt"] > 30)
        & (out["tas_kt"] < 700)
    ].reset_index(drop=True)
    if out.empty:
        return None
    t_s = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    out["vertical_rate"] = smooth_diff(t_s, out["altitude"].to_numpy(dtype=float)) * 60.0
    out = out.dropna(subset=["vertical_rate"]).reset_index(drop=True)
    # Drop physically-impossible vertical rates (sentinel-driven jumps).
    # P-3 max sustained ROC ~3500 fpm; G-IV max ~4500 fpm.  5000 fpm
    # comfortably keeps real climbs while dropping diff-from-bad-altitude
    # transients.
    out = out[out["vertical_rate"].abs() < 5_000].reset_index(drop=True)
    # Drop fixes with sentinel-flagged altitude (-5000 to -2000 ft is
    # outside any real flight envelope and indicates a sentinel pulled
    # through the m→ft conversion).
    out = out[out["altitude"] > -1000].reset_index(drop=True)
    # Drop ground fixes
    if not out.empty:
        out = out[~((out["altitude"] < 200) & (out["tas_kt"] < 50))].reset_index(drop=True)
    return out if not out.empty else None
