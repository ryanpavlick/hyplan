"""Export flight plans to MovingLines-compatible file formats.

All formats produce output that is drop-in compatible with MovingLines
(github.com/samuelleblanc/fp) so existing flight crew workflows are
preserved.

Typical usage::

    from hyplan import compute_flight_plan, DynamicAviation_B200
    from hyplan.exports import to_pilot_excel, to_foreflight_csv

    plan = compute_flight_plan(aircraft, waypoints, ...)
    to_pilot_excel(plan, "flight_plan_for_pilots.xlsx", aircraft=aircraft)
    to_foreflight_csv(plan, "flight_plan_FOREFLIGHT.csv")
"""

import datetime
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from .geometry import (
    dd_to_ddm,
    dd_to_ddms,
    dd_to_foreflight_oneline,
    dd_to_nddmm,
    magnetic_declination,
    true_to_magnetic,
)
from .units import ureg

__all__ = [
    "extract_waypoints",
    "generate_wp_names",
    "to_excel",
    "to_pilot_excel",
    "to_foreflight_csv",
    "to_honeywell_fms",
    "to_er2_csv",
    "to_icartt",
    "to_kml",
    "to_gpx",
    "to_txt",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        alt_ft = _safe_float(row.get("start_altitude"), default=0.0)
        alt_m = alt_ft * 0.3048
        alt_kft = alt_ft / 1000.0

        dist_nm = row.get("distance", 0.0) or 0.0
        dist_km = dist_nm * 1.852
        leg_time = row.get("time_to_segment", 0.0) or 0.0

        # Derive speed from distance and time
        if leg_time > 0 and dist_nm > 0:
            speed_kt = dist_nm / (leg_time / 60.0)
            speed_mps = speed_kt * 0.5144
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
        alt_ft = _safe_float(last.get("end_altitude"), default=0.0)
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


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, replacing None/NaN with *default*."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _utc_fraction(minutes_from_midnight: float) -> float:
    """Convert minutes from midnight to Excel time fraction (0-1)."""
    return minutes_from_midnight / 1440.0


def _compute_sza(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute solar zenith angle, returning -9999 if sun module unavailable."""
    try:
        from sunposition import sunpos
        ts = pd.DatetimeIndex([dt], tz='UTC')
        _, zenith, *_ = sunpos(ts, lat, lon, elevation=0)
        return float(zenith[0])
    except Exception:
        return -9999.0


def _compute_solar_azimuth(lat: float, lon: float,
                           dt: datetime.datetime) -> float:
    """Compute solar azimuth, returning -9999 if unavailable."""
    try:
        from sunposition import sunpos
        ts = pd.DatetimeIndex([dt], tz='UTC')
        azimuth, *_ = sunpos(ts, lat, lon, elevation=0)
        return float(azimuth[0])
    except Exception:
        return -9999.0


# ---------------------------------------------------------------------------
# Full working Excel  (MovingLines ``write_to_excel`` layout)
# ---------------------------------------------------------------------------

def to_excel(
    plan: gpd.GeoDataFrame,
    filepath: str,
    aircraft=None,
    takeoff_time: datetime.datetime = None,
    mission_name: str = "",
) -> None:
    """Write a full MovingLines-compatible working Excel file.

    This produces the 24-column format (A-X) that MovingLines uses as its
    internal working format, enabling round-tripping between tools.

    Args:
        plan: Flight plan GeoDataFrame from ``compute_flight_plan()``.
        filepath: Output ``.xlsx`` path.
        aircraft: Optional Aircraft object for metadata.
        takeoff_time: UTC datetime of takeoff.  If *None*, times are shown
            as elapsed minutes from 00:00 UTC.
        mission_name: Campaign / mission name string.
    """
    import xlsxwriter

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    # Compute UTC base (minutes from midnight)
    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    wb = xlsxwriter.Workbook(filepath, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet("Flight Plan")

    # Formats
    hdr_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4",
                             "font_color": "white", "font_size": 12,
                             "font_name": "Aptos Narrow"})
    cell_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow"})
    time_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                              "num_format": "hh:mm"})
    dec_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                             "num_format": "0.0"})
    dec2_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                              "num_format": "0.00"})
    int_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                             "num_format": "0"})

    # Headers (matching MovingLines column order)
    headers = [
        "WP", "Lat [+-90]", "Lon [+-180]", "Speed [m/s]", "delayT [min]",
        "Altitude [m]", "CumLegT [hh:mm]", "UTC [hh:mm]", "LocalT [hh:mm]",
        "LegT [hh:mm]", "Dist [km]", "CumDist [km]", "Dist [Nmi]",
        "CumDist [Nmi]", "Speed [kt]", "Altitude [kft]", "SZA [deg]",
        "AZI [deg]", "Bearing [deg]", "ClimbT [min]", "Comments",
        "WP names", "HdWind [kt]", "HdWind [m/s]",
    ]
    for col, h in enumerate(headers):
        ws.write(0, col, h, hdr_fmt)

    ws.freeze_panes(1, 0)

    # Data rows
    for idx, (_, wp) in enumerate(wps.iterrows()):
        row = idx + 1
        utc_min = base_minutes + wp["cum_time_min"]
        local_offset_h = wp["lon"] / 15.0  # approximate local time
        local_min = utc_min + local_offset_h * 60.0

        # Compute SZA/AZI if we have a real takeoff time
        if takeoff_time:
            wp_dt = takeoff_time + datetime.timedelta(minutes=wp["cum_time_min"])
            sza = _compute_sza(wp["lat"], wp["lon"], wp_dt)
            azi = _compute_solar_azimuth(wp["lat"], wp["lon"], wp_dt)
        else:
            sza = -9999.0
            azi = -9999.0

        ws.write(row, 0, int(wp["wp"]), int_fmt)                   # WP
        ws.write(row, 1, wp["lat"], cell_fmt)                      # Lat
        ws.write(row, 2, wp["lon"], cell_fmt)                      # Lon
        ws.write(row, 3, wp["speed_mps"], dec2_fmt)                # Speed m/s
        ws.write(row, 4, 0, int_fmt)                               # delayT
        ws.write(row, 5, wp["alt_m"] or 0, dec_fmt)                # Altitude m
        ws.write(row, 6, _utc_fraction(wp["cum_time_min"]), time_fmt)  # CumLegT
        ws.write(row, 7, _utc_fraction(utc_min), time_fmt)        # UTC
        ws.write(row, 8, _utc_fraction(local_min % 1440), time_fmt)  # LocalT
        ws.write(row, 9, _utc_fraction(wp["leg_time_min"]), time_fmt)  # LegT
        ws.write(row, 10, wp["dist_km"], dec_fmt)                  # Dist km
        ws.write(row, 11, wp["cum_dist_km"], dec_fmt)              # CumDist km
        ws.write(row, 12, wp["dist_nm"], dec_fmt)                  # Dist Nmi
        ws.write(row, 13, wp["cum_dist_nm"], dec_fmt)              # CumDist Nmi
        ws.write(row, 14, wp["speed_kt"], dec_fmt)                 # Speed kt
        ws.write(row, 15, wp["alt_kft"] or 0, dec_fmt)            # Altitude kft
        ws.write(row, 16, sza, dec_fmt)                            # SZA
        ws.write(row, 17, azi, dec_fmt)                            # AZI
        ws.write(row, 18, wp["heading"], dec_fmt)                  # Bearing
        ws.write(row, 19, 0.0, dec_fmt)                            # ClimbT
        ws.write(row, 20, wp["segment_type"], cell_fmt)            # Comments
        ws.write(row, 21, wp_names[idx] if idx < len(wp_names) else "", cell_fmt)  # WP names
        ws.write(row, 22, 0.0, dec_fmt)                            # HdWind kt
        ws.write(row, 23, 0.0, dec_fmt)                            # HdWind m/s

    # Metadata cells (column W+ area, matching MovingLines)
    meta_row = 0
    date_str = (takeoff_time.strftime("%Y-%m-%d") if takeoff_time
                else datetime.date.today().strftime("%Y-%m-%d"))
    ws.write(meta_row, 24, date_str, cell_fmt)
    ws.write(meta_row, 25, mission_name, cell_fmt)
    ws.write(meta_row, 26, "Created with", cell_fmt)
    ws.write(meta_row, 27, "hyplan", cell_fmt)

    wb.close()


# ---------------------------------------------------------------------------
# Pilot Excel  (MovingLines ``save2xl_for_pilots_xlswriter`` layout)
# ---------------------------------------------------------------------------

def to_pilot_excel(
    plan: gpd.GeoDataFrame,
    filepath: str,
    aircraft=None,
    takeoff_time: datetime.datetime = None,
    mission_name: str = "",
    coord_format: str = "DD MM",
    include_mag_heading: bool = True,
) -> None:
    """Write a simplified pilot-facing Excel file.

    Matches the MovingLines ``_for_pilots.xlsx`` format.

    Args:
        plan: Flight plan GeoDataFrame from ``compute_flight_plan()``.
        filepath: Output ``.xlsx`` path.
        aircraft: Optional Aircraft object for metadata (type, tail number).
        takeoff_time: UTC datetime of takeoff.
        mission_name: Campaign / mission name.
        coord_format: Coordinate display format — one of
            ``'DD MM'``, ``'DD MM SS'``, ``'NDDD MM.SS'``.
        include_mag_heading: Include a magnetic heading column.
    """
    import xlsxwriter

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    wb = xlsxwriter.Workbook(filepath, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet("Pilots")

    # Title row
    title_fmt = wb.add_format({
        "bold": True, "font_size": 24, "font_color": "white",
        "bg_color": "#4472C4", "font_name": "Aptos Narrow",
    })
    aircraft_name = ""
    if aircraft:
        aircraft_name = getattr(aircraft, "aircraft_type", "")
        tail = getattr(aircraft, "tail_number", "")
        if tail:
            aircraft_name += f" ({tail})"

    date_str = (takeoff_time.strftime("%Y-%m-%d") if takeoff_time
                else datetime.date.today().strftime("%Y-%m-%d"))
    title = f"{mission_name}  {aircraft_name}  {date_str}".strip()
    ws.merge_range(0, 0, 0, 6, title, title_fmt)

    # Header row
    hdr_fmt = wb.add_format({
        "bold": True, "bg_color": "#4472C4", "font_color": "white",
        "font_size": 12, "font_name": "Aptos Narrow",
    })
    headers = ["WP", "WP name", "Lat", "Lon", "Altitude [kft]", "UTC [hh:mm]"]
    if include_mag_heading:
        headers.append("Mag Heading [deg]")
    headers.append("Comments")

    for col, h in enumerate(headers):
        ws.write(1, col, h, hdr_fmt)

    ws.freeze_panes(2, 0)

    # Alternating row formats
    fmt_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
    })
    fmt_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
    })
    time_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "hh:mm",
    })
    time_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "hh:mm",
    })
    dec_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "0.00",
    })
    dec_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "0.00",
    })
    hdg_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "0.0",
    })
    hdg_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "0.0",
    })

    # Coordinate formatter
    if coord_format == "DD MM SS":
        coord_fn = dd_to_ddms
    elif coord_format == "NDDD MM.SS":
        coord_fn = dd_to_nddmm
    else:
        coord_fn = dd_to_ddm

    for idx, (_, wp) in enumerate(wps.iterrows()):
        row = idx + 2
        cf = fmt_even if idx % 2 == 0 else fmt_odd
        tf = time_even if idx % 2 == 0 else time_odd
        df = dec_even if idx % 2 == 0 else dec_odd
        hf = hdg_even if idx % 2 == 0 else hdg_odd

        lat_s, lon_s = coord_fn(wp["lat"], wp["lon"])
        utc_min = base_minutes + wp["cum_time_min"]

        col = 0
        ws.write(row, col, int(wp["wp"]), cf); col += 1
        ws.write(row, col, wp_names[idx] if idx < len(wp_names) else "", cf); col += 1
        ws.write(row, col, lat_s, cf); col += 1
        ws.write(row, col, lon_s, cf); col += 1
        ws.write(row, col, wp["alt_kft"] or 0, df); col += 1
        ws.write(row, col, _utc_fraction(utc_min), tf); col += 1

        if include_mag_heading:
            dec = magnetic_declination(wp["lat"], wp["lon"])
            mag_hdg = true_to_magnetic(wp["heading"], dec)
            ws.write(row, col, mag_hdg, hf); col += 1

        comment = wp["segment_type"]
        ws.write(row, col, comment, cf)

    # ForeFlight one-liner footer
    footer_row = len(wps) + 3
    ws.write(footer_row, 0, "One line waypoints for foreflight:", wb.add_format({"bold": True}))
    oneline_parts = []
    for _, wp in wps.iterrows():
        oneline_parts.append(dd_to_foreflight_oneline(wp["lat"], wp["lon"]))
    ws.write(footer_row + 1, 0, " ".join(oneline_parts))

    # Metadata
    meta_col = len(headers) + 1
    ws.write(1, meta_col, date_str)
    ws.write(1, meta_col + 1, mission_name)
    ws.write(1, meta_col + 2, "Created with")
    ws.write(1, meta_col + 3, "hyplan")

    wb.close()


# ---------------------------------------------------------------------------
# ForeFlight CSV
# ---------------------------------------------------------------------------

def to_foreflight_csv(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write a ForeFlight-compatible CSV file.

    Matches the MovingLines ``_FOREFLIGHT.csv`` format.  Also writes a
    companion ``_FOREFLIGHT_oneline.txt`` file.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.csv`` path.
        takeoff_time: Optional UTC takeoff time (used for waypoint naming).
    """
    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    seen_names = set()
    lines = ["Waypoint,Description,LAT,LONG"]

    for idx, (_, wp) in enumerate(wps.iterrows()):
        name = wp_names[idx] if idx < len(wp_names) else f"WP{idx:02d}"
        if name in seen_names:
            continue
        seen_names.add(name)

        alt_kft = wp["alt_kft"] or 0
        comment = (wp["segment_type"] or "").replace(",", "")
        desc = f"ALT={alt_kft:3.2f} kft {comment}".strip()
        lines.append(f"{name},{desc},{wp['lat']:+2.12f},{wp['lon']:+2.12f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Companion one-liner
    oneline_path = filepath.replace(".csv", "_oneline.txt")
    oneline_parts = []
    for _, wp in wps.iterrows():
        oneline_parts.append(dd_to_foreflight_oneline(wp["lat"], wp["lon"]))
    with open(oneline_path, "w") as f:
        f.write(" ".join(oneline_parts) + "\n")


# ---------------------------------------------------------------------------
# Honeywell FMS CSV
# ---------------------------------------------------------------------------

def to_honeywell_fms(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write a Honeywell FMS-compatible CSV file.

    Matches the MovingLines ``_Honeywell.csv`` format for Gulfstream
    III/IV avionics.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.csv`` path.
        takeoff_time: Optional UTC takeoff time (used for waypoint naming).
    """
    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    seen_names = set()
    lines = ["E,WPT,FIX,LAT,LON"]

    for idx, (_, wp) in enumerate(wps.iterrows()):
        name = wp_names[idx] if idx < len(wp_names) else f"WP{idx:02d}"
        if name in seen_names:
            continue
        seen_names.add(name)

        alt_kft = wp["alt_kft"] or 0
        comment = (wp["segment_type"] or "").replace(",", "")
        fix = f"ALT={alt_kft:3.2f} kft {comment}".strip()
        lat_s, lon_s = dd_to_nddmm(wp["lat"], wp["lon"])
        lines.append(f"x,{name},{fix},{lat_s},{lon_s}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# ER-2 CSV
# ---------------------------------------------------------------------------

def to_er2_csv(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write an ER-2-compatible CSV file.

    Matches the MovingLines ER-2 export format.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.csv`` path.
        takeoff_time: Optional UTC takeoff time.
    """
    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    lines = ["ID,Description,LAT,LONG,Altitude [kft],UTC [hh:mm],Comments"]

    for idx, (_, wp) in enumerate(wps.iterrows()):
        name = wp_names[idx] if idx < len(wp_names) else f"WP{idx:02d}"
        alt_kft = wp["alt_kft"] or 0
        utc_min = base_minutes + wp["cum_time_min"]
        utc_h = int(utc_min // 60)
        utc_m = int(utc_min % 60)
        comment = (wp["segment_type"] or "").replace(",", "")
        lines.append(
            f"{idx:2.0f},.{name},{wp['lat']:+2.12f},{wp['lon']:+2.12f},"
            f"{alt_kft:3.2f},{utc_h:2.0f}:{utc_m:02.0f},{comment}"
        )

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# ICARTT
# ---------------------------------------------------------------------------

def to_icartt(
    plan: gpd.GeoDataFrame,
    filepath: str,
    pi_name: str = "",
    institution: str = "",
    mission_name: str = "",
    flight_date: datetime.date = None,
    aircraft=None,
    takeoff_time: datetime.datetime = None,
    interval_seconds: float = 60.0,
    revision: str = "RA",
    revision_comments: str = "RA: Planned flight track - pre-flight",
    special_comments: str = "",
    normal_comments: str = "",
) -> None:
    """Write an ICARTT v1001 file.

    Matches the MovingLines ``save2ict`` / ``write_ict`` format.  Data
    is linearly interpolated along the flight plan at *interval_seconds*
    intervals.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.ict`` path.
        pi_name: Principal investigator name.
        institution: PI institution.
        mission_name: Campaign name.
        flight_date: Flight date (default today).
        aircraft: Aircraft object (for platform name).
        takeoff_time: UTC takeoff time.  If *None*, Start_UTC begins at 0.
        interval_seconds: Interpolation interval in seconds (default 60).
        revision: Revision code (RA, RB, R0, R1, ...).
        revision_comments: Revision description string.
        special_comments: Additional special comment lines.
        normal_comments: Additional normal comment lines.
    """
    if flight_date is None:
        flight_date = datetime.date.today()
    today = datetime.date.today()

    if takeoff_time:
        base_seconds = (takeoff_time.hour * 3600
                        + takeoff_time.minute * 60
                        + takeoff_time.second)
    else:
        base_seconds = 0.0

    # Interpolate along the flight plan
    wps = extract_waypoints(plan)
    interp_points = []  # (utc_s, lat, lon, alt_m, speed_mps, bearing)

    for idx in range(len(wps) - 1):
        w0 = wps.iloc[idx]
        w1 = wps.iloc[idx + 1]

        t0 = base_seconds + w0["cum_time_min"] * 60.0
        t1 = base_seconds + w1["cum_time_min"] * 60.0
        dt = t1 - t0
        if dt <= 0:
            continue

        n_steps = max(1, int(dt / interval_seconds))
        for step in range(n_steps):
            frac = step / n_steps
            t = t0 + frac * dt
            lat = w0["lat"] + frac * (w1["lat"] - w0["lat"])
            lon = w0["lon"] + frac * (w1["lon"] - w0["lon"])
            alt = (w0["alt_m"] or 0) + frac * ((w1["alt_m"] or 0) - (w0["alt_m"] or 0))
            spd = w0["speed_mps"] + frac * (w1["speed_mps"] - w0["speed_mps"])
            hdg = w0["heading"]  # heading for this leg
            interp_points.append((t, lat, lon, alt, spd, hdg))

    # Add final point
    if len(wps) > 0:
        wf = wps.iloc[-1]
        tf = base_seconds + wf["cum_time_min"] * 60.0
        interp_points.append((
            tf, wf["lat"], wf["lon"], wf["alt_m"] or 0,
            0.0, wf["heading"],
        ))

    # Compute SZA for each point
    data_rows = []
    for t, lat, lon, alt, spd, hdg in interp_points:
        if takeoff_time:
            dt_utc = datetime.datetime(
                flight_date.year, flight_date.month, flight_date.day,
                tzinfo=datetime.timezone.utc,
            ) + datetime.timedelta(seconds=t)
            sza = _compute_sza(lat, lon, dt_utc)
        else:
            sza = -9999.0
        data_rows.append((t, lat, lon, alt, spd, hdg, sza))

    # Build header
    platform_name = ""
    if aircraft:
        platform_name = getattr(aircraft, "aircraft_type", "")

    special_lines = [s for s in special_comments.strip().split("\n") if s]
    normal_lines_list = []
    normal_lines_list.append(f"PI_CONTACT_INFO: {pi_name}")
    normal_lines_list.append(f"PLATFORM: {platform_name}")
    normal_lines_list.append("LOCATION: N/A")
    normal_lines_list.append("ASSOCIATED_DATA: N/A")
    normal_lines_list.append("INSTRUMENT_INFO: Planned flight track generated by hyplan")
    normal_lines_list.append("DATA_INFO: Linearly interpolated planned flight track")
    normal_lines_list.append("UNCERTAINTY: N/A - planned track")
    normal_lines_list.append("ULOD_FLAG: -7777")
    normal_lines_list.append("ULOD_VALUE: N/A")
    normal_lines_list.append("LLOD_FLAG: -8888")
    normal_lines_list.append("LLOD_VALUE: N/A")
    normal_lines_list.append(f"DM_CONTACT_INFO: {pi_name}")
    normal_lines_list.append(f"PROJECT_INFO: {mission_name}")
    normal_lines_list.append("STIPULATIONS_ON_USE: Use as-is for flight planning only")
    normal_lines_list.append(f"OTHER_COMMENTS: {normal_comments}")
    normal_lines_list.append(f"REVISION: {revision}")
    normal_lines_list.append(revision_comments)

    n_data_vars = 6  # Lat, Lon, Alt, speed, Bearing, SZA
    n_special = len(special_lines)
    n_normal = len(normal_lines_list)

    # Count header lines: top metadata (7) + interval (1) + indep var (1)
    # + n_data (1) + scale factors (1) + missing values (1)
    # + var descriptions (n_data_vars) + n_special header (1) + special lines
    # + n_normal header (1) + normal lines + column header (1)
    n_header = (7 + 1 + 1 + 1 + 1 + 1 + n_data_vars
                + 1 + n_special + 1 + n_normal + 1)

    header = []
    header.append(f"{n_header}, 1001")
    header.append(pi_name or "N/A")
    header.append(institution or "N/A")
    header.append("Planned flight track - hyplan")
    header.append(mission_name or "N/A")
    header.append("1, 1")
    header.append(f"{flight_date.year}, {flight_date.month}, {flight_date.day}, "
                  f"{today.year}, {today.month}, {today.day}")
    header.append("0")  # data interval (0 = variable)
    header.append("Start_UTC, seconds, seconds from midnight UTC")
    header.append(str(n_data_vars))
    header.append(", ".join(["1"] * n_data_vars))  # scale factors
    header.append(", ".join(["-9999"] * n_data_vars))  # missing values
    header.append("Latitude, Degrees, North positive, geodetic latitude")
    header.append("Longitude, Degrees, East positive, geodetic longitude")
    header.append("Altitude, meters, above sea level")
    header.append("speed, meters per second, ground speed")
    header.append("Bearing, degrees, degrees from north")
    header.append("SZA, degrees, solar zenith angle")
    header.append(str(n_special))
    header.extend(special_lines)
    header.append(str(n_normal))
    header.extend(normal_lines_list)
    header.append("Start_UTC, Latitude, Longitude, Altitude, speed, Bearing, SZA")

    with open(filepath, "w") as f:
        for line in header:
            f.write(line + "\n")
        for t, lat, lon, alt, spd, hdg, sza in data_rows:
            f.write(f"{t:.0f}, {lat:.9f}, {lon:.9f}, {alt:.0f}, "
                    f"{spd:.3f}, {hdg:.3f}, {sza:.3f}\n")


# ---------------------------------------------------------------------------
# KML / KMZ
# ---------------------------------------------------------------------------

def to_kml(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime = None,
    altitude_exaggeration: float = 1.0,
) -> None:
    """Write a KML (and optionally KMZ) file for Google Earth.

    Matches the MovingLines ``save2kml`` layout.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.kml`` or ``.kmz`` path.
        takeoff_time: Optional UTC takeoff time.
        altitude_exaggeration: Multiplier for altitude in the KML
            coordinates (default 1.0 = true scale; MovingLines uses 10).
    """
    import simplekml

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    kml = simplekml.Kml()

    # Camera at first waypoint
    if len(wps) > 0:
        w0 = wps.iloc[0]
        kml.document.lookat = simplekml.LookAt(
            longitude=w0["lon"], latitude=w0["lat"],
            altitude=3000, altitudemode=simplekml.AltitudeMode.relativetoground,
            range=50000,
        )

    # Waypoint points
    for idx, (_, wp) in enumerate(wps.iterrows()):
        name = wp_names[idx] if idx < len(wp_names) else f"WP{idx}"
        alt_display = (wp["alt_m"] or 0) * altitude_exaggeration

        utc_min = base_minutes + wp["cum_time_min"]
        utc_h = int(utc_min // 60)
        utc_m = int(utc_min % 60)

        desc = (
            f"WP# {int(wp['wp'])}<br>"
            f"UTC: {utc_h:02d}:{utc_m:02d}<br>"
            f"Name: {name}<br>"
            f"CumDist: {wp['cum_dist_nm']:.1f} nm<br>"
            f"Speed: {wp['speed_kt']:.0f} kt<br>"
            f"Bearing: {wp['heading']:.0f} deg<br>"
            f"Alt: {wp['alt_kft'] or 0:.1f} kft<br>"
            f"Type: {wp['segment_type']}"
        )

        pnt = kml.newpoint(name=name, coords=[(wp["lon"], wp["lat"], alt_display)])
        pnt.altitudemode = simplekml.AltitudeMode.relativetoground
        pnt.extrude = 1
        pnt.description = desc

    # Flight path line
    coords = []
    for _, wp in wps.iterrows():
        alt_display = (wp["alt_m"] or 0) * altitude_exaggeration
        coords.append((wp["lon"], wp["lat"], alt_display))

    if coords:
        ls = kml.newlinestring(name="Flight Plan")
        ls.coords = coords
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        ls.extrude = 1
        ls.style.linestyle.width = 4.0
        ls.style.linestyle.color = simplekml.Color.red

    # Save
    if filepath.lower().endswith(".kmz"):
        kml.savekmz(filepath)
    else:
        kml.save(filepath)
        # Also save KMZ companion
        kmz_path = filepath.rsplit(".", 1)[0] + ".kmz"
        kml.savekmz(kmz_path)


# ---------------------------------------------------------------------------
# GPX
# ---------------------------------------------------------------------------

def to_gpx(
    plan: gpd.GeoDataFrame,
    filepath: str,
    mission_name: str = "",
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write a GPX route file.

    Matches the MovingLines ``save2gpx`` format.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.gpx`` path.
        mission_name: Route name.
        takeoff_time: Optional UTC takeoff time.
    """
    import gpxpy
    import gpxpy.gpx

    wps = extract_waypoints(plan)

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    gpx = gpxpy.gpx.GPX()
    route = gpxpy.gpx.GPXRoute(name=mission_name or "Flight Plan")
    gpx.routes.append(route)

    for idx, (_, wp) in enumerate(wps.iterrows()):
        utc_min = base_minutes + wp["cum_time_min"]

        rpt = gpxpy.gpx.GPXRoutePoint(
            latitude=wp["lat"],
            longitude=wp["lon"],
            elevation=wp["alt_m"] or 0,
            name=f"WP#{idx}",
            comment=wp["segment_type"] or "",
        )

        if takeoff_time:
            rpt.time = takeoff_time + datetime.timedelta(minutes=wp["cum_time_min"])

        route.points.append(rpt)

    with open(filepath, "w") as f:
        f.write(gpx.to_xml())


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def to_txt(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write a plain text waypoint file.

    Matches the MovingLines ``save2txt`` format.  Note: Lon comes before
    Lat (opposite of the Excel format).

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.txt`` path.
        takeoff_time: Optional UTC takeoff time.
    """
    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    header = (
        "#WP  Lon[+-180]  Lat[+-90]  Speed[m/s]  delayT[min]  "
        "Altitude[m]  CumLegT[H]  UTC[H]  LocalT[H]  LegT[H]  "
        "Dist[km]  CumDist[km]  Dist[Nmi]  CumDist[Nmi]  Speed[kt]  "
        "Altitude[kft]  SZA[deg]  AZI[deg]  Bearing[deg]  Climbt[min]  "
        "Comments  WPnames"
    )

    lines = [header]

    for idx, (_, wp) in enumerate(wps.iterrows()):
        utc_min = base_minutes + wp["cum_time_min"]
        utc_h = utc_min / 60.0
        local_h = utc_h + wp["lon"] / 15.0
        cum_h = wp["cum_time_min"] / 60.0
        leg_h = wp["leg_time_min"] / 60.0
        name = wp_names[idx] if idx < len(wp_names) else ""

        line = (
            f"{int(wp['wp']):<2d}  "
            f"{wp['lon']:+2.8f}  "
            f"{wp['lat']:+2.8f}  "
            f"{wp['speed_mps']:<4.2f}  "
            f"{0:<3d}  "  # delayT
            f"{wp['alt_m'] or 0:<5.1f}  "
            f"{cum_h:<2.2f}  "
            f"{utc_h:<2.2f}  "
            f"{local_h % 24:<2.2f}  "
            f"{leg_h:<2.2f}  "
            f"{wp['dist_km']:<5.1f}  "
            f"{wp['cum_dist_km']:<5.1f}  "
            f"{wp['dist_nm']:<5.1f}  "
            f"{wp['cum_dist_nm']:<5.1f}  "
            f"{wp['speed_kt']:<3.1f}  "
            f"{wp['alt_kft'] or 0:<3.2f}  "
            f"{-9999:<3.1f}  "  # SZA placeholder
            f"{-9999:<3.1f}  "  # AZI placeholder
            f"{wp['heading']:<3.1f}  "
            f"{0:<3d}  "  # ClimbT
            f"{wp['segment_type']}  "
            f"{name}"
        )
        lines.append(line)

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
