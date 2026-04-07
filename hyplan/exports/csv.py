"""CSV exports for ForeFlight, Honeywell FMS, and the NASA ER-2.

Three loosely related plain-text formats grouped together because they
all build a comma-separated waypoint listing from the same shared
:func:`~hyplan.exports._common.extract_waypoints` table.
"""

import datetime

import geopandas as gpd

from ..geometry import dd_to_foreflight_oneline, dd_to_nddmm
from ._common import extract_waypoints, generate_wp_names

__all__ = ["to_foreflight_csv", "to_honeywell_fms", "to_er2_csv"]


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
