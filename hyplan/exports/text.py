"""Plain-text exports for hyplan flight plans.

Two formats:

* :func:`to_txt` — MovingLines ``save2txt`` waypoint listing.
* :func:`to_trackair` — TrackAir scanner-planning INI file used by HyMap
  and HySpex line-scanner crews.
"""

from __future__ import annotations

import datetime

import geopandas as gpd
import numpy as np

from ._common import extract_waypoints, generate_wp_names

__all__ = ["to_txt", "to_trackair"]


def to_txt(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime | None = None,
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
        date=takeoff_time.date() if takeoff_time else None,  # type: ignore[arg-type]
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


def to_trackair(
    plan: gpd.GeoDataFrame,
    filepath: str,
    sensor=None,
    terrain_elevation_m: float = 0.0,
    author: str = "",
    mission_name: str = "",
) -> None:
    """Export flight plan to TrackAir scanner planning format.

    TrackAir is an INI-style format used with HyMap and HySpex line-scanner
    sensors.  The ``[strips]`` section lists each straight data-collection
    segment as ``N=lat1,lon1,lat2,lon2`` in decimal degrees.

    Args:
        plan: GeoDataFrame returned by ``compute_flight_plan()``.
        filepath: Output file path (typically ``*.txt``).
        sensor: Optional ``LineScanner`` instance used to populate the
            ``[spex]`` field-of-view and swath-width fields.
        terrain_elevation_m: Terrain elevation in metres used to convert MSL
            altitude to AGL.  Defaults to 0 (sea-level terrain).
        author: Name written to ``Designed by`` fields.
        mission_name: Written to ``Flight plan name``.
    """
    flight_line_rows = plan[plan["segment_type"] == "flight_line"]

    # --- Altitude AGL from first flight-line segment ---
    if len(flight_line_rows):
        alt_msl_ft = float(flight_line_rows.iloc[0]["start_altitude"])
        alt_agl_m = alt_msl_ft * 0.3048 - terrain_elevation_m
    else:
        alt_agl_m = 0.0

    # --- Sensor-derived fields ---
    if sensor is not None:
        fov = sensor.half_angle * 2
        swath_width = int(2 * np.tan(np.radians(sensor.half_angle)) * alt_agl_m)
        fov_str = str(fov)
        swath_str = str(swath_width)
    else:
        fov_str = ""
        swath_str = ""

    alt_agl_str = str(int(round(alt_agl_m))) if alt_agl_m else ""

    output_lines = [
        "[general]",
        "Project name =",
        "",
        "Coordinate system = World geographic latitude-longitude",
        "Datum =",
        "",
        f"Flight plan name = {mission_name}",
        "Export flight plan = no",
        f"Designed by = {author}",
        "",
        "[spex]",
        "'DO NOT EDIT THESE PARAMETERS HERE, USE THE SCANNER MENU.",
        "Type of planning = LINE SCANNER PLANNING",
        f"Field of view = {fov_str}",
        f"Flying height agl (meters) = {alt_agl_str}",
        f"Swath width (meters) = {swath_str}",
        "",
        f"Designed by = {author}",
        "",
        "[strips]",
    ]

    for strip_n, (_, row) in enumerate(flight_line_rows.iterrows(), start=1):
        output_lines.append(
            f"{strip_n}={row['start_lat']},{row['start_lon']},"
            f"{row['end_lat']},{row['end_lon']}"
        )

    with open(filepath, "w") as f:
        f.write("\n".join(output_lines) + "\n")
