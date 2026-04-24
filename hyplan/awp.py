"""Planning helpers for the Aerosol Wind Profiler (AWP).

These utilities turn an AWP instrument model plus a flight geometry into a
set of expected vector-profile sample locations. They intentionally stop short
of coherent Doppler retrieval physics; the focus is on geometry, spacing, and
stable-flight feasibility.

References
----------
Bedka, K., Marketon, J., Henderson, S., and Kavaya, M. (2024).
"AWP: NASA's Aerosol Wind Profiler Coherent Doppler Wind Lidar." In
Singh, U. N., Tzeremes, G., Refaat, T. F., and Ribes Pleguezuelo, P. (eds),
*Space-based Lidar Remote Sensing Techniques and Emerging Technologies*,
LIDAR 2023, Springer Aerospace Technology.
https://doi.org/10.1007/978-3-031-53618-2_3

Bedka, K. (2025). *3-D Lidar Wind Airborne Profiling Using A
Coherent-Detection Doppler Wind Lidar Designed For Space-Based Operation*.
Final Project Report for NOAA Broad Agency Announcement, "Measuring the
Atmospheric Wind Profile (3D Winds): Call for Studies and Field Measurement
Campaigns."
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import LineString, Point

from .exceptions import HyPlanTypeError, HyPlanValueError
from .flight_line import FlightLine
from .geometry import process_linestring, wrap_to_180
from .instruments.awp import AerosolWindProfiler, ProfilingLidar
from .units import ureg

__all__ = [
    "flag_awp_stable_segments",
    "awp_profile_locations_for_flight_line",
    "awp_profile_locations_for_plan",
]


_PROFILE_COLUMNS = [
    "sample_index",
    "source_segment_index",
    "source_segment_name",
    "source_segment_type",
    "distance_from_start_m",
    "elapsed_time_s",
    "time_utc",
    "platform_lat",
    "platform_lon",
    "platform_heading_deg",
    "altitude_agl_m",
    "altitude_msl_ft",
    "profile_spacing_m",
    "profile_spacing_s",
    "profile_assignment_offset_m",
    "los_surface_separation_m",
    "stable_platform_ok",
    "los1_azimuth_deg",
    "los1_lat",
    "los1_lon",
    "los2_azimuth_deg",
    "los2_lat",
    "los2_lon",
    "geometry",
]


def _empty_profile_gdf() -> gpd.GeoDataFrame:
    frame = pd.DataFrame(columns=_PROFILE_COLUMNS)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")


def _as_awp(sensor: Optional[ProfilingLidar]) -> ProfilingLidar:
    if sensor is None:
        return AerosolWindProfiler()
    if not isinstance(sensor, ProfilingLidar):
        raise HyPlanTypeError(
            "sensor must be a ProfilingLidar or None (for default AWP)"
        )
    return sensor


def _as_quantity(value, unit: str, label: str) -> Quantity:
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(f"{label} must be numeric or a pint.Quantity, got {type(value)}")


def _altitude_agl_or_default(value: Optional[Quantity], fallback: Quantity) -> Quantity:
    if value is None:
        return fallback.to("meter")
    return _as_quantity(value, "meter", "altitude_agl")


def _interpolate_geodesic(linestring: LineString, distances_m: Iterable[float]) -> list[dict[str, float]]:
    """Interpolate positions and headings along a WGS84 LineString."""
    lats, lons, azimuths, cumulative = process_linestring(linestring)
    if len(cumulative) < 2:
        lon0, lat0 = linestring.coords[0]
        return [
            {"latitude": float(lat0), "longitude": float(lon0), "heading": float(azimuths[0])}
            for _ in distances_m
        ]

    out: list[dict[str, float]] = []
    max_distance = float(cumulative[-1])
    for distance_m in distances_m:
        d = min(max(float(distance_m), 0.0), max_distance)
        seg_idx = int(np.searchsorted(cumulative, d, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(cumulative) - 2))
        seg_start = float(cumulative[seg_idx])
        seg_heading = float(azimuths[seg_idx])
        lat0 = float(lats[seg_idx])
        lon0 = float(lons[seg_idx])
        remain = d - seg_start
        lat_i, lon_i = pymap3d.vincenty.vreckon(lat0, lon0, remain, seg_heading)
        out.append(
            {
                "latitude": float(lat_i),
                "longitude": float(wrap_to_180(lon_i)),
                "heading": seg_heading,
            }
        )
    return out


def _max_heading_change(linestring: LineString) -> float:
    """Maximum change from the initial heading across a segment geometry."""
    _, _, azimuths, _ = process_linestring(linestring)
    if len(azimuths) <= 1:
        return 0.0
    ref = float(azimuths[0])
    delta = ((np.asarray(azimuths) - ref + 180.0) % 360.0) - 180.0
    return float(np.max(np.abs(delta)))


def _line_length_m(linestring: LineString) -> float:
    _, _, _, cumulative = process_linestring(linestring)
    if len(cumulative) == 0:
        return 0.0
    return float(cumulative[-1])


def _segment_groundspeed(row: pd.Series) -> Optional[Quantity]:
    groundspeed = row.get("groundspeed_kts")
    if pd.notna(groundspeed) and float(groundspeed) > 0:
        return ureg.Quantity(float(groundspeed), "knot")

    distance = row.get("distance")
    duration = row.get("time_to_segment")
    if pd.notna(distance) and pd.notna(duration) and float(duration) > 0:
        return ureg.Quantity(float(distance) / float(duration), "nautical_mile / minute")
    return None


def flag_awp_stable_segments(
    plan: gpd.GeoDataFrame,
    sensor: ProfilingLidar | None = None,
) -> gpd.GeoDataFrame:
    """Annotate a flight plan with AWP stability flags.

    Bedka (2025) gives explicit QC limits for roll (3°) and
    profile-to-profile altitude changes (0.5 km). HyPlan does not carry
    per-sample roll, so this helper uses segment heading change as a planning
    proxy for turns and unstable aircraft attitude.
    """
    if not isinstance(plan, gpd.GeoDataFrame):
        raise HyPlanTypeError("plan must be a GeoDataFrame")

    awp = _as_awp(sensor)
    flagged = plan.copy()
    heading_changes: list[float] = []
    altitude_changes_m: list[float] = []
    stable_flags: list[bool] = []

    always_unstable = {"takeoff", "climb", "descent", "approach", "loiter"}

    for _, row in flagged.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            heading_change = 0.0
            altitude_change_m = 0.0
            stable = False
        else:
            heading_change = _max_heading_change(geom)
            alt0 = row.get("start_altitude")
            alt1 = row.get("end_altitude")
            if pd.notna(alt0) and pd.notna(alt1):
                altitude_change_m = abs(
                    ureg.Quantity(float(alt1) - float(alt0), "foot").m_as("meter")
                )
            else:
                altitude_change_m = 0.0

            stable = awp.is_stable_segment(
                heading_change=heading_change * ureg.degree,
                altitude_change=altitude_change_m * ureg.meter,
            )
            if row.get("segment_type") in always_unstable:
                stable = False

        heading_changes.append(float(heading_change))
        altitude_changes_m.append(float(altitude_change_m))
        stable_flags.append(bool(stable))

    flagged["awp_heading_change_deg"] = heading_changes
    flagged["awp_altitude_change_m"] = altitude_changes_m
    flagged["awp_stable_platform_ok"] = stable_flags
    return flagged


def _profiles_for_geometry(
    geometry: LineString,
    *,
    altitude_agl: Quantity,
    altitude_msl_ft: float,
    source_segment_index: int | None,
    source_segment_name: str | None,
    source_segment_type: str | None,
    sensor: ProfilingLidar,
    ground_speed: Quantity,
    start_time: _dt.datetime | None,
    dwell_time_per_los: Quantity | None,
    nadir_dwell_time: Quantity | None,
    stable_platform_ok: bool,
) -> gpd.GeoDataFrame:
    total_length_m = _line_length_m(geometry)
    if total_length_m <= 0:
        return _empty_profile_gdf()

    spacing = sensor.vector_profile_spacing(
        ground_speed,
        dwell_time_per_los=dwell_time_per_los,
        nadir_dwell_time=nadir_dwell_time,
    )
    offset = sensor.profile_assignment_offset(
        ground_speed,
        dwell_time_per_los=dwell_time_per_los,
        nadir_dwell_time=nadir_dwell_time,
    )
    spacing_m = spacing.m_as("meter")
    offset_m = offset.m_as("meter")
    if offset_m > total_length_m:
        return _empty_profile_gdf()

    sample_distances = np.arange(offset_m, total_length_m + 1e-9, spacing_m)
    interpolated = _interpolate_geodesic(geometry, sample_distances)
    records = []
    for sample_index, (distance_m, pos) in enumerate(zip(sample_distances, interpolated), start=1):
        intercepts = sensor.los_ground_intercepts(
            pos["latitude"],
            pos["longitude"],
            pos["heading"],
            altitude_agl,
        )
        elapsed_s = distance_m / ground_speed.m_as("meter / second")
        time_utc = start_time + _dt.timedelta(seconds=float(elapsed_s)) if start_time else pd.NaT
        records.append(
            {
                "sample_index": sample_index,
                "source_segment_index": source_segment_index,
                "source_segment_name": source_segment_name,
                "source_segment_type": source_segment_type,
                "distance_from_start_m": float(distance_m),
                "elapsed_time_s": float(elapsed_s),
                "time_utc": time_utc,
                "platform_lat": pos["latitude"],
                "platform_lon": pos["longitude"],
                "platform_heading_deg": pos["heading"],
                "altitude_agl_m": altitude_agl.m_as("meter"),
                "altitude_msl_ft": float(altitude_msl_ft),
                "profile_spacing_m": spacing_m,
                "profile_spacing_s": sensor.vector_profile_time_spacing(
                    dwell_time_per_los=dwell_time_per_los,
                    nadir_dwell_time=nadir_dwell_time,
                ).m_as("second"),
                "profile_assignment_offset_m": offset_m,
                "los_surface_separation_m": sensor.los_surface_separation(altitude_agl).m_as("meter"),
                "stable_platform_ok": bool(stable_platform_ok),
                "los1_azimuth_deg": intercepts["los1"]["azimuth"],
                "los1_lat": intercepts["los1"]["latitude"],
                "los1_lon": intercepts["los1"]["longitude"],
                "los2_azimuth_deg": intercepts["los2"]["azimuth"],
                "los2_lat": intercepts["los2"]["latitude"],
                "los2_lon": intercepts["los2"]["longitude"],
                "geometry": Point(pos["longitude"], pos["latitude"]),
            }
        )

    return gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry="geometry", crs="EPSG:4326")


def awp_profile_locations_for_flight_line(
    flight_line: FlightLine,
    *,
    sensor: ProfilingLidar | None = None,
    ground_speed: Quantity | None = None,
    start_time: _dt.datetime | None = None,
    altitude_agl: Quantity | None = None,
    dwell_time_per_los: Quantity | None = None,
    nadir_dwell_time: Quantity | None = None,
) -> gpd.GeoDataFrame:
    """Predict AWP vector-profile sample locations along a single flight line.

    The spacing and LOS geometry follow the AWP observing concept summarized by
    Bedka et al. (2024) and Bedka (2025).
    """
    if not isinstance(flight_line, FlightLine):
        raise HyPlanTypeError("flight_line must be a FlightLine")

    awp = _as_awp(sensor)
    speed = _as_quantity(
        ground_speed if ground_speed is not None else awp.nominal_airspeed,
        "meter / second",
        "ground_speed",
    )
    altitude = _altitude_agl_or_default(altitude_agl, flight_line.altitude_msl)
    return _profiles_for_geometry(
        flight_line.geometry,
        altitude_agl=altitude,
        altitude_msl_ft=flight_line.altitude_msl.m_as("foot"),
        source_segment_index=None,
        source_segment_name=flight_line.site_name,
        source_segment_type="flight_line",
        sensor=awp,
        ground_speed=speed,
        start_time=start_time,
        dwell_time_per_los=dwell_time_per_los,
        nadir_dwell_time=nadir_dwell_time,
        stable_platform_ok=True,
    )


def awp_profile_locations_for_plan(
    plan: gpd.GeoDataFrame,
    *,
    sensor: ProfilingLidar | None = None,
    takeoff_time: _dt.datetime | None = None,
    dwell_time_per_los: Quantity | None = None,
    nadir_dwell_time: Quantity | None = None,
    stable_only: bool = True,
) -> gpd.GeoDataFrame:
    """Predict AWP vector-profile locations along stable segments of a plan.

    This combines the AWP LOS geometry from Bedka et al. (2024) with the
    stable-flight planning heuristics summarized in Bedka (2025).
    """
    flagged = flag_awp_stable_segments(plan, sensor=sensor)
    awp = _as_awp(sensor)

    all_profiles: list[gpd.GeoDataFrame] = []
    cumulative_seconds = 0.0
    for idx, row in flagged.iterrows():
        seg_duration_min = row.get("time_to_segment")
        seg_duration_s = float(seg_duration_min) * 60.0 if pd.notna(seg_duration_min) else 0.0

        geom = row.geometry
        stable = bool(row.get("awp_stable_platform_ok", False))
        speed = _segment_groundspeed(row)
        if (
            isinstance(geom, LineString)
            and speed is not None
            and speed.magnitude > 0
            and (stable or not stable_only)
        ):
            alt0 = row.get("start_altitude")
            alt1 = row.get("end_altitude")
            if pd.notna(alt0) and pd.notna(alt1):
                alt_ft = (float(alt0) + float(alt1)) / 2.0
                altitude_agl = ureg.Quantity(alt_ft, "foot").to("meter")
                seg_start_time = (
                    takeoff_time + _dt.timedelta(seconds=cumulative_seconds)
                    if takeoff_time is not None
                    else None
                )
                gdf = _profiles_for_geometry(
                    geom,
                    altitude_agl=altitude_agl,
                    altitude_msl_ft=alt_ft,
                    source_segment_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    source_segment_name=row.get("segment_name"),
                    source_segment_type=row.get("segment_type"),
                    sensor=awp,
                    ground_speed=speed.to("meter / second"),
                    start_time=seg_start_time,
                    dwell_time_per_los=dwell_time_per_los,
                    nadir_dwell_time=nadir_dwell_time,
                    stable_platform_ok=stable,
                )
                if not gdf.empty:
                    all_profiles.append(gdf)

        cumulative_seconds += max(seg_duration_s, 0.0)

    if not all_profiles:
        return _empty_profile_gdf()
    return gpd.GeoDataFrame(pd.concat(all_profiles, ignore_index=True), geometry="geometry", crs="EPSG:4326")
