"""Profiling lidar models for HyPlan.

This module currently implements NASA Langley's Aerosol Wind Profiler (AWP)
as a planning-oriented instrument model. The emphasis is on observation
geometry and stable-flight sampling constraints rather than Doppler retrieval
physics or atmospheric backscatter performance.

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
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import LineString, Point

from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..flight_line import FlightLine
from ..geometry import process_linestring, wrap_to_180, wrap_to_360
from ..terrain import generate_demfile, get_elevations, ray_terrain_intersection
from ..units import ureg
from ._base import Sensor

__all__ = [
    "AerosolWindProfiler",
    "flag_awp_stable_segments",
    "awp_profile_locations_for_flight_line",
    "awp_profile_locations_for_plan",
]


_LIGHT_SPEED_MPS = 299_792_458.0


def _as_quantity(value, unit: str, label: str) -> Quantity:
    """Normalize *value* to a quantity in *unit*."""
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(f"{label} must be numeric or a pint.Quantity, got {type(value)}")


class AerosolWindProfiler(Sensor):
    """NASA Langley Aerosol Wind Profiler (AWP).

    A coherent Doppler wind lidar that alternates between two line-of-sight
    (LOS) directions to recover a vector wind profile, plus a nadir dwell.
    The abstraction captures the planning geometry needed to answer questions
    like:

    - How far apart are valid vector retrievals along track?
    - Where do the LOS footprints intersect the surface?
    - How much straight-and-level flight is needed before vector retrievals
      become valid?

    All constructor parameters default to the airborne AWP configuration
    described by Bedka et al. (2024) and Bedka (2025):

    - wavelength: 2052.92 nm
    - 30° off nadir
    - dual LOS at ±45° azimuth from the aircraft nose
    - 200 Hz pulse rate
    - 500 MHz digitization rate
    - 65528 samples per pulse
    - common airborne mode of 3 s per LOS + 1 s nadir
    - QC limits of 3° roll and 0.5 km altitude change per profile

    References
    ----------
    Bedka, K., Marketon, J., Henderson, S., and Kavaya, M. (2024).
    "AWP: NASA's Aerosol Wind Profiler Coherent Doppler Wind Lidar." In
    *Space-based Lidar Remote Sensing Techniques and Emerging Technologies*.
    Springer. https://doi.org/10.1007/978-3-031-53618-2_3

    Bedka, K. (2025). *3-D Lidar Wind Airborne Profiling Using A
    Coherent-Detection Doppler Wind Lidar Designed For Space-Based Operation*.
    NOAA BAA final project report.
    """

    def __init__(
        self,
        name: str = "Aerosol Wind Profiler",
        *,
        wavelength: Quantity = 2052.92 * ureg.nanometer,
        off_nadir_angle: float = 30.0,
        los_azimuths_relative: Iterable[float] = (-45.0, 45.0),
        pulse_rate: Quantity = 200 * ureg.Hz,
        digitization_rate: Quantity = 500e6 * ureg.Hz,
        samples_per_pulse: int = 65528,
        dwell_time_per_los: Quantity = 3 * ureg.second,
        nadir_dwell_time: Quantity = 1 * ureg.second,
        blind_zone: Quantity = 200 * ureg.meter,
        max_roll_angle: Quantity = 3 * ureg.degree,
        max_heading_change: Quantity = 3 * ureg.degree,
        max_altitude_change: Quantity = 0.5 * ureg.kilometer,
        nominal_airspeed: Quantity = 225 * ureg.meter / ureg.second,
    ):
        super().__init__(name)

        self.wavelength = _as_quantity(wavelength, "nanometer", "wavelength")
        self.pulse_rate = _as_quantity(pulse_rate, "Hz", "pulse_rate")
        self.digitization_rate = _as_quantity(digitization_rate, "Hz", "digitization_rate")
        self.dwell_time_per_los = _as_quantity(dwell_time_per_los, "second", "dwell_time_per_los")
        self.nadir_dwell_time = _as_quantity(nadir_dwell_time, "second", "nadir_dwell_time")
        self.blind_zone = _as_quantity(blind_zone, "meter", "blind_zone")
        self.max_roll_angle = _as_quantity(max_roll_angle, "degree", "max_roll_angle")
        self.max_heading_change = _as_quantity(max_heading_change, "degree", "max_heading_change")
        self.max_altitude_change = _as_quantity(max_altitude_change, "meter", "max_altitude_change")
        self.nominal_airspeed = _as_quantity(nominal_airspeed, "meter / second", "nominal_airspeed")

        if not (0.0 < float(off_nadir_angle) < 90.0):
            raise HyPlanValueError("off_nadir_angle must be between 0 and 90 degrees")
        self.off_nadir_angle = float(off_nadir_angle)

        rel_az = tuple(float(v) for v in los_azimuths_relative)
        if len(rel_az) != 2:
            raise HyPlanValueError("los_azimuths_relative must contain exactly two azimuths")
        if np.isclose((rel_az[1] - rel_az[0]) % 360.0, 0.0):
            raise HyPlanValueError("The two LOS azimuths must not be identical")
        self.los_azimuths_relative = rel_az

        if self.pulse_rate.magnitude <= 0:
            raise HyPlanValueError("pulse_rate must be positive")
        if self.digitization_rate.magnitude <= 0:
            raise HyPlanValueError("digitization_rate must be positive")
        if samples_per_pulse <= 0:
            raise HyPlanValueError("samples_per_pulse must be positive")
        self.samples_per_pulse = int(samples_per_pulse)

        for label, q in (
            ("dwell_time_per_los", self.dwell_time_per_los),
            ("nadir_dwell_time", self.nadir_dwell_time),
            ("blind_zone", self.blind_zone),
            ("max_roll_angle", self.max_roll_angle),
            ("max_heading_change", self.max_heading_change),
            ("max_altitude_change", self.max_altitude_change),
            ("nominal_airspeed", self.nominal_airspeed),
        ):
            if q.magnitude <= 0:
                raise HyPlanValueError(f"{label} must be positive")

    @property
    def sample_period(self) -> Quantity:
        """ADC sample period."""
        return (1.0 / self.digitization_rate).to("second")  # type: ignore[no-any-return]

    def max_slant_range(self) -> Quantity:
        """Maximum recorded LOS range from pulse digitization."""
        distance_m = self.samples_per_pulse * _LIGHT_SPEED_MPS * self.sample_period.m_as("second") / 2.0
        return distance_m * ureg.meter  # type: ignore[no-any-return]

    def max_vertical_range(self) -> Quantity:
        """Maximum vertical profiling extent for the configured off-nadir view."""
        return self.max_slant_range() * np.cos(np.radians(self.off_nadir_angle))  # type: ignore[return-value,no-any-return]

    def vertical_bin_spacing(self, fft_samples: int = 256) -> Quantity:
        """Vertical bin spacing implied by an FFT window length.

        The range gate spans ``fft_samples`` digitized points. Converting the
        two-way light-travel time to slant range and projecting by the
        off-nadir angle reproduces the 66-132 m planning values cited for AWP
        in Bedka et al. (2024) and the Bedka (2025) final report for 256-512
        sample windows.
        """
        if fft_samples <= 0:
            raise HyPlanValueError("fft_samples must be positive")
        slant_m = fft_samples * _LIGHT_SPEED_MPS * self.sample_period.m_as("second") / 2.0
        return slant_m * np.cos(np.radians(self.off_nadir_angle)) * ureg.meter  # type: ignore[no-any-return]

    def los_ground_distance(self, altitude_agl: Quantity) -> Quantity:
        """Ground distance from nadir to the LOS surface intercept."""
        altitude_agl = _as_quantity(altitude_agl, "meter", "altitude_agl")
        return altitude_agl * np.tan(np.radians(self.off_nadir_angle))  # type: ignore[return-value,no-any-return]

    def los_orientations(self, track_heading: float) -> Tuple[float, float]:
        """Return absolute LOS azimuths (degrees true) for a given track heading."""
        abs_az = wrap_to_360(np.array(self.los_azimuths_relative) + float(track_heading))
        return float(abs_az[0]), float(abs_az[1])

    def los_surface_separation(self, altitude_agl: Quantity) -> Quantity:
        """Surface separation between the two LOS intercepts.

        For AWP's default ``±45°`` azimuth pair the included angle is ``90°``,
        so at 12 km altitude and 30° off nadir this is about 9.8 km, matching
        the ``~10 km`` separation described by Bedka (2025).
        """
        radius = self.los_ground_distance(altitude_agl).m_as("meter")
        az0, az1 = self.los_azimuths_relative
        delta = np.radians(abs(((az1 - az0) + 180.0) % 360.0 - 180.0))
        separation = np.sqrt(radius ** 2 + radius ** 2 - 2.0 * radius ** 2 * np.cos(delta))
        return separation * ureg.meter  # type: ignore[no-any-return]

    def los_ground_intercepts(
        self,
        latitude: float,
        longitude: float,
        track_heading: float,
        altitude_agl: Quantity,
    ) -> dict[str, dict[str, float]]:
        """Ground intercepts of the two LOS beams for a platform location."""
        radius_m = self.los_ground_distance(altitude_agl).m_as("meter")
        azimuths = self.los_orientations(track_heading)
        intercepts: dict[str, dict[str, float]] = {}
        for idx, azimuth in enumerate(azimuths, start=1):
            lat_i, lon_i = pymap3d.vincenty.vreckon(latitude, longitude, radius_m, azimuth)
            intercepts[f"los{idx}"] = {
                "latitude": float(lat_i),
                "longitude": float(wrap_to_180(lon_i)),
                "azimuth": float(azimuth),
            }
        return intercepts

    def vector_profile_time_spacing(
        self,
        *,
        dwell_time_per_los: Quantity | None = None,
        nadir_dwell_time: Quantity | None = None,
    ) -> Quantity:
        """Nominal time spacing between vector wind profiles.

        This follows the AWP airborne concept of operations described by
        Bedka et al. (2024) and Bedka (2025): a vector retrieval is assigned at
        the LOS1→LOS2 switch, producing a spacing of
        ``2 * dwell_time_per_los + nadir_dwell_time``.
        """
        dwell = _as_quantity(
            dwell_time_per_los if dwell_time_per_los is not None else self.dwell_time_per_los,
            "second",
            "dwell_time_per_los",
        )
        nadir = _as_quantity(
            nadir_dwell_time if nadir_dwell_time is not None else self.nadir_dwell_time,
            "second",
            "nadir_dwell_time",
        )
        return (2.0 * dwell + nadir).to("second")  # type: ignore[no-any-return]

    def profile_assignment_offset(
        self,
        ground_speed: Quantity | None = None,
        *,
        dwell_time_per_los: Quantity | None = None,
        nadir_dwell_time: Quantity | None = None,
    ) -> Quantity:
        """Distance from the start of a stable leg to the first vector profile."""
        speed = _as_quantity(
            ground_speed if ground_speed is not None else self.nominal_airspeed,
            "meter / second",
            "ground_speed",
        )
        dwell = _as_quantity(
            dwell_time_per_los if dwell_time_per_los is not None else self.dwell_time_per_los,
            "second",
            "dwell_time_per_los",
        )
        nadir = _as_quantity(
            nadir_dwell_time if nadir_dwell_time is not None else self.nadir_dwell_time,
            "second",
            "nadir_dwell_time",
        )
        return (speed * (dwell + nadir)).to("meter")  # type: ignore[no-any-return]

    def vector_profile_spacing(
        self,
        ground_speed: Quantity | None = None,
        *,
        dwell_time_per_los: Quantity | None = None,
        nadir_dwell_time: Quantity | None = None,
    ) -> Quantity:
        """Along-track spacing between adjacent vector profiles."""
        speed = _as_quantity(
            ground_speed if ground_speed is not None else self.nominal_airspeed,
            "meter / second",
            "ground_speed",
        )
        dt = self.vector_profile_time_spacing(
            dwell_time_per_los=dwell_time_per_los,
            nadir_dwell_time=nadir_dwell_time,
        )
        return (speed * dt).to("meter")  # type: ignore[no-any-return]

    def is_stable_segment(
        self,
        *,
        heading_change: Quantity | float = 0.0,
        altitude_change: Quantity | float = 0.0,
        roll_angle: Quantity | float = 0.0,
    ) -> bool:
        """Return ``True`` when a segment satisfies AWP stability heuristics."""
        heading = _as_quantity(heading_change, "degree", "heading_change")
        altitude = _as_quantity(altitude_change, "meter", "altitude_change")
        roll = _as_quantity(roll_angle, "degree", "roll_angle")
        return (
            abs(heading.m_as("degree")) <= self.max_heading_change.m_as("degree")
            and abs(altitude.m_as("meter")) <= self.max_altitude_change.m_as("meter")
            and abs(roll.m_as("degree")) <= self.max_roll_angle.m_as("degree")
        )


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
    "terrain_elevation_m",
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
    "los1_alt_m",
    "los2_azimuth_deg",
    "los2_lat",
    "los2_lon",
    "los2_alt_m",
    "geometry",
]


def _empty_profile_gdf() -> gpd.GeoDataFrame:
    frame = pd.DataFrame(columns=_PROFILE_COLUMNS)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")


def _as_awp(sensor: Optional[AerosolWindProfiler]) -> AerosolWindProfiler:
    if sensor is None:
        return AerosolWindProfiler()
    if not isinstance(sensor, AerosolWindProfiler):
        raise HyPlanTypeError(
            "sensor must be an AerosolWindProfiler or None (for default AWP)"
        )
    return sensor


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


def _profile_sampling(
    geometry: LineString,
    sensor: AerosolWindProfiler,
    ground_speed: Quantity,
    dwell_time_per_los: Quantity | None,
    nadir_dwell_time: Quantity | None,
) -> tuple[np.ndarray, list[dict[str, float]], float, float, float]:
    """Return profile sample distances, positions, and timing metadata."""
    total_length_m = _line_length_m(geometry)
    if total_length_m <= 0:
        return np.array([], dtype=float), [], 0.0, 0.0, 0.0

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
    profile_spacing_s = sensor.vector_profile_time_spacing(
        dwell_time_per_los=dwell_time_per_los,
        nadir_dwell_time=nadir_dwell_time,
    ).m_as("second")
    if offset_m > total_length_m:
        return np.array([], dtype=float), [], spacing_m, offset_m, profile_spacing_s

    sample_distances = np.arange(offset_m, total_length_m + 1e-9, spacing_m)
    interpolated = _interpolate_geodesic(geometry, sample_distances)
    return sample_distances, interpolated, spacing_m, offset_m, profile_spacing_s


def _resolve_platform_headings(
    track_headings_deg: np.ndarray,
    *,
    crab_angle_deg: float | None = None,
    heading_deg: float | None = None,
) -> np.ndarray:
    """Resolve aircraft headings from track headings and crab metadata."""
    if heading_deg is not None:
        return np.full_like(track_headings_deg, heading_deg % 360.0)
    if crab_angle_deg is not None:
        return (track_headings_deg + crab_angle_deg) % 360.0
    return track_headings_deg


def _terrain_dem_for_awp_profiles(
    positions: list[dict[str, float]],
    sensor: AerosolWindProfiler,
    altitude_msl: Quantity,
    *,
    platform_headings_deg: np.ndarray | None = None,
) -> str:
    """Create a DEM covering the line and approximate LOS intercept envelope."""
    altitude_guess = altitude_msl.to("meter")
    dem_lats = [pos["latitude"] for pos in positions]
    dem_lons = [pos["longitude"] for pos in positions]
    if platform_headings_deg is None:
        platform_headings_deg = np.asarray([pos["heading"] for pos in positions], dtype=float)
    for pos, platform_heading in zip(positions, platform_headings_deg):
        intercepts = sensor.los_ground_intercepts(
            pos["latitude"],
            pos["longitude"],
            float(platform_heading),
            altitude_guess,
        )
        dem_lats.extend([intercepts["los1"]["latitude"], intercepts["los2"]["latitude"]])
        dem_lons.extend([intercepts["los1"]["longitude"], intercepts["los2"]["longitude"]])
    return generate_demfile(np.asarray(dem_lats, dtype=float), np.asarray(dem_lons, dtype=float))


def _terrain_aware_profiles_for_geometry(
    geometry: LineString,
    *,
    altitude_msl: Quantity,
    altitude_msl_ft: float,
    source_segment_index: int | None,
    source_segment_name: str | None,
    source_segment_type: str | None,
    sensor: AerosolWindProfiler,
    ground_speed: Quantity,
    start_time: _dt.datetime | None,
    dwell_time_per_los: Quantity | None,
    nadir_dwell_time: Quantity | None,
    dem_file: str | None,
    terrain_precision: Quantity,
    stable_platform_ok: bool,
    crab_angle_deg: float | None = None,
    heading_deg: float | None = None,
) -> gpd.GeoDataFrame:
    sample_distances, interpolated, spacing_m, offset_m, profile_spacing_s = _profile_sampling(
        geometry, sensor, ground_speed, dwell_time_per_los, nadir_dwell_time
    )
    if len(sample_distances) == 0:
        return _empty_profile_gdf()

    track_headings = np.asarray([pos["heading"] for pos in interpolated], dtype=float)
    platform_headings = _resolve_platform_headings(
        track_headings,
        crab_angle_deg=crab_angle_deg,
        heading_deg=heading_deg,
    )

    if dem_file is None:
        dem_file = _terrain_dem_for_awp_profiles(
            interpolated,
            sensor,
            altitude_msl,
            platform_headings_deg=platform_headings,
        )

    platform_lats = np.asarray([pos["latitude"] for pos in interpolated], dtype=float)
    platform_lons = np.asarray([pos["longitude"] for pos in interpolated], dtype=float)
    terrain_elevation_m = get_elevations(platform_lats, platform_lons, dem_file).astype(float)

    altitude_msl_m = altitude_msl.m_as("meter")
    altitude_agl_m = altitude_msl_m - terrain_elevation_m
    if np.any(altitude_agl_m <= 0):
        raise HyPlanValueError(
            "terrain_aware=True requires platform altitude MSL to remain above terrain "
            "along the profile locations."
        )

    los_azimuths = np.asarray(
        [sensor.los_orientations(float(heading)) for heading in platform_headings],
        dtype=float,
    )
    # ray_terrain_intersection expects tilt as off-nadir angle from vertical
    # (0 deg = nadir, 90 deg = horizontal), which matches the AWP instrument model.
    depression_angle = np.full(len(platform_lats), sensor.off_nadir_angle, dtype=float)
    precision_m = terrain_precision.m_as("meter")

    los1_lat, los1_lon, los1_alt = ray_terrain_intersection(
        platform_lats,
        platform_lons,
        altitude_msl_m,
        los_azimuths[:, 0],
        depression_angle,
        precision=precision_m,
        dem_file=dem_file,
    )
    los2_lat, los2_lon, los2_alt = ray_terrain_intersection(
        platform_lats,
        platform_lons,
        altitude_msl_m,
        los_azimuths[:, 1],
        depression_angle,
        precision=precision_m,
        dem_file=dem_file,
    )

    los_surface_separation_m = np.full(len(platform_lats), np.nan, dtype=float)
    valid_pairs = (
        np.isfinite(los1_lat)
        & np.isfinite(los1_lon)
        & np.isfinite(los2_lat)
        & np.isfinite(los2_lon)
    )
    for idx in np.where(valid_pairs)[0]:
        separation_m, _ = pymap3d.vincenty.vdist(
            float(los1_lat[idx]),
            float(los1_lon[idx]),
            float(los2_lat[idx]),
            float(los2_lon[idx]),
        )
        los_surface_separation_m[idx] = float(separation_m)

    records = []
    for sample_index, distance_m in enumerate(sample_distances, start=1):
        elapsed_s = float(distance_m) / ground_speed.m_as("meter / second")
        time_utc = start_time + _dt.timedelta(seconds=float(elapsed_s)) if start_time else pd.NaT
        row = sample_index - 1
        records.append(
            {
                "sample_index": sample_index,
                "source_segment_index": source_segment_index,
                "source_segment_name": source_segment_name,
                "source_segment_type": source_segment_type,
                "distance_from_start_m": float(distance_m),
                "elapsed_time_s": float(elapsed_s),
                "time_utc": time_utc,
                "platform_lat": float(platform_lats[row]),
                "platform_lon": float(platform_lons[row]),
                "platform_heading_deg": float(platform_headings[row]),
                "terrain_elevation_m": float(terrain_elevation_m[row]),
                "altitude_agl_m": float(altitude_agl_m[row]),
                "altitude_msl_ft": float(altitude_msl_ft),
                "profile_spacing_m": float(spacing_m),
                "profile_spacing_s": float(profile_spacing_s),
                "profile_assignment_offset_m": float(offset_m),
                "los_surface_separation_m": float(los_surface_separation_m[row]),
                "stable_platform_ok": bool(stable_platform_ok),
                "los1_azimuth_deg": float(los_azimuths[row, 0]),
                "los1_lat": float(los1_lat[row]),
                "los1_lon": float(wrap_to_180(los1_lon[row])) if np.isfinite(los1_lon[row]) else np.nan,
                "los1_alt_m": float(los1_alt[row]) if np.isfinite(los1_alt[row]) else np.nan,
                "los2_azimuth_deg": float(los_azimuths[row, 1]),
                "los2_lat": float(los2_lat[row]),
                "los2_lon": float(wrap_to_180(los2_lon[row])) if np.isfinite(los2_lon[row]) else np.nan,
                "los2_alt_m": float(los2_alt[row]) if np.isfinite(los2_alt[row]) else np.nan,
                "geometry": Point(float(platform_lons[row]), float(platform_lats[row])),
            }
        )

    return gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry="geometry", crs="EPSG:4326")


def _terrain_aware_profiles_for_flight_line(
    flight_line: FlightLine,
    *,
    sensor: AerosolWindProfiler,
    ground_speed: Quantity,
    start_time: _dt.datetime | None,
    dwell_time_per_los: Quantity | None,
    nadir_dwell_time: Quantity | None,
    dem_file: str | None,
    terrain_precision: Quantity,
) -> gpd.GeoDataFrame:
    return _terrain_aware_profiles_for_geometry(
        flight_line.geometry,
        altitude_msl=flight_line.altitude_msl,
        altitude_msl_ft=flight_line.altitude_msl.m_as("foot"),
        source_segment_index=None,
        source_segment_name=flight_line.site_name,
        source_segment_type="flight_line",
        sensor=sensor,
        ground_speed=ground_speed,
        start_time=start_time,
        dwell_time_per_los=dwell_time_per_los,
        nadir_dwell_time=nadir_dwell_time,
        dem_file=dem_file,
        terrain_precision=terrain_precision,
        stable_platform_ok=True,
    )


def flag_awp_stable_segments(
    plan: gpd.GeoDataFrame,
    sensor: AerosolWindProfiler | None = None,
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
    sensor: AerosolWindProfiler,
    ground_speed: Quantity,
    start_time: _dt.datetime | None,
    dwell_time_per_los: Quantity | None,
    nadir_dwell_time: Quantity | None,
    stable_platform_ok: bool,
    crab_angle_deg: float | None = None,
    heading_deg: float | None = None,
) -> gpd.GeoDataFrame:
    sample_distances, interpolated, spacing_m, offset_m, profile_spacing_s = _profile_sampling(
        geometry,
        sensor,
        ground_speed,
        dwell_time_per_los,
        nadir_dwell_time,
    )
    if len(sample_distances) == 0:
        return _empty_profile_gdf()

    track_headings = np.asarray([pos["heading"] for pos in interpolated], dtype=float)
    platform_headings = _resolve_platform_headings(
        track_headings,
        crab_angle_deg=crab_angle_deg,
        heading_deg=heading_deg,
    )

    records = []
    for sample_index, (distance_m, pos, platform_heading) in enumerate(
        zip(sample_distances, interpolated, platform_headings),
        start=1,
    ):
        intercepts = sensor.los_ground_intercepts(
            pos["latitude"],
            pos["longitude"],
            float(platform_heading),
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
                "platform_heading_deg": float(platform_heading),
                "terrain_elevation_m": np.nan,
                "altitude_agl_m": altitude_agl.m_as("meter"),
                "altitude_msl_ft": float(altitude_msl_ft),
                "profile_spacing_m": spacing_m,
                "profile_spacing_s": profile_spacing_s,
                "profile_assignment_offset_m": offset_m,
                "los_surface_separation_m": sensor.los_surface_separation(altitude_agl).m_as("meter"),
                "stable_platform_ok": bool(stable_platform_ok),
                "los1_azimuth_deg": intercepts["los1"]["azimuth"],
                "los1_lat": intercepts["los1"]["latitude"],
                "los1_lon": intercepts["los1"]["longitude"],
                "los1_alt_m": np.nan,
                "los2_azimuth_deg": intercepts["los2"]["azimuth"],
                "los2_lat": intercepts["los2"]["latitude"],
                "los2_lon": intercepts["los2"]["longitude"],
                "los2_alt_m": np.nan,
                "geometry": Point(pos["longitude"], pos["latitude"]),
            }
        )

    return gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry="geometry", crs="EPSG:4326")


def awp_profile_locations_for_flight_line(
    flight_line: FlightLine,
    *,
    sensor: AerosolWindProfiler | None = None,
    ground_speed: Quantity | None = None,
    start_time: _dt.datetime | None = None,
    altitude_agl: Quantity | None = None,
    dwell_time_per_los: Quantity | None = None,
    nadir_dwell_time: Quantity | None = None,
    terrain_aware: bool = False,
    dem_file: str | None = None,
    terrain_precision: Quantity | float = 30.0,
) -> gpd.GeoDataFrame:
    """Predict AWP vector-profile sample locations along a single flight line.

    The spacing and LOS geometry follow the AWP observing concept summarized by
    Bedka et al. (2024) and Bedka (2025). When ``terrain_aware=True`` the
    LOS intercepts are computed by ray-terrain intersection against a DEM and
    the returned ``altitude_agl_m`` varies with local terrain beneath the line.
    """
    if not isinstance(flight_line, FlightLine):
        raise HyPlanTypeError("flight_line must be a FlightLine")

    awp = _as_awp(sensor)
    speed = _as_quantity(
        ground_speed if ground_speed is not None else awp.nominal_airspeed,
        "meter / second",
        "ground_speed",
    )
    if terrain_aware and altitude_agl is not None:
        raise HyPlanValueError(
            "terrain_aware=True derives altitude AGL from flight_line.altitude_msl and DEM terrain; "
            "do not also pass altitude_agl."
        )
    if terrain_aware:
        return _terrain_aware_profiles_for_flight_line(
            flight_line,
            sensor=awp,
            ground_speed=speed,
            start_time=start_time,
            dwell_time_per_los=dwell_time_per_los,
            nadir_dwell_time=nadir_dwell_time,
            dem_file=dem_file,
            terrain_precision=_as_quantity(terrain_precision, "meter", "terrain_precision"),
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
    sensor: AerosolWindProfiler | None = None,
    takeoff_time: _dt.datetime | None = None,
    dwell_time_per_los: Quantity | None = None,
    nadir_dwell_time: Quantity | None = None,
    stable_only: bool = True,
    terrain_aware: bool = False,
    dem_file: str | None = None,
    terrain_precision: Quantity | float = 30.0,
    wind_aware: bool = False,
) -> gpd.GeoDataFrame:
    """Predict AWP vector-profile locations along stable segments of a plan.

    This combines the AWP LOS geometry from Bedka et al. (2024) with the
    stable-flight planning heuristics summarized in Bedka (2025). When
    ``terrain_aware=True``, LOS intercepts are ray-traced into terrain using
    a DEM. When ``wind_aware=True``, aircraft heading uses plan crab-angle
    metadata (``crab_angle_deg`` or ``wind_corrected_heading``) when present,
    so the LOS geometry follows the crabbed aircraft heading rather than the
    nominal ground track.
    """
    flagged = flag_awp_stable_segments(plan, sensor=sensor)
    awp = _as_awp(sensor)
    terrain_precision_q = _as_quantity(terrain_precision, "meter", "terrain_precision")

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
                crab_angle_deg = None
                heading_deg = None
                if wind_aware:
                    crab_angle = row.get("crab_angle_deg")
                    heading = row.get("wind_corrected_heading")
                    if pd.notna(crab_angle):
                        crab_angle_deg = float(crab_angle)
                    elif pd.notna(heading):
                        heading_deg = float(heading)

                common_kwargs = {
                    "sensor": awp,
                    "ground_speed": speed.to("meter / second"),
                    "start_time": seg_start_time,
                    "dwell_time_per_los": dwell_time_per_los,
                    "nadir_dwell_time": nadir_dwell_time,
                    "stable_platform_ok": stable,
                    "crab_angle_deg": crab_angle_deg,
                    "heading_deg": heading_deg,
                }
                if terrain_aware:
                    gdf = _terrain_aware_profiles_for_geometry(
                        geom,
                        altitude_msl=ureg.Quantity(alt_ft, "foot"),
                        altitude_msl_ft=alt_ft,
                        source_segment_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        source_segment_name=row.get("segment_name"),
                        source_segment_type=row.get("segment_type"),
                        dem_file=dem_file,
                        terrain_precision=terrain_precision_q,
                        **common_kwargs,
                    )
                else:
                    gdf = _profiles_for_geometry(
                        geom,
                        altitude_agl=altitude_agl,
                        altitude_msl_ft=alt_ft,
                        source_segment_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        source_segment_name=row.get("segment_name"),
                        source_segment_type=row.get("segment_type"),
                        **common_kwargs,
                    )
                if not gdf.empty:
                    all_profiles.append(gdf)

        cumulative_seconds += max(seg_duration_s, 0.0)

    if not all_profiles:
        return _empty_profile_gdf()
    return gpd.GeoDataFrame(pd.concat(all_profiles, ignore_index=True), geometry="geometry", crs="EPSG:4326")
