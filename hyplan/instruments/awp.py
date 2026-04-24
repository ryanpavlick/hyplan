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

from typing import Iterable, Tuple

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..geometry import wrap_to_180, wrap_to_360
from ..units import ureg
from ._base import Sensor

__all__ = [
    "ProfilingLidar",
    "AerosolWindProfiler",
]


_LIGHT_SPEED_MPS = 299_792_458.0


def _as_quantity(value, unit: str, label: str) -> Quantity:
    """Normalize *value* to a quantity in *unit*."""
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(f"{label} must be numeric or a pint.Quantity, got {type(value)}")


class ProfilingLidar(Sensor):
    """Base class for airborne profiling lidars.

    Profiling lidars differ from continuous-swath sensors in HyPlan. They
    sample discrete atmospheric columns along the flight path, often by
    alternating between multiple line-of-sight (LOS) directions to recover a
    vector wind profile. This base class captures the planning geometry needed
    to answer questions like:

    - How far apart are valid profile retrievals along track?
    - Where do the LOS footprints intersect the surface?
    - How much straight-and-level flight is needed before vector retrievals
      become valid?

    The planning abstractions in this class are aligned with the AWP airborne
    observing concept described by Bedka et al. (2024) and Bedka (2025).
    """

    def __init__(
        self,
        name: str,
        *,
        wavelength: Quantity,
        off_nadir_angle: float,
        los_azimuths_relative: Iterable[float],
        pulse_rate: Quantity,
        digitization_rate: Quantity,
        samples_per_pulse: int,
        dwell_time_per_los: Quantity,
        nadir_dwell_time: Quantity,
        blind_zone: Quantity,
        max_roll_angle: Quantity,
        max_heading_change: Quantity,
        max_altitude_change: Quantity,
        nominal_airspeed: Quantity,
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


class AerosolWindProfiler(ProfilingLidar):
    """NASA Langley Aerosol Wind Profiler (AWP).

    Defaults are drawn from Bedka et al. (2024) and Bedka (2025):

    - wavelength: 2052.92 nm
    - 30° off nadir
    - dual LOS at ±45° azimuth from the aircraft nose
    - 200 Hz pulse rate
    - 500 MHz digitization rate
    - 65528 samples per pulse
    - common airborne mode of 3 s per LOS + 1 s nadir

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

    def __init__(self):
        super().__init__(
            name="Aerosol Wind Profiler",
            wavelength=2052.92 * ureg.nanometer,
            off_nadir_angle=30.0,
            los_azimuths_relative=(-45.0, 45.0),
            pulse_rate=200 * ureg.Hz,
            digitization_rate=500e6 * ureg.Hz,
            samples_per_pulse=65528,
            dwell_time_per_los=3 * ureg.second,
            nadir_dwell_time=1 * ureg.second,
            blind_zone=200 * ureg.meter,
            max_roll_angle=3 * ureg.degree,
            max_heading_change=3 * ureg.degree,
            max_altitude_change=0.5 * ureg.kilometer,
            nominal_airspeed=225 * ureg.meter / ureg.second,
        )
