"""Aircraft performance model.

Phase-aware performance model driven by CAS/Mach speed schedules (jets) or
TAS-vs-altitude schedules (turboprops).  Each aircraft carries three
performance profiles — speed schedule, vertical performance, and turn model —
plus provenance/confidence metadata.

.. rubric:: Speed schedule types

``CasMachSchedule``
    For jets: CAS below crossover altitude, Mach above.  Requires the
    :mod:`hyplan.atmosphere` ISA model for CAS↔TAS and Mach↔TAS conversion.

``TasSchedule``
    For turboprops and simple models: piecewise-linear TAS vs altitude.

Both types expose a common ``tas_at(altitude)`` method so that callers
(``compute_flight_plan``, ``flight_optimizer``) never need to know which
schedule type is in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from .airports import Airport
from .atmosphere import cas_to_tas, mach_to_tas
from .dubins3d import DubinsPath3D
from .exceptions import HyPlanTypeError, HyPlanValueError
from .units import ureg
from .waypoint import Waypoint


# ---------------------------------------------------------------------------
# Provenance types
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    """Where a performance parameter came from.

    Args:
        source_type: One of ``"poh"``, ``"afm"``, ``"brochure"``,
            ``"adsb"``, ``"mission_log"``, ``"expert"``, ``"derived"``.
        reference: Free-text citation.
        notes: Additional context.
        confidence: 0.0 (no confidence) to 1.0 (fully validated).
    """

    source_type: str
    reference: str
    notes: str = ""
    confidence: float = 0.5


@dataclass
class PerformanceConfidence:
    """Per-submodel confidence ratings (0–1)."""

    climb: float = 0.5
    cruise: float = 0.5
    descent: float = 0.5
    turns: float = 0.5


# ---------------------------------------------------------------------------
# Speed schedule types
# ---------------------------------------------------------------------------

@dataclass
class CasMachSchedule:
    """Jet speed schedule: CAS below crossover, Mach above.

    At altitudes below ``crossover_ft`` the schedule flies a constant
    calibrated airspeed (CAS); above crossover it flies a constant Mach
    number.  The true airspeed (TAS) is derived via the ISA atmosphere model.

    Args:
        cas: Calibrated airspeed target (knots).
        mach: Cruise Mach number.
        crossover_ft: Altitude (feet) where CAS and Mach targets produce
            equal TAS under ISA.
    """

    cas: Quantity
    mach: float
    crossover_ft: float

    def __post_init__(self) -> None:
        self.cas = self.cas.to(ureg.knot)

    def tas_at(self, altitude: Quantity) -> Quantity:
        """True airspeed at *altitude* under ISA."""
        alt_ft = altitude.m_as(ureg.feet)
        if alt_ft < self.crossover_ft:
            return cas_to_tas(self.cas, altitude)
        return mach_to_tas(self.mach, altitude)


@dataclass
class TasSchedule:
    """Piecewise-linear TAS-vs-altitude schedule.

    Works with any number of breakpoints:

    * 1 point → constant TAS everywhere.
    * 2+ points → linear interpolation, clamped at endpoints.

    Args:
        points: List of ``(altitude, tas)`` tuples, both
            :class:`pint.Quantity`.  Altitudes must be ascending when N ≥ 2.
    """

    points: List[Tuple[Quantity, Quantity]]

    def __post_init__(self) -> None:
        if len(self.points) < 1:
            raise HyPlanValueError("TasSchedule requires at least 1 point.")
        self._alts_ft = np.array(
            [alt.m_as(ureg.feet) for alt, _ in self.points], dtype=float
        )
        self._tas_kt = np.array(
            [spd.m_as(ureg.knot) for _, spd in self.points], dtype=float
        )
        if len(self.points) >= 2 and not np.all(np.diff(self._alts_ft) > 0):
            raise HyPlanValueError(
                "TasSchedule altitudes must be strictly ascending."
            )

    def tas_at(self, altitude: Quantity) -> Quantity:
        """Interpolated TAS at *altitude*.  Clamps at endpoints."""
        alt_ft = altitude.m_as(ureg.feet)
        return float(np.interp(alt_ft, self._alts_ft, self._tas_kt)) * ureg.knot


# Union of both schedule types — used as a type hint on Aircraft fields.
SpeedSchedule = Union[CasMachSchedule, TasSchedule]


# ---------------------------------------------------------------------------
# Vertical performance
# ---------------------------------------------------------------------------

@dataclass
class VerticalProfile:
    """Altitude-indexed vertical rate (rate of climb or rate of descent).

    Mode is auto-detected from the number of breakpoints:

    * 1 → ``"constant"`` (single rate everywhere)
    * 2 → ``"two_point"`` (linear interpolation; enables analytical
      closed-form climb integration)
    * 3+ → ``"full"`` (piecewise-linear; numerical trapezoidal integration)

    Args:
        points: List of ``(altitude, rate)`` tuples, both
            :class:`pint.Quantity`.  Rate should be positive (feet/minute).
        source: Free-text citation for traceability.
    """

    points: List[Tuple[Quantity, Quantity]]
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.points) < 1:
            raise HyPlanValueError("VerticalProfile requires at least 1 point.")
        self._alts_ft = np.array(
            [alt.m_as(ureg.feet) for alt, _ in self.points], dtype=float
        )
        self._rates_fpm = np.array(
            [rate.m_as(ureg.feet / ureg.minute) for _, rate in self.points],
            dtype=float,
        )
        if len(self.points) >= 2 and not np.all(np.diff(self._alts_ft) > 0):
            raise HyPlanValueError(
                "VerticalProfile altitudes must be strictly ascending."
            )

        n = len(self.points)
        if n == 1:
            self._mode = "constant"
        elif n == 2:
            self._mode = "two_point"
        else:
            self._mode = "full"

    def rate_at(self, altitude: Quantity) -> Quantity:
        """Interpolated vertical rate at *altitude*.  Clamps at endpoints."""
        alt_ft = altitude.m_as(ureg.feet)
        fpm = float(np.interp(alt_ft, self._alts_ft, self._rates_fpm))
        return fpm * ureg.feet / ureg.minute

    @property
    def sea_level_rate(self) -> Quantity:
        """Rate at the lowest altitude breakpoint (first row)."""
        return self._rates_fpm[0] * ureg.feet / ureg.minute

    @property
    def ceiling_rate(self) -> Quantity:
        """Rate at the highest altitude breakpoint (last row)."""
        return self._rates_fpm[-1] * ureg.feet / ureg.minute


# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------

@dataclass
class PhaseBankAngles:
    """Bank angle limits by flight phase (degrees)."""

    climb_deg: float = 20.0
    cruise_deg: float = 25.0
    descent_deg: float = 20.0
    approach_deg: float = 15.0


@dataclass
class TurnModel:
    """Turn performance model with phase-specific bank angles.

    Args:
        bank_by_phase: Per-phase bank angle limits.
        max_bank_deg: Absolute maximum bank angle (degrees).
    """

    bank_by_phase: PhaseBankAngles = field(default_factory=PhaseBankAngles)
    max_bank_deg: float = 30.0


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------

class Aircraft:
    """Aircraft performance model.

    Holds identity, geometric constraints, phase-specific speed schedules,
    vertical performance profiles, turn model, and provenance metadata.

    Args:
        aircraft_type: Aircraft model name (e.g. ``"Gulfstream V"``).
        tail_number: Tail number or ``"Unknown"``.
        operator: Operating organization.
        service_ceiling: Maximum operational altitude.
        approach_speed: Landing approach speed.
        climb_schedule: Speed schedule for climb phase.
        cruise_schedule: Speed schedule for cruise phase.
        descent_schedule: Speed schedule for descent phase.
        climb_profile: Rate-of-climb vs altitude.
        descent_profile: Rate-of-descent vs altitude.
        turn_model: Turn performance / bank angles.
        confidence: Per-submodel confidence ratings.
        sources: List of provenance records.
        range: Maximum flight range (optional, metadata only).
        endurance: Maximum flight duration (optional, metadata only).
        useful_payload: Payload capacity (optional, metadata only).
    """

    def __init__(
        self,
        aircraft_type: str,
        tail_number: str,
        operator: str,
        service_ceiling: Quantity,
        approach_speed: Quantity,
        climb_schedule: SpeedSchedule,
        cruise_schedule: SpeedSchedule,
        descent_schedule: SpeedSchedule,
        climb_profile: VerticalProfile,
        descent_profile: VerticalProfile,
        turn_model: TurnModel,
        confidence: Optional[PerformanceConfidence] = None,
        sources: Optional[List[SourceRecord]] = None,
        range: Optional[Quantity] = None,
        endurance: Optional[Quantity] = None,
        useful_payload: Optional[Quantity] = None,
    ):
        if not isinstance(aircraft_type, str):
            raise HyPlanTypeError("Aircraft type must be a string.")
        if not isinstance(tail_number, str):
            raise HyPlanTypeError("Tail number must be a string.")
        if not isinstance(operator, str):
            raise HyPlanTypeError("Operator must be a string.")

        self.aircraft_type = aircraft_type
        self.tail_number = tail_number
        self.operator = operator
        self.service_ceiling = service_ceiling.to(ureg.feet)
        self.approach_speed = approach_speed.to(ureg.knot)

        self.climb_schedule = climb_schedule
        self.cruise_schedule = cruise_schedule
        self.descent_schedule = descent_schedule

        self.climb_profile = climb_profile
        self.descent_profile = descent_profile

        self.turn_model = turn_model
        self.confidence = confidence or PerformanceConfidence()
        self.sources = sources or []

        self.range = range.to(ureg.nautical_mile) if range is not None else None
        self.endurance = endurance.to(ureg.hour) if endurance is not None else None
        self.useful_payload = (
            useful_payload.to(ureg.pound) if useful_payload is not None else None
        )

    # ------------------------------------------------------------------
    # Public API (preserved for flight_plan.py / flight_optimizer.py)
    # ------------------------------------------------------------------

    @property
    def max_bank_angle(self) -> float:
        """Maximum bank angle in degrees (for Dubins path geometry)."""
        return self.turn_model.max_bank_deg

    def cruise_speed_at(self, altitude: Quantity) -> Quantity:
        """True airspeed at *altitude* using the cruise speed schedule."""
        return self.cruise_schedule.tas_at(altitude)

    def rate_of_climb(self, altitude: Quantity) -> Quantity:
        """Rate of climb at *altitude* from the climb profile."""
        return self.climb_profile.rate_at(altitude)

    def descent_speed_at(self, altitude: Quantity) -> Quantity:
        """True airspeed during descent at *altitude*."""
        return self.descent_schedule.tas_at(altitude)

    def pitch_limits(self, speed: Optional[Quantity] = None) -> tuple:
        """Derive pitch-angle limits from climb/descent rates and TAS.

        Returns ``(pitch_min, pitch_max)`` in degrees.  ``pitch_min`` is
        negative (descent), ``pitch_max`` is positive (climb).
        """
        tas = speed if speed is not None else self.cruise_speed_at(self.service_ceiling)
        tas_mps = tas.m_as(ureg.meter / ureg.second)

        climb_rate = self.climb_profile.sea_level_rate
        descent_rate = self.descent_profile.sea_level_rate

        climb_mps = climb_rate.m_as(ureg.meter / ureg.minute) / 60.0
        descent_mps = descent_rate.m_as(ureg.meter / ureg.minute) / 60.0

        pitch_max = float(np.degrees(np.arctan(climb_mps / tas_mps)))
        pitch_min = -float(np.degrees(np.arctan(descent_mps / tas_mps)))
        return pitch_min, pitch_max

    # ------------------------------------------------------------------
    # Climb
    # ------------------------------------------------------------------

    def _climb(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """Estimate time and horizontal distance during a climb.

        Dispatch strategy depends on the climb profile mode:

        * ``"constant"`` — single ROC, simple division.
        * ``"two_point"`` — analytical log formula (linear ROC model).
        * ``"full"`` — numerical trapezoidal integration.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)

        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.cruise_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)

        if end_altitude > self.service_ceiling:
            raise HyPlanValueError("End altitude cannot exceed the service ceiling.")
        if end_altitude <= start_altitude:
            return 0 * ureg.minute, 0 * ureg.nautical_mile

        roc_start = self.climb_profile.rate_at(start_altitude)
        roc_end = self.climb_profile.rate_at(end_altitude)
        mode = self.climb_profile._mode

        if mode == "constant":
            roc = self.climb_profile.sea_level_rate
            time_to_climb = ((end_altitude - start_altitude) / roc).to(ureg.minute)

        elif mode == "two_point":
            roc_sl = self.climb_profile.sea_level_rate
            roc_ceil = self.climb_profile.ceiling_rate
            delta_roc = (roc_sl - roc_ceil).magnitude

            if delta_roc < 1e-6:
                time_to_climb = ((end_altitude - start_altitude) / roc_sl).to(
                    ureg.minute
                )
            else:
                C = self.service_ceiling
                time_to_climb = (
                    C / (roc_sl - roc_ceil) * np.log(roc_start / roc_end)
                ).to(ureg.minute)

        else:  # "full"
            n_steps = 64
            alts_ft = np.linspace(
                start_altitude.magnitude, end_altitude.magnitude, n_steps + 1
            )
            rocs_fpm = np.array([
                self.climb_profile.rate_at(
                    ureg.Quantity(a, "feet")
                ).m_as(ureg.feet / ureg.minute)
                for a in alts_ft
            ])
            minutes = float(np.trapezoid(1.0 / rocs_fpm, alts_ft))
            time_to_climb = minutes * ureg.minute

        # Horizontal distance using average climb angle
        avg_roc = (roc_start + roc_end) / 2
        climb_angle = np.arctan(avg_roc / true_air_speed).to(ureg.radian)
        horizontal_speed = (true_air_speed * np.cos(climb_angle)).to(
            ureg.nautical_mile / ureg.hour
        )
        horizontal_distance = (horizontal_speed * time_to_climb).to(
            ureg.nautical_mile
        )

        return time_to_climb, horizontal_distance

    def climb_altitude_profile(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        n_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate altitude-vs-time curve during a climb.

        Returns ``(times, altitudes)`` as numpy arrays in minutes and feet.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)

        if end_altitude <= start_altitude:
            return np.array([0.0]), np.array([start_altitude.magnitude])

        mode = self.climb_profile._mode

        if mode == "full" or mode == "constant" and len(self.climb_profile.points) > 2:
            # Numerical integration for multi-point or full profiles
            altitudes = np.linspace(
                start_altitude.magnitude, end_altitude.magnitude, n_points
            )
            rocs = np.array([
                self.climb_profile.rate_at(
                    ureg.Quantity(a, "feet")
                ).m_as(ureg.feet / ureg.minute)
                for a in altitudes
            ])
            inv_roc = 1.0 / rocs
            dh = np.diff(altitudes)
            seg = 0.5 * (inv_roc[:-1] + inv_roc[1:]) * dh
            times = np.concatenate([[0.0], np.cumsum(seg)])
            return times, altitudes

        if mode == "constant":
            roc_sl = self.climb_profile.sea_level_rate.m_as(
                ureg.feet / ureg.minute
            )
            total_time = (end_altitude - start_altitude).magnitude / roc_sl
            times = np.linspace(0, total_time, n_points)
            altitudes = start_altitude.magnitude + roc_sl * times
            return times, altitudes

        # two_point: analytical exponential profile
        h0 = start_altitude.magnitude
        C = self.service_ceiling.magnitude
        roc_sl = self.climb_profile.sea_level_rate.m_as(ureg.feet / ureg.minute)
        roc_ceil = self.climb_profile.ceiling_rate.m_as(ureg.feet / ureg.minute)
        delta_roc = roc_sl - roc_ceil

        if delta_roc < 1e-6:
            total_time = (end_altitude - start_altitude).magnitude / roc_sl
            times = np.linspace(0, total_time, n_points)
            altitudes = h0 + roc_sl * times
        else:
            alpha = delta_roc / C
            h_eq = C * roc_sl / delta_roc
            roc_h0 = self.climb_profile.rate_at(start_altitude).m_as(
                ureg.feet / ureg.minute
            )
            roc_h1 = self.climb_profile.rate_at(end_altitude).m_as(
                ureg.feet / ureg.minute
            )
            total_time = (1 / alpha) * np.log(roc_h0 / roc_h1)
            times = np.linspace(0, total_time, n_points)
            altitudes = h_eq - (h_eq - h0) * np.exp(-alpha * times)

        return times, altitudes

    # ------------------------------------------------------------------
    # Descent
    # ------------------------------------------------------------------

    def _descend(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """Estimate time and horizontal distance during descent.

        Uses the descent profile (altitude-indexed ROD).  Integration
        strategy matches the climb profile mode.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)

        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.descent_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)

        if start_altitude <= end_altitude:
            return 0 * ureg.minute, 0 * ureg.nautical_mile

        altitude_difference = start_altitude - end_altitude
        mode = self.descent_profile._mode

        if mode == "constant":
            rod = self.descent_profile.sea_level_rate
            time_to_descend = (altitude_difference / rod).to(ureg.minute)
        else:
            # Numerical integration for two_point and full modes
            n_steps = 64
            # Note: descending, so we integrate from high to low
            alts_ft = np.linspace(
                end_altitude.magnitude, start_altitude.magnitude, n_steps + 1
            )
            rods_fpm = np.array([
                self.descent_profile.rate_at(
                    ureg.Quantity(a, "feet")
                ).m_as(ureg.feet / ureg.minute)
                for a in alts_ft
            ])
            minutes = float(np.trapezoid(1.0 / rods_fpm, alts_ft))
            time_to_descend = minutes * ureg.minute

        descent_rate_avg = self.descent_profile.rate_at(
            (start_altitude + end_altitude) / 2
        )
        descent_angle = np.arctan(descent_rate_avg / true_air_speed).to(ureg.radian)
        horizontal_speed = (true_air_speed * np.cos(descent_angle)).to(
            ureg.nautical_mile / ureg.hour
        )
        horizontal_distance = (horizontal_speed * time_to_descend).to(
            ureg.nautical_mile
        )

        return time_to_descend, horizontal_distance

    # ------------------------------------------------------------------
    # 3D path planning
    # ------------------------------------------------------------------

    def time_to_takeoff(
        self,
        airport: Airport,
        waypoint: Waypoint,
        wind: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Calculate time from takeoff to the first waypoint.

        Uses 3D Dubins path planning for the departure including climb.
        """
        _, departure_heading = pymap3d.vincenty.vdist(
            airport.latitude, airport.longitude,
            waypoint.latitude, waypoint.longitude,
        )
        airport_waypoint = Waypoint(
            latitude=airport.latitude,
            longitude=airport.longitude,
            heading=departure_heading,
            altitude_msl=airport.elevation,
        )
        return self.time_to_cruise(airport_waypoint, waypoint, wind=wind)

    def time_to_return(
        self,
        waypoint: Waypoint,
        airport: Airport,
        wind: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Calculate time from the last waypoint back to the airport.

        Uses 3D Dubins path planning for the return including descent.
        """
        _, arrival_heading = pymap3d.vincenty.vdist(
            waypoint.latitude, waypoint.longitude,
            airport.latitude, airport.longitude,
        )
        airport_waypoint = Waypoint(
            latitude=airport.latitude,
            longitude=airport.longitude,
            heading=(arrival_heading + 180.0) % 360.0,
            altitude_msl=airport.elevation,
        )
        return self.time_to_cruise(waypoint, airport_waypoint, wind=wind)

    def time_to_cruise(
        self,
        start_waypoint: Waypoint,
        end_waypoint: Waypoint,
        true_air_speed: Optional[Quantity] = None,
        wind: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Calculate time to fly between two waypoints.

        Uses 3D Dubins path planning with pitch constraints derived from
        climb/descent performance.  Returns a dict with ``total_time``,
        ``phases``, and ``dubins_path``.

        Args:
            wind: Optional ``(u_east, v_north)`` wind vector in m/s.
                When provided, horizontal turning arcs become trochoids
                and the returned path length / timing account for wind.
        """
        true_air_speed = true_air_speed or self.cruise_speed_at(
            end_waypoint.altitude_msl
        )

        start_altitude = start_waypoint.altitude_msl.to(ureg.feet)
        end_altitude = end_waypoint.altitude_msl.to(ureg.feet)

        pitch_min, pitch_max = self.pitch_limits(true_air_speed)

        path = DubinsPath3D(
            start=start_waypoint,
            end=end_waypoint,
            speed=true_air_speed,
            bank_angle=self.max_bank_angle,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            wind=wind,
        )

        distance = path.length.to(ureg.nautical_mile)
        total_time = (distance / true_air_speed).to(ureg.minute)

        phases = self._phases_from_3d_path(
            path, true_air_speed, start_altitude, end_altitude, total_time,
        )

        return {
            "total_time": total_time,
            "phases": phases,
            "dubins_path": path,
        }

    def _phases_from_3d_path(
        self,
        path3d,
        true_air_speed: Quantity,
        start_altitude: Quantity,
        end_altitude: Quantity,
        total_time: Quantity,
    ) -> dict:
        """Split a 3D Dubins path into climb/cruise/descent phases."""
        pts = path3d.points  # (N, 5): lat, lon, alt_m, heading, pitch
        alts_m = pts[:, 2]
        n = len(alts_m)

        if n < 2:
            return {
                "cruise": {
                    "start_altitude": start_altitude,
                    "end_altitude": end_altitude,
                    "start_time": 0 * ureg.minute,
                    "end_time": total_time,
                    "distance": path3d.length.to(ureg.nautical_mile),
                }
            }

        pitches = pts[:, 4]
        seg_pitches = (pitches[:-1] + pitches[1:]) / 2.0
        pitch_threshold_deg = 0.001
        segment_types = np.where(
            seg_pitches > pitch_threshold_deg,
            1,
            np.where(seg_pitches < -pitch_threshold_deg, -1, 0),
        )

        phases = {}
        i = 0
        phase_idx = 0

        while i < len(segment_types):
            seg_type = segment_types[i]
            j = i
            while j < len(segment_types) and segment_types[j] == seg_type:
                j += 1

            frac_start = i / (n - 1)
            frac_end = j / (n - 1)
            phase_time_start = total_time * frac_start
            phase_time_end = total_time * frac_end
            phase_distance = path3d.length.to(ureg.nautical_mile) * (
                frac_end - frac_start
            )

            phase_start_alt = ureg.Quantity(float(alts_m[i]), "meter").to(ureg.feet)
            phase_end_alt = ureg.Quantity(
                float(alts_m[min(j, n - 1)]), "meter"
            ).to(ureg.feet)

            if seg_type == 1:
                label = "cruise_climb"
            elif seg_type == -1:
                label = "cruise_descent"
            else:
                label = "cruise"

            key = label if label not in phases else f"{label}_{phase_idx}"

            phases[key] = {
                "start_altitude": phase_start_alt,
                "end_altitude": phase_end_alt,
                "start_time": phase_time_start.to(ureg.minute),
                "end_time": phase_time_end.to(ureg.minute),
                "distance": phase_distance,
            }

            phase_idx += 1
            i = j

        return phases


# %% Concrete aircraft models
#
# The aircraft subclasses live in hyplan.aircraft_models.  They are
# re-exported below so ``from hyplan.aircraft import NASA_ER2`` keeps working.
from .aircraft_models import (  # noqa: E402
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    NASA_C20A,
    NASA_P3,
    NASA_WB57,
    NASA_B777,
    DynamicAviation_DH8,
    DynamicAviation_A90,
    DynamicAviation_B200,
    C130,
    BAe146,
    Learjet,
    TwinOtter,
)

__all__ = [
    "Aircraft",
    "CasMachSchedule",
    "TasSchedule",
    "VerticalProfile",
    "TurnModel",
    "PhaseBankAngles",
    "SourceRecord",
    "PerformanceConfidence",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_C20A",
    "NASA_P3",
    "NASA_WB57",
    "NASA_B777",
    "DynamicAviation_DH8",
    "DynamicAviation_A90",
    "DynamicAviation_B200",
    "C130",
    "BAe146",
    "Learjet",
    "TwinOtter",
]
