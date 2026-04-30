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
from typing import List, Literal, Optional, Tuple, Union

import logging
import warnings

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from ..airports import Airport
from ..atmosphere import cas_to_tas, mach_to_tas
from ..dubins3d import DubinsPath2D
from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..units import ureg
from ..waypoint import Waypoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance types
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    """Where a performance parameter came from.

    Args:
        source_type: One of ``"poh"``, ``"afm"``, ``"brochure"``,
            ``"adsb"``, ``"iwg1"``, ``"mission_log"``, ``"expert"``,
            ``"derived"``.  ``"iwg1"`` denotes calibration from NASA's
            Inter-agency Working Group 1 in-situ flight log format
            (per-sortie .txt CSVs of measured TAS, wind, attitude,
            etc.; see ``hyplan.aircraft.iwg1.load_iwg1``).
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
        self.cas = self.cas.to(ureg.knot)  # type: ignore[assignment]

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
        return float(np.interp(alt_ft, self._alts_ft, self._tas_kt)) * ureg.knot  # type: ignore[no-any-return]


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
        return fpm * ureg.feet / ureg.minute  # type: ignore[no-any-return]

    @property
    def sea_level_rate(self) -> Quantity:
        """Rate at the lowest altitude breakpoint (first row)."""
        return self._rates_fpm[0] * ureg.feet / ureg.minute  # type: ignore[no-any-return]

    @property
    def ceiling_rate(self) -> Quantity:
        """Rate at the highest altitude breakpoint (last row)."""
        return self._rates_fpm[-1] * ureg.feet / ureg.minute  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Approach profile
# ---------------------------------------------------------------------------

# 1 nautical mile = 6076.115485564 feet (exact, from international foot defn).
_FEET_PER_NMI = 6076.115485564


@dataclass
class ApproachProfile:
    """Generic terminal-arrival template (not a published-procedure model).

    Models the standardized terminal descent below ``top_of_approach_agl``:
    a TAS schedule keyed by altitude AGL, plus a constant glideslope.
    Horizontal distance to the runway is derived from those two; it is
    not stored, so the geometry stays self-consistent.

    All altitudes are AGL.  Convert to MSL at consumer sites by adding
    :py:attr:`hyplan.airports.Airport.elevation`.

    Args:
        speed_schedule: Piecewise-linear TAS-vs-AGL-altitude schedule.
            Lowest breakpoint should be 0 ft (touchdown).  Highest
            breakpoint must equal ``top_of_approach_agl``.
        top_of_approach_agl: Altitude AGL at which the cruise descent
            hands off to the terminal regime.
        glideslope_deg: Constant glideslope (degrees).  Default 3.0
            matches a standard ILS; calibrated values are typed in
            directly per aircraft.
    """

    speed_schedule: TasSchedule
    top_of_approach_agl: Quantity
    glideslope_deg: float = 3.0

    def __post_init__(self) -> None:
        top_ft = self.top_of_approach_agl.m_as(ureg.feet)
        if top_ft <= 0:
            raise HyPlanValueError(
                "ApproachProfile.top_of_approach_agl must be strictly positive."
            )
        max_schedule_ft = float(self.speed_schedule._alts_ft.max())
        if abs(max_schedule_ft - top_ft) > 1.0:
            raise HyPlanValueError(
                f"ApproachProfile speed_schedule's highest breakpoint ({max_schedule_ft:.0f} ft) "
                f"must equal top_of_approach_agl ({top_ft:.0f} ft)."
            )
        if not (0.0 < self.glideslope_deg < 90.0):
            raise HyPlanValueError(
                f"ApproachProfile.glideslope_deg must be in (0, 90); got {self.glideslope_deg}."
            )

    @property
    def touchdown_speed(self) -> Quantity:
        """TAS at altitude 0 ft AGL — the touchdown / threshold speed."""
        return self.speed_schedule.tas_at(0 * ureg.feet)

    @property
    def approx_approach_distance_nmi(self) -> float:
        """Horizontal distance from top_of_approach to threshold (nmi).

        Derived geometrically as ``h / tan(glideslope)``.  Approximate
        because real terminal procedures include level segments,
        intercept arcs, and procedure turns — this models a clean
        constant-glideslope final.
        """
        h_ft = self.top_of_approach_agl.m_as(ureg.feet)
        return (h_ft / np.tan(np.radians(self.glideslope_deg))) / _FEET_PER_NMI

    def tas_at(self, altitude_agl: Quantity) -> Quantity:
        """Scheduled TAS at *altitude_agl*."""
        return self.speed_schedule.tas_at(altitude_agl)

    def approx_vertical_rate_at(
        self,
        altitude_agl: Quantity,
        groundspeed: Optional[Quantity] = None,
    ) -> Quantity:
        """Approximate vertical rate on a constant-glideslope path.

        Uses ``VS = groundspeed × tan(glideslope)`` when ``groundspeed``
        is provided; otherwise falls back to ``VS = TAS × tan(glideslope)``
        (still-air approximation).  The fallback overestimates VS in a
        headwind and underestimates in a tailwind.
        """
        speed = groundspeed if groundspeed is not None else self.tas_at(altitude_agl)
        speed_fpm = speed.m_as(ureg.feet / ureg.minute)
        vs_fpm = speed_fpm * np.tan(np.radians(self.glideslope_deg))
        return vs_fpm * ureg.feet / ureg.minute  # type: ignore[no-any-return]

    def time_to_touchdown(self, groundspeed: Optional[Quantity] = None) -> Quantity:
        """Integrate 1/VS from top_of_approach down to 0 ft AGL.

        Uses the same TAS-vs-groundspeed convention as
        :meth:`approx_vertical_rate_at`.  When ``groundspeed`` is None,
        scheduled TAS is used at every altitude (still-air approximation).
        """
        # Integration grid: union of speed_schedule breakpoints, in ascending altitude.
        alts_ft = np.asarray(self.speed_schedule._alts_ft, dtype=float)
        # Integrand: 1 / VS(altitude), in minutes per foot.
        if groundspeed is None:
            speeds_fpm = np.asarray(self.speed_schedule._tas_kt, dtype=float) * (
                _FEET_PER_NMI / 60.0
            )
        else:
            gs_fpm = groundspeed.m_as(ureg.feet / ureg.minute)
            speeds_fpm = np.full_like(alts_ft, gs_fpm)
        vs_fpm = speeds_fpm * np.tan(np.radians(self.glideslope_deg))
        if np.any(vs_fpm <= 0):
            raise HyPlanValueError(
                "ApproachProfile.time_to_touchdown requires positive vertical rates "
                "at every breakpoint."
            )
        # trapezoidal integration of 1/VS over altitude (ft) → minutes.
        minutes = float(np.trapezoid(1.0 / vs_fpm, alts_ft))
        return minutes * ureg.minute  # type: ignore[no-any-return]


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

    def for_phase(self, phase: str) -> float:
        """Return the bank angle for ``phase``.

        Args:
            phase: One of ``"climb"``, ``"cruise"``, ``"descent"``,
                ``"approach"``.

        Raises:
            HyPlanValueError: For any other ``phase`` string.
        """
        try:
            return getattr(self, f"{phase}_deg")
        except AttributeError:
            raise HyPlanValueError(
                f"Unknown phase {phase!r}; expected one of "
                "'climb', 'cruise', 'descent', 'approach'."
            )


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
        descent_profile: Rate-of-descent vs altitude.  When
            ``approach_profile`` is set, this profile is intended to
            cover the cruise-altitude → top-of-approach (MSL) regime
            only; the terminal descent below top-of-approach is owned
            by ``approach_profile``.  When ``approach_profile`` is
            ``None``, ``descent_profile`` continues to cover the full
            cruise-to-touchdown range as before (legacy behavior).
        turn_model: Turn performance / bank angles.
        engine_type: Propulsion category — ``"jet"``, ``"turboprop"``,
            or ``"piston"``.
        confidence: Per-submodel confidence ratings.
        sources: List of provenance records.
        range: Maximum flight range (optional, metadata only).
        endurance: Maximum flight duration (optional, metadata only).
        useful_payload: Payload capacity (optional, metadata only).
        approach_profile: Optional terminal-arrival template covering
            top-of-approach → touchdown.  When set, the planner can
            estimate terminal-segment timing geometrically from the
            speed schedule and glideslope.  When ``None``, the legacy
            scalar ``approach_speed`` and ``descent_profile`` are used
            for arrival behavior.
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
        engine_type: Literal["jet", "turboprop", "piston"],
        confidence: Optional[PerformanceConfidence] = None,
        sources: Optional[List[SourceRecord]] = None,
        range: Optional[Quantity] = None,
        endurance: Optional[Quantity] = None,
        useful_payload: Optional[Quantity] = None,
        approach_profile: Optional[ApproachProfile] = None,
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
        self.engine_type = engine_type
        self.confidence = confidence or PerformanceConfidence()
        self.sources = sources or []

        self.range = range.to(ureg.nautical_mile) if range is not None else None
        self.endurance = endurance.to(ureg.hour) if endurance is not None else None
        self.useful_payload = (
            useful_payload.to(ureg.pound) if useful_payload is not None else None
        )

        if approach_profile is not None and not isinstance(approach_profile, ApproachProfile):
            raise HyPlanTypeError(
                f"approach_profile must be ApproachProfile or None, got {type(approach_profile).__name__}."
            )
        self.approach_profile = approach_profile

        self._validate_schedule_compatibility()

    # ------------------------------------------------------------------
    # Schedule / engine-type compatibility
    # ------------------------------------------------------------------

    # Compatibility matrix: engine_type → schedule_type → level.
    #   "typical"  — expected combination, no warning.
    #   "unusual"  — allowed but surprising; emits a warning.
    #   "forbid"   — raises HyPlanValueError.
    _SCHEDULE_COMPAT = {
        "jet": {CasMachSchedule: "typical", TasSchedule: "typical"},
        "turboprop": {CasMachSchedule: "unusual", TasSchedule: "typical"},
        "piston": {CasMachSchedule: "forbid", TasSchedule: "typical"},
    }

    def _validate_schedule_compatibility(self) -> None:
        """Check engine_type vs speed-schedule combinations.

        Warns for unusual pairings, raises for invalid ones.
        """
        schedules = {
            "climb_schedule": self.climb_schedule,
            "cruise_schedule": self.cruise_schedule,
            "descent_schedule": self.descent_schedule,
        }
        compat = self._SCHEDULE_COMPAT.get(self.engine_type, {})
        for name, sched in schedules.items():
            level = compat.get(type(sched))
            if level == "forbid":
                raise HyPlanValueError(
                    f"{name}: {type(sched).__name__} is not valid for "
                    f"engine_type={self.engine_type!r}."
                )
            if level == "unusual":
                warnings.warn(
                    f"{self.aircraft_type}: {name} uses "
                    f"{type(sched).__name__} with engine_type="
                    f"{self.engine_type!r} — this is unusual.",
                    stacklevel=3,
                )

    @property
    def speed_model_fidelity(self) -> str:
        """Describe the fidelity level of the cruise speed model.

        Returns one of:

        * ``"cas_mach"`` — CAS/Mach schedule (atmosphere-aware).
        * ``"simplified_tas"`` — piecewise-linear TAS approximation.
        """
        if isinstance(self.cruise_schedule, CasMachSchedule):
            return "cas_mach"
        return "simplified_tas"

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

    def climb_speed_at(self, altitude: Quantity) -> Quantity:
        """True airspeed during climb at *altitude*.

        Mirrors :meth:`descent_speed_at` for the climb schedule.  Most
        aircraft factories alias ``climb_schedule`` to ``cruise_schedule``,
        in which case this returns the same value as :meth:`cruise_speed_at`.
        """
        return self.climb_schedule.tas_at(altitude)

    def rate_of_climb(self, altitude: Quantity) -> Quantity:
        """Rate of climb at *altitude* from the climb profile."""
        return self.climb_profile.rate_at(altitude)

    def descent_speed_at(self, altitude: Quantity) -> Quantity:
        """True airspeed during descent at *altitude*."""
        return self.descent_schedule.tas_at(altitude)

    def approach_speed_at(self, altitude_agl: Quantity) -> Quantity:
        """True airspeed at *altitude_agl* during the terminal approach.

        When :attr:`approach_profile` is set, this returns the schedule
        value at *altitude_agl*.  When it isn't, this returns the legacy
        scalar :attr:`approach_speed` for any altitude (the existing
        single-speed approximation).
        """
        if self.approach_profile is None:
            return self.approach_speed
        return self.approach_profile.tas_at(altitude_agl)

    def approach_vertical_rate_at(
        self,
        altitude_agl: Quantity,
        groundspeed: Optional[Quantity] = None,
    ) -> Optional[Quantity]:
        """Approximate vertical rate on the terminal approach.

        When :attr:`approach_profile` is set, returns the geometric
        rate from :meth:`ApproachProfile.approx_vertical_rate_at`
        (using the optional *groundspeed* override or scheduled TAS in
        still air).  Returns ``None`` when no approach profile is
        configured — callers should fall back to legacy descent
        behavior in that case rather than synthesizing from
        :attr:`descent_profile`, which keeps the regime split crisp.
        """
        if self.approach_profile is None:
            return None
        return self.approach_profile.approx_vertical_rate_at(
            altitude_agl, groundspeed=groundspeed
        )

    def pitch_limits(self, speed: Optional[Quantity] = None) -> tuple:
        """Derive pitch-angle limits from climb/descent rates and TAS.

        Returns ``(pitch_min, pitch_max)`` in degrees.  ``pitch_min`` is
        negative (descent), ``pitch_max`` is positive (climb).
        """
        tas = speed if speed is not None else self.cruise_speed_at(self.service_ceiling)  # type: ignore[arg-type]
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
        start_altitude = start_altitude.to(ureg.feet)  # type: ignore[assignment]
        end_altitude = end_altitude.to(ureg.feet)  # type: ignore[assignment]

        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.climb_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)  # type: ignore[assignment]

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
        start_altitude = start_altitude.to(ureg.feet)  # type: ignore[assignment]
        end_altitude = end_altitude.to(ureg.feet)  # type: ignore[assignment]

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
        start_altitude = start_altitude.to(ureg.feet)  # type: ignore[assignment]
        end_altitude = end_altitude.to(ureg.feet)  # type: ignore[assignment]

        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.descent_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)  # type: ignore[assignment]

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
        return self.time_to_cruise(
            airport_waypoint, waypoint, wind=wind, phase="climb",
        )

    def time_to_return(
        self,
        waypoint: Waypoint,
        airport: Airport,
        wind: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Calculate time from the last waypoint back to the airport.

        Uses 3D Dubins path planning for the return including descent.

        When :attr:`approach_profile` is set, the Dubins descent is
        targeted at top-of-approach MSL (= ``airport.elevation +
        approach_profile.top_of_approach_agl``) and a terminal
        ``"approach"`` segment is appended using
        :meth:`ApproachProfile.time_to_touchdown`.  When no profile is
        set, the legacy single-leg-to-runway behavior is preserved.
        """
        _, arrival_heading = pymap3d.vincenty.vdist(
            waypoint.latitude, waypoint.longitude,
            airport.latitude, airport.longitude,
        )
        arrival_heading_deg = (arrival_heading + 180.0) % 360.0

        if self.approach_profile is None:
            # Legacy: Dubins descent all the way to runway elevation.
            airport_waypoint = Waypoint(
                latitude=airport.latitude,
                longitude=airport.longitude,
                heading=arrival_heading_deg,
                altitude_msl=airport.elevation,
            )
            return self.time_to_cruise(
                waypoint, airport_waypoint, wind=wind, phase="descent",
            )

        # New: Dubins descent stops at the FAF (final approach fix); the
        # ApproachProfile owns the terminal segment from FAF to airport.
        # The FAF is positioned upwind of the runway by the geometric
        # approach distance — without this offset the Dubins endpoint
        # would sit directly above the airport, leaving no horizontal
        # room for a non-degenerate approach geometry.
        approach_distance_nmi = self.approach_profile.approx_approach_distance_nmi
        approach_distance_m = approach_distance_nmi * 1852.0
        faf_lat, faf_lon = pymap3d.vincenty.vreckon(
            airport.latitude, airport.longitude,
            approach_distance_m, arrival_heading_deg,
        )
        faf_lat = float(faf_lat)
        faf_lon = ((float(faf_lon) + 180.0) % 360.0) - 180.0  # wrap to [-180, 180)

        top_of_approach_msl = airport.elevation + self.approach_profile.top_of_approach_agl
        top_of_approach_waypoint = Waypoint(
            latitude=faf_lat,
            longitude=faf_lon,
            heading=arrival_heading_deg,
            altitude_msl=top_of_approach_msl,
        )
        cruise_descent = self.time_to_cruise(
            waypoint, top_of_approach_waypoint, wind=wind, phase="descent",
        )
        approach_time = self.approach_profile.time_to_touchdown().to(ureg.minute)
        approach_distance = approach_distance_nmi * ureg.nautical_mile

        # Real terminal geometry: 2-point line from FAF to airport,
        # used by process_flight_phase verbatim (no Dubins-slicing).
        from shapely.geometry import LineString
        approach_geom = LineString([(faf_lon, faf_lat), (airport.longitude, airport.latitude)])

        total_time = (cruise_descent["total_time"] + approach_time).to(ureg.minute)
        phases = dict(cruise_descent["phases"])
        phases["approach"] = {
            "start_altitude": top_of_approach_msl.to(ureg.feet),
            "end_altitude": airport.elevation.to(ureg.feet),
            "start_time": cruise_descent["total_time"].to(ureg.minute),
            "end_time": total_time,
            "distance": approach_distance,
            "geometry": approach_geom,
            "start_lat": faf_lat,
            "start_lon": faf_lon,
            "end_lat": airport.latitude,
            "end_lon": airport.longitude,
            "start_heading": arrival_heading_deg,
            "end_heading": arrival_heading_deg,
        }

        return {
            "total_time": total_time,
            "phases": phases,
            "dubins_path": cruise_descent["dubins_path"],
        }

    def _hybrid_path(
        self,
        start_waypoint: Waypoint,
        end_waypoint: Waypoint,
        *,
        cruise_altitude: Optional[Quantity] = None,
        true_air_speed: Optional[Quantity] = None,
        wind: Optional[Tuple[float, float]] = None,
        phase: str = "cruise",
    ) -> dict:
        """Solve a hybrid horizontal-Dubins + integrated-vertical path.

        The horizontal layout comes from a 2D Dubins solver.  Turn
        radius is set from ``turn_model.bank_by_phase.for_phase(phase)``
        and the cruise TAS at ``cruise_altitude``; the vertical profile
        comes from integrating ``climb_profile`` / ``descent_profile``
        against altitude (:meth:`_climb` / :meth:`_descend` return
        ``(time, horizontal_distance)`` for the requested altitude
        range).

        ``phase`` selects which entry of
        :class:`PhaseBankAngles` drives the Dubins arc radius.  The
        default ``"cruise"`` is the right choice for any path whose
        horizontal turning happens at cruise altitude (inter-line
        transits, single-segment cruise legs).  Callers solving paths
        whose turn arcs happen during climb-out (takeoff phase) or
        descent (return phase) should pass ``"climb"`` or ``"descent"``
        respectively, so the radius reflects the gentler bank that
        aircraft actually fly during those phases.

        ``cruise_altitude`` defaults to ``max(start.alt, end.alt)``.
        Climb covers ``start.alt → cruise_alt``; descent covers
        ``cruise_alt → end.alt``; cruise fills whatever horizontal
        distance is left.  When the climb + descent horizontal distance
        exceeds the 2D Dubins length (short-leg edge case), the
        horizontal extents are scaled to fit and there's no cruise
        segment — the aircraft "spirals up / down" within the leg
        at full climb / descent times.

        Returns a dict with:

        * ``total_time``: ``Quantity`` (minutes).
        * ``phases``: ``dict`` keyed by ``"climb"`` / ``"cruise"`` /
          ``"descent"`` (only the present phases), each carrying
          ``start_altitude``, ``end_altitude``, ``start_time``,
          ``end_time``, ``distance``, ``geometry``,
          ``start_lat / start_lon / end_lat / end_lon``,
          ``start_heading / end_heading``.
        * ``dubins_path``: the :class:`DubinsPath2D` instance (also
          accessible via the ``horizontal_path`` key).
        * ``horizontal_path``: alias of ``dubins_path``.

        Wind handling: the 2D path uses trochoidal geometry when
        ``wind`` is supplied.  Vertical integration (``_climb`` /
        ``_descend``) is still-air-only — that's consistent with the
        rest of the timing pipeline and good enough for HyPlan's
        mission-planning use case.
        """
        start_alt = start_waypoint.altitude_msl.to(ureg.feet)  # type: ignore[union-attr]
        end_alt = end_waypoint.altitude_msl.to(ureg.feet)  # type: ignore[union-attr]

        if cruise_altitude is None:
            # Default: the higher of the two endpoints.
            cruise_altitude = start_alt if start_alt >= end_alt else end_alt
        cruise_altitude = cruise_altitude.to(ureg.feet)

        cruise_tas = (
            true_air_speed
            if true_air_speed is not None
            else self.cruise_speed_at(cruise_altitude)
        )

        bank_deg = self.turn_model.bank_by_phase.for_phase(phase)
        h_path = DubinsPath2D(
            start_waypoint, end_waypoint,
            speed=cruise_tas, bank_angle=bank_deg, wind=wind,
        )
        L_m = h_path.length.m_as(ureg.meter)
        L_nmi = L_m / 1852.0

        # Climb segment (start.alt -> cruise.alt).
        if start_alt < cruise_altitude:
            climb_t_q, climb_d_q = self._climb(start_alt, cruise_altitude)
            climb_time_min = climb_t_q.m_as(ureg.minute)
            climb_dist_nmi = climb_d_q.m_as(ureg.nautical_mile)
        else:
            climb_time_min = 0.0
            climb_dist_nmi = 0.0

        # Descent segment (cruise.alt -> end.alt).
        if end_alt < cruise_altitude:
            desc_t_q, desc_d_q = self._descend(cruise_altitude, end_alt)
            descent_time_min = desc_t_q.m_as(ureg.minute)
            descent_dist_nmi = desc_d_q.m_as(ureg.nautical_mile)
        else:
            descent_time_min = 0.0
            descent_dist_nmi = 0.0

        # Short-leg edge cases.  When the climb (or descent) horizontal
        # distance alone exceeds the 2D Dubins length, the aircraft is
        # not transiting forward during that phase — it's spiraling up
        # at departure (or down at arrival).  Use orbit geometry for the
        # vertical phase and a full-length cruise for the transit.  When
        # *both* climb and descent distances exceed the leg length (rare
        # degenerate case), fall back to proportional scaling.
        cruise_tas_kt = cruise_tas.m_as(ureg.knot)
        eps_short = 1e-6
        short_climb = (
            climb_dist_nmi > L_nmi + eps_short
            and descent_dist_nmi <= eps_short
        )
        short_descent = (
            descent_dist_nmi > L_nmi + eps_short
            and climb_dist_nmi <= eps_short
        )
        short_mixed = (
            climb_dist_nmi + descent_dist_nmi > L_nmi + eps_short
            and not short_climb
            and not short_descent
        )

        if short_climb:
            climb_dist_nmi = 0.0
            cruise_dist_nmi = L_nmi
            cruise_time_min = (
                cruise_dist_nmi / cruise_tas_kt * 60.0
                if cruise_tas_kt > 0
                else 0.0
            )
        elif short_descent:
            descent_dist_nmi = 0.0
            cruise_dist_nmi = L_nmi
            cruise_time_min = (
                cruise_dist_nmi / cruise_tas_kt * 60.0
                if cruise_tas_kt > 0
                else 0.0
            )
        elif short_mixed:
            scale = L_nmi / (climb_dist_nmi + descent_dist_nmi)
            climb_dist_nmi *= scale
            descent_dist_nmi *= scale
            cruise_dist_nmi = 0.0
            cruise_time_min = 0.0
        else:
            cruise_dist_nmi = max(0.0, L_nmi - climb_dist_nmi - descent_dist_nmi)
            cruise_time_min = (
                cruise_dist_nmi / cruise_tas_kt * 60.0
                if cruise_tas_kt > 0
                else 0.0
            )

        total_time_min = climb_time_min + cruise_time_min + descent_time_min

        # Build phases dict with explicit per-phase geometry.
        phases: dict = {}
        cum_dist_m = 0.0
        cum_time_min = 0.0
        nmi_to_m = 1852.0
        eps = 1e-6

        if climb_time_min > eps or climb_dist_nmi > eps:
            if short_climb:
                # Spiral-up at departure: orbit geometry over start
                # waypoint.  Use the climb bank + climb-schedule TAS
                # at the *midpoint* altitude — a single-orbit
                # representation of what physically is a climbing
                # helix.  Climb bank (typ. 11° for ER-2) is gentler
                # than cruise bank (20°), giving a wider, more
                # realistic ground track.
                from ..planning.segments import loiter_orbit_geometry
                mid_alt = (start_alt + cruise_altitude) / 2.0
                orbit_wp = Waypoint(
                    latitude=start_waypoint.latitude,
                    longitude=start_waypoint.longitude,
                    heading=start_waypoint.heading,
                    altitude_msl=mid_alt,
                )
                orbit_geom = loiter_orbit_geometry(
                    orbit_wp, self, phase="climb",
                )
                track_dist_nmi = climb_time_min / 60.0 * cruise_tas_kt
                s_lat = e_lat = start_waypoint.latitude
                s_lon = e_lon = start_waypoint.longitude
                s_hdg = e_hdg = start_waypoint.heading
                phases["climb"] = {
                    "start_altitude": start_alt,
                    "end_altitude": cruise_altitude,
                    "start_time": cum_time_min * ureg.minute,
                    "end_time": (cum_time_min + climb_time_min) * ureg.minute,
                    "distance": track_dist_nmi * ureg.nautical_mile,
                    "geometry": orbit_geom,
                    "start_lat": s_lat, "start_lon": s_lon,
                    "end_lat": e_lat, "end_lon": e_lon,
                    "start_heading": s_hdg, "end_heading": e_hdg,
                }
            else:
                end_d_m = climb_dist_nmi * nmi_to_m
                s_lat, s_lon, s_hdg = h_path.sample_at_distance(0.0)
                e_lat, e_lon, e_hdg = h_path.sample_at_distance(end_d_m)
                phases["climb"] = {
                    "start_altitude": start_alt,
                    "end_altitude": cruise_altitude,
                    "start_time": cum_time_min * ureg.minute,
                    "end_time": (cum_time_min + climb_time_min) * ureg.minute,
                    "distance": climb_dist_nmi * ureg.nautical_mile,
                    "geometry": h_path.sublinestring(0.0, end_d_m),
                    "start_lat": s_lat, "start_lon": s_lon,
                    "end_lat": e_lat, "end_lon": e_lon,
                    "start_heading": s_hdg, "end_heading": e_hdg,
                }
                cum_dist_m = end_d_m
            cum_time_min += climb_time_min

        if cruise_time_min > eps or cruise_dist_nmi > eps:
            end_d_m = cum_dist_m + cruise_dist_nmi * nmi_to_m
            s_lat, s_lon, s_hdg = h_path.sample_at_distance(cum_dist_m)
            e_lat, e_lon, e_hdg = h_path.sample_at_distance(end_d_m)
            phases["cruise"] = {
                "start_altitude": cruise_altitude,
                "end_altitude": cruise_altitude,
                "start_time": cum_time_min * ureg.minute,
                "end_time": (cum_time_min + cruise_time_min) * ureg.minute,
                "distance": cruise_dist_nmi * ureg.nautical_mile,
                "geometry": h_path.sublinestring(cum_dist_m, end_d_m),
                "start_lat": s_lat, "start_lon": s_lon,
                "end_lat": e_lat, "end_lon": e_lon,
                "start_heading": s_hdg, "end_heading": e_hdg,
            }
            cum_dist_m = end_d_m
            cum_time_min += cruise_time_min

        if descent_time_min > eps or descent_dist_nmi > eps:
            if short_descent:
                # Spiral-down at arrival: mirror of spiral-up.  Use
                # the descent bank + descent-schedule TAS at the
                # midpoint altitude.
                from ..planning.segments import loiter_orbit_geometry
                mid_alt = (cruise_altitude + end_alt) / 2.0
                orbit_wp = Waypoint(
                    latitude=end_waypoint.latitude,
                    longitude=end_waypoint.longitude,
                    heading=end_waypoint.heading,
                    altitude_msl=mid_alt,
                )
                orbit_geom = loiter_orbit_geometry(
                    orbit_wp, self, phase="descent",
                )
                track_dist_nmi = descent_time_min / 60.0 * cruise_tas_kt
                s_lat = e_lat = end_waypoint.latitude
                s_lon = e_lon = end_waypoint.longitude
                s_hdg = e_hdg = end_waypoint.heading
                phases["descent"] = {
                    "start_altitude": cruise_altitude,
                    "end_altitude": end_alt,
                    "start_time": cum_time_min * ureg.minute,
                    "end_time": (cum_time_min + descent_time_min) * ureg.minute,
                    "distance": track_dist_nmi * ureg.nautical_mile,
                    "geometry": orbit_geom,
                    "start_lat": s_lat, "start_lon": s_lon,
                    "end_lat": e_lat, "end_lon": e_lon,
                    "start_heading": s_hdg, "end_heading": e_hdg,
                }
            else:
                end_d_m = L_m
                s_lat, s_lon, s_hdg = h_path.sample_at_distance(cum_dist_m)
                e_lat, e_lon, e_hdg = h_path.sample_at_distance(end_d_m)
                phases["descent"] = {
                    "start_altitude": cruise_altitude,
                    "end_altitude": end_alt,
                    "start_time": cum_time_min * ureg.minute,
                    "end_time": (cum_time_min + descent_time_min) * ureg.minute,
                    "distance": descent_dist_nmi * ureg.nautical_mile,
                    "geometry": h_path.sublinestring(cum_dist_m, end_d_m),
                    "start_lat": s_lat, "start_lon": s_lon,
                    "end_lat": e_lat, "end_lon": e_lon,
                    "start_heading": s_hdg, "end_heading": e_hdg,
                }
            cum_time_min += descent_time_min

        # Always emit at least one phase.  Pure cruise at equal altitudes
        # falls into the cruise branch above; if even that is empty
        # (degenerate zero-length path), synthesize a trivial cruise.
        if not phases:
            s_lat, s_lon, s_hdg = h_path.sample_at_distance(0.0)
            phases["cruise"] = {
                "start_altitude": cruise_altitude,
                "end_altitude": cruise_altitude,
                "start_time": 0 * ureg.minute,
                "end_time": 0 * ureg.minute,
                "distance": 0 * ureg.nautical_mile,
                "geometry": h_path.geometry,
                "start_lat": s_lat, "start_lon": s_lon,
                "end_lat": s_lat, "end_lon": s_lon,
                "start_heading": s_hdg, "end_heading": s_hdg,
            }

        return {
            "total_time": total_time_min * ureg.minute,
            "phases": phases,
            "dubins_path": h_path,
            "horizontal_path": h_path,
        }

    def time_to_cruise(
        self,
        start_waypoint: Waypoint,
        end_waypoint: Waypoint,
        true_air_speed: Optional[Quantity] = None,
        wind: Optional[Tuple[float, float]] = None,
        phase: str = "cruise",
    ) -> dict:
        """Calculate time to fly between two waypoints.

        Hybrid 2D Dubins (horizontal layout) + integrated vertical
        profile.  Returns a dict with ``total_time``, ``phases``, and
        ``dubins_path`` (the 2D path; legacy key name kept for
        backward compatibility — use ``horizontal_path`` in new code).

        Args:
            wind: Optional ``(u_east, v_north)`` wind vector in m/s.
                When provided, horizontal turning arcs become trochoids
                and the 2D path length / timing account for wind drift.
                Vertical integration is still-air.
            phase: Which entry of
                :class:`PhaseBankAngles` drives the horizontal Dubins
                turn radius — see :meth:`_hybrid_path`.  Defaults to
                ``"cruise"``.
        """
        return self._hybrid_path(
            start_waypoint, end_waypoint,
            true_air_speed=true_air_speed, wind=wind, phase=phase,
        )

