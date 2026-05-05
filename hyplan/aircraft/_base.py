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
import math
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
        return float((h_ft / np.tan(np.radians(self.glideslope_deg))) / _FEET_PER_NMI)

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
        return vs_fpm * ureg.feet / ureg.minute

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
        return minutes * ureg.minute


# ---------------------------------------------------------------------------
# Climb plan
# ---------------------------------------------------------------------------

@dataclass
class ClimbPlan:
    """Staged climb plan for the takeoff phase.

    Real high-altitude aircraft step-climb out of the weight-limited
    ceiling: they climb to an intermediate altitude, level off briefly
    to burn fuel and reduce gross weight, then continue climbing.  A
    NASA ER-2 sortie typically holds for ~25 minutes around FL356
    before completing the climb to FL650.

    Each entry in ``pauses`` is ``(level_off_altitude, hold_duration)``
    — at the level-off altitude the aircraft holds (level orbit) for
    ``hold_duration``, gaining time but no forward distance.  Pauses
    are applied in altitude order; pauses outside the
    ``[start_altitude, cruise_altitude]`` range during planning are
    silently skipped.

    Pass an instance to :func:`hyplan.planning.compute_flight_plan`
    via the ``climb_plan`` keyword to make the takeoff-phase planner
    use :meth:`Aircraft.step_climb` instead of plain :meth:`_climb`.
    """

    pauses: list = field(default_factory=list)


@dataclass
class ClimbOutPolicy:
    """Documents the typical pre-cruise climb-out behaviour an
    :class:`Aircraft`'s ``climb_profile`` is calibrated to absorb.

    Aircraft factories may set this to make the absorption posture
    queryable rather than buried in comments or docstrings.

    .. note::
        ``Aircraft._hybrid_path`` does **not** consume this field
        today (Phase 1).  It's metadata that names what the
        wall-clock-fit ``climb_profile`` already encodes.  The Phase 3
        ``compute_flight_plan(climb_plan="auto")`` sentinel will
        eventually read ``explicit_climb_plan`` from here and disable
        the implicit absorption.

    Args:
        absorbed_in_climb_profile: When ``True``, the aircraft's
            ``climb_profile`` is tuned so that integrating ``_climb()``
            over a typical mission band reproduces the wall-clock TOC,
            *including* the holds listed in ``typical_holds``.  Calling
            ``_climb()`` directly will therefore over-state pure
            active-climb time by approximately ``typical_overhead_min``.
            When ``False``, ``climb_profile`` represents active climb
            only and the holds belong in an explicit ``ClimbPlan``.
        typical_holds: Ordered ``(altitude, hold_duration)`` pairs
            that the calibrated profile assumes a typical sortie
            spends at level-offs / orbits.  Informational; not
            consumed by the planner.
        typical_overhead_min: Total wall-clock minutes of pre-cruise
            level-offs / `.delay` orbits the ``climb_profile`` assumes.
            Used by reviewers to sanity-check the calibration story;
            not consumed by the planner.
        notes: Free-text caveats — which calibration set, what
            fraction of the cruise-altitude TOC residual is mission-
            specific vs aircraft-intrinsic, when the absorption
            posture breaks down.
        explicit_climb_plan: Optional ready-to-use :class:`ClimbPlan`
            that, when supplied as ``compute_flight_plan(
            climb_plan=...)``, *replaces* the implicit absorption.
            Callers using this **must** also set the aircraft's
            ``climb_path_angle_max_deg`` (e.g. to 6.0) to disable the
            spiral-up absorption — otherwise the holds are
            double-counted with the climb_profile's bake-in.
    """

    absorbed_in_climb_profile: bool
    typical_holds: List[Tuple[Quantity, Quantity]] = field(default_factory=list)
    typical_overhead_min: float = 0.0
    notes: str = ""
    explicit_climb_plan: Optional[ClimbPlan] = None


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
            return float(getattr(self, f"{phase}_deg"))
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
        max_load_factor: Structural load-factor budget (g).  Used by
            :meth:`Aircraft.max_bank_under_budget` to cap the bank
            angle when the implicit pitch from the climb / descent
            profile would otherwise push the aircraft past its design
            envelope.  Default ``2.5`` covers normal-category
            certification (FAR 23 § 23.337); transport-category bizjets
            and survey aircraft typically operate well within this
            limit.  Utility / acrobatic aircraft can push higher
            (~3.8 / ~6.0); raise the value when modeling those.
    """

    bank_by_phase: PhaseBankAngles = field(default_factory=PhaseBankAngles)
    max_bank_deg: float = 30.0
    max_load_factor: float = 2.5


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
        descent_path_angle_max_deg: Maximum sustainable flight path
            angle during descent (degrees).  When set, ``_hybrid_path``
            steepens the descent to fit available lateral distance
            rather than spiralling at end of leg.
        climb_path_angle_max_deg: Maximum sustainable flight path
            angle during climb (degrees).  When set, ``_hybrid_path``
            steepens the climb to fit available lateral distance
            rather than spiralling at departure.  When ``None`` and
            ``typical_climb_out.absorbed_in_climb_profile`` is True,
            the spiral-up regime acts as the absorption mechanism for
            mission-typical level-offs / `.delay` orbits.
        typical_climb_out: Optional :class:`ClimbOutPolicy` documenting
            the pre-cruise climb-out behaviour the calibrated
            ``climb_profile`` is tuned to absorb.  Pure metadata in
            v1.5 — names the absorption posture so it's queryable
            rather than buried in comments.
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
        descent_path_angle_max_deg: Optional[float] = None,
        climb_path_angle_max_deg: Optional[float] = None,
        typical_climb_out: Optional[ClimbOutPolicy] = None,
        stall_speed_cas: Optional[Quantity] = None,
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

        # Maximum sustainable flight path angle during descent (degrees).
        # When set, _hybrid_path will steepen the descent (scaling
        # descent_profile VS uniformly) to fit available lateral distance,
        # rather than falling back to the spiral-down regime.  None
        # preserves the legacy "spiral if descent doesn't fit" behavior.
        if descent_path_angle_max_deg is not None and descent_path_angle_max_deg <= 0:
            raise HyPlanValueError(
                "descent_path_angle_max_deg must be positive."
            )
        self.descent_path_angle_max_deg = descent_path_angle_max_deg

        # Maximum sustainable flight path angle during climb (degrees).
        # When set, _hybrid_path will steepen the climb (scaling
        # climb_profile VS uniformly) to fit available lateral
        # distance, rather than falling back to the spiral-up regime.
        # This means modeled climb time represents *pure active climb*;
        # mission-specific holds (level-offs, .delay orbits) must be
        # added explicitly via ClimbPlan pauses or Waypoint.delay.
        # None preserves the legacy "spiral if climb doesn't fit"
        # behavior.
        if climb_path_angle_max_deg is not None and climb_path_angle_max_deg <= 0:
            raise HyPlanValueError(
                "climb_path_angle_max_deg must be positive."
            )
        self.climb_path_angle_max_deg = climb_path_angle_max_deg

        # Typical pre-cruise climb-out policy.  Pure metadata in
        # Phase 1 — names what `climb_profile` is calibrated to absorb
        # so the absorption posture is queryable rather than buried in
        # comments.  Phase 3 will teach `compute_flight_plan` to
        # consume `explicit_climb_plan` from here when the caller
        # passes ``climb_plan="auto"``.
        if typical_climb_out is not None and not isinstance(
            typical_climb_out, ClimbOutPolicy
        ):
            raise HyPlanTypeError(
                f"typical_climb_out must be ClimbOutPolicy or None, got "
                f"{type(typical_climb_out).__name__}."
            )
        self.typical_climb_out = typical_climb_out

        # Published Vs (calibrated airspeed) at landing config, MLW.
        # Single CAS value; TAS at altitude is derived via standard
        # atmosphere via :meth:`stall_speed_at`.  Convention: cite Vs0
        # (landing config); weight scales as sqrt(W/W_ref) and is not
        # modeled here — callers wanting a tighter floor should pass a
        # larger ``margin`` to :meth:`min_safe_speed_at`.
        if stall_speed_cas is not None:
            stall_speed_cas = stall_speed_cas.to(ureg.knot)
            if stall_speed_cas.magnitude <= 0:
                raise HyPlanValueError(
                    "stall_speed_cas must be positive."
                )
        self.stall_speed_cas = stall_speed_cas

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

    def stall_speed_at(self, altitude: Quantity) -> Quantity:
        """True airspeed at stall at *altitude*.

        Stall is published as a single :attr:`stall_speed_cas` value
        (calibrated airspeed at landing config, MLW).  TAS at altitude
        is derived via standard atmosphere — Vs in CAS is approximately
        invariant with altitude (stall is a fixed-AoA, fixed-q event)
        but TAS scales as 1/sqrt(density), so TAS at FL400 is roughly
        twice the SL value.

        Raises:
            HyPlanValueError: If :attr:`stall_speed_cas` is None for
                this aircraft.
        """
        if self.stall_speed_cas is None:
            raise HyPlanValueError(
                f"{self.aircraft_type} has no calibrated stall_speed_cas; "
                f"cannot compute TAS at stall."
            )
        from ..atmosphere import cas_to_tas
        return cas_to_tas(self.stall_speed_cas, altitude).to(ureg.knot)

    def min_safe_speed_at(
        self,
        altitude: Quantity,
        *,
        margin: float = 1.3,
    ) -> Quantity:
        """Minimum safe true airspeed at *altitude*, with margin above stall.

        ``margin`` defaults to 1.3, mirroring the FAR Part 25 rule that
        V_ref >= 1.3 * Vs0.  Pass a tighter margin (e.g., 1.2) for
        attentive level orbits in benign conditions, or a looser one
        (1.4-1.5) for night IFR / unstable atmospheres.

        Returns ``margin * stall_speed_at(altitude)`` — useful for
        science planners deciding whether a slow-survey speed at a
        given altitude is acceptable.

        Raises:
            HyPlanValueError: If :attr:`stall_speed_cas` is None or
                ``margin`` is non-positive.
        """
        if margin <= 0:
            raise HyPlanValueError(
                f"margin must be positive, got {margin}."
            )
        return margin * self.stall_speed_at(altitude)

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

    def climb_gradient_at(self, altitude: Quantity) -> float:
        """Climb gradient at *altitude* — dimensionless rise/run.

        Computed from the integrated climb performance: the climb rate
        from :attr:`climb_profile` divided by the climb-schedule TAS.
        Useful for terrain-aware planning ("can the aircraft clear
        a 10,000 ft ridge in 50 nmi?") since obstacle clearance is
        naturally expressed as a horizontal-vs-vertical ratio.

        Returns 0.0 if TAS is zero (defensive — physically unreachable).
        """
        rate_mps = self.climb_profile.rate_at(altitude).m_as(
            ureg.meter / ureg.second,
        )
        tas_mps = self.climb_speed_at(altitude).m_as(
            ureg.meter / ureg.second,
        )
        if tas_mps <= 0.0:
            return 0.0
        return float(rate_mps / tas_mps)

    def descent_gradient_at(self, altitude: Quantity) -> float:
        """Descent gradient at *altitude* — dimensionless drop/run, positive.

        Mirror of :meth:`climb_gradient_at` for descent, returning a
        positive value (the magnitude of the descent slope).
        """
        rate_mps = self.descent_profile.rate_at(altitude).m_as(
            ureg.meter / ureg.second,
        )
        tas_mps = self.descent_speed_at(altitude).m_as(
            ureg.meter / ureg.second,
        )
        if tas_mps <= 0.0:
            return 0.0
        return float(abs(rate_mps) / tas_mps)

    def _implicit_pitch_for_phase(
        self,
        phase: str,
        start_alt: Quantity,
        cruise_altitude: Quantity,
        end_alt: Quantity,
    ) -> float:
        """Estimate the implicit pitch (deg) for a phase used in
        :meth:`_hybrid_path`.

        Uses the rate-vs-altitude profile (climb / descent) at the
        midpoint altitude of the phase, divided by the phase's TAS,
        to get the steady-state pitch.  Returns 0° for cruise / approach
        (level flight assumption).
        """
        if phase == "climb" and start_alt < cruise_altitude:
            mid_alt = (start_alt + cruise_altitude) / 2.0
            rate_q = self.climb_profile.rate_at(mid_alt)
            tas_q = self.climb_speed_at(mid_alt)
        elif phase == "descent" and end_alt < cruise_altitude:
            mid_alt = (cruise_altitude + end_alt) / 2.0
            rate_q = self.descent_profile.rate_at(mid_alt)
            tas_q = self.descent_speed_at(mid_alt)
        else:
            return 0.0
        rate_mps = rate_q.m_as(ureg.meter / ureg.second)
        tas_mps = tas_q.m_as(ureg.meter / ureg.second)
        if tas_mps <= 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(abs(rate_mps), tas_mps)))

    def max_bank_under_budget(self, pitch_deg: float = 0.0) -> float:
        """Maximum bank angle (deg) consistent with the load-factor budget.

        For a steady banked climb / descent at pitch angle ``pitch_deg``,
        lift balance gives ``n = 1 / (cos(bank) · cos(pitch))``.  Solving
        for the bank that drives ``n`` to ``turn_model.max_load_factor``:

            cos(bank_max) = 1 / (n_max · cos(pitch))

        Returns ``0.0`` when the implied pitch alone exceeds the
        budget (i.e. the aircraft can't sustain level flight at that
        pitch — physically unreachable, included as a defensive
        guard).  Returns ``90.0`` when the budget is unbounded
        (``n_max <= 0`` is treated as "no limit").

        The default ``pitch_deg=0`` covers level cruise; callers in
        the climb / descent paths supply the implicit pitch from the
        rate-vs-altitude profile.

        For the calibrated HyPlan aircraft library this returns large
        values (60-67° depending on aircraft and pitch) — well above
        every aircraft's calibrated ``bank_by_phase`` entry — so the
        budget is effectively a defensive ceiling.  It only narrows
        the chosen bank when callers force unusually aggressive
        manoeuvres.
        """
        n_max = self.turn_model.max_load_factor
        if n_max <= 0.0:
            return 90.0
        cos_pitch = math.cos(math.radians(pitch_deg))
        if cos_pitch <= 0.0:
            return 0.0
        inv_cos_bank = 1.0 / (n_max * cos_pitch)
        if inv_cos_bank >= 1.0:
            # Pitch alone consumes the entire load budget; no lateral
            # margin remains for banking.
            return 0.0
        return float(np.degrees(np.arccos(inv_cos_bank)))

    # ------------------------------------------------------------------
    # Climb
    # ------------------------------------------------------------------

    def _climb(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
        wind_along_track: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """Estimate time and horizontal distance during a continuous climb.

        Integrates ``climb_profile`` against altitude (mode-dispatched):

        * ``"constant"`` — single ROC, simple division.
        * ``"two_point"`` — analytical log formula (linear ROC model).
        * ``"full"`` — numerical trapezoidal integration.

        ``wind_along_track`` is the signed wind component projected onto
        the ground track (positive = tailwind, negative = headwind).
        When supplied, horizontal distance is integrated as
        ``ground_speed × time`` where
        ``ground_speed = TAS · cos(climb_angle) + tailwind_component``.
        Default ``None`` is still-air behavior (backwards-compatible
        with v1.3 and earlier).

        For staged climbs with intermediate level-off pauses (e.g., a
        weight-driven hold during climb-out), see :meth:`step_climb`.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)

        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.climb_speed_at(avg_alt)
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
        if wind_along_track is not None:
            # Add tailwind (signed) onto still-air horizontal speed to get
            # ground speed; integrate ground distance over time_to_climb.
            wind_kt = wind_along_track.m_as(ureg.knot)
            ground_speed_kt = max(
                0.0,
                horizontal_speed.m_as(ureg.knot) + wind_kt,
            )
            horizontal_speed = ground_speed_kt * (
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

    def step_climb(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        pauses: List[Tuple[Quantity, Quantity]],
        wind_along_track: Optional[Quantity] = None,
    ) -> Tuple[Quantity, Quantity]:
        """Total time and forward distance for a staged climb with pauses.

        Real high-altitude aircraft step-climb out of weight-limited
        ceiling: they climb to an intermediate altitude, level off
        briefly to burn fuel and reduce gross weight, then continue
        climbing.  For a NASA ER-2 sortie this typically looks like a
        25-minute hold at FL611 climbing slowly under reduced weight,
        before final climb to the FL650 cruise altitude.

        Each entry in ``pauses`` is ``(level_off_altitude, hold_duration)``.
        At each pause altitude, the aircraft holds (level orbit) for
        ``hold_duration`` adding only to total time — zero forward
        distance, since the aircraft is presumed to be orbiting at one
        location during the hold.  Climb segments between pauses use
        the aircraft's calibrated :attr:`climb_profile` via
        :meth:`_climb`.

        Pauses are applied in altitude order; pauses outside the
        ``[start_altitude, end_altitude]`` range are silently skipped.

        Args:
            start_altitude: Starting altitude (e.g., airport elevation).
            end_altitude: Final altitude (e.g., cruise altitude).
            pauses: List of ``(altitude, hold_duration)`` tuples.  Pass
                an empty list to recover the no-pause behavior of
                :meth:`_climb`.

        Returns:
            Tuple of ``(total_time, total_forward_distance)`` as
            :class:`pint.Quantity`.

        Example:
            ER-2 NM17 B planned climb-out: 25-min hold at FL611
            climbing to FL650.

            >>> ac = NASA_ER2()
            >>> t, d = ac.step_climb(
            ...     start_altitude=6_187 * ureg.foot,
            ...     end_altitude=65_000 * ureg.foot,
            ...     pauses=[(35_600 * ureg.foot, 25 * ureg.minute)],
            ... )
        """
        ft = ureg.foot
        nmi = ureg.nautical_mile
        minute = ureg.minute

        # Distance and active-climb time come from a single _climb call
        # over the full [start, end] range so the result is consistent
        # with `_hybrid_path`'s baseline (no-ClimbPlan) climb_dist.
        # Summing per-sub-segment _climb calls would over-estimate
        # distance (each sub-segment uses its own midpoint TAS, and the
        # upper sub-segment's higher TAS inflates its forward distance),
        # which would spuriously trigger `_hybrid_path`'s short_climb
        # regime — turning the climb into a spiral-up plus a full-leg
        # cruise on top.
        active_t, total_dist = self._climb(
            start_altitude, end_altitude,
            wind_along_track=wind_along_track,
        )
        total_time = active_t.to(minute)

        # Holds add time only; they hold ground position so contribute
        # zero forward distance.  Pauses outside [start, end] are
        # silently skipped (matches the docstring contract).
        if pauses:
            pauses_sorted = sorted(pauses, key=lambda p: p[0].m_as(ft))
            for level_alt, hold_dur in pauses_sorted:
                if (
                    level_alt.m_as(ft) <= start_altitude.m_as(ft)
                    or level_alt.m_as(ft) > end_altitude.m_as(ft)
                ):
                    continue
                total_time = total_time + hold_dur.to(minute)

        return total_time, total_dist.to(nmi)

    # ------------------------------------------------------------------
    # Descent
    # ------------------------------------------------------------------

    def _descend(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
        wind_along_track: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """Estimate time and horizontal distance during descent.

        Uses the descent profile (altitude-indexed ROD).  Integration
        strategy matches the climb profile mode.

        ``wind_along_track`` is the signed wind component projected onto
        the ground track (positive = tailwind, negative = headwind).
        When supplied, horizontal distance is integrated as
        ``ground_speed × time``; default ``None`` is still-air
        (backwards-compatible).
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
        if wind_along_track is not None:
            wind_kt = wind_along_track.m_as(ureg.knot)
            ground_speed_kt = max(
                0.0,
                horizontal_speed.m_as(ureg.knot) + wind_kt,
            )
            horizontal_speed = ground_speed_kt * (
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
        climb_plan: Union["ClimbPlan", str, None] = "auto",
    ) -> dict:
        """Calculate time from takeoff to the first waypoint.

        Builds the path via :meth:`_hybrid_path` with ``phase="climb"``
        — 2D Dubins horizontally, integrated ``climb_profile``
        vertically — so the climb-out timing reflects the aircraft's
        calibrated rate-vs-altitude curve, not a constant pitch.

        ``climb_plan`` controls the pre-cruise hold model:

        * ``"auto"`` (default): use this aircraft's
          ``typical_climb_out.explicit_climb_plan`` if defined,
          otherwise no holds.
        * :class:`ClimbPlan`: caller-supplied pauses, used as-is.
        * ``None``: no holds; pure active-climb integration.
        """
        if isinstance(climb_plan, str):
            if climb_plan != "auto":
                raise HyPlanValueError(
                    f"climb_plan must be a ClimbPlan, None, or \"auto\"; "
                    f"got {climb_plan!r}."
                )
            climb_plan = (
                self.typical_climb_out.explicit_climb_plan
                if self.typical_climb_out is not None
                else None
            )
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
            airport_waypoint, waypoint,
            wind=wind, phase="climb",
            climb_plan=climb_plan,
        )

    def time_to_return(
        self,
        waypoint: Waypoint,
        airport: Airport,
        wind: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Calculate time from the last waypoint back to the airport.

        Builds the path via :meth:`_hybrid_path` with ``phase="descent"``
        — 2D Dubins horizontally, integrated ``descent_profile``
        vertically.

        When :attr:`approach_profile` is set, the descent is targeted at
        top-of-approach MSL (= ``airport.elevation +
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
        climb_plan: Optional["ClimbPlan"] = None,
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

        Wind handling: when ``wind`` is supplied, both the 2D path
        (trochoidal geometry) and the vertical integration consume it.
        The vertical phases project the ``(u, v)`` tuple onto the
        great-circle bearing from start to end and pass the signed
        along-track component to :meth:`_climb` / :meth:`_descend`,
        so a headwind shrinks the climb's forward distance and a
        tailwind extends it.
        """
        start_alt = start_waypoint.altitude_msl.to(ureg.feet)  # type: ignore[union-attr]
        end_alt = end_waypoint.altitude_msl.to(ureg.feet)  # type: ignore[union-attr]

        if cruise_altitude is None:
            # Default: the higher of the two endpoints.
            cruise_altitude = start_alt if start_alt >= end_alt else end_alt
        cruise_altitude = cruise_altitude.to(ureg.feet)

        # Service-ceiling check.  Aircraft can sometimes operate
        # transiently above their certified ceiling (e.g., research
        # missions push to "absolute" ceiling); we warn rather than
        # raise so that planning continues, but the user sees that
        # the aircraft model isn't calibrated up there and the
        # extrapolated speed / climb-rate values are best-effort.
        if (
            self.service_ceiling is not None
            and cruise_altitude > self.service_ceiling
        ):
            warnings.warn(
                f"{self.aircraft_type}: requested cruise altitude "
                f"{cruise_altitude.m_as(ureg.feet):.0f} ft exceeds "
                f"service ceiling "
                f"{self.service_ceiling.m_as(ureg.feet):.0f} ft.  "
                "Speed and climb / descent profiles are extrapolated "
                "outside their calibrated range.",
                stacklevel=3,
            )

        cruise_tas = (
            true_air_speed
            if true_air_speed is not None
            else self.cruise_speed_at(cruise_altitude)
        )

        # Bank for the horizontal Dubins arc.  Start from the
        # phase-specific calibrated value, then clip against the
        # load-factor budget given the implicit pitch from the
        # rate-vs-altitude profile of this phase.  For the
        # operating envelope of every calibrated aircraft in the
        # current library, this clip is a no-op (calibrated banks
        # consume <50% of the structural budget).  It binds only
        # when callers force aggressive banks via custom
        # ``bank_by_phase`` values, e.g. modelling a fighter or
        # acrobatic aircraft outside the survey/transport regime.
        bank_calibrated_deg = self.turn_model.bank_by_phase.for_phase(phase)
        bank_max_deg = self.max_bank_under_budget(
            self._implicit_pitch_for_phase(phase, start_alt, cruise_altitude, end_alt),
        )
        bank_deg = min(bank_calibrated_deg, bank_max_deg)
        h_path = DubinsPath2D(
            start_waypoint, end_waypoint,
            speed=cruise_tas, bank_angle=bank_deg, wind=wind,
        )
        L_m = h_path.length.m_as(ureg.meter)
        L_nmi = L_m / 1852.0

        # Project wind onto the great-circle bearing from start to end
        # so the vertical phases see a signed along-track component.
        # A headwind (positive_along_track < 0) shrinks the climb's
        # forward distance; a tailwind extends it.
        wind_along_q: Optional[Quantity] = None
        if wind is not None:
            u_mps, v_mps = wind
            _, az_fwd = pymap3d.vincenty.vdist(
                start_waypoint.latitude, start_waypoint.longitude,
                end_waypoint.latitude, end_waypoint.longitude,
            )
            az_rad = math.radians(float(az_fwd))
            tailwind_mps = u_mps * math.sin(az_rad) + v_mps * math.cos(az_rad)
            wind_along_q = tailwind_mps * (ureg.meter / ureg.second)

        # Climb segment (start.alt -> cruise.alt).  When ``climb_plan``
        # is supplied (and the leg actually climbs), use ``step_climb``
        # so the level-off pauses contribute hold time without forward
        # distance.  Each pause will be emitted as its own
        # ``"loiter"``-tagged phase in the phases dict below.
        climb_pauses_in_range: list = []
        if (
            climb_plan is not None
            and phase == "climb"
            and start_alt < cruise_altitude
        ):
            climb_pauses_in_range = sorted(
                (
                    (lvl.to(ureg.feet), dur.to(ureg.minute))
                    for lvl, dur in climb_plan.pauses
                    if lvl > start_alt and lvl < cruise_altitude
                ),
                key=lambda p: p[0].m_as(ureg.feet),
            )

        if start_alt < cruise_altitude:
            if climb_pauses_in_range:
                climb_t_q, climb_d_q = self.step_climb(
                    start_alt, cruise_altitude, climb_pauses_in_range,
                    wind_along_track=wind_along_q,
                )
            else:
                climb_t_q, climb_d_q = self._climb(
                    start_alt, cruise_altitude,
                    wind_along_track=wind_along_q,
                )
            climb_time_min = climb_t_q.m_as(ureg.minute)
            climb_dist_nmi = climb_d_q.m_as(ureg.nautical_mile)
        else:
            climb_time_min = 0.0
            climb_dist_nmi = 0.0

        # Descent segment (cruise.alt -> end.alt).
        if end_alt < cruise_altitude:
            desc_t_q, desc_d_q = self._descend(
                cruise_altitude, end_alt,
                wind_along_track=wind_along_q,
            )
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

        # Path-angle-constrained climb: when the preferred climb
        # would overrun the lateral leg (short_climb regime) and the
        # aircraft has a configured max FPA, steepen the climb to fit
        # rather than spiraling at departure.  Modeled climb time
        # represents *pure active climb*; mission-specific holds must
        # be added explicitly (ClimbPlan / Waypoint.delay).
        if short_climb and self.climb_path_angle_max_deg is not None:
            dh_ft = (cruise_altitude - start_alt).m_as(ureg.foot)
            L_ft = L_nmi * 6076.115485564
            if L_ft > 0:
                fpa_req_deg = math.degrees(math.atan(dh_ft / L_ft))
                if fpa_req_deg <= self.climb_path_angle_max_deg:
                    scale = L_nmi / climb_dist_nmi
                    climb_time_min *= scale
                    climb_dist_nmi = L_nmi
                    short_climb = False  # fits as steepened climb

        # Path-angle-constrained descent (mirror): when the preferred
        # descent would overrun the lateral leg (short_descent regime)
        # and the aircraft has a configured max FPA, steepen the
        # descent to fit rather than spiraling at end of leg.
        if short_descent and self.descent_path_angle_max_deg is not None:
            dh_ft = (cruise_altitude - end_alt).m_as(ureg.foot)
            L_ft = L_nmi * 6076.115485564
            if L_ft > 0:
                fpa_req_deg = math.degrees(math.atan(dh_ft / L_ft))
                if fpa_req_deg <= self.descent_path_angle_max_deg:
                    scale = L_nmi / descent_dist_nmi
                    descent_time_min *= scale
                    descent_dist_nmi = L_nmi
                    short_descent = False  # fits as steepened descent

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
            elif climb_pauses_in_range:
                # Staged climb: emit one "climb_<i>" sub-phase per
                # climb segment plus one "climb_pause_<i>" loiter-orbit
                # phase per pause.  Forward distance accumulates only
                # during climb segments; orbits hold ground position.
                from ..planning.segments import loiter_orbit_geometry
                cum_d_m_climb = 0.0
                cum_t_min_climb = 0.0
                prev_alt_q = start_alt
                for i, (level_alt, hold_dur) in enumerate(climb_pauses_in_range, start=1):
                    sub_t_q, sub_d_q = self._climb(
                        prev_alt_q, level_alt,
                        wind_along_track=wind_along_q,
                    )
                    sub_t_min = sub_t_q.m_as(ureg.minute)
                    sub_d_nmi = sub_d_q.m_as(ureg.nautical_mile)
                    s_d_m = cum_d_m_climb
                    e_d_m = min(L_m, cum_d_m_climb + sub_d_nmi * nmi_to_m)
                    s_lat, s_lon, s_hdg = h_path.sample_at_distance(s_d_m)
                    e_lat, e_lon, e_hdg = h_path.sample_at_distance(e_d_m)
                    phases[f"climb_{i}"] = {
                        "start_altitude": prev_alt_q,
                        "end_altitude": level_alt,
                        "start_time": (cum_time_min + cum_t_min_climb) * ureg.minute,
                        "end_time": (cum_time_min + cum_t_min_climb + sub_t_min) * ureg.minute,
                        "distance": sub_d_nmi * ureg.nautical_mile,
                        "geometry": h_path.sublinestring(s_d_m, e_d_m),
                        "start_lat": s_lat, "start_lon": s_lon,
                        "end_lat": e_lat, "end_lon": e_lon,
                        "start_heading": s_hdg, "end_heading": e_hdg,
                    }
                    cum_d_m_climb = e_d_m
                    cum_t_min_climb += sub_t_min
                    # Pause orbit at level_alt — hold time, zero forward.
                    orbit_wp = Waypoint(
                        latitude=e_lat, longitude=e_lon,
                        heading=e_hdg, altitude_msl=level_alt,
                    )
                    orbit_geom = loiter_orbit_geometry(
                        orbit_wp, self, phase="climb",
                    )
                    hold_min = hold_dur.m_as(ureg.minute)
                    phases[f"climb_pause_{i}"] = {
                        "start_altitude": level_alt,
                        "end_altitude": level_alt,
                        "start_time": (cum_time_min + cum_t_min_climb) * ureg.minute,
                        "end_time": (cum_time_min + cum_t_min_climb + hold_min) * ureg.minute,
                        "distance": 0.0 * ureg.nautical_mile,
                        "geometry": orbit_geom,
                        "segment_type": "loiter",
                        "start_lat": e_lat, "start_lon": e_lon,
                        "end_lat": e_lat, "end_lon": e_lon,
                        "start_heading": e_hdg, "end_heading": e_hdg,
                    }
                    cum_t_min_climb += hold_min
                    prev_alt_q = level_alt
                # Final climb from last pause to cruise altitude.
                if prev_alt_q < cruise_altitude:
                    sub_t_q, sub_d_q = self._climb(
                        prev_alt_q, cruise_altitude,
                        wind_along_track=wind_along_q,
                    )
                    sub_t_min = sub_t_q.m_as(ureg.minute)
                    sub_d_nmi = sub_d_q.m_as(ureg.nautical_mile)
                    s_d_m = cum_d_m_climb
                    e_d_m = min(L_m, cum_d_m_climb + sub_d_nmi * nmi_to_m)
                    s_lat, s_lon, s_hdg = h_path.sample_at_distance(s_d_m)
                    e_lat, e_lon, e_hdg = h_path.sample_at_distance(e_d_m)
                    final_idx = len(climb_pauses_in_range) + 1
                    phases[f"climb_{final_idx}"] = {
                        "start_altitude": prev_alt_q,
                        "end_altitude": cruise_altitude,
                        "start_time": (cum_time_min + cum_t_min_climb) * ureg.minute,
                        "end_time": (cum_time_min + cum_t_min_climb + sub_t_min) * ureg.minute,
                        "distance": sub_d_nmi * ureg.nautical_mile,
                        "geometry": h_path.sublinestring(s_d_m, e_d_m),
                        "start_lat": s_lat, "start_lon": s_lon,
                        "end_lat": e_lat, "end_lon": e_lon,
                        "start_heading": s_hdg, "end_heading": e_hdg,
                    }
                    cum_d_m_climb = e_d_m
                cum_dist_m = cum_d_m_climb
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
        climb_plan: Optional["ClimbPlan"] = None,
    ) -> dict:
        """Calculate time to fly between two waypoints.

        Hybrid 2D Dubins (horizontal layout) + integrated vertical
        profile.  Returns a dict with ``total_time``, ``phases``, and
        ``dubins_path`` (the 2D path; legacy key name kept for
        backward compatibility — use ``horizontal_path`` in new code).

        Args:
            wind: Optional ``(u_east, v_north)`` wind vector in m/s.
                When provided, horizontal turning arcs become trochoids,
                the 2D path length and timing account for wind drift,
                and the vertical phases project the wind onto the
                great-circle bearing for ground-speed-corrected
                forward distance.
            phase: Which entry of
                :class:`PhaseBankAngles` drives the horizontal Dubins
                turn radius — see :meth:`_hybrid_path`.  Defaults to
                ``"cruise"``.
            climb_plan: Optional :class:`ClimbPlan` with level-off
                pauses to insert during the climb.  Only takes effect
                when ``phase == "climb"`` and the leg actually climbs.
        """
        return self._hybrid_path(
            start_waypoint, end_waypoint,
            true_air_speed=true_air_speed, wind=wind, phase=phase,
            climb_plan=climb_plan,
        )

