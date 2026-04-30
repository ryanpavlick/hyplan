"""Concrete aircraft definitions for HyPlan.

Each class is a thin :class:`~hyplan.aircraft.Aircraft` subclass that fills
in the performance parameters for a specific airborne science platform.

For backwards compatibility every class defined here is re-exported from
:mod:`hyplan.aircraft`, so existing code that does
``from hyplan.aircraft import NASA_ER2`` continues to work.
"""

from __future__ import annotations

from ._base import (
    Aircraft,
    ApproachProfile,
    CasMachSchedule,
    TasSchedule,
    VerticalProfile,
    TurnModel,
    PhaseBankAngles,
    PerformanceConfidence,
    SourceRecord,
)
from ..units import ureg

__all__ = [
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_C20A",
    "NASA_P3",
    "NASA_WB57",
    "NASA_B777",
    "Dash8",
    "KingAirA90",
    "KingAirB200",
    "C130",
    "BAe146",
    "Learjet",
    "TwinOtter",
]

# ---------------------------------------------------------------------------
# Helper: build a descent TasSchedule from a cruise TasSchedule with
# speed reduction applied.
# ---------------------------------------------------------------------------

def _descent_schedule_from_cruise(
    cruise_schedule: TasSchedule, speed_reduction_kt: float
) -> TasSchedule:
    """Build a descent speed schedule by reducing cruise TAS at all points."""
    return TasSchedule(
        points=[
            (alt, max(0, spd.m_as(ureg.knot) - speed_reduction_kt) * ureg.knot)
            for alt, spd in cruise_schedule.points
        ]
    )


# ---------------------------------------------------------------------------
# NASA high-altitude research aircraft
# ---------------------------------------------------------------------------

class NASA_ER2(Aircraft):
    """NASA ER-2 high-altitude research aircraft.

    Operates at 70,000 ft, acquiring data above 95% of the Earth's
    atmosphere.  Based at NASA Armstrong Flight Research Center (AFRC).

    Speed schedule (Moving Lines brochure): TAS = 70 + alt_m * 0.0071 (m/s).

    Vertical-rate and approach behavior calibrated from 17 NASA AFRC IWG1
    in-situ flight logs (2023-02 to 2025-08, ~64 000 cruise fixes above
    60 kft).  See [notebooks/er2_calibration/iwg1_calibration.ipynb] for
    the full derivation: per-altitude-bin |VS| medians, breakpoint
    selection rules, and validation against per-sortie observed timing.

    Vertical-rate highlights from the calibration:

    * Step climb at 19-21 kft (fuel-burn-driven; the average operational
      profile across the 17-sortie set shows VS dropping from ~4400 fpm
      to ~540 fpm in the step band, recovering immediately at 23 kft to
      ~4800 fpm, then declining through the cruise band).
    * Two-regime descent: peak idle-power |VS| ~3675 fpm at top-of-
      descent, decaying to ~840 fpm at top-of-approach as the aircraft
      configures for the terminal pattern.
    * Empirical 2.5° glideslope on the terminal approach (shallower
      than standard 3° ILS — ER-2's approach geometry as flown across
      the IWG1 sortie set; touchdown estimate uses 6 sorties with
      fixes ≤ 50 ft AGL after ground-taxi trim).

    See also:
        `https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC <https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC>`_
    """

    def __init__(self):
        # Distinct climb / cruise / descent TAS schedules from IWG1 per-altitude-bin
        # medians (n=17 sorties, 2-kft bins, all bins with n >= 30 fixes).
        # Pre-Item-4 these were aliased to a single brochure 2-point linear curve;
        # the IWG1 data shows climb / descent are within ~5 kt of each other at
        # any given altitude (both reflect pitched flight) but cruise sits ~10-20
        # kt higher in the 40-60 kft band.
        climb_schedule = TasSchedule(points=[
            (    0 * ureg.feet, 120 * ureg.knot),  # extrapolated; 3 kft = 134 kt
            (20000 * ureg.feet, 224 * ureg.knot),  # IWG1 19 kft median
            (40000 * ureg.feet, 308 * ureg.knot),  # IWG1 41 kft median
            (60000 * ureg.feet, 397 * ureg.knot),  # IWG1 61 kft median
            (70000 * ureg.feet, 410 * ureg.knot),  # extrapolated above 67 kft
        ])
        cruise_schedule = TasSchedule(points=[
            (    0 * ureg.feet, 130 * ureg.knot),  # brochure-equivalent low-alt
            (50000 * ureg.feet, 374 * ureg.knot),  # IWG1 51 kft median
            (60000 * ureg.feet, 388 * ureg.knot),  # IWG1 59 kft median
            (65000 * ureg.feet, 401 * ureg.knot),  # IWG1 65 kft median (n=51383)
            (70000 * ureg.feet, 410 * ureg.knot),  # extrapolated above 67 kft
        ])
        descent_schedule = TasSchedule(points=[
            (    0 * ureg.feet,  90 * ureg.knot),  # IWG1 1 kft median (touchdown)
            (20000 * ureg.feet, 221 * ureg.knot),  # IWG1 19 kft median
            (40000 * ureg.feet, 318 * ureg.knot),  # IWG1 41 kft median
            (60000 * ureg.feet, 399 * ureg.knot),  # IWG1 59 kft median
            (70000 * ureg.feet, 410 * ureg.knot),  # extrapolated above 67 kft
        ])
        super().__init__(
            aircraft_type="ER-2",
            tail_number="NASA 806",
            operator="NASA AFRC",
            service_ceiling=70000 * ureg.feet,
            approach_speed=130 * ureg.knot,  # legacy scalar; approach_profile is preferred
            climb_schedule=climb_schedule,
            cruise_schedule=cruise_schedule,
            descent_schedule=descent_schedule,
            # Calibrated 8-point climb profile (was 2-point linear).
            # Step climb at 19-21 kft is the load-out / fuel-burn level-off.
            # The post-step recovery anchor at 23 kft (4778 fpm) is critical
            # — the actual aircraft recovers to its peak climb rate
            # immediately after the step.  Without it, linear interpolation
            # 21 kft → 55 kft would imply a constant ~700 fpm climb
            # through the entire 21-55 kft band, off by 4× from reality.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 5000 * ureg.feet / ureg.minute),  # SL anchor (brochure)
                (17000 * ureg.feet, 4432 * ureg.feet / ureg.minute),  # steady-climb anchor
                (19000 * ureg.feet, 1050 * ureg.feet / ureg.minute),  # step start (n=17 median)
                (21000 * ureg.feet,  540 * ureg.feet / ureg.minute),  # step bottom (n=17 median)
                (23000 * ureg.feet, 4778 * ureg.feet / ureg.minute),  # post-step recovery
                (35000 * ureg.feet, 2693 * ureg.feet / ureg.minute),  # mid-climb anchor
                (49000 * ureg.feet, 1290 * ureg.feet / ureg.minute),  # upper-mid anchor
                (66000 * ureg.feet,  200 * ureg.feet / ureg.minute),  # operational ceiling
            ]),
            # Calibrated 6-point descent profile (was 3-point).
            # Same lesson as the climb_profile recalibration: 3 anchors
            # spanning 5 kft to 66 kft hide the structure.  IWG1 per-bin
            # medians show distinct regimes — slow approach-prep below
            # 13 kft (~1000-1700 fpm), steady high-altitude descent at
            # 25-55 kft (~3000-3500 fpm), and a peak around 41 kft.
            # Without intermediate anchors the linear interp from
            # 5242 ft (low) to 41000 ft (peak) implies a constant
            # ~2000 fpm across the entire 5-41 kft range, ~30% too slow.
            # Covers cruise -> top-of-approach MSL only; approach_profile
            # owns the terminal segment from there to touchdown.
            descent_profile=VerticalProfile(points=[
                ( 5242 * ureg.feet, 1024 * ureg.feet / ureg.minute),  # top-of-approach MSL (slowing for approach)
                (13000 * ureg.feet, 1728 * ureg.feet / ureg.minute),  # mid-low transition
                (25000 * ureg.feet, 3008 * ureg.feet / ureg.minute),  # steady high regime (start)
                (41000 * ureg.feet, 3456 * ureg.feet / ureg.minute),  # peak |VS|
                (55000 * ureg.feet, 3264 * ureg.feet / ureg.minute),  # still high
                (66000 * ureg.feet, 3500 * ureg.feet / ureg.minute),  # top-of-descent
            ]),
            # Calibrated terminal-arrival profile (3 kft AGL -> touchdown).
            # 2.5° glideslope is the empirical median over 846 IWG1
            # approach-phase fixes; touchdown 65 kt is the per-sortie
            # median TAS in the lowest 50 ft AGL band, n=6 sorties
            # contributing.  Plausible for ER-2's flare regime — the
            # airframe has no conventional flaps, so it decelerates
            # close to stall in ground effect before touchdown.
            approach_profile=ApproachProfile(
                speed_schedule=TasSchedule(points=[
                    (   0 * ureg.feet,  65 * ureg.knot),  # touchdown
                    ( 200 * ureg.feet,  71 * ureg.knot),  # interpolated
                    (1000 * ureg.feet,  94 * ureg.knot),  # interpolated
                    (3000 * ureg.feet, 151 * ureg.knot),  # top-of-approach
                ]),
                top_of_approach_agl=3000 * ureg.feet,
                glideslope_deg=2.51,
            ),
            # max_bank_deg=30 is the brochure / envelope ceiling — well-
            # supported by IWG1 (p90 < 30° in every altitude band).
            # bank_by_phase carries the calibrated *typical-operations*
            # medians from the IWG1 sortie set (n=12 686 turn fixes).
            # These are metadata today: compute_flight_plan's Dubins solver
            # consumes max_bank_angle (the scalar envelope), and
            # loiter_orbit_geometry consumes bank_by_phase.cruise_deg.
            turn_model=TurnModel(
                max_bank_deg=30.0,
                bank_by_phase=PhaseBankAngles(
                    climb_deg=11.0,     # 10-30 kft band p50 (climb regime)
                    cruise_deg=20.0,    # 50-70 kft band p50 (n=9782 fixes)
                    descent_deg=16.0,   # 30-50 kft band p50 (descent transit)
                    approach_deg=9.0,   # 0-10 kft band p50 (terminal area)
                ),
            ),
            engine_type="jet",
            range=5000 * ureg.nautical_mile,
            endurance=8 * ureg.hour,
            useful_payload=2900 * ureg.pound,
            sources=[
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science, Moving Lines project",
                    confidence=0.6,
                ),
                SourceRecord(
                    source_type="iwg1",
                    reference=(
                        "NASA AFRC IWG1 in-situ flight logs, n=17 sorties "
                        "2023-02 to 2025-08; calibrated climb step, "
                        "two-regime descent, and approach_profile with "
                        "2.6° empirical glideslope"
                    ),
                    confidence=0.8,
                ),
            ],
        )


# ---------------------------------------------------------------------------
# Gulfstream business jets (NASA)
# ---------------------------------------------------------------------------

class NASA_GIII(Aircraft):
    """NASA Gulfstream III (NASA 520) research aircraft.

    Operated by NASA Langley Research Center (LaRC).

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 292 * ureg.knot),
            (45000 * ureg.feet, 459 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="Gulfstream III",
            tail_number="NASA 520",
            operator="NASA LaRC",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 49),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 4000 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=3767 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            useful_payload=2610 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.5, cruise=0.5, descent=0.4, turns=0.5,
            ),
            sources=[SourceRecord(
                source_type="brochure",
                reference="NASA Airborne Science fact sheet; EUROCONTROL GLF3",
                confidence=0.5,
            )],
        )


class NASA_GIV(Aircraft):
    """NASA Gulfstream IV (NASA 817) research aircraft.

    Twin turbofan operated by NASA Armstrong Flight Research Center (AFRC).

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 292 * ureg.knot),
            (45000 * ureg.feet, 459 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="Gulfstream IV",
            tail_number="NASA 817",
            operator="NASA AFRC",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 49),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 4000 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=5130 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            useful_payload=5610 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.4, cruise=0.4, descent=0.35, turns=0.5,
            ),
            sources=[SourceRecord(
                source_type="brochure",
                reference="NASA Airborne Science fact sheet; EUROCONTROL GLF4",
                notes="[ESTIMATED — same as GIII, needs GIV AFM data]",
                confidence=0.4,
            )],
        )


class NASA_GV(Aircraft):
    """NASA Gulfstream V research aircraft.

    Operated by NASA Armstrong Flight Research Center (AFRC).
    Service ceiling 51,000 ft, cruise speed 500 kt (Mach 0.80).
    Currently undergoing modifications expected to conclude ~August 2026.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_V_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_V_-_AFRC>`_
    """

    def __init__(self):
        super().__init__(
            aircraft_type="Gulfstream V",
            tail_number="Unknown",
            operator="NASA AFRC",
            service_ceiling=51000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            climb_schedule=CasMachSchedule(
                cas=280 * ureg.knot, mach=0.74, crossover_ft=28000,
            ),
            cruise_schedule=CasMachSchedule(
                cas=300 * ureg.knot, mach=0.80, crossover_ft=30000,
            ),
            descent_schedule=CasMachSchedule(
                cas=290 * ureg.knot, mach=0.78, crossover_ft=30000,
            ),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 3800 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 3200 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2400 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 700 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 300 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1200 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1800 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2200 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2500 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2200 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 1800 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(
                bank_by_phase=PhaseBankAngles(
                    climb_deg=20, cruise_deg=25, descent_deg=20, approach_deg=15,
                ),
                max_bank_deg=30.0,
            ),
            engine_type="jet",
            range=5500 * ureg.nautical_mile,
            endurance=13 * ureg.hour,
            confidence=PerformanceConfidence(
                climb=0.45, cruise=0.50, descent=0.35, turns=0.50,
            ),
            sources=[
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet, GV at AFRC",
                    confidence=0.5,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="EUROCONTROL Aircraft Performance Database, GLF5",
                    confidence=0.5,
                ),
            ],
        )


class NASA_C20A(Aircraft):
    """NASA C-20A (Gulfstream III variant, NASA 502) research aircraft.

    Obtained from the U.S. Air Force in 2003. Primary platform for
    UAVSAR missions. Operated by NASA AFRC.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 292 * ureg.knot),
            (45000 * ureg.feet, 460 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="C-20A",
            tail_number="NASA 502",
            operator="NASA AFRC",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 49),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 3500 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=3400 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=2500 * ureg.pound,
        )


# ---------------------------------------------------------------------------
# NASA turboprops
# ---------------------------------------------------------------------------

class NASA_P3(Aircraft):
    """NASA P-3 Orion (NASA 426) airborne science laboratory.

    Four-engine turboprop capable of long-duration flights (8–14 hours)
    and large payloads up to 18,000 lbs. Operated by NASA Wallops Flight
    Facility (WFF).

    Speed profile from Moving Lines: TAS = 110 + alt_m * 0.007 (m/s),
    capped at 155 m/s (~301 kt) above ~21,000 ft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/P-3_Orion <https://airbornescience.nasa.gov/aircraft/P-3_Orion>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 214 * ureg.knot),
            (21091 * ureg.feet, 301 * ureg.knot),
            (32000 * ureg.feet, 301 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="P-3 Orion",
            tail_number="NASA 426",
            operator="NASA WFF",
            service_ceiling=32000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 29),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 3500 * ureg.feet / ureg.minute),
                (32000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=3800 * ureg.nautical_mile,
            endurance=12 * ureg.hour,
            useful_payload=18000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.45, cruise=0.40, descent=0.35, turns=0.50,
            ),
            sources=[SourceRecord(
                source_type="brochure",
                reference="NASA Airborne Science fact sheet; Moving Lines TAS formula",
                confidence=0.5,
            )],
        )


class NASA_WB57(Aircraft):
    """NASA WB-57 (NASA 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field.
    Operates up to 60,000 ft with 8,800 lbs useful payload.

    See also:
        `https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC <https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 136 * ureg.knot),
            (60000 * ureg.feet, 410 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="WB-57",
            tail_number="NASA 927",
            operator="NASA JSC",
            service_ceiling=60000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=cruise,
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 5000 * ureg.feet / ureg.minute),
                (60000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=2500 * ureg.nautical_mile,
            endurance=6.5 * ureg.hour,
            useful_payload=8800 * ureg.pound,
        )


class NASA_B777(Aircraft):
    """NASA Boeing 777 long-range research aircraft.

    Operated by NASA Langley Research Center (LaRC). Very large payload
    capacity (75,000 lbs) and long endurance (18 hours).
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 350 * ureg.knot),
            (43000 * ureg.feet, 487 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="B777",
            tail_number="Unknown",
            operator="NASA LaRC",
            service_ceiling=43000 * ureg.feet,
            approach_speed=150 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 30),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2500 * ureg.feet / ureg.minute),
                (43000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=9000 * ureg.nautical_mile,
            endurance=18 * ureg.hour,
            useful_payload=75000 * ureg.pound,
        )


# ---------------------------------------------------------------------------
# King Air and Dash 8 turboprops
# ---------------------------------------------------------------------------

class Dash8(Aircraft):
    """DHC-8 Dash 8 twin-turboprop aircraft."""

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 170 * ureg.knot),
            (25000 * ureg.feet, 243 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="Dash 8",
            tail_number="Unknown",
            operator="Unknown",
            service_ceiling=25000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 15),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=950 * ureg.nautical_mile,
            endurance=5 * ureg.hour,
            useful_payload=15000 * ureg.pound,
        )


class KingAirA90(Aircraft):
    """Beechcraft King Air A90 twin-turboprop aircraft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 170 * ureg.knot),
            (30000 * ureg.feet, 230 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="King Air 90",
            tail_number="Unknown",
            operator="Unknown",
            service_ceiling=30000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 10),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1800 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1500 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=2950 * ureg.pound,
        )


class KingAirB200(Aircraft):
    """Beechcraft King Air B200 twin-turboprop aircraft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 185 * ureg.knot),
            (35000 * ureg.feet, 250 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="Unknown",
            operator="Unknown",
            service_ceiling=35000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 10),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1632 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=4250 * ureg.pound,
        )


# ---------------------------------------------------------------------------
# Other research / military aircraft
# ---------------------------------------------------------------------------

class C130(Aircraft):
    """C-130H Hercules four-engine turboprop transport / research aircraft.

    Speed profile from Moving Lines: TAS = 130 + alt_m * 0.0075 (m/s),
    capped at 175 m/s (~340 kt) above ~19,685 ft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF <https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 253 * ureg.knot),
            (19685 * ureg.feet, 340 * ureg.knot),
            (25000 * ureg.feet, 340 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="C-130H Hercules",
            tail_number="Unknown",
            operator="Various",
            service_ceiling=25000 * ureg.feet,
            approach_speed=115 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 29),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=20.0),
            engine_type="turboprop",
            range=2500 * ureg.nautical_mile,
            endurance=10 * ureg.hour,
            useful_payload=45000 * ureg.pound,
        )


class BAe146(Aircraft):
    """BAe-146-301 atmospheric research aircraft (G-LUXE).

    Operated by the UK FAAM. Speed profile from Moving Lines:
    TAS = 130 + alt_m * 0.002 (m/s).

    See also:
        `https://faam.ac.uk/ <https://faam.ac.uk/>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 253 * ureg.knot),
            (28000 * ureg.feet, 286 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="BAe-146",
            tail_number="Unknown",
            operator="FAAM",
            service_ceiling=28000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 29),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1000 * ureg.feet / ureg.minute),
                (28000 * ureg.feet, 100 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1000 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=20.0),
            engine_type="jet",
            range=1800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=10000 * ureg.pound,
        )


class Learjet(Aircraft):
    """Learjet high-altitude research aircraft.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        `https://airbornescience.nasa.gov/aircraft/Learjet_25 <https://airbornescience.nasa.gov/aircraft/Learjet_25>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 194 * ureg.knot),
            (35000 * ureg.feet, 430 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="Learjet",
            tail_number="Unknown",
            operator="Various",
            service_ceiling=35000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 39),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 4000 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=1500 * ureg.nautical_mile,
            endurance=4 * ureg.hour,
            useful_payload=3000 * ureg.pound,
        )


class TwinOtter(Aircraft):
    """DHC-6 Twin Otter STOL twin-turboprop utility aircraft.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        `https://airbornescience.nasa.gov/aircraft/Twin_Otter_-_CIRPAS_-_NPS <https://airbornescience.nasa.gov/aircraft/Twin_Otter_-_CIRPAS_-_NPS>`_
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 97 * ureg.knot),
            (10000 * ureg.feet, 150 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="Unknown",
            operator="Various",
            service_ceiling=25000 * ureg.feet,
            approach_speed=70 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=_descent_schedule_from_cruise(cruise, 8),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 430 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 50 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 430 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=25.0),
            engine_type="turboprop",
            range=800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=4000 * ureg.pound,
        )
