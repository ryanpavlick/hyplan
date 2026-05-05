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
    ClimbOutPolicy,
    ClimbPlan,
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

    Speed schedules, vertical-rate profile, and approach behavior calibrated
    from 17 NASA AFRC IWG1
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
            # Active-climb-only median fit (138 mission sorties, IWG1).
            # Filter: climb-phase fixes with vertical_rate >= 1500 fpm
            # AND outside ±2 kft bands around {FL240, FL260, FL356}
            # (the known typical level-off / weight-band hold altitudes).
            # Median per 5-kft bin; smoothed slightly at the FL15-25
            # plateau to keep VS strictly monotonic above the SL peak.
            #
            # Replaces the earlier wall-clock-fit p75-mixed profile
            # (`6016 fpm @ FL150` vs the brochure SL ROC of 5000 fpm
            # was a tell that the values were absorbing level-off
            # fixes rather than representing pure climb performance).
            #
            # The typical pre-cruise mission overhead these anchors
            # used to absorb is now represented explicitly via
            # `typical_climb_out.explicit_climb_plan` — see below.
            #
            # See `notebooks/er2_calibration/iwg1_calibration.ipynb` §5b
            # for the active-only derivation.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 5000 * ureg.feet / ureg.minute),  # SL anchor (brochure)
                (15000 * ureg.feet, 3800 * ureg.feet / ureg.minute),  # active median (smoothed for monotonicity)
                (25000 * ureg.feet, 3789 * ureg.feet / ureg.minute),  # active median (FL20-25 active band)
                (35000 * ureg.feet, 2455 * ureg.feet / ureg.minute),  # active median
                (45000 * ureg.feet, 1789 * ureg.feet / ureg.minute),  # active median
                (55000 * ureg.feet, 1615 * ureg.feet / ureg.minute),  # active median
                (66000 * ureg.feet,  200 * ureg.feet / ureg.minute),  # operational ceiling (brochure)
            ]),
            # Median-based descent profile (replaces earlier peak-based
            # 6-anchor calibration).  iwg1_calibration §9 showed the
            # peak-based TOD anchor over-estimated descent rate at top
            # by 4-7x against the observed median (mod 3500 fpm vs obs
            # ~600 fpm at 60-66 kft, n=5793).  Three anchors from the
            # median rule: bottom = median |VS| at top-of-approach band,
            # mid = peak median |VS| in 25-45 kft band (the steep
            # regime), top = median |VS| across cruise altitudes.
            descent_profile=VerticalProfile(points=[
                ( 5260 * ureg.feet,  735 * ureg.feet / ureg.minute),  # top-of-approach MSL
                (35000 * ureg.feet, 3022 * ureg.feet / ureg.minute),  # steep-regime peak
                (66000 * ureg.feet, 1444 * ureg.feet / ureg.minute),  # top-of-descent
            ]),
            # Descent_profile retained from the 22-sortie calibration.
            # The 136-sortie recalibration shifts these anchors (5240/878,
            # 43000/2880, 66000/1290) to better match the fleet median,
            # but regresses NM17 B's descent residual by +20 min — descent
            # VS has high mission-specific variability (lateral leg
            # length, ATC routing) that no single profile captures.  The
            # 22-sortie subset happens to fit our planned-vs-flown
            # validation pairs (NM17 B, CO07v4, CO06) better.
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
            # medians from the IWG1 sortie set (n=12 686 turn fixes) and
            # is consumed by Aircraft._hybrid_path (per-phase turn radius)
            # and loiter_orbit_geometry (phase-aware orbit radius).
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
            # Calibrated max descent flight path angle.  IWG1 cross-fleet
            # descent FPA distribution: median 2°, p90 7°, p99 10°.  6° is
            # a conservative envelope that captures normal operations
            # (covers p90 in every altitude band except 10-15 kft where
            # p90 reaches 9°) and avoids the sustained steep descents
            # only seen in approach-prep / abort / unusual profiles.
            # When a leg is shorter than the preferred descent distance,
            # _hybrid_path scales descent VS up to fit rather than
            # spiraling at end of leg.
            descent_path_angle_max_deg=6.0,
            # climb_path_angle_max_deg=6.0 disables _hybrid_path's
            # short_climb spiral-up absorption — the climb either
            # fits in the leg (steepened to FPA <= 6°) or, on legs
            # too short to fit, the planner falls back to the
            # spiral-up regime regardless of this cap.  With the
            # active-climb-only climb_profile (Phase 2), pure
            # _climb() integration is now what the model represents,
            # and the typical pre-cruise mission overhead is
            # injected explicitly via the typical_climb_out below.
            climb_path_angle_max_deg=6.0,
            typical_climb_out=ClimbOutPolicy(
                # Phase 3: climb_profile is active-only and the
                # planner reads `explicit_climb_plan` via
                # `compute_flight_plan(climb_plan="auto")` (the
                # default).  This recovers empirical-typical
                # wall-clock TOC honestly: the holds appear as
                # explicit `loiter` segments in the plan dataframe,
                # not absorbed into the climb_profile values.
                absorbed_in_climb_profile=False,
                typical_holds=[
                    (24_000 * ureg.feet,  1 * ureg.minute),  # FL240 brief level-off
                    (26_000 * ureg.feet,  1 * ureg.minute),  # FL260 brief level-off
                    (35_600 * ureg.feet, 12 * ureg.minute),  # FL356 weight-band .delay (median)
                ],
                typical_overhead_min=14.0,
                notes=(
                    "climb_profile is active-climb-only (138-sortie IWG1 "
                    "median per 5-kft bin, VS >= 1500 fpm filter, hold "
                    "bands excluded). Pre-cruise mission overhead is "
                    "injected via explicit_climb_plan when the caller "
                    "uses climb_plan='auto' (the default).  The 12-min "
                    "FL356 hold is the median of the 138-sortie cache; "
                    "individual sorties range 0-25+ min.  Power users "
                    "wanting pure aircraft physics pass climb_plan=None "
                    "to bypass the typical-mission absorption.  See "
                    "notebooks/er2_calibration/planned_vs_flown.ipynb §16."
                ),
                explicit_climb_plan=ClimbPlan(pauses=[
                    # Median weight-band .delay duration across the
                    # 138-sortie cache.  Single representative pause
                    # at FL356 — brief FL240/FL260 level-offs are
                    # too short to be worth modelling explicitly and
                    # contribute <2 min combined.
                    (35_600 * ureg.feet, 12 * ureg.minute),
                ]),
            ),
            sources=[
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet, ER-2 at AFRC",
                    confidence=0.6,
                ),
                SourceRecord(
                    source_type="iwg1",
                    reference=(
                        "NASA AFRC IWG1 in-situ flight logs, n=138 mission "
                        "sorties 2023-03 to 2026-05 (NASA 806 + 809); "
                        "calibrated climb (active-only median, VS >= 1500 fpm "
                        "outside known hold bands), descent (median), "
                        "approach_profile with empirical glideslope, and "
                        "descent_path_angle_max_deg (p90 envelope)"
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
                reference="NASA Airborne Science fact sheet, P-3 Orion at WFF",
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

    Operated by the UK FAAM.

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
