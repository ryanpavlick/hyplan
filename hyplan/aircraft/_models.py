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
    60 kft).  See [notebooks/calibration/er2/iwg1_calibration.ipynb] for
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
            # Active-climb median (VS >= 1500 fpm, ±2-kft bands around
            # {FL240, FL260, FL356} excluded — known weight-management
            # hold altitudes), 5-kft bins, n>=100/bin across 199 IWG1
            # sorties.  Ships per-bin medians directly; non-monotone
            # mid-altitude bumps reflect real hold-band-adjacent climb
            # behaviour and are preserved.  The SL anchor is data-driven
            # (3245 fpm) — the ER-2 is U-2-derived but isn't a U-2,
            # and the active-climb data shows actual SL ROC well below
            # the U-2 brochure 5000 fpm value.
            #
            # See `notebooks/calibration/er2/iwg1_calibration.ipynb` §5b
            # for the active-only derivation and §6b for the IQR fit
            # validation.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 3245 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 4290 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 4083 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 3378 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 3841 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 3621 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 3212 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2453 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2072 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1724 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 1618 * ureg.feet / ureg.minute),
                (55000 * ureg.feet, 1607 * ureg.feet / ureg.minute),
                # Operational ceiling residual; FL600+ active-climb bins
                # are too sparse (n<30) for a reliable empirical fit.
                (66000 * ureg.feet,  200 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins
            # across 199 sorties.  Replaces the earlier sparse 3-anchor
            # construction whose bottom anchor (735 fpm) was driven by
            # approach-speed fixes (low-VS gentle final descent), not
            # active descent — predictions for any FL030-FL250 transit
            # were systematically too slow by 800-1500 fpm.  The new
            # bin-median profile sits inside every IQR for FL000-FL550
            # where data is dense; FL600+ has sparser coverage and
            # natural steepening in the data is preserved here.  See
            # ``notebooks/calibration/er2/iwg1_calibration.ipynb`` §6b
            # for the IQR + median visualization.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1630 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1871 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2042 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2312 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2477 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2632 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2720 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2770 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2755 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 2783 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 3051 * ureg.feet / ureg.minute),
                (55000 * ureg.feet, 2928 * ureg.feet / ureg.minute),
                (60000 * ureg.feet, 2194 * ureg.feet / ureg.minute),
                (65000 * ureg.feet, 1849 * ureg.feet / ureg.minute),
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
                    "notebooks/calibration/er2/planned_vs_flown.ipynb §16."
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
        # Calibrated against 153 IWG1 sorties from NASA 520, combining
        # the local "n520NA_g3_alltracks.csv" delivery (2025-07 through
        # 2026-04) with the public NASA ASP archive
        # (asp-archive.arc.nasa.gov/N520NA, FY2024-FY2026).  See
        # ``notebooks/calibration/giii/iwg1_calibration.ipynb``.
        super().__init__(
            aircraft_type="Gulfstream III",
            tail_number="NASA 520",
            operator="NASA LaRC",
            # p99 of per-sortie peak altitude across 153 sorties; the
            # broader sample shows the 45000 ft AFM ceiling is reached
            # in some campaigns.
            service_ceiling=45000 * ureg.feet,
            approach_speed=139 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 150 * ureg.knot),  # rotation
                (10000 * ureg.feet, 339 * ureg.knot),
                (20000 * ureg.feet, 421 * ureg.knot),
                (30000 * ureg.feet, 446 * ureg.knot),
                (40000 * ureg.feet, 436 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (25000 * ureg.feet, 406 * ureg.knot),
                (30000 * ureg.feet, 472 * ureg.knot),
                (35000 * ureg.feet, 449 * ureg.knot),
                (40000 * ureg.feet, 455 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 180 * ureg.knot),  # final approach
                (10000 * ureg.feet, 329 * ureg.knot),
                (20000 * ureg.feet, 410 * ureg.knot),
                (30000 * ureg.feet, 453 * ureg.knot),
                (40000 * ureg.feet, 463 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins, n>=30/bin.
            # Peak ROC near FL050 (250-KCAS-below-FL100 constraint at SL);
            # profile is non-monotone by design.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2113 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2616 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2253 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2135 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1916 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1926 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1853 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1688 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1840 * ureg.feet / ureg.minute),
                (45000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            # Descent VS peaks near FL150 (VMO descent in CAS) and
            # declines in upper levels (Mach-limited descent).
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1778 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1907 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2155 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2492 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2425 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2365 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2481 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2088 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2007 * ureg.feet / ureg.minute),
            ]),
            # p90 |Roll| during turn-state fixes (gate >5°, n=40,523):
            # 30°, the operational ceiling the data shows the aircraft
            # actually willing to use.  Median (20°) is dragged down by
            # small in-cruise course corrections; p90 captures the
            # bank used during real survey-line transitions.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=3767 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            useful_payload=2610 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 520 IWG1 calibration, n=153 sorties (in-house + ASP archive FY2024-FY2026)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet; EUROCONTROL GLF3",
                    confidence=0.5,
                ),
            ],
            # Vs0 at landing config, MLW per Gulfstream III AFM.  IWG1
            # data never sees stall (slowest cruise TAS is ~250 kt at
            # FL040), so this is brochure-derived not measured.
            stall_speed_cas=105 * ureg.knot,
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
        # Vertical profiles, approach, bank, and stall calibrated against
        # 100 IWG1 sorties from NASA 95 (data/n95na_2019.csv,
        # n95na_2020_2022.csv, n95na_2023_2026.csv), 2019 through 2026.
        # See ``notebooks/calibration/gv/iwg1_calibration.ipynb`` for the
        # active-only fits.  CasMachSchedule speed parameters retained
        # — the data confirms M0.80 cruise above ~FL300, validating the
        # existing schedule shape.
        super().__init__(
            aircraft_type="Gulfstream V",
            tail_number="NASA 95",
            operator="NASA AFRC",
            # Certified 51 kft is achievable; this dataset shows the
            # JSC G-V mostly cruises at FL400 (p99 of peak alt = 45 kft)
            # for science reasons, but the airframe is capable above.
            service_ceiling=51000 * ureg.feet,
            approach_speed=126 * ureg.knot,
            climb_schedule=CasMachSchedule(
                cas=280 * ureg.knot, mach=0.74, crossover_ft=28000,
            ),
            cruise_schedule=CasMachSchedule(
                cas=300 * ureg.knot, mach=0.80, crossover_ft=30000,
            ),
            descent_schedule=CasMachSchedule(
                cas=290 * ureg.knot, mach=0.78, crossover_ft=30000,
            ),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins,
            # n>=30/bin.  Replaces the earlier brochure-grade points
            # which over-stated SL ROC (3800 -> ~2200 actual).
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2196 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2521 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2284 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2266 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2003 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1896 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1804 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1734 * ureg.feet / ureg.minute),
                # Residual rate at certified ceiling so the integrator
                # terminates cleanly.
                (51000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1703 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1806 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1873 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2209 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2331 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2481 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2432 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2091 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1824 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(
                bank_by_phase=PhaseBankAngles(
                    climb_deg=20, cruise_deg=25, descent_deg=20, approach_deg=15,
                ),
                # AFM normal-ops bank (30°), not the data p90 (27°,
                # n=40,626 turn fixes).  AFRC G-V missions in this
                # dataset are transit / sampling, not survey grids,
                # so the IWG1 sample doesn't see the tight-bank
                # flight-line transitions a science planner might use.
                max_bank_deg=30.0,
            ),
            engine_type="jet",
            range=5500 * ureg.nautical_mile,
            endurance=13 * ureg.hour,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.5, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 95 IWG1 calibration, n=101 sorties (in-house + ASP archive FY2019-FY2021)",
                    confidence=0.85,
                ),
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
            # Vs0 at landing config, MLW per Gulfstream V AFM.
            # Brochure-derived; IWG1 data never sees stall.
            stall_speed_cas=104 * ureg.knot,
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
        # Calibrated against 252 IWG1 sorties from NASA 426 (P3-B),
        # combining a local "p3_alltracks.csv" delivery with the public
        # NASA ASP archive (asp-archive.arc.nasa.gov/N426NA, FY2003
        # through FY2025).  Filter MAX_PEAK_ALT_FT=32000 in the
        # calibration notebook excludes 4 sorties peaking 35-55 kft
        # — those are tail-number reassignments where N426NA was
        # recorded for a different (jet-class) airframe; not P-3 data.
        # See ``notebooks/calibration/p3/iwg1_calibration.ipynb``.
        super().__init__(
            aircraft_type="P-3 Orion",
            tail_number="NASA 426",
            operator="NASA WFF",
            # p99 of per-sortie peak altitude across 252 valid sorties.
            service_ceiling=27000 * ureg.feet,
            approach_speed=132 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 110 * ureg.knot),  # rotation
                ( 5000 * ureg.feet, 250 * ureg.knot),
                (10000 * ureg.feet, 264 * ureg.knot),
                (15000 * ureg.feet, 277 * ureg.knot),
                (20000 * ureg.feet, 288 * ureg.knot),
                (25000 * ureg.feet, 306 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (15000 * ureg.feet, 320 * ureg.knot),
                (20000 * ureg.feet, 333 * ureg.knot),
                (25000 * ureg.feet, 346 * ureg.knot),
                (28000 * ureg.feet, 346 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 130 * ureg.knot),  # final approach
                ( 5000 * ureg.feet, 257 * ureg.knot),
                (10000 * ureg.feet, 283 * ureg.knot),
                (15000 * ureg.feet, 307 * ureg.knot),
                (20000 * ureg.feet, 330 * ureg.knot),
                (25000 * ureg.feet, 345 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins,
            # n>=30/bin.  Flat ~1640 fpm through FL150 reflects the
            # P-3's even turboprop power band; brochure 3500 fpm is
            # full-power MTOW SL ROC, rarely sustained in actual sorties.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1647 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1663 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1640 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1596 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1532 * ureg.feet / ureg.minute),
                # Residual at the brochure ceiling so the integrator
                # terminates cleanly.
                (30000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1672 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1737 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1716 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1706 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1638 * ureg.feet / ureg.minute),
            ]),
            # AFM normal-ops bank (30°), not the data p90 (27°,
            # n=687,102 turn fixes).  Most P-3 missions are transit /
            # sampling profiles, not survey grids; airframe is
            # capable of 30° normal ops.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=3800 * ureg.nautical_mile,
            endurance=12 * ureg.hour,
            useful_payload=18000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 426 IWG1 calibration, n=252 sorties (in-house + ASP archive FY2003-FY2025; 4 mislabeled high-alt sorties filtered)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet, P-3 Orion at WFF",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=95 * ureg.knot,
        )


class NASA_WB57(Aircraft):
    """NASA WB-57 (NASA 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field.
    Operates up to 60,000 ft with 8,800 lbs useful payload.

    See also:
        `https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC <https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC>`_
    """

    def __init__(self):
        # Calibrated against 100 IWG1 sorties combined from NASA 926
        # (data/n926na_alltracks.csv) and NASA 927
        # (data/n927na_alltracks.csv), 2018-11 through 2024.  See
        # ``notebooks/calibration/wb57/iwg1_calibration.ipynb`` for the
        # active-only fits and per-phase TAS / bank derivations.
        super().__init__(
            aircraft_type="WB-57",
            tail_number="NASA 926/927",
            operator="NASA JSC",
            # p99 of per-sortie peak altitude across 100 sorties.
            service_ceiling=63000 * ureg.feet,
            approach_speed=117 * ureg.knot,
            # Climb-phase TAS medians.  SL anchor at typical jet
            # rotation TAS (150 kt) since the SL climb-phase bin is
            # contaminated by takeoff-roll fixes still accelerating.
            # FL600 bin (n=257) is sparse and dips relative to FL500;
            # interpolation handles the gap cleanly.
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 150 * ureg.knot),
                (10000 * ureg.feet, 200 * ureg.knot),
                (20000 * ureg.feet, 234 * ureg.knot),
                (30000 * ureg.feet, 276 * ureg.knot),
                (40000 * ureg.feet, 330 * ureg.knot),
                (50000 * ureg.feet, 374 * ureg.knot),
            ]),
            # Cruise-phase TAS medians at the typical cruise band.
            # FL450 is the dominant cruise altitude (n=349,982 cruise
            # fixes vs n=51,103 at FL500); only ~22% of sorties peak
            # above FL500 and only ~5% reach FL600.  The schedule
            # anchors on FL400 onward to match the actual operational
            # envelope.
            cruise_schedule=TasSchedule(points=[
                (40000 * ureg.feet, 337 * ureg.knot),
                (45000 * ureg.feet, 348 * ureg.knot),
                (50000 * ureg.feet, 366 * ureg.knot),
                (55000 * ureg.feet, 382 * ureg.knot),
                (60000 * ureg.feet, 381 * ureg.knot),
                (62000 * ureg.feet, 381 * ureg.knot),
            ]),
            # Descent-phase TAS medians; SL anchor at observed
            # final-approach TAS (140 kt).
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 140 * ureg.knot),
                (10000 * ureg.feet, 205 * ureg.knot),
                (20000 * ureg.feet, 241 * ureg.knot),
                (30000 * ureg.feet, 278 * ureg.knot),
                (40000 * ureg.feet, 326 * ureg.knot),
                (50000 * ureg.feet, 372 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins,
            # n>=30/bin.  Peak ROC near FL050-100 (~2900 fpm),
            # declining through the cruise band.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2137 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2889 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2843 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2788 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2680 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2514 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2194 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1790 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1614 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1674 * ureg.feet / ureg.minute),
                # Residual rate at the certified ceiling so the
                # integrator terminates cleanly.
                (65000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1638 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1864 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1949 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2004 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1996 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2196 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2396 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2463 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2191 * ureg.feet / ureg.minute),
            ]),
            # p90 |Roll| during turn-state fixes (gate >5°,
            # n=156,946): 33°.  Median (19°) is dragged down by
            # small in-cruise course corrections.
            turn_model=TurnModel(max_bank_deg=33.0),
            engine_type="jet",
            range=2500 * ureg.nautical_mile,
            endurance=6.5 * ureg.hour,
            useful_payload=8800 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 926+927 IWG1 calibration, n=100 sorties (2018-11 to 2024)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet, WB-57 at JSC",
                    confidence=0.5,
                ),
            ],
            # Vs0 at landing config, MLW per WB-57 AFM (B-57 / Canberra
            # derivative).  Brochure-derived; IWG1 data never sees stall.
            stall_speed_cas=100 * ureg.knot,
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
        # Calibrated against 272 NASA ICARTT sorties from multiple
        # B-200 / UC-12 (military variant) campaigns: ACTAMERICA
        # Hskping (NASA 529 LaRC), DISCOVER-AQ California / Colorado /
        # Texas APPLANIX, KORUS-AQ B200 NAV, LMOS UC12 NAV, BlueFlux
        # AIMMS-20 (ICT identifies the BlueFlux platform as B-200
        # despite the data folder labeling).  See
        # ``notebooks/calibration/b200/iwg1_calibration.ipynb``.
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="multi-tail",
            operator="NASA (multiple)",
            # p99 of per-sortie peak altitude across 272 sorties.  The
            # 35000 ft brochure ceiling is rarely flown; typical
            # operational peaks land at FL280-FL300.
            service_ceiling=30000 * ureg.feet,
            approach_speed=113 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 110 * ureg.knot),  # rotation
                ( 5000 * ureg.feet, 157 * ureg.knot),
                (10000 * ureg.feet, 193 * ureg.knot),
                (15000 * ureg.feet, 205 * ureg.knot),
                (20000 * ureg.feet, 206 * ureg.knot),
                (25000 * ureg.feet, 212 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (10000 * ureg.feet, 220 * ureg.knot),
                (15000 * ureg.feet, 230 * ureg.knot),
                (20000 * ureg.feet, 239 * ureg.knot),
                (25000 * ureg.feet, 238 * ureg.knot),
                (28000 * ureg.feet, 238 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 130 * ureg.knot),  # final approach
                ( 5000 * ureg.feet, 178 * ureg.knot),
                # FL100 descent bin median is 221 kt; clip to cruise
                # FL100 (220 kt) so descent <= cruise at every shared
                # altitude — the 1-kt difference is below the bin
                # measurement precision.
                (10000 * ureg.feet, 220 * ureg.knot),
                (15000 * ureg.feet, 240 * ureg.knot),
                (20000 * ureg.feet, 253 * ureg.knot),
                (25000 * ureg.feet, 251 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1000 fpm), 5-kft bins, n>=30/bin.
            # Threshold lowered from the 1500 fpm default so the climb
            # bins extend through FL250 — above FL150 the B-200's
            # active-climb VS is normally 1000-1300 fpm.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1267 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1320 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1346 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1190 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1068 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1031 * ureg.feet / ureg.minute),
                # Residual at service_ceiling so the integrator
                # terminates cleanly there.
                (30000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1000 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1198 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1320 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1397 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1551 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1852 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1406 * ureg.feet / ureg.minute),
            ]),
            # AFM normal-ops bank.  Data p90=30° agrees (n=1.14M turn
            # fixes across 272 sorties).
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1632 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=4250 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference="Multi-campaign ICARTT calibration, n=272 sorties (ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS, BlueFlux)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="Beechcraft King Air B200 AFM",
                    confidence=0.5,
                ),
            ],
            # Vs0 at landing config, MLW per Beechcraft B200 AFM.
            stall_speed_cas=75 * ureg.knot,
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
        # Calibrated against 87 IWG1 sorties from NASA Wallops C-130H
        # (NASA 436) flying the ACT-America campaign 2016-2019.  Data
        # downloaded from the public NASA ASP archive at
        # https://asp-archive.arc.nasa.gov/ACTAMERICA/N436NA/ via
        # ``notebooks/calibration/c130/_fetch_act_america.py``.  See
        # ``notebooks/calibration/c130/iwg1_calibration.ipynb`` for the
        # active-only fits and per-phase TAS / bank / approach
        # derivations.
        super().__init__(
            aircraft_type="C-130H Hercules",
            tail_number="NASA 436",
            operator="NASA WFF",
            # p99 of per-sortie peak altitude across 87 sorties; the
            # 33000 ft brochure ceiling is rarely flown — typical
            # ACT-America cruise band is FL200-FL280.
            service_ceiling=28000 * ureg.feet,
            approach_speed=126 * ureg.knot,
            # Climb-phase TAS medians.  SL anchor at typical C-130
            # rotation TAS (~110 kt) since the SL climb-phase bin is
            # contaminated by takeoff-roll fixes still accelerating.
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 110 * ureg.knot),
                ( 5000 * ureg.feet, 234 * ureg.knot),
                (10000 * ureg.feet, 239 * ureg.knot),
                (15000 * ureg.feet, 240 * ureg.knot),
                (20000 * ureg.feet, 250 * ureg.knot),
                (25000 * ureg.feet, 257 * ureg.knot),
            ]),
            # Cruise-phase TAS medians at the typical cruise band.
            # Below FL200 cruise-labeled bins are mostly transient
            # level-offs during step climbs.
            cruise_schedule=TasSchedule(points=[
                (20000 * ureg.feet, 305 * ureg.knot),
                (25000 * ureg.feet, 302 * ureg.knot),
                (28000 * ureg.feet, 302 * ureg.knot),
            ]),
            # Descent-phase TAS medians; SL anchor at observed
            # final-approach TAS (130 kt).
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 130 * ureg.knot),
                ( 5000 * ureg.feet, 260 * ureg.knot),
                (10000 * ureg.feet, 283 * ureg.knot),
                (15000 * ureg.feet, 304 * ureg.knot),
                (20000 * ureg.feet, 311 * ureg.knot),
                (25000 * ureg.feet, 301 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins,
            # n>=30/bin.  Flat across the climb envelope (~1620-1640
            # fpm SL through FL150) reflects the C-130's even turboprop
            # power band; brochure 2000 fpm SL ROC is rarely sustained
            # in actual ACT-America sorties.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1623 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1627 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1643 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1624 * ureg.feet / ureg.minute),
                # Residual at service_ceiling so the integrator
                # terminates cleanly there.  Active-climb VS data only
                # extends to FL150 (above that the aircraft is climbing
                # below the 1500-fpm active threshold).
                (28000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1634 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1695 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1700 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1644 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1580 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1563 * ureg.feet / ureg.minute),
            ]),
            # AFM normal-ops bank (30°), not the data p90 (27°).
            # ACT-America is a transit / vertical-profile mission,
            # not a survey grid, so the IWG1 sample never sees the
            # tight-bank flight-line transitions a science planner
            # might reasonably plan for.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=2500 * ureg.nautical_mile,
            endurance=10 * ureg.hour,
            useful_payload=45000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 436+439 IWG1 calibration, n=91 sorties (ASP archive ACT-America 2016-2019; both NASA WFF tails)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="C-130H AFM",
                    confidence=0.5,
                ),
            ],
            # Vs0 at landing config, MLW per C-130H AFM.
            # Brochure-derived; IWG1 data never sees stall.
            stall_speed_cas=100 * ureg.knot,
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
        # Calibrated against 17 ICARTT sorties from FIREX-AQ N48RF
        # (NOAA DHC-6 Twin Otter), summer 2019.  AIMSS Probe in-situ
        # data — TAS, attitude, wind, pressure.  Twin Otter climbs
        # slowly compared to jets/turboprops, so the active-climb
        # threshold in the calibration notebook is 500 fpm (vs the
        # 1500 fpm used for jets and 1000 fpm used for B-200).
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="N48RF",
            operator="NOAA",
            # p99 of per-sortie peak across 17 sorties; 25000 ft
            # brochure ceiling not approached in the FIREX-AQ data
            # (typical ops cruise FL080-FL120).
            service_ceiling=15000 * ureg.feet,
            approach_speed=99 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet,  70 * ureg.knot),  # rotation
                ( 5000 * ureg.feet, 118 * ureg.knot),
                (10000 * ureg.feet, 128 * ureg.knot),
                (12000 * ureg.feet, 128 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                ( 5000 * ureg.feet, 138 * ureg.knot),
                ( 8000 * ureg.feet, 141 * ureg.knot),
                (10000 * ureg.feet, 141 * ureg.knot),
                (12000 * ureg.feet, 141 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet,  90 * ureg.knot),
                ( 5000 * ureg.feet, 143 * ureg.knot),
                (10000 * ureg.feet, 148 * ureg.knot),
                (12000 * ureg.feet, 148 * ureg.knot),
            ]),
            # Active-climb median (VS >= 500 fpm), 5-kft bins.  The
            # FIREX-AQ data peaks at FL100-FL130; lowering the
            # threshold further wouldn't add bins since most sorties
            # don't climb actively above FL100.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  770 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  769 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  665 * ureg.feet / ureg.minute),
                (25000 * ureg.feet,  200 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 500 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  663 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  733 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  657 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=4000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference="FIREX-AQ N48RF Twin Otter ICARTT calibration, n=17 sorties (2019)",
                    confidence=0.85,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="DHC-6 Twin Otter Series 300 manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=58 * ureg.knot,  # Vs0 landing config, MTOW
        )
