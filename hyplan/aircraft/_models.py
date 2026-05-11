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
    "DLR_HALO",
    "NASA_B777",
    "NASA_C20A",
    "NASA_C130",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_P3",
    "NASA_WB57",
    "NCAR_GV",
    "NERC_DO228",
    "NOAA_GIV",
    "NOAA_WP3D",
    "SAFIRE_ATR42",
    "AWI_BaslerBT67",
    "BAS_TwinOtter",
    "FAAM_BAe146",
    "KingAir350",
    "KingAirA90",
    "KingAirB200",
    "NOAA_TwinOtter",
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
    from cached NASA AFRC IWG1 in-situ flight logs covering NASA 806 and
    NASA 809, with continuous fiscal-year coverage 2012-2026 (618 sorties
    loaded from a 629-file cache pulled from the public NASA ASP archive
    plus a local in-house delivery; see
    ``notebooks/calibration/NASA_ER2/_fetch_asp.py`` for the fetcher).
    See [notebooks/calibration/NASA_ER2/calibration.ipynb] for the full
    derivation: per-altitude-bin |VS| medians, breakpoint selection rules,
    and validation against per-sortie observed timing.

    Vertical-rate highlights from the calibration:

    * Weight-management level-offs and holds during climb-out are modeled
      explicitly via ``typical_climb_out``; the ``climb_profile`` itself
      is active-climb-only performance, not wall-clock climb-out timing.
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

    def __init__(self) -> None:
        # Distinct climb / cruise / descent TAS schedules from IWG1 per-
        # altitude-bin medians across 618 NASA 806 + 809 sorties (2012-2026
        # continuous fiscal-year coverage; bins with n >= 30 fixes).  The
        # IWG1 data shows climb / descent are within ~5 kt of each other at
        # any given altitude (both reflect pitched flight); cruise sits ~5 kt
        # higher in the 60-65 kft band.  Endpoints at 0 and 70 kft are
        # extrapolations beyond the bins where data is dense.
        climb_schedule = TasSchedule(points=[
            (    0 * ureg.feet, 130 * ureg.knot),  # extrapolated; 2.5 kft bin median = 141
            (20000 * ureg.feet, 230 * ureg.knot),  # IWG1 17.5-22.5 kft band (220-235)
            (40000 * ureg.feet, 306 * ureg.knot),  # IWG1 37.5-42.5 kft band (291-321)
            (60000 * ureg.feet, 393 * ureg.knot),  # IWG1 57.5-62.5 kft band (390-396)
            (70000 * ureg.feet, 400 * ureg.knot),  # extrapolated; 67.5 kft bin median = 398
        ])
        cruise_schedule = TasSchedule(points=[
            (    0 * ureg.feet, 130 * ureg.knot),  # brochure-equivalent low-alt anchor
            (50000 * ureg.feet, 375 * ureg.knot),  # IWG1 47.5-52.5 kft band (361-390)
            (60000 * ureg.feet, 397 * ureg.knot),  # IWG1 57.5-62.5 kft band (395-399)
            (65000 * ureg.feet, 400 * ureg.knot),  # IWG1 62.5-67.5 kft band (399-401)
            (70000 * ureg.feet, 401 * ureg.knot),  # extrapolated above 67 kft
        ])
        descent_schedule = TasSchedule(points=[
            (    0 * ureg.feet,  95 * ureg.knot),  # touchdown extrapolation; 2.5 kft = 117
            (20000 * ureg.feet, 224 * ureg.knot),  # IWG1 17.5-22.5 kft band (215-234)
            (40000 * ureg.feet, 312 * ureg.knot),  # IWG1 37.5-42.5 kft band (297-327)
            (60000 * ureg.feet, 394 * ureg.knot),  # IWG1 57.5-62.5 kft band (392-396)
            (70000 * ureg.feet, 400 * ureg.knot),  # extrapolated above 67 kft
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
            # hold altitudes), 5-kft bins across the cached IWG1 sortie
            # set.  Ships per-bin medians directly; non-monotone
            # mid-altitude bumps reflect real hold-band-adjacent climb
            # behaviour and are preserved.
            #
            # See `notebooks/calibration/NASA_ER2/calibration.ipynb` §5
            # for the active-only derivation and §6b for the IQR fit
            # validation.
            climb_profile=VerticalProfile(points=[
                ( 5000 * ureg.feet, 3553 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 3876 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 3851 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 3526 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 3423 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 3121 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2338 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2035 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1761 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 1668 * ureg.feet / ureg.minute),
                (55000 * ureg.feet, 1612 * ureg.feet / ureg.minute),
                # Operational ceiling residual; FL600+ active-climb bins
                # are too sparse (n<30) for a reliable empirical fit.
                (66000 * ureg.feet,  200 * ureg.feet / ureg.minute),
            ]),
            # Three-anchor descent profile derived from the 618-sortie
            # 2012-2026 IWG1 cache: gentle |VS| at top-of-approach
            # (just-begun descent), peak |VS| mid-descent, and the
            # high-altitude initial-descent rate near the operating
            # ceiling.  Bottom anchor is keyed to top_of_approach_msl
            # (representative airport elevation + 3 kft AGL = ~5265 ft)
            # so the descent-vs-approach handoff has a clean meaning.
            # See ``notebooks/calibration/NASA_ER2/calibration.ipynb``
            # for the derivation.
            descent_profile=VerticalProfile(points=[
                ( 5265 * ureg.feet,  473 * ureg.feet / ureg.minute),
                (37500 * ureg.feet, 2570 * ureg.feet / ureg.minute),
                (66000 * ureg.feet, 1502 * ureg.feet / ureg.minute),
            ]),
            # Calibrated terminal-arrival profile (3 kft AGL -> touchdown).
            # 2.53° glideslope is the empirical median over 66 499 IWG1
            # approach-phase fixes from the expanded 618-sortie cache;
            # touchdown 72 kt is the per-sortie median TAS in the lowest
            # 50 ft AGL band, n=89 sorties contributing.  The 65→72 kt
            # bump versus the prior 199-sortie fit reflects the larger
            # operational sample (the small earlier sample skewed light /
            # below typical ER-2 flare).  Top-of-approach 157 kt is the
            # median at the 2.5-3.5 kft AGL band (n=8499 fixes).
            approach_profile=ApproachProfile(
                speed_schedule=TasSchedule(points=[
                    (   0 * ureg.feet,  72 * ureg.knot),  # touchdown (n=89 sorties)
                    ( 200 * ureg.feet,  78 * ureg.knot),  # interpolated
                    (1000 * ureg.feet, 100 * ureg.knot),  # interpolated
                    (3000 * ureg.feet, 157 * ureg.knot),  # top-of-approach band median
                ]),
                top_of_approach_agl=3000 * ureg.feet,
                glideslope_deg=2.53,
            ),
            # max_bank_deg=30 is the brochure / envelope ceiling — still
            # well-supported by the expanded IWG1 cache (p99 < 30° in every
            # altitude band).  bank_by_phase carries the calibrated
            # *typical-operations* medians from the 618-sortie set and is
            # consumed by Aircraft._hybrid_path (per-phase turn radius) and
            # loiter_orbit_geometry (phase-aware orbit radius).
            turn_model=TurnModel(
                max_bank_deg=30.0,
                bank_by_phase=PhaseBankAngles(
                    climb_deg=14.0,     # climb-phase p50 (n=93 412 fixes)
                    cruise_deg=20.0,    # cruise-phase p50 (n=549 320 fixes)
                    descent_deg=13.0,   # descent-phase p50 (n=119 513 fixes)
                    approach_deg=11.0,  # 0-10 kft band p50 (terminal area)
                ),
            ),
            engine_type="jet",
            range=5000 * ureg.nautical_mile,
            calibration_status="calibrated",
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
            # active-climb-only climb_profile, pure
            # _climb() integration is now what the model represents,
            # and the typical pre-cruise mission overhead is
            # injected explicitly via the typical_climb_out below.
            climb_path_angle_max_deg=6.0,
            typical_climb_out=ClimbOutPolicy(
                # climb_profile is active-only (|VS| >= 1500 fpm) and
                # the planner reads `explicit_climb_plan` via
                # `compute_flight_plan(climb_plan="auto")` to add the
                # typical climb-out overhead the active-only model
                # doesn't capture.
                absorbed_in_climb_profile=False,
                typical_holds=[
                    # No discretionary altitude-band hold concentrates
                    # in the modern 618-sortie sample.  Brief level-offs
                    # at FL240/FL260 occur in 10-20% of sorties (median
                    # 2-3 min when present) but aren't typical.  The
                    # historical FL356 12-min weight-management hold
                    # characteristic of HS3 / ATTREX (2012-2015) appears
                    # in only 0.8% of sorties post-2016.
                ],
                typical_overhead_min=13.0,
                notes=(
                    "Per-sortie climb-out overhead derived from the "
                    "618-sortie 2012-2026 IWG1 cache (355 with a clear "
                    "top-of-climb level-off).  Total inactive-climb "
                    "(|VS| < 1500 fpm) time from takeoff to TOC has "
                    "median 19.8 min; the active-only climb_profile "
                    "integration accounts for ~7 min via its 200 fpm "
                    "anchor at FL660, leaving ~13 min of marginal "
                    "overhead.  This overhead concentrates in the "
                    "FL500-FL650 approach-to-ceiling slowdown band "
                    "(98.9% of sorties spend ≥1 min between FL550-"
                    "FL600 with |VS| < 1500 fpm; median 5.83 min "
                    "there alone) — airframe physics rather than "
                    "pilot procedure.  explicit_climb_plan places a "
                    "single representative pause at FL550 so consumers "
                    "of climb_plan='auto' get the right total time "
                    "with a sensible loiter-row location.  Power users "
                    "wanting pure active-climb physics pass "
                    "climb_plan=None.  See _fetch_asp.py for the "
                    "data source and calibration.ipynb for the full "
                    "active-climb derivation."
                ),
                explicit_climb_plan=ClimbPlan(pauses=[
                    # FL550 is the lowest band of the FL500-FL650
                    # approach-to-ceiling slowdown.  Duration matches
                    # the observed-vs-active-climb gap (~13 min) at
                    # the total-time level.  Single anchor; the actual
                    # in-flight slowdown is diffuse across FL500-FL650
                    # but a single loiter-row keeps the flight-plan
                    # dataframe tractable for downstream consumers.
                    (55_000 * ureg.feet, 13 * ureg.minute),
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
                    url="https://www.nasa.gov/centers-and-facilities/armstrong/er-2/",
                    reference=(
                        "NASA AFRC IWG1 in-situ flight logs from cached "
                        "NASA 806 + 809 sorties; "
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

    def __init__(self) -> None:
        # Calibrated against 152 IWG1 sorties from NASA 520, combining
        # the local "n520NA_g3_alltracks.csv" delivery (2025-07 through
        # 2026-04) with the public NASA ASP archive
        # (asp-archive.arc.nasa.gov/N520NA, FY2024-FY2026).  See
        # ``notebooks/calibration/NASA_GIII/calibration.ipynb``.
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
            calibration_status="calibrated",
            endurance=7.5 * ureg.hour,
            useful_payload=2610 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 520 IWG1 calibration, n=152 sorties (in-house + ASP archive FY2024-FY2026)",
                    confidence=0.85,
                    url="https://airbornescience.nasa.gov/data",
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

    .. warning::

        **Uncalibrated.**  Performance values come from manufacturer
        brochures / type-certificate data; no in-situ flight-data fit
        has been performed.  Treat planning output as a best-effort
        starting point.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC>`_
    """

    def __init__(self) -> None:
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
            calibration_status="uncalibrated",
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

    def __init__(self) -> None:
        # Vertical profiles, approach, bank, and stall calibrated against
        # 84 IWG1 sorties from NASA 95 (data/n95na_2019.csv,
        # n95na_2020_2022.csv, n95na_2023_2026.csv), 2019 through 2026.
        # 169 candidate files were filtered: most were ASP-archive
        # ferry/test entries with no airborne fixes or invalid altitude;
        # the 84 retained sorties are the valid science-mission set.
        # See ``notebooks/calibration/NASA_GV/calibration.ipynb`` for the
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
            calibration_status="calibrated",
            endurance=13 * ureg.hour,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.5, descent=0.85, turns=0.8,
            ),
            sources=[
                SourceRecord(
                    source_type="iwg1",
                    reference="NASA 95 IWG1 calibration, n=84 sorties (in-house + ASP archive 2019-2026)",
                    confidence=0.85,
                    url="https://airbornescience.nasa.gov/data",
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


class NCAR_GV(Aircraft):
    """NSF/NCAR HIAPER Gulfstream V research aircraft (N677F).

    Operated by NSF NCAR Earth Observing Laboratory (EOL).  Same
    airframe family as NASA_GV but a separate operational tail with
    different mission profile and avionics.  HIAPER routinely cruises
    FL410-FL510 on long-duration atmospheric campaigns (HIPPO,
    SOCRATES, ORCAS, ATTREX) and carries a different flight-data
    suite (high-rate 25-Hz NetCDF).

    See also:
        `https://www.eol.ucar.edu/observing_facilities/hiaper`
    """

    def __init__(self) -> None:
        # Calibrated against 22 NSF-GV ICARTT NAV sorties from the DC3
        # campaign (May-Jun 2012), fetched from the NASA LaRC Airborne
        # Science Data archive.  The DC3 RAF-NAV product omits TASX
        # (true airspeed); TAS is reconstructed via the wind triangle
        # from in-situ groundspeed + true heading + WSC/WDC wind
        # speed/direction.
        #
        # Service ceiling, range, endurance, and the CAS/Mach speed
        # schedule are kept at the NASA_GV (same airframe) calibrated
        # values: the DC3 campaign was a deep-convection sample that
        # cruised mostly FL370-FL410, so the dataset doesn't reach
        # HIAPER's typical FL490+ atmospheric-research ceiling.
        # Cruise TAS at FL300 (~468 kt observed) is consistent with
        # the Mach-0.80 cruise schedule already in NASA_GV.
        #
        # Climb / descent profiles are HIAPER-specific (DC3-derived)
        # and run notably softer than NASA_GV at low altitude (15-25%
        # lower SL ROC), reflecting HIAPER's heavier atmospheric-
        # chemistry payload.  Above FL350 only a few hundred fixes/bin
        # were available; the FL510 terminator point matches the
        # NASA_GV convention so the integrator terminates cleanly.
        #
        # See ``notebooks/calibration/NCAR_GV/calibrate.py`` for the
        # recipe.  Refresh with HIPPO/SOCRATES/ORCAS data when the
        # NCAR EOL ORDER pathway is set up — those campaigns will
        # tighten the FL410-FL510 climb / descent shape.
        super().__init__(
            aircraft_type="Gulfstream V",
            tail_number="N677F",
            operator="NSF/NCAR EOL",
            service_ceiling=51000 * ureg.feet,
            approach_speed=141 * ureg.knot,
            climb_schedule=CasMachSchedule(
                cas=280 * ureg.knot, mach=0.74, crossover_ft=28000,
            ),
            cruise_schedule=CasMachSchedule(
                cas=300 * ureg.knot, mach=0.80, crossover_ft=30000,
            ),
            descent_schedule=CasMachSchedule(
                cas=290 * ureg.knot, mach=0.78, crossover_ft=30000,
            ),
            # DC3-derived bin medians (active VS >= 1500 fpm).  HIAPER
            # climbs ~15-25% slower than NASA 95 at the same altitudes
            # (heavier payload + atmospheric-research mission profile).
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1834 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2004 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1738 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1693 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1689 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1580 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1535 * ureg.feet / ureg.minute),
                # Residual rate at certified ceiling so the integrator
                # terminates cleanly.  DC3 sample doesn't reach FL510;
                # value mirrored from NASA_GV.
                (51000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1553 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1642 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1712 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1802 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1851 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1967 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1855 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1826 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1969 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(
                bank_by_phase=PhaseBankAngles(
                    climb_deg=20, cruise_deg=25, descent_deg=20, approach_deg=15,
                ),
                # AFM normal-ops bank.  DC3 bank-angle p90 was 27.1°
                # (n=22 sorties of survey + transit), p99 = 29.8°.
                max_bank_deg=30.0,
            ),
            engine_type="jet",
            range=6500 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=14 * ureg.hour,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.7, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference=(
                        "NSF/NCAR HIAPER (N677F) DC3 RAF-NAV ICARTT calibration; "
                        "22 sorties from NASA LaRC ASD archive; TAS reconstructed "
                        "via wind triangle from in-situ groundspeed + heading + "
                        "WSC/WDC wind components"
                    ),
                    confidence=0.85,
                    url="https://www-air.larc.nasa.gov/missions/dc3-seac4rs/",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference=(
                        "NSF/NCAR HIAPER Investigator Handbook; CasMachSchedule "
                        "values mirrored from calibrated NASA_GV (same airframe), "
                        "validated by DC3 cruise TAS observations at FL300"
                    ),
                    confidence=0.7,
                ),
            ],
            stall_speed_cas=104 * ureg.knot,
        )


class NASA_C20A(Aircraft):
    """NASA C-20A (Gulfstream III variant, NASA 502) research aircraft.

    Obtained from the U.S. Air Force in 2003. Primary platform for
    UAVSAR missions. Operated by NASA AFRC.

    .. note::

        **Inferred.**  Performance is mirrored from the calibrated
        ``NASA_GIII`` model (same type certificate).  C-20A-specific
        IWG1 calibration is deferred pending data access.  Output
        is more reliable than a brochure-only model but may not
        capture C-20A-specific operational differences.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC>`_
    """

    def __init__(self) -> None:
        # Inferred from NASA_GIII calibration (152 IWG1 sorties).
        # The C-20A is a Gulfstream III military variant — same
        # airframe, same engines (Spey Mk.511-8), same type
        # certificate as the civilian G-III used by NASA LaRC.  No
        # public C-20A IWG1/ICARTT data is available (NASA AFRC
        # publishes UAVSAR remote-sensing products, not housekeeping
        # nav), so the cruise / climb / descent / approach / bank
        # values here mirror the calibrated NASA_GIII directly.
        # When per-tail C-20A data becomes available it should
        # replace this; differences from the LaRC G-III would mostly
        # come from operational profile (UAVSAR survey grids vs
        # transit) rather than aircraft physics.
        super().__init__(
            aircraft_type="C-20A",
            tail_number="NASA 502",
            operator="NASA AFRC",
            service_ceiling=45000 * ureg.feet,
            approach_speed=139 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 150 * ureg.knot),
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
                (    0 * ureg.feet, 180 * ureg.knot),
                (10000 * ureg.feet, 329 * ureg.knot),
                (20000 * ureg.feet, 410 * ureg.knot),
                (30000 * ureg.feet, 453 * ureg.knot),
                (40000 * ureg.feet, 463 * ureg.knot),
            ]),
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
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=3400 * ureg.nautical_mile,
            calibration_status="inferred",
            endurance=6 * ureg.hour,
            useful_payload=2500 * ureg.pound,
            confidence=PerformanceConfidence(
                # Inferred from G-III; not directly calibrated against
                # C-20A data.  Confidence reflects airframe-equivalence
                # assumption rather than measurement.
                climb=0.7, cruise=0.7, descent=0.7, turns=0.7,
            ),
            sources=[
                SourceRecord(
                    source_type="inferred",
                    reference="Mirrors NASA_GIII calibration (152 IWG1 sorties); C-20A is a G-III military variant",
                    confidence=0.7,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet; EUROCONTROL GLF3",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=105 * ureg.knot,
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

    def __init__(self) -> None:
        # Calibrated against 252 IWG1 sorties from NASA 426 (P3-B),
        # combining a local "p3_alltracks.csv" delivery with the public
        # NASA ASP archive (asp-archive.arc.nasa.gov/N426NA, FY2003
        # through FY2025).  Filter MAX_PEAK_ALT_FT=32000 in the
        # calibration notebook excludes 4 sorties peaking 35-55 kft
        # — those are tail-number reassignments where N426NA was
        # recorded for a different (jet-class) airframe; not P-3 data.
        # See ``notebooks/calibration/NASA_P3/calibration.ipynb``.
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
            calibration_status="calibrated",
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
                    url="https://airbornescience.nasa.gov/data",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="NASA Airborne Science fact sheet, P-3 Orion at WFF",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=95 * ureg.knot,
        )


class NOAA_WP3D(Aircraft):
    """NOAA WP-3D Orion atmospheric chemistry / Hurricane Hunter.

    Lockheed WP-3D Orion operated by the NOAA Aircraft Operations
    Center (AOC).  Two tails are typically active for atmospheric
    chemistry research: N42RF "Kermit" and N43RF "Miss Piggy".
    Same airframe family as ``NASA_P3`` (LaRC's NASA 426); calibrated
    separately because the operator, mission profile, and outfit
    differ.  NOAA WP-3D missions tracked here are NOAA CSL chemistry
    campaigns, not the hurricane-research deployments.

    See also:
        `https://www.omao.noaa.gov/aircraft/lockheed-wp-3d-orion`
    """

    def __init__(self) -> None:
        # Calibrated against 18 merged ICARTT sorties from the NOAA
        # CSL archive at ``data/WP3D/NOAA_CSL/``: ARCPAC 2008,
        # CalNex 2010, SENEX 2013, SONGNEX 2015 (4 campaigns of 12-27
        # sortie-dates each; only sorties peaking >FL080 are kept,
        # which excludes pure boundary-layer Arctic profiles).
        #
        # Per-sortie data is split across three ICARTT files
        # (AircraftMet, AircraftPos, AircraftMis) merged on the
        # AOCTimewave UTC-seconds-past-midnight column inside the
        # calibration script.  See
        # ``notebooks/calibration/NOAA_WP3D/calibrate.py`` for the recipe.
        #
        # The climb / descent profiles are essentially identical to
        # the calibrated ``NASA_P3`` (same airframe, similar
        # operational profile) — within ±5 fpm at every altitude bin.
        # Cruise TAS schedule runs slightly slower than NASA_P3
        # (NOAA mission profile uses lower-cruise patterns).
        super().__init__(
            aircraft_type="P-3 Orion (WP-3D)",
            tail_number="N42RF + N43RF",
            operator="NOAA AOC",
            # p99 of per-sortie peak altitude across 96 sorties (18 CSL +
            # 78 HRD).  WP-3D brochure ceiling is ~28000 ft; the
            # combined dataset reaches FL275 in chemistry missions
            # while hurricane penetrations stay FL010-FL120.
            service_ceiling=27600 * ureg.feet,
            # Same airframe as NASA_P3 (calibrated approach 132 kt);
            # the HRD "last 60 s of airborne segment" heuristic
            # captures hurricane-eye exit descents at high TAS rather
            # than the actual final approach, so use NASA_P3 value.
            approach_speed=132 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 228 * ureg.knot),
                ( 5000 * ureg.feet, 242 * ureg.knot),
                (10000 * ureg.feet, 259 * ureg.knot),
                (15000 * ureg.feet, 274 * ureg.knot),
                (20000 * ureg.feet, 290 * ureg.knot),
                (25000 * ureg.feet, 302 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 224 * ureg.knot),
                ( 5000 * ureg.feet, 248 * ureg.knot),
                (10000 * ureg.feet, 254 * ureg.knot),
                (15000 * ureg.feet, 326 * ureg.knot),
                (20000 * ureg.feet, 344 * ureg.knot),
                (25000 * ureg.feet, 343 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 226 * ureg.knot),
                ( 5000 * ureg.feet, 249 * ureg.knot),
                (10000 * ureg.feet, 275 * ureg.knot),
                (15000 * ureg.feet, 304 * ureg.knot),
                (20000 * ureg.feet, 336 * ureg.knot),
                (25000 * ureg.feet, 344 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins, n>=30.
            # Climb peaks at FL050 (~1700 fpm) and decays smoothly.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1698 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1671 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1624 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1568 * ureg.feet / ureg.minute),
                (30000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1767 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1818 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1845 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1808 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1803 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1629 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=3800 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=12 * ureg.hour,
            useful_payload=18000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference=(
                        "NOAA WP-3D (N42RF + N43RF) calibration, n=96 "
                        "sorties total: NOAA CSL chemistry ICARTT (18 "
                        "sorties: ARCPAC 2008, CalNex 2010, SENEX 2013, "
                        "SONGNEX 2015) + HRD AOML hurricane field-program "
                        "1-second flight-level (78 sorties 2021-2025)"
                    ),
                    confidence=0.85,
                    url="https://csl.noaa.gov/groups/csl4/measurements/",
                ),
                SourceRecord(
                    source_type="text",
                    reference=(
                        "NOAA WP-3D (N42RF + N43RF) HRD AOML hurricane "
                        "field-program 1-second flight-level data, "
                        "n=78 sorties 2021-2025"
                    ),
                    confidence=0.85,
                    url="https://www.aoml.noaa.gov/ftp/hrd/data/flightlevel/",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="Lockheed WP-3D Orion AOC operating handbook",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=95 * ureg.knot,
        )


class NOAA_GIV(Aircraft):
    """NOAA Gulfstream IV-SP "Gonzo" (N49RF) hurricane synoptic-surveillance jet.

    Operated by NOAA Aircraft Operations Center.  Hurricane synoptic
    surveillance — high-altitude (FL420-FL450) sonde drops around
    developing tropical cyclones in the Atlantic and East Pacific
    basins.  Distinct mission profile and operator from NASA's
    brochure-only ``NASA_GIV``; this is the first calibrated G-IV
    variant in HyPlan.

    See also:
        `https://www.aoc.noaa.gov/aircraft-gulfstream-iv.html`
    """

    def __init__(self) -> None:
        # Calibrated against 93 NOAA HRD AOML hurricane field-program
        # 1-sec flight-level files (2021-2025, N-prefix), one per
        # synoptic-surveillance sortie.  See
        # ``notebooks/calibration/NOAA_GIV/calibrate.py`` and the loader at
        # ``notebooks/calibration/_hrd_loader.py:load_p3_1sec``.
        super().__init__(
            aircraft_type="Gulfstream IV-SP",
            tail_number="N49RF",
            operator="NOAA AOC",
            # Operational p99 of per-sortie peak altitude.  G-IV-SP
            # certified ceiling 45000 ft; routinely operates at
            # certified max for hurricane synoptic surveillance.
            service_ceiling=47500 * ureg.feet,
            approach_speed=146 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 220 * ureg.knot),
                (10000 * ureg.feet, 344 * ureg.knot),
                (20000 * ureg.feet, 421 * ureg.knot),
                (30000 * ureg.feet, 456 * ureg.knot),
                (40000 * ureg.feet, 438 * ureg.knot),
                (45000 * ureg.feet, 436 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 258 * ureg.knot),
                (10000 * ureg.feet, 316 * ureg.knot),
                (20000 * ureg.feet, 425 * ureg.knot),
                (30000 * ureg.feet, 476 * ureg.knot),
                (40000 * ureg.feet, 443 * ureg.knot),
                (45000 * ureg.feet, 435 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 175 * ureg.knot),
                (10000 * ureg.feet, 324 * ureg.knot),
                (20000 * ureg.feet, 423 * ureg.knot),
                (30000 * ureg.feet, 482 * ureg.knot),
                (40000 * ureg.feet, 457 * ureg.knot),
                (45000 * ureg.feet, 441 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins.
            # Climb peaks ~2350 fpm at FL050, decays to ~1650 fpm at
            # FL400 — typical Mach-limited climb at the upper levels.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2116 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2344 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1877 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1878 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1642 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1589 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1632 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1646 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1655 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1616 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1745 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1978 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2476 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2502 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2436 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2402 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2137 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1942 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1646 * ureg.feet / ureg.minute),
            ]),
            # HRD .1sec.txt format does not ship roll; AFM normal-ops
            # 30° default applies.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=4220 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=8.5 * ureg.hour,
            useful_payload=2500 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.6,
            ),
            sources=[
                SourceRecord(
                    source_type="text",
                    reference=(
                        "NOAA G-IV-SP (N49RF) HRD AOML hurricane "
                        "synoptic-surveillance 1-second flight-level "
                        "data, n=93 sorties 2021-2025"
                    ),
                    confidence=0.85,
                    url="https://www.aoml.noaa.gov/ftp/hrd/data/flightlevel/",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="Gulfstream G-IV / G-IV-SP Performance Manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=110 * ureg.knot,
        )


class NASA_WB57(Aircraft):
    """NASA WB-57F (NASA 926 / 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field.
    Operates up to 60 000 ft with 8 800 lbs useful payload.

    Calibrated against 127 sorties combining in-house IWG1 deliveries
    (NASA 926 + 927, 2018-2026) with the ACCLIP 2022 deployment's
    MMS-1HZ ICARTT data from NASA LaRC ASDC (collection
    ``ACCLIP_MetNav_AircraftInSitu_WB57_Data``; covers Lait's GSFC
    flight-planner WB-57 tuning window).  See
    ``notebooks/calibration/NASA_WB57/`` for ``_fetch_acclip.py`` (the
    LaRC ASDC fetcher), ``calibrate.py`` (the IWG1 + ICARTT-aware
    fitter), and ``calibration.ipynb`` (per-bin diagnostics).

    See also:
        `https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC <https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC>`_
    """

    def __init__(self) -> None:
        # Calibrated against 127 sorties combined from:
        #   * IWG1 in-house delivery: NASA 926 + 927, 2018-2026
        #     (data/n926na_alltracks.csv + data/n927na_alltracks.csv,
        #     split into per-sortie n92[67]_*.txt files).
        #   * ACCLIP 2022 MMS-1HZ ICARTT: 27 daily files from NASA 926
        #     at NASA LaRC ASDC (collection
        #     ACCLIP_MetNav_AircraftInSitu_WB57_Data); covers Lait's
        #     GSFC-flight-planner WB-57 tuning window (ChangeLog
        #     2022-08-02 / 2022-08-16: "improved wb57 tuning to
        #     acclip 2022").  See
        #     ``notebooks/calibration/NASA_WB57/_fetch_acclip.py``.
        # The two formats produce canonical-schema DataFrames and are
        # combined into a single calibration sample (see
        # ``notebooks/calibration/NASA_WB57/calibrate.py``).
        super().__init__(
            aircraft_type="WB-57",
            tail_number="NASA 926/927",
            operator="NASA JSC",
            # p99 of per-sortie peak altitude across 127 sorties.
            service_ceiling=64000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            # Climb-phase TAS medians.  SL anchor at typical jet
            # rotation TAS (150 kt) since the SL climb-phase bin is
            # contaminated by takeoff-roll fixes still accelerating.
            # ACCLIP 2022 added an FL600 anchor (n>200 fixes) that
            # was sparse in the IWG1-only fit.
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 150 * ureg.knot),
                (10000 * ureg.feet, 203 * ureg.knot),
                (20000 * ureg.feet, 236 * ureg.knot),
                (30000 * ureg.feet, 280 * ureg.knot),
                (40000 * ureg.feet, 334 * ureg.knot),
                (50000 * ureg.feet, 382 * ureg.knot),
                (60000 * ureg.feet, 402 * ureg.knot),
            ]),
            # Cruise-phase TAS medians at the typical cruise band.
            # FL450 is the dominant cruise altitude (n=420 498 cruise
            # fixes vs n=113 354 at FL500 in the expanded sample);
            # the schedule anchors on FL400 onward to match the actual
            # operational envelope.
            cruise_schedule=TasSchedule(points=[
                (40000 * ureg.feet, 337 * ureg.knot),
                (45000 * ureg.feet, 353 * ureg.knot),
                (50000 * ureg.feet, 383 * ureg.knot),
                (55000 * ureg.feet, 384 * ureg.knot),
                (60000 * ureg.feet, 389 * ureg.knot),
                (62000 * ureg.feet, 389 * ureg.knot),
            ]),
            # Descent-phase TAS medians; SL anchor at observed
            # final-approach TAS (140 kt).
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 140 * ureg.knot),
                (10000 * ureg.feet, 207 * ureg.knot),
                (20000 * ureg.feet, 243 * ureg.knot),
                (30000 * ureg.feet, 281 * ureg.knot),
                (40000 * ureg.feet, 330 * ureg.knot),
                (50000 * ureg.feet, 389 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins,
            # n>=30/bin.  Peak ROC near FL050-100 (~2775 fpm),
            # declining through the cruise band.  The ACCLIP data
            # added an FL500 bin (~1616 fpm) that was below n=30 in
            # the IWG1-only sample.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2274 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2775 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2794 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2771 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2688 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2468 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2133 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1751 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1605 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1641 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 1616 * ureg.feet / ureg.minute),
                # Residual rate at the certified ceiling so the
                # integrator terminates cleanly.
                (65000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1500 fpm), 5-kft bins.
            # ACCLIP added FL500 / FL550 / FL600 bins from the high-
            # altitude descent legs.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1651 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1829 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1918 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1968 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 2021 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 2194 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 2299 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 2266 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 2140 * ureg.feet / ureg.minute),
                (45000 * ureg.feet, 1946 * ureg.feet / ureg.minute),
                (50000 * ureg.feet, 1670 * ureg.feet / ureg.minute),
                (55000 * ureg.feet, 1613 * ureg.feet / ureg.minute),
                (60000 * ureg.feet, 1643 * ureg.feet / ureg.minute),
            ]),
            # p90 |Roll| during turn-state fixes (gate >5°,
            # n=187 410): 31.7° in the expanded sample.  Median (19°)
            # is dragged down by small in-cruise course corrections.
            turn_model=TurnModel(max_bank_deg=32.0),
            engine_type="jet",
            range=2500 * ureg.nautical_mile,
            calibration_status="calibrated",
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
                    url="https://airbornescience.nasa.gov/data",
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

    .. warning::

        **Uncalibrated.**  Performance values come from manufacturer
        brochures / type-certificate data; no in-situ flight-data fit
        has been performed.  Treat planning output as a best-effort
        starting point.

    See also:
        `https://airbornescience.nasa.gov/aircraft/B-777_-_LaRC <https://airbornescience.nasa.gov/aircraft/B-777_-_LaRC>`_
    """

    def __init__(self) -> None:
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
            calibration_status="uncalibrated",
            endurance=18 * ureg.hour,
            useful_payload=75000 * ureg.pound,
            sources=[
                SourceRecord(
                    source_type="brochure",
                    reference="Boeing 777-200ER Performance Manual; EUROCONTROL B772",
                    confidence=0.5,
                    url="https://airbornescience.nasa.gov/aircraft/B-777_-_LaRC",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# King Air turboprops
# ---------------------------------------------------------------------------

class KingAirA90(Aircraft):
    """Beechcraft King Air A90 twin-turboprop aircraft.

    Calibrated against 428 ADS-B sorties from `airplanes.live` daily-
    trace archives (2026-04-09 → 2026-05-08; 25 active US 65-A90 /
    65-A90-1 civilian tails) after filtering skydive jump-run profiles
    out of the raw 643-sortie sample.  No public IWG1-grade A-90 data
    is currently available, so this calibration is the highest-quality
    A90 envelope publicly available.  See
    ``notebooks/calibration/KingAirA90/calibrate.py`` for the recipe
    and ``_fetch_airplanes_live.py`` for the trace fetcher.

    Confidence is moderate (~0.65) — TAS is approximated by
    groundspeed (still-air baseline), and the active US A90 fleet
    skews toward skydive / charter / freight operations rather than
    research missions.  v1.6.1 will switch to MERRA-2 wind-triangle
    TAS reconstruction.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90>`_
    """

    def __init__(self) -> None:
        # Calibrated against 428 ADS-B sorties from
        # `airplanes.live globe-history archive (2026-04-09 → 2026-05-08
        # 30-day window) covering 25 of the 124 active US 65-A90 /
        # 65-A90-1 (civilian) tails in the FAA registry.  Skydive jump-
        # run profiles (short duration + high peak altitude + sustained
        # descent > 2500 fpm) were filtered out in the calibrate.py
        # pipeline; the retained sample is dominated by corporate /
        # charter / freight A90 ops.  See
        # ``notebooks/calibration/KingAirA90/calibrate.py``.
        #
        # Brochure cross-check: A90 POH gives ceiling 26,400 ft
        # (Part 23), max cruise 226 KTAS @ FL150-160, ROC 1,800 fpm SL.
        # Calibrated values track these closely (cruise 222 kt @ FL160;
        # ceiling op-p99 = 25 kft).  Approach speed 149 kt is the
        # median TAS in the last 500 ft AGL during sustained descent
        # — operationally biased above the AFM Vref 95-100 KIAS, but
        # representative of the science / charter / skydive-shuttle
        # mission mix.
        super().__init__(
            aircraft_type="King Air A90",
            tail_number="multi-tail",
            operator="multi-operator (A90 fleet aggregate)",
            service_ceiling=25000 * ureg.feet,  # op-p99 vs 26.4k POH
            approach_speed=149 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                ( 1800 * ureg.feet, 138 * ureg.knot),
                (11800 * ureg.feet, 158 * ureg.knot),
                (17800 * ureg.feet, 183 * ureg.knot),
                (19800 * ureg.feet, 180 * ureg.knot),
                (21800 * ureg.feet, 194 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                ( 2000 * ureg.feet, 143 * ureg.knot),
                (10000 * ureg.feet, 166 * ureg.knot),
                (12000 * ureg.feet, 190 * ureg.knot),
                (16000 * ureg.feet, 222 * ureg.knot),  # max cruise band
                (20000 * ureg.feet, 219 * ureg.knot),
                (22000 * ureg.feet, 194 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                ( 2000 * ureg.feet, 149 * ureg.knot),
                ( 4000 * ureg.feet, 148 * ureg.knot),
                (10000 * ureg.feet, 190 * ureg.knot),
                (14000 * ureg.feet, 190 * ureg.knot),
                (16000 * ureg.feet, 170 * ureg.knot),
                (22000 * ureg.feet, 179 * ureg.knot),
            ]),
            climb_profile=VerticalProfile(points=[
                ( 1800 * ureg.feet, 1024 * ureg.feet / ureg.minute),
                ( 3800 * ureg.feet,  832 * ureg.feet / ureg.minute),
                ( 9800 * ureg.feet,  896 * ureg.feet / ureg.minute),
                (13800 * ureg.feet,  640 * ureg.feet / ureg.minute),
                (17800 * ureg.feet,  704 * ureg.feet / ureg.minute),
                (21800 * ureg.feet,  512 * ureg.feet / ureg.minute),
                (26400 * ureg.feet,  100 * ureg.feet / ureg.minute),  # Part 23 ceiling anchor
            ]),
            descent_profile=VerticalProfile(points=[
                ( 2000 * ureg.feet,  640 * ureg.feet / ureg.minute),
                ( 4000 * ureg.feet,  640 * ureg.feet / ureg.minute),
                ( 6000 * ureg.feet,  704 * ureg.feet / ureg.minute),
                (12000 * ureg.feet,  576 * ureg.feet / ureg.minute),
                (16000 * ureg.feet,  960 * ureg.feet / ureg.minute),
                (22000 * ureg.feet,  960 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1275 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=6 * ureg.hour,
            stall_speed_cas=75 * ureg.knot,  # Vs0 (full flaps, landing config)
            useful_payload=2400 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.6, cruise=0.7, descent=0.6, turns=0.5,
            ),
            sources=[
                SourceRecord(
                    source_type="adsb",
                    reference=(
                        "King Air A90 ADS-B calibration, n=428 sorties "
                        "from airplanes.live globe-history archive "
                        "(2026-04-09 → 2026-05-08, 25 active US 65-A90 / "
                        "65-A90-1 tails); skydive jump-run profiles "
                        "filtered.  TAS approximated via groundspeed "
                        "(still-air baseline); v1.6.1 will switch to "
                        "MERRA-2 wind-triangle reconstruction."
                    ),
                    confidence=0.65,
                    url="https://www.airplanes.live/",
                ),
            ],
        )


class KingAirB200(Aircraft):
    """Beechcraft King Air B200 twin-turboprop aircraft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200>`_
    """

    def __init__(self) -> None:
        # Calibrated against 250 NASA ICARTT sorties from multiple
        # B-200 / UC-12 (military variant) campaigns: ACTAMERICA
        # Hskping (NASA 529 LaRC), DISCOVER-AQ California / Colorado /
        # Texas APPLANIX, KORUS-AQ B200 NAV, LMOS UC12 NAV.  See
        # ``notebooks/calibration/KingAirB200/calibration.ipynb``.
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="multi-tail",
            operator="NASA (multiple)",
            # p99 of per-sortie peak altitude across 250 sorties.  The
            # 35000 ft brochure ceiling is rarely flown; typical
            # operational peaks land at FL280-FL300.
            service_ceiling=30000 * ureg.feet,
            approach_speed=112 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 110 * ureg.knot),  # rotation
                ( 5000 * ureg.feet, 185 * ureg.knot),
                (10000 * ureg.feet, 195 * ureg.knot),
                (15000 * ureg.feet, 205 * ureg.knot),
                (20000 * ureg.feet, 206 * ureg.knot),
                (25000 * ureg.feet, 212 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (10000 * ureg.feet, 224 * ureg.knot),
                (15000 * ureg.feet, 230 * ureg.knot),
                (20000 * ureg.feet, 239 * ureg.knot),
                (25000 * ureg.feet, 238 * ureg.knot),
                (28000 * ureg.feet, 238 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 130 * ureg.knot),  # final approach
                ( 5000 * ureg.feet, 207 * ureg.knot),
                (10000 * ureg.feet, 224 * ureg.knot),
                (15000 * ureg.feet, 240 * ureg.knot),
                (20000 * ureg.feet, 253 * ureg.knot),
                (25000 * ureg.feet, 251 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1000 fpm), 5-kft bins, n>=30/bin.
            # Threshold lowered from the 1500 fpm default so the climb
            # bins extend through FL250 — above FL150 the B-200's
            # active-climb VS is normally 1000-1300 fpm.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1411 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1439 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1347 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1190 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1068 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1031 * ureg.feet / ureg.minute),
                # Residual at service_ceiling so the integrator
                # terminates cleanly there.
                (30000 * ureg.feet,  500 * ureg.feet / ureg.minute),
            ]),
            # Active-descent median (|VS| >= 1000 fpm), 5-kft bins.
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1212 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1310 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1397 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1551 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1852 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1406 * ureg.feet / ureg.minute),
            ]),
            # AFM normal-ops bank.  Data p90=26° (n turn fixes across
            # 250 sorties), so AFM 30° is the binding ceiling.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1632 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=6 * ureg.hour,
            useful_payload=4250 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference="Multi-campaign ICARTT calibration (ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS)",
                    confidence=0.85,
                    url="https://www-air.larc.nasa.gov/missions.htm",
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


class KingAir350(Aircraft):
    """Beechcraft King Air 350 / 350i twin-turboprop research aircraft.

    Stretched 200-series airframe with PT6A-60A engines (or
    Blackhawk XP-67A upgrade with PT6A-67A).  Larger cabin, higher
    MTOW (15,000 lb), longer range, and higher service ceiling
    than the King Air 200.

    Notable research operators include the University of Wyoming
    (UWKA-2 / N2UW since 2024, replacing UW's earlier King Air
    200T), NCAR, and several university-operated tails.

    Calibrated against 22 ADS-B sorties from the University of
    Wyoming UWKA-2 (N2UW / hex A18F28) pulled from the
    `airplanes.live` globe-history archive.  18 active days
    clustered into 5 science-campaign windows in 2025-01, 2025-03,
    2025-06, 2025-07/08, 2025-10, and 2026-04 (10k trace rows
    total).  See
    ``notebooks/calibration/KingAir350/calibrate.py`` for the
    fitting recipe.

    Confidence is moderate-low (~0.5) — sample is single-tail and
    sparse, with science-mission profile bias (high-altitude
    measurement legs rather than long-range cruise).  UWKA-2 has
    the Blackhawk XP-67A engine upgrade (PT6A-67A), so calibrated
    values may run a few percent above stock B300 with PT6A-60A.
    Approach speed is held to brochure 110 kt because the
    calibrated 162 kt reflects science-leg descent TAS rather than
    a normal Vref pattern speed.

    See also:
        `https://www.uwyo.edu/atsc/research-facilities/uwka/king-air.html`
    """

    def __init__(self) -> None:
        # Hybrid calibration:
        #   - service_ceiling, climb_schedule, climb_profile, cruise
        #     peak, cruise_schedule shape: data-fit from UWKA-2 ADS-B.
        #   - approach_speed: brochure 110 kt (calibrated 162 kt is
        #     science-leg descent TAS, not pattern Vref).
        #   - descent_schedule, descent_profile: brochure-shaped (the
        #     22-sortie sample produces single-bin artifacts at the
        #     upper levels that don't reflect normal descent ops).
        #
        # Brochure cross-check: B300 / King Air 350 ceiling 35,000 ft,
        # max cruise 312 KTAS @ FL280, long-range 274 KTAS @ FL350.
        # Calibrated cruise peak 318 KTAS @ FL330 — a few percent
        # above stock factory data, consistent with the Blackhawk
        # XP-67A upgrade on UWKA-2.
        super().__init__(
            aircraft_type="King Air 350",
            tail_number="N2UW",
            operator="University of Wyoming (UWKA-2)",
            service_ceiling=35000 * ureg.feet,
            approach_speed=110 * ureg.knot,  # brochure (calibrated 162 is science-leg)
            climb_schedule=TasSchedule(points=[
                ( 4000 * ureg.feet, 165 * ureg.knot),
                (10000 * ureg.feet, 199 * ureg.knot),
                (22000 * ureg.feet, 245 * ureg.knot),
                (28000 * ureg.feet, 245 * ureg.knot),
                (34000 * ureg.feet, 303 * ureg.knot),
            ]),
            # The MAD-based fit also surfaces a 188 kt dip at 12 000 ft that
            # reflects a brief level-off during a step climb in the sample,
            # not the underlying climb schedule.  Dropped here so the
            # schedule stays monotonically rising.
            cruise_schedule=TasSchedule(points=[
                ( 5000 * ureg.feet, 220 * ureg.knot),  # brochure low-alt anchor
                (10000 * ureg.feet, 245 * ureg.knot),
                (20000 * ureg.feet, 290 * ureg.knot),
                (28000 * ureg.feet, 312 * ureg.knot),  # brochure max cruise
                (33000 * ureg.feet, 318 * ureg.knot),  # calibrated peak
                (35000 * ureg.feet, 274 * ureg.knot),  # brochure long-range
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 130 * ureg.knot),
                (10000 * ureg.feet, 230 * ureg.knot),
                (20000 * ureg.feet, 270 * ureg.knot),
                (35000 * ureg.feet, 290 * ureg.knot),
            ]),
            climb_profile=VerticalProfile(points=[
                ( 4000 * ureg.feet, 1088 * ureg.feet / ureg.minute),
                ( 6000 * ureg.feet, 1856 * ureg.feet / ureg.minute),
                ( 8000 * ureg.feet, 1344 * ureg.feet / ureg.minute),
                (18000 * ureg.feet, 1344 * ureg.feet / ureg.minute),
                (34000 * ureg.feet,  384 * ureg.feet / ureg.minute),
                (35000 * ureg.feet,  100 * ureg.feet / ureg.minute),  # Part 23 ceiling
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=2100 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=5 * ureg.hour,
            useful_payload=2970 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.55, cruise=0.65, descent=0.4, turns=0.5,
            ),
            sources=[
                SourceRecord(
                    source_type="adsb",
                    reference=(
                        "King Air 350 ADS-B calibration, n=22 sorties "
                        "from airplanes.live globe-history archive "
                        "(University of Wyoming UWKA-2, N2UW; "
                        "single-tail, science-mission profile)"
                    ),
                    confidence=0.55,
                    url="https://www.airplanes.live/",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference=(
                        "Beechcraft King Air 350i AFM + Blackhawk XP-67A "
                        "upgrade datasheet; UW UWKA-2 facility specs "
                        "(approach_speed and descent_profile retained "
                        "from brochure due to single-bin sample artifacts)"
                    ),
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=75 * ureg.knot,
        )


# ---------------------------------------------------------------------------
# Other research / military aircraft
# ---------------------------------------------------------------------------

class NASA_C130(Aircraft):
    """NASA C-130H Hercules (NASA 436) four-engine turboprop research aircraft.

    Operated by NASA Wallops Flight Facility.  Calibration is specific
    to NASA 436's ACT-America mission profile; other C-130 operators
    (USAF, NCAR, NRL, FAA) fly different envelopes and would need
    their own calibration.

    See also:
        `https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF <https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF>`_
    """

    def __init__(self) -> None:
        # Calibrated against 91 IWG1 sorties from NASA Wallops C-130H
        # (NASA 436) flying the ACT-America campaign 2016-2019.  Data
        # downloaded from the public NASA ASP archive at
        # https://asp-archive.arc.nasa.gov/ACTAMERICA/N436NA/ via
        # ``notebooks/calibration/NASA_C130/_fetch_act_america.py``.  See
        # ``notebooks/calibration/NASA_C130/calibration.ipynb`` for the
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
            calibration_status="calibrated",
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
                    url="https://airbornescience.nasa.gov/data",
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


class NOAA_TwinOtter(Aircraft):
    """NOAA DHC-6 Twin Otter (N48RF + N46RF) STOL twin-turboprop.

    Operated by the NOAA Chemical Sciences Laboratory and NOAA AOC
    for boundary-layer atmospheric chemistry, lidar, and aerosol
    sampling missions.  Calibration is specific to NOAA's mission
    profile (slow-cruise dwell-time over plumes, FL060-FL150 ops);
    CIRPAS / Kenn Borek / commercial Twin Otters fly different
    envelopes and would need their own calibration class.

    See also:
        `https://www.omao.noaa.gov/aircraft/de-havilland-dhc-6-twin-otter`
    """

    def __init__(self) -> None:
        # Calibrated against 164 ICARTT sorties spanning two NOAA tails
        # (N48RF + N46RF) and seven campaigns:
        #   * FIREX-AQ 2019 (N48RF, ~17 sorties)
        #   * NOAA CSL archive (N46RF):
        #       - TopDown 2014, UWFPS 2017, CalFiDE 2022,
        #       - AEROMMA 2023, AMMBEC 2024, USOS 2024
        # AIMSS Probe / CUPiDS-AircraftData in-situ data — TAS,
        # attitude, wind, pressure.  Twin Otter climbs slowly
        # compared to jets/turboprops, so the active-climb threshold
        # is 500 fpm (vs the 1500 fpm used for jets and 1000 fpm
        # used for B-200).
        #
        # Per-file unit detection in the calibration script handles
        # inconsistent m/s vs kt labeling between PIs (some campaigns
        # advertise ``TrueAirSpd, m/s`` while values are in kt; the
        # aircraft's physical envelope at ~85 m/s lets us flag and
        # re-interpret the mislabeled files).
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="N48RF + N46RF",
            operator="NOAA",
            # p99 of per-sortie peak altitudes across 164 sorties.
            # 25000 ft brochure ceiling not approached in any campaign
            # — typical sub-orbital chemistry / lidar ops cruise
            # FL060-FL150.
            service_ceiling=17500 * ureg.feet,
            approach_speed=105 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet,  70 * ureg.knot),  # rotation (brochure)
                ( 5000 * ureg.feet, 119 * ureg.knot),
                (10000 * ureg.feet, 126 * ureg.knot),
                (12000 * ureg.feet, 126 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 126 * ureg.knot),
                ( 5000 * ureg.feet, 133 * ureg.knot),
                (10000 * ureg.feet, 142 * ureg.knot),
                (12000 * ureg.feet, 142 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 109 * ureg.knot),
                ( 5000 * ureg.feet, 133 * ureg.knot),
                (10000 * ureg.feet, 142 * ureg.knot),
                (12000 * ureg.feet, 142 * ureg.knot),
            ]),
            # Active-climb median (VS >= 500 fpm), 5-kft bins from 164
            # sorties.  Climb peaks at ~800 fpm @ FL050 then decays
            # to ~600 fpm at FL150 (limit of the operational envelope).
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  727 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  799 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  721 * ureg.feet / ureg.minute),
                (15000 * ureg.feet,  593 * ureg.feet / ureg.minute),
                # Residual rate near brochure ceiling so the
                # integrator terminates cleanly above the operational
                # cruise band.
                (25000 * ureg.feet,  200 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  785 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  808 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  824 * ureg.feet / ureg.minute),
                (15000 * ureg.feet,  689 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=800 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=6 * ureg.hour,
            useful_payload=4000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="icartt",
                    reference=(
                        "NOAA Twin Otter (N48RF + N46RF) ICARTT calibration, "
                        "n=164 sorties across FIREX-AQ 2019, TopDown 2014, "
                        "UWFPS 2017, CalFiDE 2022, AEROMMA 2023, AMMBEC 2024, "
                        "USOS 2024"
                    ),
                    confidence=0.85,
                    url="https://csl.noaa.gov/groups/csl4/measurements/",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="DHC-6 Twin Otter Series 300 manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=58 * ureg.knot,  # Vs0 landing config, MTOW
        )


# ---------------------------------------------------------------------------
# International research aircraft (UK / EU / DLR)
# ---------------------------------------------------------------------------

class BAS_TwinOtter(Aircraft):
    """British Antarctic Survey DHC-6-300 Twin Otter (MASIN-equipped).

    BAS operates two Twin Otters (VP-FBL, VP-FBB) from Rothera Research
    Station for Antarctic atmospheric science and from Ny-Ålesund for
    Arctic missions.  Same airframe class as `NOAA_TwinOtter` but the
    polar operating profile (sustained low-altitude survey, cold-soak,
    high-wind ops, short-field gravel landings) is distinct enough to
    warrant a separate calibration.

    See also:
        `https://www.bas.ac.uk/team/operations-team/operational-delivery/airborne-science/`
    """

    def __init__(self) -> None:
        # Calibrated against 105 valid BAS MASIN sorties spanning five
        # CEDA archives (2010-2022) covering the full polar operating
        # envelope:
        #   * OFCAP 2010-2011    (23 sorties, sub-Antarctic Falklands)
        #   * ACCACIA 2013       (34 sorties, high Arctic)
        #   * ORCHESTRA 2017-18  (22 sorties, Southern Ocean)
        #   * IGP 2018           (14 sorties, Iceland-Greenland Seas)
        #   * ArcticCyclones 2022 (15 sorties, summer Arctic cyclones)
        # OFCAP files use the JAVAD GPS suffix; ACCACIA / ORCHESTRA use
        # OXTS.  IGP and ArcticCyclones ship the QC subset only — TAS
        # reconstructed via wind triangle and VS from gps_alt finite
        # difference.  See ``notebooks/calibration/BAS_TwinOtter/calibrate.py``.
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="VP-FBL + VP-FBB",
            operator="BAS",
            # Operational p99 of per-sortie peak altitudes; polar surveys
            # spend most time below FL150.  Brochure ceiling 25000 ft.
            service_ceiling=14600 * ureg.feet,
            approach_speed=95 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 113 * ureg.knot),
                ( 5000 * ureg.feet, 118 * ureg.knot),
                (10000 * ureg.feet, 123 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 120 * ureg.knot),
                ( 5000 * ureg.feet, 138 * ureg.knot),
                (10000 * ureg.feet, 141 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 126 * ureg.knot),
                ( 5000 * ureg.feet, 137 * ureg.knot),
                (10000 * ureg.feet, 141 * ureg.knot),
            ]),
            # Active-climb median (VS >= 500 fpm) — same gate as NOAA TO.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 655 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 640 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 623 * ureg.feet / ureg.minute),
                # Residual rate above operational ceiling for clean
                # integrator termination (no data above FL110).
                (20000 * ureg.feet, 200 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 661 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 770 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 679 * ureg.feet / ureg.minute),
            ]),
            # ROLL p90 = 22° (>5° gate) from OXTS-equipped OFCAP / ACCACIA /
            # ORCHESTRA flights; capped at AFM normal-ops 30° floor.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=800 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=6 * ureg.hour,
            useful_payload=4000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.75,
            ),
            sources=[
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "BAS MASIN core data via CEDA, n=105 sorties: "
                        "OFCAP 2010-2011 (23) + ACCACIA 2013 (34) + "
                        "ORCHESTRA 2017-18 (22) + IGP 2018 (14) + "
                        "ArcticCyclones 2022 (15)"
                    ),
                    confidence=0.85,
                    url="https://catalogue.ceda.ac.uk/uuid/8be3dd7cdf44090d89aeb8f105421506",
                    doi="10.5285/8be3dd7cdf44090d89aeb8f105421506",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="DHC-6 Twin Otter Series 300 manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=58 * ureg.knot,
        )


class FAAM_BAe146(Aircraft):
    """FAAM BAe-146-301 atmospheric research aircraft (G-LUXE).

    The Facility for Airborne Atmospheric Measurements operates G-LUXE
    on behalf of NERC and the UK Met Office.  Four-engine regional jet
    with a uniquely flexible mission envelope: low-altitude (200 ft AGL)
    boundary-layer surveys to FL350 troposphere/stratosphere transits.

    See also:
        `https://www.faam.ac.uk/`
    """

    def __init__(self) -> None:
        # Calibrated against 125 valid FAAM Core Data Product 1 Hz NetCDFs
        # downloaded from CEDA (2017-2024 ASMM-tagged science campaigns
        # spanning 27 distinct projects: DCMEX, ACSIS/ARNA, MPHASE,
        # CCREST, CLARIFY, ACRUISE, ICE-D, WESCON, NAWDEX, STANCO, EMERGE,
        # ACAO, MEWS, AMCCA, VaMaMe, GRIM-SAF, PICASSO, plus FAAM
        # test/training).  See ``notebooks/calibration/FAAM_BAe146/calibrate.py``.
        #
        # FAAM files are exceptionally clean: every nav variable carries
        # units, WOW_IND gives a perfect on-ground filter, HGT_RADR feeds
        # the radar-altimeter approach gate.  Approach TAS comes from
        # the final 60 s of each sortie's airborne segment (FAAM flies
        # low-altitude science surveys at cruise speed, so MSL/AGL
        # alone don't isolate the landing).
        super().__init__(
            aircraft_type="BAe-146-301",
            tail_number="G-LUXE",
            operator="FAAM",
            service_ceiling=34500 * ureg.feet,
            approach_speed=121 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 218 * ureg.knot),
                ( 5000 * ureg.feet, 238 * ureg.knot),
                (10000 * ureg.feet, 262 * ureg.knot),
                (15000 * ureg.feet, 282 * ureg.knot),
                (20000 * ureg.feet, 302 * ureg.knot),
                (25000 * ureg.feet, 326 * ureg.knot),
                (30000 * ureg.feet, 350 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 216 * ureg.knot),
                ( 5000 * ureg.feet, 237 * ureg.knot),
                (10000 * ureg.feet, 257 * ureg.knot),
                (15000 * ureg.feet, 283 * ureg.knot),
                (20000 * ureg.feet, 308 * ureg.knot),
                (25000 * ureg.feet, 333 * ureg.knot),
                (30000 * ureg.feet, 369 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 215 * ureg.knot),
                ( 5000 * ureg.feet, 238 * ureg.knot),
                (10000 * ureg.feet, 261 * ureg.knot),
                (15000 * ureg.feet, 285 * ureg.knot),
                (20000 * ureg.feet, 306 * ureg.knot),
                (25000 * ureg.feet, 329 * ureg.knot),
                (30000 * ureg.feet, 360 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins, n>=30/bin.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1657 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1783 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1778 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1679 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1617 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1631 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1654 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1662 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1718 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1723 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1902 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1837 * ureg.feet / ureg.minute),
            ]),
            # p90 |Roll| during turn-state fixes (>5°): 33° from 125
            # sorties — the survey-mission bank the aircraft routinely
            # uses, above the 30° AFM normal-ops floor.
            turn_model=TurnModel(max_bank_deg=33.0),
            engine_type="jet",
            range=1800 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=5 * ureg.hour,
            useful_payload=8500 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.85, cruise=0.85, descent=0.85, turns=0.85,
            ),
            sources=[
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "FAAM Core Data Product 1 Hz, n=125 sorties from "
                        "CEDA archive 2017-2024 (27 ASMM-tagged campaigns)"
                    ),
                    confidence=0.85,
                    url="https://catalogue.ceda.ac.uk/uuid/86433ad261a64a82a3b7b56ed29e4717",
                    doi="10.5285/86433ad261a64a82a3b7b56ed29e4717",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="BAe-146-301 Aircraft Flight Manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=98 * ureg.knot,
        )


class SAFIRE_ATR42(Aircraft):
    """SAFIRE ATR-42-320 atmospheric research aircraft (F-HMTO).

    SAFIRE (Service des Avions Français Instrumentés pour la Recherche
    en Environnement) operates an ATR-42-320 turboprop for European
    atmospheric science campaigns.  Small, slow, payload-heavy — common
    European partner platform alongside FAAM and DLR HALO.

    See also:
        `https://www.safire.fr/`
    """

    def __init__(self) -> None:
        # Calibrated against 44 valid sorties combining two archives:
        #   * CEDA EUFAR (28 flights, 7 transnational-access projects:
        #     geomad, i-wake2, icare-qad, micwa, olacta2, tetrad, walitemp)
        #   * AERIS EUREC4A 2020 (19 flights, Caribbean trade-cumulus)
        #
        # CEDA EUFAR files lack TAS shipped directly — TAS is reconstructed
        # via wind triangle from position derivatives and the wind vector.
        # EUREC4A AERIS files ship native TAS and use it directly.  See
        # ``notebooks/calibration/SAFIRE_ATR42/calibrate.py``.
        super().__init__(
            aircraft_type="ATR-42-320",
            tail_number="F-HMTO",
            operator="SAFIRE",
            service_ceiling=24700 * ureg.feet,
            approach_speed=117 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 167 * ureg.knot),
                ( 5000 * ureg.feet, 180 * ureg.knot),
                (10000 * ureg.feet, 195 * ureg.knot),
                (15000 * ureg.feet, 195 * ureg.knot),
                (20000 * ureg.feet, 191 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 185 * ureg.knot),
                ( 5000 * ureg.feet, 205 * ureg.knot),
                (10000 * ureg.feet, 213 * ureg.knot),
                (15000 * ureg.feet, 227 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 186 * ureg.knot),
                ( 5000 * ureg.feet, 207 * ureg.knot),
                (10000 * ureg.feet, 224 * ureg.knot),
                (15000 * ureg.feet, 238 * ureg.knot),
                (20000 * ureg.feet, 248 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1000 fpm) — turboprop gate, lower
            # than jets but higher than the Twin Otter.  ATR-42 climb is
            # genuinely flat in the operational envelope.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1134 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1185 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1109 * ureg.feet / ureg.minute),
                # Residual rate near brochure ceiling for integrator
                # termination above the operational cruise band.
                (20000 * ureg.feet,  600 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1272 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1328 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1437 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1250 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),  # p90 28°, AFM floor 30°
            engine_type="turboprop",
            range=900 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=5 * ureg.hour,
            useful_payload=5500 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.75, cruise=0.75, descent=0.75, turns=0.7,
            ),
            sources=[
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "SAFIRE ATR-42 EUFAR Transnational Access core nav "
                        "via CEDA (28 flights across 7 TA projects)"
                    ),
                    confidence=0.75,
                    url="https://catalogue.ceda.ac.uk/uuid/3d27ed6d44614ea6a9f4f59ff7e4e1ec",
                ),
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "SAFIRE ATR-42 EUREC4A 2020 core L2 1 Hz "
                        "via AERIS (19 flights, native TAS)"
                    ),
                    confidence=0.85,
                    url="https://eurec4a.aeris-data.fr/",
                    doi="10.25326/162",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="ATR-42-320 Aircraft Flight Manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=83 * ureg.knot,
        )


class NERC_DO228(Aircraft):
    """NERC ARSF Dornier Do228-101 atmospheric / remote-sensing aircraft.

    The NERC Airborne Research and Survey Facility operated D-CALM as a
    medium-tropospheric, non-pressurised twin-turboprop research platform.
    This model is calibrated from public CEDA NERC/ARSF DO228 datasets,
    principally ACTIVE and Eyjafjallajokull core aircraft measurements.

    The calibration is useful for timing / reachability but deliberately
    carries lower confidence than platforms with native TAS and attitude:
    the CEDA files provide position, altitude, U/V wind, pressure, and
    temperature, so TAS is reconstructed by wind triangle and turn/bank
    behaviour remains brochure/default.

    See also:
        `https://catalogue.ceda.ac.uk/uuid/d2c5c36981824b71a98a2906394d61f3/`
    """

    def __init__(self) -> None:
        # Calibrated against 34 valid CEDA NetCDF sorties:
        #   * ACTIVE 2005-2006 Dornier-D-Calm core instruments (27 sorties)
        #   * Eyjafjallajokull 2010 NERC ARSF aircraft 1 Hz core data (7 sorties)
        #
        # Both archives ship lat/lon/altitude and U/V wind; TAS is
        # reconstructed from finite-difference groundspeed minus wind.
        # SOLAS-SLATEA (7 NASA Ames files) and NAMBLEX GPS-only (3 files)
        # were collected for independent geometry/profile checks but are
        # not used in the TAS schedules because they lack the full wind/TAS
        # state needed for a consistent fit.
        super().__init__(
            aircraft_type="Dornier Do228-101",
            tail_number="D-CALM",
            operator="NERC ARSF",
            # CEDA platform record: normal operational science ceiling 15 kft,
            # maximum ceiling 22 kft depending on oxygen / instrument limits.
            # The calibration set reaches ~FL230 p99, so use the platform
            # max ceiling as the model envelope.
            service_ceiling=22000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 146 * ureg.knot),
                ( 5000 * ureg.feet, 150 * ureg.knot),
                (10000 * ureg.feet, 162 * ureg.knot),
                (15000 * ureg.feet, 149 * ureg.knot),
                (20000 * ureg.feet, 140 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 168 * ureg.knot),
                ( 5000 * ureg.feet, 174 * ureg.knot),
                (10000 * ureg.feet, 176 * ureg.knot),
                (15000 * ureg.feet, 162 * ureg.knot),
                (20000 * ureg.feet, 146 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 171 * ureg.knot),
                ( 5000 * ureg.feet, 190 * ureg.knot),
                (10000 * ureg.feet, 197 * ureg.knot),
                (15000 * ureg.feet, 160 * ureg.knot),
                (20000 * ureg.feet, 183 * ureg.knot),
            ]),
            # Active-climb median (VS >= 700 fpm), 5-kft bins.  The
            # threshold is lower than the generic turboprop 1000 fpm gate
            # because this DO228 dataset often climbs in the 750-950 fpm
            # range during research operations.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  926 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  942 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  827 * ureg.feet / ureg.minute),
                (15000 * ureg.feet,  755 * ureg.feet / ureg.minute),
                # Residual rate at the platform envelope ceiling so the
                # integrator terminates cleanly above dense data coverage.
                (22000 * ureg.feet,  300 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  965 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1014 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  880 * ureg.feet / ureg.minute),
                (15000 * ureg.feet,  927 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1318 * ureg.feet / ureg.minute),
            ]),
            # The public CEDA DO228 core files used here do not expose roll
            # or bank angle.  Keep the normal-category AFM/default turn
            # envelope until an attitude-capable D-CALM archive is found.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            # CEDA platform record: max range 2600 km, max-payload range
            # 1800 km, normal measurement-flight range 1500 km.  HyPlan's
            # range field is the aircraft envelope, not typical sortie use.
            range=1400 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=5 * ureg.hour,
            useful_payload=3516 * ureg.pound,  # 1595 kg platform max payload
            confidence=PerformanceConfidence(
                climb=0.7, cruise=0.65, descent=0.7, turns=0.5,
            ),
            sources=[
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "CEDA NERC/ARSF DO228 calibration: ACTIVE core "
                        "instruments (27 sorties, 2005-2006) + "
                        "Eyjafjallajokull ARSF 1 Hz core data (7 sorties, 2010)"
                    ),
                    url="https://catalogue.ceda.ac.uk/uuid/d2c5c36981824b71a98a2906394d61f3",
                    doi="10.5285/d2c5c36981824b71a98a2906394d61f3",
                    notes=(
                        "TAS reconstructed from finite-difference groundspeed "
                        "and U/V wind. SOLAS-SLATEA and NAMBLEX GPS-only "
                        "archives were collected as supporting validation "
                        "sources but not used for TAS schedule fitting."
                    ),
                    confidence=0.7,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference=(
                        "CEDA platform record: NERC ARSF Dornier Do228-101 "
                        "D-CALM Aircraft"
                    ),
                    confidence=0.5,
                ),
            ],
        )


class AWI_BaslerBT67(Aircraft):
    """AWI Polar 5 / Polar 6 Basler BT-67 polar research aircraft.

    The Alfred Wegener Institute operates two broadly similar Basler BT-67
    aircraft, Polar 5 and Polar 6, for Arctic and Antarctic airborne science.
    This model intentionally represents the combined AWI BT-67 operating
    envelope rather than one tail: the calibration uses public PANGAEA
    Polar 5 and Polar 6 nav/met records from ACLOUD 2017 and HALO-AC3 2022.

    The PANGAEA products include native TAS, ground speed, attitude, U/V/W
    wind, pressure, and temperature.  Vertical rate is derived from 1 Hz
    altitude, which is quantized to whole metres in these public files; the
    climb/descent profiles are therefore intentionally simple and lower
    confidence than the TAS schedules and turn envelope.

    See also:
        `https://www.awi.de/en/fleet-stations/aircraft/polar-5-6.html <https://www.awi.de/en/fleet-stations/aircraft/polar-5-6.html>`_
    """

    def __init__(self) -> None:
        # Calibrated against 78 valid public PANGAEA sorties currently collected
        # under data/BT67/AWI_Polar:
        #   * ACLOUD 2017 1 Hz wind/temperature (47 downloaded sorties;
        #     23 Polar 5 + 24 Polar 6)
        #   * HALO-AC3 2022 wind/temperature (31 valid sorties after a
        #     short-flight filter; 16 Polar 5 + 15 Polar 6)
        #
        # The data coverage is dense below FL150 and sparse above; retain
        # the Basler BT-67 25 kft service ceiling as the model envelope,
        # but treat the high-altitude schedule/profile points as residual
        # extrapolation for clean planner integration.
        super().__init__(
            aircraft_type="Basler BT-67",
            tail_number="Polar 5 + Polar 6",
            operator="AWI",
            service_ceiling=25000 * ureg.feet,
            approach_speed=90 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 126 * ureg.knot),
                ( 5000 * ureg.feet, 136 * ureg.knot),
                (10000 * ureg.feet, 146 * ureg.knot),
                (15000 * ureg.feet, 180 * ureg.knot),
                (20000 * ureg.feet, 180 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 127 * ureg.knot),
                ( 5000 * ureg.feet, 158 * ureg.knot),
                (10000 * ureg.feet, 186 * ureg.knot),
                (15000 * ureg.feet, 198 * ureg.knot),
                (20000 * ureg.feet, 198 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 138 * ureg.knot),
                ( 5000 * ureg.feet, 176 * ureg.knot),
                (10000 * ureg.feet, 191 * ureg.knot),
                (15000 * ureg.feet, 213 * ureg.knot),
                (20000 * ureg.feet, 213 * ureg.knot),
            ]),
            # Active-climb median (VS >= 500 fpm) from 1 Hz altitude
            # differences.  The 1 m altitude quantization yields ~197 fpm
            # increments, so use a smooth operational profile rather than
            # overfitting small per-bin fluctuations.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  771 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  755 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  702 * ureg.feet / ureg.minute),
                (25000 * ureg.feet,  250 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet,  646 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet,  715 * ureg.feet / ureg.minute),
                (10000 * ureg.feet,  689 * ureg.feet / ureg.minute),
                (15000 * ureg.feet,  666 * ureg.feet / ureg.minute),
                (20000 * ureg.feet,  700 * ureg.feet / ureg.minute),
            ]),
            # p90 |roll| during turn-state fixes (>5°) is ~23° across the
            # collected PANGAEA records.  Keep the normal-category 30° floor
            # as the max-bank envelope for planning.
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="turboprop",
            range=1600 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=6.5 * ureg.hour,
            useful_payload=4000 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.65, cruise=0.8, descent=0.65, turns=0.75,
            ),
            sources=[
                SourceRecord(
                    source_type="pangaea",
                    reference=(
                        "AWI Polar 5/Polar 6 Basler BT-67 nav/met: "
                        "ACLOUD 2017 1 Hz wind-temperature data "
                        "(doi:10.1594/PANGAEA.902849; 47 downloaded sorties) "
                        "+ HALO-AC3 2022 wind-temperature series "
                        "(doi:10.1594/PANGAEA.968911; 31 valid sorties)"
                    ),
                    url="https://www.pangaea.de/?q=Polar+5+Polar+6+Basler",
                    doi="10.1594/PANGAEA.902849",
                    notes=(
                        "Model combines Polar 5 and Polar 6 because both are "
                        "AWI Basler BT-67 aircraft and the public PANGAEA "
                        "records are balanced across tails. Native TAS and "
                        "roll are used directly; vertical rates are derived "
                        "from 1 Hz whole-metre altitude."
                    ),
                    confidence=0.75,
                ),
                SourceRecord(
                    source_type="brochure",
                    reference=(
                        "AWI Polar 5/6 platform description and Basler BT-67 "
                        "published aircraft specifications"
                    ),
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=75 * ureg.knot,
        )


class DLR_HALO(Aircraft):
    """DLR HALO (D-ADLR) Gulfstream G550 high-altitude long-range research aircraft.

    HALO (High Altitude and LOng range) is operated by DLR
    Flugexperimente at Oberpfaffenhofen.  Same airframe family as
    `NCAR_GV` (HIAPER) — both Gulfstream V/G550 — but operated
    independently and typically configured for different payloads.

    See also:
        `https://www.dlr.de/en/research-and-transfer/research-infrastructure/halo`
    """

    def __init__(self) -> None:
        # Calibrated against 18 HALO-AC3 BAHAMAS sorties (March-April
        # 2022, Arctic).  Single-campaign dataset; confidence 0.7
        # reflects the narrower operating-envelope sample compared to
        # multi-year multi-campaign calibrations.  BAHAMAS is HALO's
        # native sensor system and ships everything we need:
        # IRS-derived TAS, attitude, and vertical velocity at 10 Hz
        # (downsampled to 1 Hz for calibration).  See
        # ``notebooks/calibration/DLR_HALO/calibrate.py``.
        super().__init__(
            aircraft_type="Gulfstream G550",
            tail_number="D-ADLR",
            operator="DLR",
            # Operational p99 of per-sortie peak altitudes; HALO-AC3
            # mostly stayed at FL400-FL410 for science legs.  G550
            # certified ceiling is 51000 ft.
            service_ceiling=44300 * ureg.feet,
            approach_speed=128 * ureg.knot,
            climb_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 207 * ureg.knot),
                (10000 * ureg.feet, 297 * ureg.knot),
                (20000 * ureg.feet, 349 * ureg.knot),
                (30000 * ureg.feet, 419 * ureg.knot),
                (40000 * ureg.feet, 455 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 158 * ureg.knot),
                (10000 * ureg.feet, 302 * ureg.knot),
                (20000 * ureg.feet, 457 * ureg.knot),
                (30000 * ureg.feet, 457 * ureg.knot),
                (35000 * ureg.feet, 464 * ureg.knot),
                (40000 * ureg.feet, 455 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (    0 * ureg.feet, 134 * ureg.knot),
                (10000 * ureg.feet, 298 * ureg.knot),
                (20000 * ureg.feet, 352 * ureg.knot),
                (30000 * ureg.feet, 422 * ureg.knot),
                (40000 * ureg.feet, 454 * ureg.knot),
            ]),
            # Active-climb median (VS >= 1500 fpm), 5-kft bins.  Peak
            # ROC ~2500 fpm at FL050, decays to ~1650 fpm at FL350.
            climb_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 2360 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 2493 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2413 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 2133 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1905 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1770 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1748 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1651 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (    0 * ureg.feet, 1572 * ureg.feet / ureg.minute),
                ( 5000 * ureg.feet, 1749 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 1767 * ureg.feet / ureg.minute),
                (15000 * ureg.feet, 1848 * ureg.feet / ureg.minute),
                (20000 * ureg.feet, 1895 * ureg.feet / ureg.minute),
                (25000 * ureg.feet, 1931 * ureg.feet / ureg.minute),
                (30000 * ureg.feet, 1991 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 1844 * ureg.feet / ureg.minute),
                (40000 * ureg.feet, 1817 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),  # p90 27°, AFM floor 30°
            engine_type="jet",
            range=6750 * ureg.nautical_mile,
            calibration_status="calibrated",
            endurance=10 * ureg.hour,
            useful_payload=6172 * ureg.pound,
            confidence=PerformanceConfidence(
                climb=0.7, cruise=0.7, descent=0.7, turns=0.65,
            ),
            sources=[
                SourceRecord(
                    source_type="netcdf",
                    reference=(
                        "DLR HALO BAHAMAS 1 Hz, n=18 HALO-AC3 sorties "
                        "(March-April 2022, Arctic)"
                    ),
                    confidence=0.7,
                    url="https://halo-db.pa.op.dlr.de/mission/120",
                    doi="10.1594/PANGAEA.967719",
                ),
                SourceRecord(
                    source_type="brochure",
                    reference="Gulfstream G550 Performance Manual",
                    confidence=0.5,
                ),
            ],
            stall_speed_cas=110 * ureg.knot,
        )
