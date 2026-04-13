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
    "DynamicAviation_DH8",
    "DynamicAviation_A90",
    "DynamicAviation_B200",
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

    Speed profile from Moving Lines: TAS = 70 + alt_m * 0.0071 (m/s).

    See also:
        https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 136 * ureg.knot),
            (70000 * ureg.feet, 431 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="ER-2",
            tail_number="NASA 806",
            operator="NASA AFRC",
            service_ceiling=70000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            climb_schedule=cruise,
            cruise_schedule=cruise,
            descent_schedule=cruise,
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 5000 * ureg.feet / ureg.minute),
                (70000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            turn_model=TurnModel(max_bank_deg=30.0),
            engine_type="jet",
            range=5000 * ureg.nautical_mile,
            endurance=8 * ureg.hour,
            useful_payload=2900 * ureg.pound,
            sources=[SourceRecord(
                source_type="brochure",
                reference="NASA Airborne Science, Moving Lines project",
                confidence=0.6,
            )],
        )


# ---------------------------------------------------------------------------
# Gulfstream business jets (NASA)
# ---------------------------------------------------------------------------

class NASA_GIII(Aircraft):
    """NASA Gulfstream III (NASA 520) research aircraft.

    Operated by NASA Langley Research Center (LaRC).

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC
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
        https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC
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
        https://airbornescience.nasa.gov/aircraft/Gulfstream_V_-_AFRC
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
        https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC
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
        https://airbornescience.nasa.gov/aircraft/P-3_Orion
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
        https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC
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
# Dynamic Aviation contract aircraft
# ---------------------------------------------------------------------------

class DynamicAviation_DH8(Aircraft):
    """Dynamic Aviation DHC-8 Dash 8 twin-turboprop aircraft.

    See also:
        https://www.dynamicaviation.com/fleet-dash-8
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 170 * ureg.knot),
            (25000 * ureg.feet, 243 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="Dash 8",
            tail_number="Unknown",
            operator="Dynamic Aviation",
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


class DynamicAviation_A90(Aircraft):
    """Dynamic Aviation Beechcraft King Air A90 twin-turboprop aircraft.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 170 * ureg.knot),
            (30000 * ureg.feet, 230 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="King Air 90",
            tail_number="Unknown",
            operator="Dynamic Aviation",
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


class DynamicAviation_B200(Aircraft):
    """Dynamic Aviation Beechcraft King Air B200 twin-turboprop aircraft.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200
    """

    def __init__(self):
        cruise = TasSchedule(points=[
            (0 * ureg.feet, 185 * ureg.knot),
            (35000 * ureg.feet, 250 * ureg.knot),
        ])
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="Unknown",
            operator="Dynamic Aviation",
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
        https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF
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
        https://faam.ac.uk/
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
        https://airbornescience.nasa.gov/aircraft/Learjet_25
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
        https://airbornescience.nasa.gov/aircraft/Twin_Otter_-_CIRPAS_-_NPS
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
            service_ceiling=10000 * ureg.feet,
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
            turn_model=TurnModel(max_bank_deg=15.0),
            engine_type="turboprop",
            range=800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            useful_payload=4000 * ureg.pound,
        )
