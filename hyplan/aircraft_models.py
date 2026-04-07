"""Concrete aircraft definitions for HyPlan.

Each class is a thin :class:`~hyplan.aircraft.Aircraft` subclass that fills
in the performance parameters for a specific airborne science platform.
The base class lives in :mod:`hyplan.aircraft` and is responsible for all
the climb / cruise / descent calculations; this module only carries the
data.

For backwards compatibility every class defined here is re-exported from
:mod:`hyplan.aircraft`, so existing code that does
``from hyplan.aircraft import NASA_ER2`` continues to work.
"""

from __future__ import annotations

from .aircraft import Aircraft
from .units import ureg

__all__ = [
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
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


class NASA_ER2(Aircraft):
    """
    NASA ER-2 high-altitude research aircraft.

    Operates at 70,000 ft, acquiring data above 95% of the Earth's atmosphere.
    Based at NASA Armstrong Flight Research Center (AFRC).

    Speed profile from Moving Lines: TAS = 70 + alt_m * 0.0071 (m/s).
    Linear increase to ceiling with no cap (cap altitude exceeds ceiling).

    See also:
        https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="ER-2",
            tail_number="NASA 806",
            service_ceiling=70000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=5000 * ureg.feet / ureg.minute,
            cruise_speed=410 * ureg.knot,
            range=5000 * ureg.nautical_mile,
            endurance=8 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=2900 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=0 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 136 * ureg.knot),
                (70000 * ureg.feet, 431 * ureg.knot),
            ],
        )

class NASA_GIII(Aircraft):
    """
    NASA Gulfstream III (NASA 520) research aircraft.

    Operated by NASA Langley Research Center (LaRC) for earth science
    research and remote sensing missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Gulfstream III",
            tail_number="NASA 520",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=459 * ureg.knot,
            range=3767 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=2610 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 459 * ureg.knot),
            ],
        )

class NASA_GIV(Aircraft):
    """
    NASA Gulfstream IV (NASA 817) research aircraft.

    Twin turbofan business-class aircraft operated by NASA Armstrong
    Flight Research Center (AFRC). Useful payload of 5,610 lbs.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Gulfstream IV",
            tail_number="NASA 817",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=459 * ureg.knot,
            range=5130 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=5610 * ureg.pound,
            vx=150 * ureg.knot,
            vy=170 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 459 * ureg.knot),
            ],
        )

class NASA_C20A(Aircraft):
    """
    NASA C-20A (Gulfstream III variant, NASA 502) research aircraft.

    A business jet structurally modified and instrumented by NASA Armstrong
    Flight Research Center to serve as a multi-role cooperative research
    platform. Obtained from the U.S. Air Force in 2003. Primary platform
    for UAVSAR missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="C-20A",
            tail_number="NASA 502",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=3500 * ureg.feet / ureg.minute,
            cruise_speed=460 * ureg.knot,
            range=3400 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=2500 * ureg.pound,
            vx=150 * ureg.knot,
            vy=170 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 460 * ureg.knot),
            ],
        )

class NASA_P3(Aircraft):
    """
    NASA P-3 Orion (NASA 426) airborne science laboratory.

    A four-engine turboprop aircraft capable of long-duration flights
    (8-14 hours) and large payloads up to 14,700 lbs. Operated by
    NASA Wallops Flight Facility (WFF). Supports ecology, geography,
    hydrology, meteorology, oceanography, atmospheric chemistry,
    cryospheric research, and satellite calibration/validation.

    Speed profile from Moving Lines: TAS = 110 + alt_m * 0.007 (m/s),
    capped at 155 m/s (~301 kt) above ~21,000 ft.

    See also:
        https://airbornescience.nasa.gov/aircraft/P-3_Orion
    """
    def __init__(self):
        super().__init__(
            aircraft_type="P-3 Orion",
            tail_number="NASA 426",
            service_ceiling=32000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=3500 * ureg.feet / ureg.minute,
            cruise_speed=301 * ureg.knot,
            range=3800 * ureg.nautical_mile,
            endurance=12 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=18000 * ureg.pound,
            vx=135 * ureg.knot,
            vy=155 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 214 * ureg.knot),
                (21091 * ureg.feet, 301 * ureg.knot),
                (32000 * ureg.feet, 301 * ureg.knot),
            ],
        )

class NASA_WB57(Aircraft):
    """
    NASA WB-57 (NASA 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field. Three
    WB-57 aircraft have been flying research missions since the early
    1970s. Operates up to 60,000 ft with 8,800 lbs useful payload.

    Speed profile assumed similar to ER-2 (linear increase to ceiling).

    See also:
        https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="WB-57",
            tail_number="NASA 927",
            service_ceiling=60000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=5000 * ureg.feet / ureg.minute,
            cruise_speed=410 * ureg.knot,
            range=2500 * ureg.nautical_mile,
            endurance=6.5 * ureg.hour,
            operator="NASA JSC",
            max_bank_angle=30.0,
            useful_payload=8800 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=0 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 136 * ureg.knot),
                (60000 * ureg.feet, 410 * ureg.knot),
            ],
        )

class NASA_B777(Aircraft):
    """
    NASA Boeing 777 long-range research aircraft.

    Operated by NASA Langley Research Center (LaRC). Very large payload
    capacity (75,000 lbs) and long endurance (18 hours), suitable for
    global-scale remote sensing missions.
    """
    def __init__(self):
        super().__init__(
            aircraft_type="B777",
            tail_number="Unknown",
            service_ceiling=43000 * ureg.feet,
            approach_speed=150 * ureg.knot,
            best_rate_of_climb=2500 * ureg.feet / ureg.minute,
            cruise_speed=487 * ureg.knot,
            range=9000 * ureg.nautical_mile,
            endurance=18 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=75000 * ureg.pound,
            vx=160 * ureg.knot,
            vy=180 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=30 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 350 * ureg.knot),
                (43000 * ureg.feet, 487 * ureg.knot),
            ],
        )

class DynamicAviation_DH8(Aircraft):
    """
    Dynamic Aviation DHC-8 Dash 8 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. Medium-lift platform suitable for both
    high and low altitude missions with 15,000 lbs useful payload.

    See also:
        https://www.dynamicaviation.com/fleet-dash-8
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Dash 8",
            tail_number="Unknown",
            service_ceiling=25000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=243 * ureg.knot,
            range=950 * ureg.nautical_mile,
            endurance=5 * ureg.hour,
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=15000 * ureg.pound,
            vx=110 * ureg.knot,
            vy=130 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=15 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 170 * ureg.knot),
                (25000 * ureg.feet, 243 * ureg.knot),
            ],
        )

class DynamicAviation_A90(Aircraft):
    """
    Dynamic Aviation Beechcraft King Air A90 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. Two-engine turboprop used for sensor
    integration, flight testing, and airborne science support.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90
    """
    def __init__(self):
        super().__init__(
            aircraft_type="King Air 90",
            tail_number="Unknown",
            service_ceiling=30000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            best_rate_of_climb=1800 * ureg.feet / ureg.minute,
            cruise_speed=230 * ureg.knot,
            range=1500 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=2950 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=10 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 170 * ureg.knot),
                (30000 * ureg.feet, 230 * ureg.knot),
            ],
        )

class DynamicAviation_B200(Aircraft):
    """
    Dynamic Aviation Beechcraft King Air B200 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. An all-metal twin-turboprop capable of
    operating from a wide variety of civilian and military airports.
    Well suited for aerial remote sensing, chase aircraft support,
    and technology demonstration missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200
    """
    def __init__(self):
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="Unknown",
            service_ceiling=35000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=250 * ureg.knot,
            range=1632 * ureg.nautical_mile,
            endurance=6 * ureg.hour,  # Estimated
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=4250 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=10 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 185 * ureg.knot),
                (35000 * ureg.feet, 250 * ureg.knot),
            ],
        )

class C130(Aircraft):
    """
    C-130H Hercules four-engine turboprop transport / research aircraft.

    Used by multiple agencies (NSF/NCAR, NASA WFF, NOAA) for airborne
    science. In a typical research configuration carries 13,000 lbs of
    payload with 8-9 hour endurance. The NASA C-130H (N436NA) at Wallops
    Flight Facility supports airborne scientific research and cargo.

    Speed profile from Moving Lines: TAS = 130 + alt_m * 0.0075 (m/s),
    capped at 175 m/s (~340 kt) above ~19,685 ft.

    See also:
        https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF
    """
    def __init__(self):
        super().__init__(
            aircraft_type="C-130H Hercules",
            tail_number="Unknown",
            service_ceiling=25000 * ureg.feet,
            approach_speed=115 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=340 * ureg.knot,
            range=2500 * ureg.nautical_mile,
            endurance=10 * ureg.hour,
            operator="Various",
            max_bank_angle=20.0,
            useful_payload=45000 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100 * ureg.feet / ureg.minute,
            descent_rate=2000 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 253 * ureg.knot),
                (19685 * ureg.feet, 340 * ureg.knot),
                (25000 * ureg.feet, 340 * ureg.knot),
            ],
        )

class BAe146(Aircraft):
    """
    BAe-146-301 atmospheric research aircraft (G-LUXE).

    Operated by the UK Facility for Airborne Atmospheric Measurements
    (FAAM). Can fly with up to 4 tonnes of scientific instruments,
    from 50 ft over the sea to 35,000 ft, with flights lasting 1-6
    hours covering up to 2,000 nautical miles. Has completed 1,400+
    science flights across 30 countries.

    Speed profile from Moving Lines: TAS = 130 + alt_m * 0.002 (m/s),
    capped at 150 m/s (~292 kt) above ~32,808 ft (above ceiling, so
    effectively linear to ceiling).

    See also:
        https://faam.ac.uk/
    """
    def __init__(self):
        super().__init__(
            aircraft_type="BAe-146",
            tail_number="Unknown",
            service_ceiling=28000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            best_rate_of_climb=1000 * ureg.feet / ureg.minute,
            cruise_speed=286 * ureg.knot,
            range=1800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="FAAM",
            max_bank_angle=20.0,
            useful_payload=10000 * ureg.pound,
            vx=130 * ureg.knot,
            vy=145 * ureg.knot,
            roc_at_service_ceiling=100 * ureg.feet / ureg.minute,
            descent_rate=1000 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 253 * ureg.knot),
                (28000 * ureg.feet, 286 * ureg.knot),
            ],
        )

class Learjet(Aircraft):
    """
    Learjet high-altitude research aircraft.

    Used for atmospheric and remote sensing research by various
    operators. NASA has operated multiple Learjet variants (23, 24D,
    25, 35) for high-altitude atmospheric science.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        https://airbornescience.nasa.gov/aircraft/Learjet_25
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Learjet",
            tail_number="Unknown",
            service_ceiling=35000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=430 * ureg.knot,
            range=1500 * ureg.nautical_mile,
            endurance=4 * ureg.hour,
            operator="Various",
            max_bank_angle=30.0,
            useful_payload=3000 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=39 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 194 * ureg.knot),
                (35000 * ureg.feet, 430 * ureg.knot),
            ],
        )

class TwinOtter(Aircraft):
    """
    DHC-6 Twin Otter STOL twin-turboprop utility aircraft.

    Common low-altitude research platform. The NPS CIRPAS Twin Otter
    (based at Naval Postgraduate School, Monterey, CA) has supported
    atmospheric and oceanographic research since 1998 for ONR, NSF,
    DOE, NOAA, NASA, and others. Twin Otter International Ltd. (TOIL)
    also operates a fleet for medium-lift, slow-flight research.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        https://airbornescience.nasa.gov/aircraft/Twin_Otter_-_CIRPAS_-_NPS
    """
    def __init__(self):
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="Unknown",
            service_ceiling=10000 * ureg.feet,
            approach_speed=70 * ureg.knot,
            best_rate_of_climb=430 * ureg.feet / ureg.minute,
            cruise_speed=150 * ureg.knot,
            range=800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="Various",
            max_bank_angle=15.0,
            useful_payload=4000 * ureg.pound,
            vx=75 * ureg.knot,
            vy=85 * ureg.knot,
            roc_at_service_ceiling=50 * ureg.feet / ureg.minute,
            descent_rate=430 * ureg.feet / ureg.minute,
            descent_speed_reduction=8 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 97 * ureg.knot),
                (10000 * ureg.feet, 150 * ureg.knot),
            ],
        )
