"""Tests for hyplan.aircraft."""

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.aircraft import (
    CasMachSchedule,
    TasSchedule,
    VerticalProfile,
    TurnModel,
    PhaseBankAngles,
    PerformanceConfidence,
    SourceRecord,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    DynamicAviation_B200 as B200,
    C130,
    TwinOtter,
)
from hyplan.exceptions import HyPlanValueError


# ---------------------------------------------------------------------------
# TasSchedule
# ---------------------------------------------------------------------------

class TestTasSchedule:
    def test_single_point_constant(self):
        s = TasSchedule(points=[(0 * ureg.feet, 200 * ureg.knot)])
        assert s.tas_at(0 * ureg.feet).m_as("knot") == pytest.approx(200)
        assert s.tas_at(30000 * ureg.feet).m_as("knot") == pytest.approx(200)

    def test_two_point_interpolation(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 180 * ureg.knot),
            (30000 * ureg.feet, 300 * ureg.knot),
        ])
        assert s.tas_at(15000 * ureg.feet).m_as("knot") == pytest.approx(240)

    def test_clamps_below(self):
        s = TasSchedule(points=[
            (5000 * ureg.feet, 200 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        assert s.tas_at(0 * ureg.feet).m_as("knot") == pytest.approx(200)

    def test_clamps_above(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 200 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        assert s.tas_at(40000 * ureg.feet).m_as("knot") == pytest.approx(280)

    def test_requires_ascending_altitudes(self):
        with pytest.raises(HyPlanValueError):
            TasSchedule(points=[
                (20000 * ureg.feet, 280 * ureg.knot),
                (10000 * ureg.feet, 200 * ureg.knot),
            ])

    def test_requires_at_least_one_point(self):
        with pytest.raises(HyPlanValueError):
            TasSchedule(points=[])

    def test_unit_conversion(self):
        s = TasSchedule(points=[
            (0 * ureg.meter, 100 * ureg.meter / ureg.second),
            (3000 * ureg.meter, 120 * ureg.meter / ureg.second),
        ])
        result = s.tas_at(1500 * ureg.meter)
        assert result.check("[length] / [time]")

    def test_three_point(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 180 * ureg.knot),
            (15000 * ureg.feet, 260 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        # Midpoint of first segment
        assert s.tas_at(7500 * ureg.feet).m_as("knot") == pytest.approx(220)


# ---------------------------------------------------------------------------
# CasMachSchedule
# ---------------------------------------------------------------------------

class TestCasMachSchedule:
    def test_below_crossover_uses_cas(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_low = s.tas_at(10000 * ureg.feet)
        assert tas_low.m_as("knot") > 280  # TAS > CAS at altitude

    def test_above_crossover_uses_mach(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_high = s.tas_at(40000 * ureg.feet)
        # M0.80 at FL400 (above tropopause, isothermal) → ~460 kt
        assert 440 < tas_high.m_as("knot") < 480

    def test_speed_continuity_near_crossover(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_below = s.tas_at(27900 * ureg.feet).m_as("knot")
        tas_above = s.tas_at(28100 * ureg.feet).m_as("knot")
        # CAS and Mach targets don't perfectly match at crossover_ft,
        # but the transition should not be wildly discontinuous.
        assert abs(tas_below - tas_above) < 60

    def test_sea_level_cas_approx_tas(self):
        s = CasMachSchedule(cas=250 * ureg.knot, mach=0.78, crossover_ft=30000)
        tas = s.tas_at(0 * ureg.feet)
        # At sea level, CAS ≈ TAS
        assert abs(tas.m_as("knot") - 250) < 1


# ---------------------------------------------------------------------------
# VerticalProfile
# ---------------------------------------------------------------------------

class TestVerticalProfile:
    def test_constant_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "constant"
        assert vp.rate_at(15000 * ureg.feet).m_as("feet/minute") == pytest.approx(2000)

    def test_two_point_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "two_point"
        # Midpoint interpolation
        mid = vp.rate_at(17500 * ureg.feet).m_as("feet/minute")
        assert mid == pytest.approx(1050)

    def test_full_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 3800 * ureg.feet / ureg.minute),
            (10000 * ureg.feet, 3200 * ureg.feet / ureg.minute),
            (20000 * ureg.feet, 2400 * ureg.feet / ureg.minute),
            (30000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "full"

    def test_sea_level_rate(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp.sea_level_rate.m_as("feet/minute") == pytest.approx(2000)

    def test_ceiling_rate(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp.ceiling_rate.m_as("feet/minute") == pytest.approx(100)

    def test_requires_ascending_altitudes(self):
        with pytest.raises(HyPlanValueError):
            VerticalProfile(points=[
                (20000 * ureg.feet, 1000 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            ])

    def test_requires_at_least_one_point(self):
        with pytest.raises(HyPlanValueError):
            VerticalProfile(points=[])

    def test_clamps_at_endpoints(self):
        vp = VerticalProfile(points=[
            (5000 * ureg.feet, 1800 * ureg.feet / ureg.minute),
            (25000 * ureg.feet, 500 * ureg.feet / ureg.minute),
        ])
        assert vp.rate_at(0 * ureg.feet).m_as("feet/minute") == pytest.approx(1800)
        assert vp.rate_at(40000 * ureg.feet).m_as("feet/minute") == pytest.approx(500)


# ---------------------------------------------------------------------------
# TurnModel
# ---------------------------------------------------------------------------

class TestTurnModel:
    def test_defaults(self):
        tm = TurnModel()
        assert tm.max_bank_deg == 30.0
        assert tm.bank_by_phase.cruise_deg == 25.0
        assert tm.bank_by_phase.climb_deg == 20.0

    def test_custom_bank_angles(self):
        tm = TurnModel(
            bank_by_phase=PhaseBankAngles(
                climb_deg=15, cruise_deg=20, descent_deg=15, approach_deg=10,
            ),
            max_bank_deg=25.0,
        )
        assert tm.max_bank_deg == 25.0
        assert tm.bank_by_phase.cruise_deg == 20.0


# ---------------------------------------------------------------------------
# Aircraft instantiation
# ---------------------------------------------------------------------------

class TestAircraftInstantiation:
    def test_b200(self):
        ac = B200()
        assert ac.aircraft_type == "King Air 200"
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_er2(self):
        ac = NASA_ER2()
        assert ac.service_ceiling.m_as("feet") > 60000

    def test_giii(self):
        ac = NASA_GIII()
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_giv(self):
        ac = NASA_GIV()
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_gv(self):
        ac = NASA_GV()
        assert ac.aircraft_type == "Gulfstream V"
        assert ac.service_ceiling.m_as("feet") == pytest.approx(51000)

    def test_c130(self):
        ac = C130()
        assert ac.aircraft_type is not None

    def test_twin_otter(self):
        ac = TwinOtter()
        assert ac.cruise_speed_at(5000 * ureg.feet).magnitude > 0


# ---------------------------------------------------------------------------
# Aircraft performance methods
# ---------------------------------------------------------------------------

class TestAircraftPerformance:
    def test_cruise_speed_at(self):
        ac = B200()
        speed = ac.cruise_speed_at(ureg.Quantity(20000, "feet"))
        assert speed.magnitude > 0
        assert speed.check("[length] / [time]")

    def test_cruise_speed_varies_with_altitude(self):
        ac = B200()
        low = ac.cruise_speed_at(ureg.Quantity(5000, "feet"))
        high = ac.cruise_speed_at(ureg.Quantity(25000, "feet"))
        assert low.magnitude != high.magnitude

    def test_cruise_speed_at_sea_level(self):
        ac = B200()
        speed = ac.cruise_speed_at(ureg.Quantity(0, "feet"))
        assert speed.magnitude > 0

    def test_cruise_speed_at_ceiling(self):
        ac = B200()
        speed = ac.cruise_speed_at(ac.service_ceiling)
        assert speed.magnitude > 0

    def test_rate_of_climb_decreases(self):
        ac = B200()
        roc_low = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        roc_high = ac.rate_of_climb(ureg.Quantity(20000, "feet"))
        assert roc_low.magnitude > roc_high.magnitude

    def test_rate_of_climb_at_sea_level(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        assert roc.m_as("feet/minute") == pytest.approx(
            ac.climb_profile.sea_level_rate.m_as("feet/minute"), rel=1e-6
        )

    def test_rate_of_climb_at_ceiling(self):
        ac = B200()
        roc = ac.rate_of_climb(ac.service_ceiling)
        assert roc.m_as("feet/minute") == pytest.approx(
            ac.climb_profile.ceiling_rate.m_as("feet/minute"), rel=1e-6
        )

    def test_rate_of_climb_mid_altitude(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(15000, "feet"))
        assert roc.m_as("feet/minute") > ac.climb_profile.ceiling_rate.m_as("feet/minute")
        assert roc.m_as("feet/minute") < ac.climb_profile.sea_level_rate.m_as("feet/minute")

    def test_pitch_limits(self):
        ac = B200()
        pmin, pmax = ac.pitch_limits()
        assert pmin < 0  # descent
        assert pmax > 0  # climb
        assert abs(pmin) < 90
        assert abs(pmax) < 90

    def test_pitch_limits_with_speed(self):
        ac = B200()
        pmin1, pmax1 = ac.pitch_limits()
        pmin2, pmax2 = ac.pitch_limits(speed=ureg.Quantity(100, "knot"))
        # Slower speed -> steeper pitch possible
        assert abs(pmax2) > abs(pmax1)

    def test_descent_speed_at(self):
        ac = B200()
        cruise = ac.cruise_speed_at(ureg.Quantity(10000, "feet"))
        descent = ac.descent_speed_at(ureg.Quantity(10000, "feet"))
        # Descent speed should be <= cruise speed
        assert descent.magnitude <= cruise.magnitude

    def test_max_bank_angle(self):
        ac = B200()
        assert isinstance(ac.max_bank_angle, float)
        assert 0 < ac.max_bank_angle < 90

    def test_endurance_exists_and_reasonable(self):
        ac = B200()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs > 0
        assert endurance_hrs < 24

    def test_endurance_er2(self):
        ac = NASA_ER2()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs >= 6

    def test_climb_altitude_profile(self):
        ac = B200()
        times, altitudes = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet")
        )
        assert len(times) > 0
        assert len(altitudes) > 0
        assert altitudes[-1] == pytest.approx(10000, rel=0.1)


# ---------------------------------------------------------------------------
# Climb and descent integration
# ---------------------------------------------------------------------------

class TestClimbAndDescend:
    def test_climb_time_positive(self):
        ac = B200()
        time, dist = ac._climb(
            ureg.Quantity(0, "feet"), ureg.Quantity(20000, "feet")
        )
        assert time.magnitude > 0
        assert dist.magnitude > 0

    def test_climb_no_change(self):
        ac = B200()
        time, dist = ac._climb(
            ureg.Quantity(10000, "feet"), ureg.Quantity(10000, "feet")
        )
        assert time.magnitude == 0
        assert dist.magnitude == 0

    def test_descend_time_positive(self):
        ac = B200()
        time, dist = ac._descend(
            ureg.Quantity(20000, "feet"), ureg.Quantity(0, "feet")
        )
        assert time.magnitude > 0
        assert dist.magnitude > 0

    def test_descend_no_change(self):
        ac = B200()
        time, dist = ac._descend(
            ureg.Quantity(10000, "feet"), ureg.Quantity(10000, "feet")
        )
        assert time.magnitude == 0
        assert dist.magnitude == 0

    def test_climb_higher_takes_longer(self):
        ac = B200()
        t1, _ = ac._climb(ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet"))
        t2, _ = ac._climb(ureg.Quantity(0, "feet"), ureg.Quantity(20000, "feet"))
        assert t2.magnitude > t1.magnitude


# ---------------------------------------------------------------------------
# ER-2 performance (high-altitude, TAS speed profile)
# ---------------------------------------------------------------------------

class TestER2Performance:
    def test_cruise_speed_at_sea_level(self):
        ac = NASA_ER2()
        speed = ac.cruise_speed_at(ureg.Quantity(0, "feet"))
        assert speed.magnitude > 0

    def test_cruise_speed_at_ceiling(self):
        ac = NASA_ER2()
        speed = ac.cruise_speed_at(ac.service_ceiling)
        assert speed.magnitude > 0

    def test_cruise_speed_increases_with_altitude(self):
        ac = NASA_ER2()
        low = ac.cruise_speed_at(ureg.Quantity(10000, "feet"))
        high = ac.cruise_speed_at(ureg.Quantity(60000, "feet"))
        assert high.magnitude > low.magnitude

    def test_pitch_limits(self):
        ac = NASA_ER2()
        pmin, pmax = ac.pitch_limits()
        assert pmin < 0
        assert pmax > 0

    def test_max_bank_angle(self):
        ac = NASA_ER2()
        assert 0 < ac.max_bank_angle < 90


# ---------------------------------------------------------------------------
# GV (CAS/Mach schedule) specific tests
# ---------------------------------------------------------------------------

class TestGVPerformance:
    """Tests specific to the NASA GV with CAS/Mach speed schedules."""

    def test_cruise_speed_below_crossover(self):
        ac = NASA_GV()
        # Below crossover (30000 ft), uses CAS
        speed = ac.cruise_speed_at(ureg.Quantity(20000, "feet"))
        assert speed.m_as("knot") > 300  # TAS > CAS at altitude

    def test_cruise_speed_above_crossover(self):
        ac = NASA_GV()
        # Above crossover (30000 ft), uses Mach
        speed = ac.cruise_speed_at(ureg.Quantity(40000, "feet"))
        assert 440 < speed.m_as("knot") < 500

    def test_climb_profile_full_mode(self):
        ac = NASA_GV()
        assert ac.climb_profile._mode == "full"

    def test_descent_profile_full_mode(self):
        ac = NASA_GV()
        assert ac.descent_profile._mode == "full"

    def test_climb_roc_decreases_with_altitude(self):
        ac = NASA_GV()
        roc_low = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        roc_high = ac.rate_of_climb(ureg.Quantity(40000, "feet"))
        assert roc_low.m_as("feet/minute") > roc_high.m_as("feet/minute")

    def test_climb_time_positive(self):
        ac = NASA_GV()
        time, dist = ac._climb(
            ureg.Quantity(0, "feet"), ureg.Quantity(40000, "feet")
        )
        assert time.m_as("minute") > 0
        assert dist.m_as("nautical_mile") > 0

    def test_descend_time_positive(self):
        ac = NASA_GV()
        time, dist = ac._descend(
            ureg.Quantity(40000, "feet"), ureg.Quantity(0, "feet")
        )
        assert time.m_as("minute") > 0
        assert dist.m_as("nautical_mile") > 0

    def test_climb_altitude_profile_monotone(self):
        ac = NASA_GV()
        times, alts = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(40000, "feet"), n_points=32
        )
        assert len(times) == len(alts) == 32
        assert times[0] == 0.0
        assert np.all(np.diff(times) > 0)
        assert np.all(np.diff(alts) > 0)
        assert alts[-1] == pytest.approx(40000, rel=1e-6)

    def test_phase_bank_angles(self):
        ac = NASA_GV()
        assert ac.turn_model.bank_by_phase.climb_deg == 20
        assert ac.turn_model.bank_by_phase.cruise_deg == 25
        assert ac.turn_model.bank_by_phase.descent_deg == 20
        assert ac.turn_model.bank_by_phase.approach_deg == 15


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_confidence_defaults(self):
        pc = PerformanceConfidence()
        assert pc.climb == 0.5
        assert pc.cruise == 0.5

    def test_source_record(self):
        sr = SourceRecord(
            source_type="poh",
            reference="Beechcraft King Air B200 POH",
            confidence=0.8,
        )
        assert sr.source_type == "poh"
        assert sr.confidence == 0.8

    def test_gv_has_sources(self):
        ac = NASA_GV()
        assert len(ac.sources) >= 1
        assert ac.confidence.cruise == 0.50
