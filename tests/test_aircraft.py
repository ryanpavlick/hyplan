"""Tests for hyplan.aircraft."""

import pytest
from hyplan.units import ureg
from hyplan.aircraft import (
    NASA_ER2,
    NASA_GIII,
    DynamicAviation_B200 as B200,
    C130,
    TwinOtter,
)


class TestAircraftInstantiation:
    def test_b200(self):
        ac = B200()
        assert ac.aircraft_type == "King Air 200"
        assert ac.cruise_speed.magnitude > 0

    def test_er2(self):
        ac = NASA_ER2()
        assert ac.service_ceiling.m_as("feet") > 60000

    def test_giii(self):
        ac = NASA_GIII()
        assert ac.cruise_speed.magnitude > 0

    def test_c130(self):
        ac = C130()
        assert ac.aircraft_type is not None

    def test_twin_otter(self):
        ac = TwinOtter()
        assert ac.cruise_speed.magnitude > 0


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
        # Speed should differ at different altitudes
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

    def test_rate_of_climb_at_ceiling(self):
        ac = B200()
        roc = ac.rate_of_climb(ac.service_ceiling)
        assert roc.magnitude == pytest.approx(
            ac.roc_at_service_ceiling.magnitude, rel=1e-6
        )

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

    def test_climb_altitude_profile(self):
        ac = B200()
        times, altitudes = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet")
        )
        assert len(times) > 0
        assert len(altitudes) > 0
        # Returns numpy arrays (values in feet)
        assert altitudes[-1] == pytest.approx(10000, rel=0.1)

    def test_max_bank_angle(self):
        ac = B200()
        assert isinstance(ac.max_bank_angle, float)
        assert 0 < ac.max_bank_angle < 90

    def test_endurance_exists_and_reasonable(self):
        ac = B200()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs > 0
        assert endurance_hrs < 24  # no aircraft flies more than a day

    def test_endurance_er2(self):
        ac = NASA_ER2()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs >= 6  # ER-2 has ~8 hr endurance

    def test_rate_of_climb_at_sea_level(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        # At sea level, ROC should equal best_rate_of_climb
        assert roc.magnitude == pytest.approx(
            ac.best_rate_of_climb.magnitude, rel=1e-6
        )

    def test_rate_of_climb_mid_altitude(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(15000, "feet"))
        # Should be between ceiling ROC and sea-level ROC
        assert roc.magnitude > ac.roc_at_service_ceiling.magnitude
        assert roc.magnitude < ac.best_rate_of_climb.magnitude


class TestClimbAndDescend:
    """Tests for the _climb and _descend internal methods."""

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


class TestER2Performance:
    """Exercise performance methods on the ER-2 (speed-profile aircraft)."""

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
        # ER-2 TAS increases with altitude per its speed profile
        assert high.magnitude > low.magnitude

    def test_pitch_limits(self):
        ac = NASA_ER2()
        pmin, pmax = ac.pitch_limits()
        assert pmin < 0
        assert pmax > 0

    def test_max_bank_angle(self):
        ac = NASA_ER2()
        assert 0 < ac.max_bank_angle < 90
