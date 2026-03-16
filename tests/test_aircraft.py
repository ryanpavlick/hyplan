"""Tests for hyplan.aircraft."""

import pytest
from hyplan.units import ureg
from hyplan.aircraft import (
    Aircraft,
    NASA_ER2,
    NASA_GIII,
    DynamicAviation_B200,
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
        assert ac.service_ceiling.to("feet").magnitude > 60000

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

    def test_climb_altitude_profile(self):
        ac = B200()
        times, altitudes = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet")
        )
        assert len(times) > 0
        assert len(altitudes) > 0
        # Returns numpy arrays (values in feet)
        assert altitudes[-1] == pytest.approx(10000, rel=0.1)
