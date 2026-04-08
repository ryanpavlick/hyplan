"""Tests for hyplan.units."""

import math

import pytest
from hyplan.units import (
    ureg,
    convert_distance,
    convert_speed,
    convert_angle,
    convert_time,
    altitude_to_flight_level,
)


class TestConvertDistance:
    def test_meters_to_kilometers(self):
        assert convert_distance(1000, "meters", "kilometers") == pytest.approx(1.0)

    def test_nautical_miles_to_meters(self):
        assert convert_distance(1, "nautical_miles", "meters") == pytest.approx(1852.0)

    def test_miles_to_feet(self):
        assert convert_distance(1, "miles", "feet") == pytest.approx(5280.0)

    def test_roundtrip(self):
        result = convert_distance(
            convert_distance(123.4, "meters", "feet"), "feet", "meters"
        )
        assert result == pytest.approx(123.4)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            convert_distance(1, "meters", "parsecs")


class TestConvertSpeed:
    def test_knots_to_mps(self):
        assert convert_speed(1, "knots", "mps") == pytest.approx(0.514444, rel=1e-3)

    def test_kph_to_mph(self):
        assert convert_speed(100, "kph", "mph") == pytest.approx(62.1371, rel=1e-3)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            convert_speed(1, "mps", "warp")


class TestConvertAngle:
    def test_degrees_to_radians(self):
        assert convert_angle(180.0, "degrees", "radians") == pytest.approx(math.pi)

    def test_radians_to_degrees(self):
        assert convert_angle(math.pi, "radians", "degrees") == pytest.approx(180.0)

    def test_arcminutes_to_degrees(self):
        assert convert_angle(60.0, "arcminutes", "degrees") == pytest.approx(1.0)

    def test_arcseconds_to_arcminutes(self):
        assert convert_angle(60.0, "arcseconds", "arcminutes") == pytest.approx(1.0)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            convert_angle(1.0, "degrees", "gradians")


class TestConvertTime:
    def test_seconds_to_minutes(self):
        assert convert_time(60.0, "seconds", "minutes") == pytest.approx(1.0)

    def test_hours_to_minutes(self):
        assert convert_time(1.5, "hours", "minutes") == pytest.approx(90.0)

    def test_days_to_hours(self):
        assert convert_time(2.0, "days", "hours") == pytest.approx(48.0)

    def test_roundtrip(self):
        result = convert_time(
            convert_time(123.4, "seconds", "hours"), "hours", "seconds"
        )
        assert result == pytest.approx(123.4)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            convert_time(1.0, "seconds", "fortnights")


class TestAltitudeToFlightLevel:
    def test_sea_level(self):
        assert altitude_to_flight_level(0) == "FL000"

    def test_10000_meters(self):
        fl = altitude_to_flight_level(10000)
        # 10000 m ≈ 32808 ft → FL328
        assert fl == "FL328"

    def test_with_pint_quantity(self):
        alt = ureg.Quantity(20000, "feet")
        assert altitude_to_flight_level(alt) == "FL200"

    def test_with_pressure_correction(self):
        # Higher pressure → lower flight level
        fl_standard = altitude_to_flight_level(10000)
        fl_high_pressure = altitude_to_flight_level(10000, pressure=1030.0)
        assert fl_standard != fl_high_pressure
