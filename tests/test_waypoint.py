"""Tests for hyplan.waypoint."""

import warnings

import pytest

from hyplan.units import ureg
from hyplan.waypoint import Waypoint, is_waypoint
from hyplan.exceptions import HyPlanValueError, HyPlanTypeError


class TestWaypointConstruction:
    def test_basic(self):
        wp = Waypoint(34.0, -118.0, 90.0)
        assert wp.latitude == 34.0
        assert wp.longitude == -118.0
        assert wp.heading == 90.0
        assert wp.altitude_msl is None

    def test_with_float_altitude(self):
        wp = Waypoint(34.0, -118.0, 0.0, altitude_msl=5000.0)
        assert wp.altitude_msl.magnitude == 5000.0
        assert wp.altitude_msl.units == ureg.meter

    def test_with_quantity_altitude(self):
        alt = ureg.Quantity(20000, "feet")
        wp = Waypoint(34.0, -118.0, 0.0, altitude_msl=alt)
        assert wp.altitude_msl.to(ureg.foot).magnitude == pytest.approx(20000, rel=1e-6)

    def test_heading_wrapping(self):
        wp = Waypoint(0.0, 0.0, 361.0)
        assert wp.heading == pytest.approx(1.0)

    def test_negative_heading_wrapping(self):
        wp = Waypoint(0.0, 0.0, -10.0)
        assert wp.heading == pytest.approx(350.0)

    def test_name_default(self):
        wp = Waypoint(34.0, -118.0, 0.0)
        assert "34.00" in wp.name
        assert "-118.00" in wp.name

    def test_name_custom(self):
        wp = Waypoint(34.0, -118.0, 0.0, name="WP1")
        assert wp.name == "WP1"

    def test_optional_fields(self):
        wp = Waypoint(
            34.0, -118.0, 0.0,
            speed=ureg.Quantity(75, "m/s"),
            delay=ureg.Quantity(30, "s"),
            headwind=ureg.Quantity(10, "m/s"),
            segment_type="pattern",
        )
        assert wp.speed.magnitude == pytest.approx(75.0)
        assert wp.delay.magnitude == pytest.approx(30.0)
        assert wp.headwind.magnitude == pytest.approx(10.0)
        assert wp.segment_type == "pattern"

    def test_speed_bare_float(self):
        wp = Waypoint(34.0, -118.0, 0.0, speed=100.0)
        assert wp.speed.magnitude == pytest.approx(100.0)
        assert wp.speed.units == ureg.meter / ureg.second


class TestWaypointValidation:
    def test_negative_altitude_raises(self):
        with pytest.raises(HyPlanValueError, match="non-negative"):
            Waypoint(34.0, -118.0, 0.0, altitude_msl=-100.0)

    def test_negative_quantity_altitude_raises(self):
        with pytest.raises(HyPlanValueError, match="non-negative"):
            Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(-500, "feet"))

    def test_extreme_altitude_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Waypoint(34.0, -118.0, 0.0, altitude_msl=25000.0)
            assert len(w) == 1
            assert "22,000" in str(w[0].message)

    def test_latitude_too_high(self):
        with pytest.raises(HyPlanValueError, match="Latitude"):
            Waypoint(91.0, 0.0, 0.0)

    def test_latitude_too_low(self):
        with pytest.raises(HyPlanValueError, match="Latitude"):
            Waypoint(-91.0, 0.0, 0.0)

    def test_longitude_too_high(self):
        with pytest.raises(HyPlanValueError, match="Longitude"):
            Waypoint(0.0, 181.0, 0.0)

    def test_longitude_too_low(self):
        with pytest.raises(HyPlanValueError, match="Longitude"):
            Waypoint(0.0, -181.0, 0.0)

    def test_heading_not_numeric_raises(self):
        with pytest.raises(HyPlanTypeError, match="Heading"):
            Waypoint(0.0, 0.0, "north")

    def test_bad_altitude_type_raises(self):
        with pytest.raises(HyPlanTypeError, match="altitude_msl"):
            Waypoint(0.0, 0.0, 0.0, altitude_msl="high")

    def test_boundary_lat_lon(self):
        # Exact boundaries should be valid
        wp = Waypoint(90.0, 180.0, 0.0)
        assert wp.latitude == 90.0
        wp2 = Waypoint(-90.0, -180.0, 0.0)
        assert wp2.longitude == -180.0


class TestWaypointOffsetNorthEast:
    def test_offset_north(self):
        wp = Waypoint(34.0, -118.0, 90.0, altitude_msl=5000.0, name="A")
        moved = wp.offset_north_east(1000.0, 0.0)
        # Should be ~0.009° north
        assert moved.latitude > 34.0
        assert moved.longitude == pytest.approx(-118.0, abs=0.001)
        # Preserves metadata
        assert moved.heading == 90.0
        assert moved.altitude_msl.magnitude == 5000.0
        assert moved.name == "A"

    def test_offset_east(self):
        wp = Waypoint(34.0, -118.0, 0.0)
        moved = wp.offset_north_east(0.0, 1000.0)
        assert moved.latitude == pytest.approx(34.0, abs=0.001)
        assert moved.longitude > -118.0

    def test_offset_with_quantity(self):
        wp = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))
        moved = wp.offset_north_east(
            ureg.Quantity(1, "km"),
            ureg.Quantity(-500, "meter"),
        )
        assert moved.latitude > 34.0
        assert moved.longitude < -118.0

    def test_preserves_segment_type(self):
        wp = Waypoint(0.0, 0.0, 45.0, segment_type="pattern",
                      speed=ureg.Quantity(75, "m/s"))
        moved = wp.offset_north_east(100.0, 100.0)
        assert moved.segment_type == "pattern"
        assert moved.speed.magnitude == pytest.approx(75.0)

    def test_zero_offset(self):
        wp = Waypoint(34.0, -118.0, 0.0)
        moved = wp.offset_north_east(0.0, 0.0)
        assert moved.latitude == pytest.approx(34.0, abs=1e-6)
        assert moved.longitude == pytest.approx(-118.0, abs=1e-6)


class TestWaypointToDict:
    def test_round_trip_fields(self):
        wp = Waypoint(34.0, -118.0, 90.0, altitude_msl=5000.0, name="test")
        d = wp.to_dict()
        assert d["latitude"] == 34.0
        assert d["longitude"] == -118.0
        assert d["heading"] == 90.0
        assert d["altitude_msl"].magnitude == 5000.0
        assert d["name"] == "test"


class TestIsWaypoint:
    def test_waypoint_instance(self):
        wp = Waypoint(0.0, 0.0, 0.0)
        assert is_waypoint(wp)

    def test_non_waypoint(self):
        assert not is_waypoint("not a waypoint")
        assert not is_waypoint(42)
        assert not is_waypoint(None)

    def test_duck_type(self):
        """An object with the right attributes should pass."""
        from types import SimpleNamespace
        from shapely.geometry import Point
        fake = SimpleNamespace(
            latitude=0, longitude=0, heading=0, altitude_msl=None,
            geometry=Point(0, 0),
        )
        assert is_waypoint(fake)
