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
        assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(20000, rel=1e-6)

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
            segment_type="pattern",
        )
        assert wp.speed.magnitude == pytest.approx(75.0)
        assert wp.delay.magnitude == pytest.approx(30.0)
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

    def test_includes_all_optional_fields(self):
        """to_dict must surface speed, delay, and segment_type, not just the first 5 fields."""
        wp = Waypoint(
            34.0, -118.0, 90.0,
            altitude_msl=ureg.Quantity(5000, "foot"),
            name="rendezvous",
            speed=ureg.Quantity(180, "knot"),
            delay=ureg.Quantity(5, "minute"),
            segment_type="hold",
        )
        d = wp.to_dict()
        assert set(d.keys()) == {
            "latitude", "longitude", "heading",
            "altitude_msl", "name",
            "speed", "delay", "segment_type",
        }
        assert d["speed"].m_as(ureg.knot) == pytest.approx(180.0)
        assert d["delay"].m_as(ureg.minute) == pytest.approx(5.0)
        assert d["segment_type"] == "hold"

    def test_optional_fields_default_to_none(self):
        wp = Waypoint(34.0, -118.0, 90.0)
        d = wp.to_dict()
        assert d["altitude_msl"] is None
        assert d["speed"] is None
        assert d["delay"] is None
        assert d["segment_type"] is None
        assert d["name"] is not None  # auto-generated coord-based name


class TestWaypointFromDict:
    def test_round_trip_identity(self):
        """Waypoint -> to_dict -> from_dict produces an equivalent Waypoint."""
        original = Waypoint(
            34.0, -118.0, 90.0,
            altitude_msl=ureg.Quantity(5000, "foot"),
            name="origin",
            speed=ureg.Quantity(180, "knot"),
            delay=ureg.Quantity(5, "minute"),
            segment_type="hold",
        )
        restored = Waypoint.from_dict(original.to_dict())
        assert restored.latitude == original.latitude
        assert restored.longitude == original.longitude
        assert restored.heading == original.heading
        assert restored.altitude_msl.m_as(ureg.foot) == pytest.approx(
            original.altitude_msl.m_as(ureg.foot)
        )
        assert restored.name == original.name
        assert restored.speed.m_as(ureg.knot) == pytest.approx(
            original.speed.m_as(ureg.knot)
        )
        assert restored.delay.m_as(ureg.minute) == pytest.approx(
            original.delay.m_as(ureg.minute)
        )
        assert restored.segment_type == original.segment_type

    def test_minimal_dict(self):
        """A dict with only the three required fields builds a default Waypoint."""
        wp = Waypoint.from_dict({"latitude": 0.0, "longitude": 0.0, "heading": 0.0})
        assert wp.latitude == 0.0
        assert wp.altitude_msl is None
        assert wp.speed is None
        assert wp.delay is None
        assert wp.segment_type is None

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            Waypoint.from_dict({"longitude": 0.0, "heading": 0.0})

    def test_round_trip_preserves_loiter_delay(self):
        """A bare Waypoint with loiter delay survives a to_dict/from_dict cycle."""
        wp = Waypoint(
            34.4, -119.8, 0.0,
            altitude_msl=ureg.Quantity(8_000, "foot"),
            delay=ureg.Quantity(45, "second"),
            name="LOITER_45S",
        )
        restored = Waypoint.from_dict(wp.to_dict())
        assert restored.delay.m_as(ureg.second) == pytest.approx(45.0)
        assert restored.name == "LOITER_45S"


class TestRelativeTo:
    """Tests for Waypoint.relative_to — geodesic offset from anchor."""

    def test_east_offset_at_equator(self):
        # 60 nmi east at the equator → ≈ 1° of longitude.
        anchor = Waypoint(0.0, 0.0, 0.0, name="EQ0")
        wp = Waypoint.relative_to(anchor, bearing=90.0, distance=60.0)
        assert wp.latitude == pytest.approx(0.0, abs=1e-3)
        assert wp.longitude == pytest.approx(1.0, abs=0.01)

    def test_north_offset_one_degree(self):
        # 60 nmi north at the equator → ≈ 1° of latitude.
        wp = Waypoint.relative_to((0.0, 0.0), bearing=0.0, distance=60.0)
        assert wp.latitude == pytest.approx(1.0, abs=0.01)
        assert wp.longitude == pytest.approx(0.0, abs=1e-3)

    def test_anchor_can_be_tuple(self):
        wp = Waypoint.relative_to((34.0, -118.0), bearing=270.0, distance=100.0)
        # 100 nmi west → longitude shifts westward, latitude ~unchanged
        assert wp.longitude < -118.0
        assert wp.latitude == pytest.approx(34.0, abs=0.5)

    def test_anchor_can_be_waypoint(self):
        edw = Waypoint(34.92, -117.87, heading=0.0, name="EDW")
        wp = Waypoint.relative_to(edw, bearing=90.0, distance=200.0)
        # 200 nmi true east; verify against vreckon (which returns
        # longitude in [0, 360); the classmethod wraps to [-180, 180)).
        import pymap3d.vincenty as vinc
        from hyplan.geometry import wrap_to_180
        ref_lat, ref_lon = vinc.vreckon(34.92, -117.87, 200 * 1852.0, 90.0)
        assert wp.latitude == pytest.approx(float(ref_lat), abs=1e-5)
        assert wp.longitude == pytest.approx(float(wrap_to_180(float(ref_lon))), abs=1e-5)

    def test_distance_quantity_kilometers(self):
        wp = Waypoint.relative_to(
            (0.0, 0.0),
            bearing=90.0,
            distance=ureg.Quantity(111.32, "kilometer"),
        )
        # ~1° east at the equator
        assert wp.longitude == pytest.approx(1.0, abs=0.005)

    def test_distance_quantity_nm_matches_float_nm(self):
        # Float-as-nautical-miles must agree with Quantity(...) in nm.
        wp_float = Waypoint.relative_to((40.0, -100.0), bearing=45.0, distance=150.0)
        wp_q = Waypoint.relative_to(
            (40.0, -100.0),
            bearing=45.0,
            distance=ureg.Quantity(150, "nautical_mile"),
        )
        assert wp_float.latitude == pytest.approx(wp_q.latitude, abs=1e-6)
        assert wp_float.longitude == pytest.approx(wp_q.longitude, abs=1e-6)

    def test_heading_defaults_to_bearing(self):
        wp = Waypoint.relative_to((0.0, 0.0), bearing=137.0, distance=10.0)
        assert wp.heading == 137.0

    def test_heading_can_override(self):
        wp = Waypoint.relative_to(
            (0.0, 0.0), bearing=137.0, distance=10.0, heading=270.0,
        )
        assert wp.heading == 270.0

    def test_optional_fields_propagate(self):
        wp = Waypoint.relative_to(
            (34.0, -118.0),
            bearing=0.0,
            distance=30.0,
            altitude_msl=ureg.Quantity(35_000, "foot"),
            name="WP_N30",
            speed=ureg.Quantity(420, "knot"),
            delay=ureg.Quantity(120, "second"),
            segment_type="pattern",
        )
        assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(35_000)
        assert wp.name == "WP_N30"
        assert wp.speed.m_as(ureg.knot) == pytest.approx(420)
        assert wp.delay.m_as(ureg.second) == pytest.approx(120)
        assert wp.segment_type == "pattern"

    def test_bearing_wraps_to_0_360(self):
        # bearing=450 should be equivalent to bearing=90
        wp_450 = Waypoint.relative_to((0.0, 0.0), bearing=450.0, distance=60.0)
        wp_90  = Waypoint.relative_to((0.0, 0.0), bearing=90.0,  distance=60.0)
        assert wp_450.latitude == pytest.approx(wp_90.latitude, abs=1e-6)
        assert wp_450.longitude == pytest.approx(wp_90.longitude, abs=1e-6)

    def test_round_trip_via_reverse_bearing(self):
        # offset 100 nmi at heading 45°, then offset back at 225° → original.
        start = Waypoint(40.0, -100.0, heading=0.0)
        away = Waypoint.relative_to(start, bearing=45.0, distance=100.0)
        # Reverse bearing on a great circle isn't simply +180°; use vdist
        # to get the back-azimuth.
        import pymap3d.vincenty as vinc
        _dist, back_az = vinc.vdist(away.latitude, away.longitude,
                                    start.latitude, start.longitude)
        back = Waypoint.relative_to(away, bearing=float(back_az), distance=100.0)
        assert back.latitude == pytest.approx(start.latitude, abs=1e-4)
        assert back.longitude == pytest.approx(start.longitude, abs=1e-4)


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
