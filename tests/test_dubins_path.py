"""Tests for hyplan.dubins_path."""

import pytest
from hyplan.units import ureg
from hyplan.dubins_path import Waypoint, DubinsPath


class TestWaypoint:
    def test_creation(self):
        wp = Waypoint(latitude=34.0, longitude=-118.0, heading=90.0)
        assert wp.latitude == 34.0
        assert wp.longitude == -118.0
        assert wp.heading == 90.0

    def test_with_altitude(self):
        wp = Waypoint(
            latitude=34.0,
            longitude=-118.0,
            heading=45.0,
            altitude_msl=ureg.Quantity(5000, "meter"),
        )
        assert wp.altitude_msl is not None

    def test_geometry(self):
        wp = Waypoint(latitude=34.0, longitude=-118.0, heading=0.0)
        geom = wp.geometry
        assert geom.x == pytest.approx(-118.0)
        assert geom.y == pytest.approx(34.0)

    def test_to_dict(self):
        wp = Waypoint(latitude=34.0, longitude=-118.0, heading=0.0, name="WP1")
        d = wp.to_dict()
        assert d["latitude"] == 34.0
        assert d["name"] == "WP1"


class TestDubinsPath:
    def test_straight_path(self):
        start = Waypoint(latitude=34.0, longitude=-118.0, heading=0.0)
        end = Waypoint(latitude=34.1, longitude=-118.0, heading=0.0)
        path = DubinsPath(
            start=start,
            end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            step_size=500.0,
        )
        assert path.length.magnitude > 0

    def test_turning_path(self):
        start = Waypoint(latitude=34.0, longitude=-118.0, heading=0.0)
        end = Waypoint(latitude=34.0, longitude=-117.9, heading=180.0)
        path = DubinsPath(
            start=start,
            end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            step_size=500.0,
        )
        assert path.geometry is not None
        assert path.length.magnitude > 0

    def test_to_dict(self):
        start = Waypoint(latitude=34.0, longitude=-118.0, heading=0.0,
                         altitude_msl=ureg.Quantity(5000, "meter"))
        end = Waypoint(latitude=34.05, longitude=-118.0, heading=0.0,
                       altitude_msl=ureg.Quantity(5000, "meter"))
        path = DubinsPath(
            start=start,
            end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            step_size=500.0,
        )
        d = path.to_dict()
        assert "distance" in d
