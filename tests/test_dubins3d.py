"""Tests for hyplan.dubins3d (3D Dubins paths with pitch constraints)."""

import math
import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.dubins3d import DubinsPath3D, _Dubins2D, _TrochoidDubins2D, _VerticalDubins
from hyplan.exceptions import HyPlanRuntimeError


class TestDubins2DInternal:
    """Verify the internal 2D solver matches expected behavior."""

    def test_straight_path(self):
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([10.0, 0.0, 0.0])
        d = _Dubins2D(qi, qf, 1.0)
        assert d.maneuver.length == pytest.approx(10.0, rel=1e-3)

    def test_path_type_is_string(self):
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([10.0, 5.0, math.pi / 2])
        d = _Dubins2D(qi, qf, 2.0)
        assert len(d.maneuver.case) == 3

    def test_sampling(self):
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([10.0, 0.0, 0.0])
        d = _Dubins2D(qi, qf, 1.0)
        start = d.get_coordinates_at(0.0)
        assert start[0] == pytest.approx(0.0, abs=1e-6)
        assert start[1] == pytest.approx(0.0, abs=1e-6)
        end = d.get_coordinates_at(d.maneuver.length)
        assert end[0] == pytest.approx(10.0, abs=1e-2)
        assert end[1] == pytest.approx(0.0, abs=1e-2)

    def test_disable_ccc(self):
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1.0, 0.0, math.pi])
        d = _Dubins2D(qi, qf, 1.0, disable_ccc=True)
        assert d.maneuver.case not in ("RLR", "LRL")


class TestVerticalDubins:
    """Verify the pitch-constrained vertical solver."""

    def test_level_flight(self):
        # Same altitude, zero pitch -> essentially straight
        qi = np.array([0.0, 1000.0, 0.0])
        qf = np.array([5000.0, 1000.0, 0.0])
        d = _VerticalDubins(qi, qf, 500.0, (-0.5, 0.5))
        assert d.maneuver.case != "XXX"
        assert d.maneuver.length < math.inf

    def test_climb(self):
        qi = np.array([0.0, 1000.0, 0.0])
        qf = np.array([10000.0, 2000.0, 0.0])
        d = _VerticalDubins(qi, qf, 2000.0, (-0.3, 0.3))
        assert d.maneuver.case != "XXX"
        assert d.maneuver.length >= 10000.0

    def test_steep_climb_uses_clamped_path(self):
        # Large altitude change with tight pitch limits — solver clamps to
        # pitch boundary producing a long straight segment
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([10.0, 10000.0, 0.0])
        d = _VerticalDubins(qi, qf, 5000.0, (-0.05, 0.05))
        assert d.maneuver.case != "XXX"
        # Path should be much longer than horizontal distance due to shallow climb
        assert d.maneuver.length > 100000


    def test_infeasible_vertical_raises(self):
        """Infeasible pitch-constrained path should raise HyPlanRuntimeError."""
        # Start/end pitches outside limits + large radius + short distance
        # makes all CSC paths infeasible
        qi = np.array([0.0, 0.0, 0.25])
        qf = np.array([5.0, -1000.0, -0.25])
        with pytest.raises(HyPlanRuntimeError, match="feasible"):
            _VerticalDubins(qi, qf, 10000.0, (-0.1, 0.1))


class TestDubinsPath3D:
    """Integration tests for the full 3D solver."""

    def test_level_path(self):
        """Same altitude — should behave like 2D Dubins."""
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=5000.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
        )
        assert path.length.magnitude > 0
        assert path.geometry is not None
        assert path.geometry_3d is not None

    def test_climbing_path(self):
        """Path with altitude change."""
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=3000.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            pitch_min=-15.0, pitch_max=15.0,
        )
        assert path.length.magnitude > 0
        # 3D path should be at least as long as horizontal distance
        pts = path.points
        assert pts[0, 2] == pytest.approx(3000.0, abs=1)
        assert pts[-1, 2] == pytest.approx(5000.0, abs=100)

    def test_descending_path(self):
        """Descending path."""
        start = Waypoint(34.0, -118.0, 90.0, altitude_msl=8000.0)
        end = Waypoint(34.0, -117.8, 90.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            pitch_min=-15.0, pitch_max=15.0,
        )
        assert path.length.magnitude > 0
        pts = path.points
        assert pts[0, 2] == pytest.approx(8000.0, abs=1)
        assert pts[-1, 2] == pytest.approx(5000.0, abs=100)

    def test_turning_with_altitude_change(self):
        """Path with both heading change and altitude change."""
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=3000.0)
        end = Waypoint(34.05, -117.95, 180.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            pitch_min=-15.0, pitch_max=15.0,
        )
        assert path.length.magnitude > 0
        assert len(path.points) > 2

    def test_points_shape(self):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=5000.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
            n_samples=50,
        )
        assert path.points.shape == (50, 5)

    def test_to_dict(self):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=5000.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
        )
        d = path.to_dict()
        assert "distance" in d
        assert "geometry_3d" in d

    def test_missing_altitude_raises(self):
        start = Waypoint(34.0, -118.0, 0.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        with pytest.raises(Exception):
            DubinsPath3D(
                start=start, end=end,
                speed=ureg.Quantity(150, "knot"),
                bank_angle=25.0,
            )

    def test_min_turn_radius(self):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=5000.0)
        end = Waypoint(34.1, -118.0, 0.0, altitude_msl=5000.0)
        path = DubinsPath3D(
            start=start, end=end,
            speed=ureg.Quantity(150, "knot"),
            bank_angle=25.0,
        )
        assert path.min_turn_radius.magnitude > 0

    def test_infeasible_pitch_raises(self):
        """Zero pitch limits with altitude change should raise."""
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=100.0)
        end = Waypoint(34.001, -118.0, 0.0, altitude_msl=15000.0)
        with pytest.raises(Exception, match="feasible"):
            DubinsPath3D(
                start=start, end=end,
                speed=ureg.Quantity(150, "knot"),
                bank_angle=25.0,
                pitch_min=0.0, pitch_max=0.0,
            )


class TestDubinsSegmentValid:
    def test_valid_segment(self):
        from hyplan.dubins3d import _DubinsSegment
        seg = _DubinsSegment(1.0, 2.0, 1.0, 4.0, "LSL")
        assert seg.valid

    def test_invalid_xxx(self):
        from hyplan.dubins3d import _DubinsSegment
        seg = _DubinsSegment(math.inf, math.inf, math.inf, math.inf, "XXX")
        assert not seg.valid

    def test_invalid_inf_length(self):
        from hyplan.dubins3d import _DubinsSegment
        seg = _DubinsSegment(1.0, 2.0, 1.0, math.inf, "LSL")
        assert not seg.valid


# ---------------------------------------------------------------------------
# Wind-aware trochoid Dubins tests
# ---------------------------------------------------------------------------

class TestTrochoidDubins2D:
    """Verify the wind-aware 2D solver."""

    def test_zero_wind_matches_standard(self):
        """With zero wind, trochoid solver should match standard solver."""
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1000.0, 0.0, 0.0])
        rho = 200.0
        airspeed = 100.0

        d_std = _Dubins2D(qi, qf, rho)
        d_troc = _TrochoidDubins2D(qi, qf, rho, airspeed, 0.0, 0.0)

        assert d_troc.maneuver.length == pytest.approx(d_std.maneuver.length, rel=1e-3)

    def test_headwind_increases_time(self):
        """Headwind should increase path time."""
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1000.0, 0.0, 0.0])
        rho = 200.0
        airspeed = 100.0

        d_still = _TrochoidDubins2D(qi, qf, rho, airspeed, 0.0, 0.0)
        d_hw = _TrochoidDubins2D(qi, qf, rho, airspeed, -20.0, 0.0)

        assert d_hw.total_time > d_still.total_time

    def test_tailwind_decreases_time(self):
        """Tailwind should decrease path time."""
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1000.0, 0.0, 0.0])
        rho = 200.0
        airspeed = 100.0

        d_still = _TrochoidDubins2D(qi, qf, rho, airspeed, 0.0, 0.0)
        d_tw = _TrochoidDubins2D(qi, qf, rho, airspeed, 20.0, 0.0)

        assert d_tw.total_time < d_still.total_time

    def test_start_position_correct(self):
        """Ground track should start at qi."""
        qi = np.array([100.0, 200.0, 0.5])
        qf = np.array([1100.0, 200.0, 0.5])
        d = _TrochoidDubins2D(qi, qf, 200.0, 100.0, 15.0, -10.0)

        p0 = d.get_coordinates_at(0.0)
        assert p0[0] == pytest.approx(qi[0], abs=1.0)
        assert p0[1] == pytest.approx(qi[1], abs=1.0)


class TestDubinsPath3DWind:
    """Verify DubinsPath3D with wind parameter."""

    @pytest.fixture
    def waypoints(self):
        start = Waypoint(latitude=34.0, longitude=-118.5,
                         altitude_msl=6000 * ureg.meter, heading=90.0)
        end = Waypoint(latitude=34.0, longitude=-118.3,
                       altitude_msl=6000 * ureg.meter, heading=90.0)
        return start, end

    def test_still_air_unchanged(self, waypoints):
        """wind=None should give same result as before."""
        start, end = waypoints
        path_none = DubinsPath3D(start, end, 130.0, 25.0)
        path_zero = DubinsPath3D(start, end, 130.0, 25.0, wind=(0.0, 0.0))

        assert path_none.length.m_as(ureg.meter) == pytest.approx(
            path_zero.length.m_as(ureg.meter), rel=0.05
        )

    def test_headwind_longer(self, waypoints):
        """Headwind path should be longer than still-air."""
        start, end = waypoints
        path_still = DubinsPath3D(start, end, 130.0, 25.0)
        path_hw = DubinsPath3D(start, end, 130.0, 25.0, wind=(-20.0, 0.0))

        assert path_hw.length > path_still.length

    def test_tailwind_shorter(self, waypoints):
        """Tailwind path should be shorter than still-air."""
        start, end = waypoints
        path_still = DubinsPath3D(start, end, 130.0, 25.0)
        path_tw = DubinsPath3D(start, end, 130.0, 25.0, wind=(20.0, 0.0))

        assert path_tw.length < path_still.length

    def test_endpoints_match(self, waypoints):
        """Start and end lat/lon should match the waypoints."""
        start, end = waypoints
        path = DubinsPath3D(start, end, 130.0, 25.0, wind=(15.0, -10.0))

        assert path.points[0, 0] == pytest.approx(34.0, abs=0.01)
        assert path.points[0, 1] == pytest.approx(-118.5, abs=0.01)
        assert path.points[-1, 0] == pytest.approx(34.0, abs=0.01)
        assert path.points[-1, 1] == pytest.approx(-118.3, abs=0.01)

    def test_geometry_is_linestring(self, waypoints):
        """Path geometry should be a valid LineString."""
        start, end = waypoints
        path = DubinsPath3D(start, end, 130.0, 25.0, wind=(10.0, 10.0))

        assert path.geometry.is_valid
        assert path.geometry_3d.is_valid
        assert len(path.points) > 2

    def test_altitude_change_with_wind(self):
        """Wind should not affect altitude profile (wind is horizontal)."""
        start = Waypoint(latitude=34.0, longitude=-118.5,
                         altitude_msl=5000 * ureg.meter, heading=90.0)
        end = Waypoint(latitude=34.0, longitude=-118.3,
                       altitude_msl=6000 * ureg.meter, heading=90.0)

        path_still = DubinsPath3D(start, end, 130.0, 25.0)
        path_wind = DubinsPath3D(start, end, 130.0, 25.0, wind=(15.0, 0.0))

        # Both should reach the same final altitude
        assert path_still.points[-1, 2] == pytest.approx(6000.0, rel=0.01)
        assert path_wind.points[-1, 2] == pytest.approx(6000.0, rel=0.01)
