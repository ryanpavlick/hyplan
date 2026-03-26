"""Tests for hyplan.dubins3d (3D Dubins paths with pitch constraints)."""

import math
import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.dubins3d import DubinsPath3D, _Dubins2D, _VerticalDubins
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
