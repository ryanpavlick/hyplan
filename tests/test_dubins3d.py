"""Tests for hyplan.dubins3d (3D Dubins paths with pitch constraints)."""

import math

import numpy as np
import pytest

from hyplan.dubins3d import DubinsPath2D, _Dubins2D, _TrochoidDubins2D
from hyplan.exceptions import HyPlanRuntimeError, HyPlanValueError
from hyplan.waypoint import Waypoint


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

    def test_coincident_pose_gives_full_loop(self):
        """Same position and heading still yields the 2πρ loop."""
        qi = np.array([100.0, 200.0, 0.0])
        rho = 500.0
        d = _Dubins2D(qi, qi.copy(), rho)
        assert d.maneuver.length == pytest.approx(2 * math.pi * rho, rel=1e-9)

    def test_jet_scale_near_pose_keeps_goal_heading(self):
        """At a jet's 36 km rhomin, a sub-meter offset with a 0.3 rad
        heading change must be solved properly — the old meters-scaled
        angle gate (~21° at this rhomin) snapped it to a same-heading
        2πρ loop that lands with the wrong final heading."""
        rho = 36_000.0
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([0.2, 0.0, 0.3])
        d = _Dubins2D(qi, qf, rho)
        assert d.maneuver.case != "RRR"
        end = d.get_coordinates_at(d.maneuver.length)
        assert end[2] == pytest.approx(0.3, abs=1e-6)

    def test_uas_scale_coincident_pose_still_loops(self):
        """A small-UAS rhomin must not shrink the degenerate gate below
        sensible numerical noise."""
        rho = 20.0
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1e-4, 0.0, 1e-7])
        d = _Dubins2D(qi, qf, rho)
        assert d.maneuver.case == "RRR"
        assert d.maneuver.length == pytest.approx(2 * math.pi * rho, rel=1e-9)


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

    def test_ccc_fallback_for_close_parallel_lines(self):
        """When start/end positions are closer than ~4r with opposite
        headings, the geometric optimum is CCC (RLR/LRL).  Verify that
        the trochoid solver picks a CCC mode (proper trochoid root or
        the air-frame-with-drift fallback) instead of the much longer
        BSB-only result.

        Geometry models an ER-2-style racetrack between adjacent
        parallel flight lines: 5 nmi lateral spacing, opposite headings,
        7.2 nmi turn radius, 425 kt TAS, 26 kt crosswind.  In this
        regime the BSB-only result is ~2x the CCC length.
        """
        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])           # north-bound
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])    # south-bound
        rhomin = 7.2 * nmi
        airspeed = 218.6   # 425 kt
        wind_u = 13.4      # 26 kt eastward

        d_ccc = _TrochoidDubins2D(qi, qf, rhomin, airspeed, wind_u, 0.0)
        d_bsb = _TrochoidDubins2D(
            qi, qf, rhomin, airspeed, wind_u, 0.0, disable_ccc=True,
        )

        assert d_ccc._mode in ("ccc_trochoid", "ccc_air_drift")
        assert d_ccc.maneuver.case in ("RLR", "LRL")
        assert d_bsb._mode == "bsb"
        # CCC must be substantially shorter than BSB-only.
        assert d_ccc.total_time < 0.7 * d_bsb.total_time, (
            f"CCC {d_ccc.total_time:.1f}s should be << "
            f"BSB-only {d_bsb.total_time:.1f}s"
        )

    def test_ccc_lands_at_target(self):
        """The CCC trochoid ground track must land at the requested
        goal position to within sampling precision."""
        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])
        d = _TrochoidDubins2D(qi, qf, 7.2 * nmi, 218.6, 13.4, 0.0)
        assert d._mode in ("ccc_trochoid", "ccc_air_drift")
        end = d.get_coordinates_at(d.total_time)
        # Position landing within 2 m (numerical noise from iterative solve).
        assert end[0] == pytest.approx(qf[0], abs=2.0)
        assert end[1] == pytest.approx(qf[1], abs=2.0)

    def test_ccc_disabled_falls_back_to_bsb(self):
        """`disable_ccc=True` must reproduce the legacy BSB-only behavior."""
        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])
        d = _TrochoidDubins2D(
            qi, qf, 7.2 * nmi, 218.6, 13.4, 0.0, disable_ccc=True,
        )
        assert d._mode == "bsb"
        assert d.maneuver.case == "TRO"

    def test_ccc_trochoid_solver_converges_in_low_wind(self):
        """The proper trochoidal CCC root-find should produce a valid
        result in the low-wind regime, agreeing with the air-frame
        Dubins-with-drift fallback to within ~0.1% on time (both methods
        are correct in the limit vw / Va → 0; they differ only in
        floating-point accumulation)."""
        from hyplan._trochoid_solver import solve_ccc_trochoid
        from hyplan.dubins3d import _try_ccc_with_drift

        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])
        rhomin = 7.2 * nmi
        airspeed = 218.6
        wind_u = 13.4

        tro_sol = solve_ccc_trochoid(qi, qf, rhomin, airspeed, wind_u, 0.0)
        assert tro_sol is not None
        assert tro_sol["family"] in ("LRL", "RLR")

        _, t_air = _try_ccc_with_drift(qi, qf, rhomin, airspeed, wind_u, 0.0)
        # Both methods solve the same physical problem; they should
        # agree to within numerical noise.
        assert abs(tro_sol["total_time"] - t_air) / t_air < 1e-3


class TestDubinsPath2DSpeedValidation:
    """DubinsPath2D must reject non-positive speeds up front."""

    @pytest.fixture
    def waypoints(self):
        start = Waypoint(34.0, -118.0, 90.0, name="A")
        end = Waypoint(34.0, -117.5, 90.0, name="B")
        return start, end

    def test_zero_speed_raises(self, waypoints):
        start, end = waypoints
        with pytest.raises(HyPlanValueError, match="speed"):
            DubinsPath2D(start, end, speed=0.0, bank_angle=30.0)

    def test_negative_speed_raises(self, waypoints):
        start, end = waypoints
        with pytest.raises(HyPlanValueError, match="speed"):
            DubinsPath2D(start, end, speed=-50.0, bank_angle=30.0)

    def test_zero_speed_quantity_raises(self, waypoints):
        from hyplan.units import ureg
        start, end = waypoints
        with pytest.raises(HyPlanValueError, match="speed"):
            DubinsPath2D(start, end, speed=ureg.Quantity(0, "m/s"), bank_angle=30.0)

    def test_coincident_waypoints_give_valid_linestring(self):
        """Same start/end pose yields the 2πρ loop and a valid geometry."""
        wp = Waypoint(34.0, -118.0, 90.0, name="A")
        path = DubinsPath2D(wp, wp, speed=100.0, bank_angle=30.0)
        assert path.length.m_as("meter") == pytest.approx(
            2 * math.pi * path.min_turn_radius.m_as("meter"), rel=1e-6,
        )
        assert path.geometry.is_valid
        assert len(path.geometry.coords) >= 2

    def test_degenerate_sample_points_yield_valid_linestring(self):
        """A forced zero-length path must sample to a 2-point LineString
        instead of crashing LineString construction."""
        from shapely.geometry import LineString

        start = Waypoint(34.0, -118.0, 90.0, name="A")
        end = Waypoint(34.0, -117.5, 90.0, name="B")
        path = DubinsPath2D(start, end, speed=100.0, bank_angle=30.0)
        path._length_m = 0.0
        pts = path._sample_points(10)
        assert pts.shape == (2, 3)
        geom = LineString(np.column_stack([pts[:, 1], pts[:, 0]]))
        assert not geom.is_empty
        assert len(geom.coords) == 2


class TestDubinsPath2DSublinestring:
    def test_matches_per_point_sampling(self):
        """Batched sublinestring must match per-point sample_at_distance."""
        start = Waypoint(34.0, -118.0, 90.0, name="A")
        end = Waypoint(34.3, -117.5, 0.0, name="B")
        path = DubinsPath2D(start, end, speed=100.0, bank_angle=30.0)

        n = 12
        length_m = path.length.m_as("meter")
        sub = path.sublinestring(0.1 * length_m, 0.9 * length_m, n_samples=n)
        coords = list(sub.coords)
        assert len(coords) == n

        distances = np.linspace(0.1 * length_m, 0.9 * length_m, n)
        for (lon, lat), d in zip(coords, distances, strict=True):
            exp_lat, exp_lon, _ = path.sample_at_distance(float(d))
            assert lat == pytest.approx(exp_lat, abs=1e-9)
            assert lon == pytest.approx(exp_lon, abs=1e-9)

    def test_matches_per_point_sampling_with_wind(self):
        start = Waypoint(34.0, -118.0, 90.0, name="A")
        end = Waypoint(34.3, -117.5, 0.0, name="B")
        path = DubinsPath2D(
            start, end, speed=100.0, bank_angle=30.0, wind=(10.0, -5.0),
        )

        n = 8
        length_m = path.length.m_as("meter")
        sub = path.sublinestring(0.0, length_m, n_samples=n)
        coords = list(sub.coords)
        distances = np.linspace(0.0, length_m, n)
        for (lon, lat), d in zip(coords, distances, strict=True):
            exp_lat, exp_lon, _ = path.sample_at_distance(float(d))
            assert lat == pytest.approx(exp_lat, abs=1e-9)
            assert lon == pytest.approx(exp_lon, abs=1e-9)


class TestTrochoidGroundLengthCache:
    def test_cached_value_stable(self):
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([5000.0, 1000.0, 0.5])
        d = _TrochoidDubins2D(qi, qf, 200.0, 100.0, 10.0, -5.0)
        first = d.ground_length
        assert first > 0
        assert d.ground_length == first


class TestTrochoidSolverFailureHandling:
    """A failed BSB solve must never beat valid candidates."""

    @staticmethod
    def _patch_failed_bsb(monkeypatch):
        """Patch solve_trochoid so it reports failure (total_time = inf)."""
        import hyplan._trochoid_solver as ts

        real_solve = ts.solve_trochoid

        def failed(qi, qf, rhomin, airspeed, wind_u, wind_v):
            sol = dict(real_solve(qi, qf, rhomin, airspeed, wind_u, wind_v))
            sol["total_time"] = math.inf
            return sol

        monkeypatch.setattr(ts, "solve_trochoid", failed)

    def test_solve_trochoid_failure_keeps_inf(self, monkeypatch):
        """With every BSB family suppressed, total_time stays inf
        (not the old 0.0 sentinel)."""
        import hyplan._trochoid_solver as ts

        monkeypatch.setattr(ts, "_try_analytical", lambda *a, **k: None)
        monkeypatch.setattr(ts, "_try_numerical", lambda *a, **k: None)
        qi = np.array([0.0, 0.0, 0.0])
        qf = np.array([1000.0, 0.0, 0.0])
        sol = ts.solve_trochoid(qi, qf, 200.0, 100.0, 10.0, 0.0)
        assert math.isinf(sol["total_time"])

    def test_failed_bsb_never_wins_selection(self, monkeypatch):
        """When BSB fails but CCC candidates exist, the solver must pick
        a CCC mode instead of a zero-time failed BSB."""
        self._patch_failed_bsb(monkeypatch)

        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])
        d = _TrochoidDubins2D(qi, qf, 7.2 * nmi, 218.6, 13.4, 0.0)
        assert d._mode in ("ccc_trochoid", "ccc_air_drift")
        assert math.isfinite(d.total_time)
        assert d.total_time > 0

    def test_all_candidates_failing_raises(self, monkeypatch):
        self._patch_failed_bsb(monkeypatch)

        nmi = 1852.0
        qi = np.array([0.0, 0.0, math.pi / 2])
        qf = np.array([5.0 * nmi, 0.0, -math.pi / 2])
        with pytest.raises(HyPlanRuntimeError, match="No feasible"):
            _TrochoidDubins2D(
                qi, qf, 7.2 * nmi, 218.6, 13.4, 0.0, disable_ccc=True,
            )
