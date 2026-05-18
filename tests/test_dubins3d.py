"""Tests for hyplan.dubins3d (3D Dubins paths with pitch constraints)."""

import math

import numpy as np
import pytest

from hyplan.dubins3d import _Dubins2D, _TrochoidDubins2D


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
