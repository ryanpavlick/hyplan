"""Tests for the ADS-B trajectory fitting pipeline.

Tests use synthetic trajectory data — no real ADS-B files or ``traffic``
installation required for the core algorithm tests.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from hyplan.aircraft._base import (
    Aircraft,
    PerformanceConfidence,
    SourceRecord,
    TasSchedule,
    VerticalProfile,
)
from hyplan.aircraft.adsb.fitting import (
    _rdp_simplify,
    _reject_outliers,
    fit_schedules,
)
from hyplan.aircraft.adsb.models import FitResult, FlightPhaseData, ScheduleFitMetrics
from hyplan.aircraft.adsb.phases import (
    _find_runs,
    _merge_short_phases,
    _rolling_median,
    label_phases,
)
from hyplan.units import ureg
from hyplan.winds import ConstantWindField, StillAirField


# ===================================================================
# Helpers — synthetic data generation
# ===================================================================


def _make_timestamps(n: int, start: str = "2024-06-01T12:00:00") -> np.ndarray:
    """Create n timestamps at 5-second intervals."""
    t0 = np.datetime64(start)
    return t0 + np.arange(n) * np.timedelta64(5, "s")


class _FakeFlight:
    """Minimal flight-like object with a .data DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.data = df
        if "icao24" in df.columns:
            self.icao24 = df["icao24"].iloc[0]
        if "callsign" in df.columns:
            self.callsign = df["callsign"].iloc[0]


def _make_climb_df(n: int = 100, start_alt: float = 2000.0, end_alt: float = 35000.0):
    """Synthetic steady climb trajectory."""
    ts = _make_timestamps(n)
    alts = np.linspace(start_alt, end_alt, n)
    vs = np.full(n, (end_alt - start_alt) / (n * 5 / 60))  # ft/min
    gs = np.linspace(250, 400, n)
    return pd.DataFrame({
        "timestamp": ts,
        "latitude": np.linspace(34.0, 36.0, n),
        "longitude": np.linspace(-118.0, -116.0, n),
        "altitude": alts,
        "groundspeed": gs,
        "track": np.full(n, 45.0),
        "vertical_rate": vs,
        "icao24": "abc123",
        "callsign": "TEST01",
    })


def _make_cruise_df(n: int = 100, altitude: float = 35000.0, speed: float = 450.0):
    """Synthetic level cruise trajectory."""
    ts = _make_timestamps(n)
    return pd.DataFrame({
        "timestamp": ts,
        "latitude": np.linspace(36.0, 38.0, n),
        "longitude": np.linspace(-116.0, -114.0, n),
        "altitude": np.full(n, altitude),
        "groundspeed": np.full(n, speed),
        "track": np.full(n, 45.0),
        "vertical_rate": np.random.normal(0, 50, n),
        "icao24": "abc123",
        "callsign": "TEST01",
    })


def _make_descent_df(n: int = 100, start_alt: float = 35000.0, end_alt: float = 2000.0):
    """Synthetic steady descent trajectory."""
    ts = _make_timestamps(n)
    alts = np.linspace(start_alt, end_alt, n)
    vs = np.full(n, (end_alt - start_alt) / (n * 5 / 60))  # negative ft/min
    gs = np.linspace(400, 250, n)
    return pd.DataFrame({
        "timestamp": ts,
        "latitude": np.linspace(38.0, 40.0, n),
        "longitude": np.linspace(-114.0, -112.0, n),
        "altitude": alts,
        "groundspeed": gs,
        "track": np.full(n, 45.0),
        "vertical_rate": vs,
        "icao24": "abc123",
        "callsign": "TEST01",
    })


def _make_full_flight_df(
    n_climb: int = 80,
    n_cruise: int = 200,
    n_descent: int = 80,
):
    """Synthetic climb-cruise-descent trajectory."""
    climb = _make_climb_df(n_climb)
    cruise = _make_cruise_df(n_cruise)
    descent = _make_descent_df(n_descent)

    # Adjust timestamps to be continuous
    dt = np.timedelta64(5, "s")
    cruise["timestamp"] = climb["timestamp"].iloc[-1] + dt + np.arange(n_cruise) * dt
    descent["timestamp"] = cruise["timestamp"].iloc[-1] + dt + np.arange(n_descent) * dt

    return pd.concat([climb, cruise, descent], ignore_index=True)


# ===================================================================
# TestLabelPhases
# ===================================================================


class TestLabelPhases:
    """Test the heuristic phase labeler."""

    def test_steady_climb(self):
        df = _make_climb_df(n=50, start_alt=5000, end_alt=30000)
        flight = _FakeFlight(df)
        result = label_phases(flight)
        assert "phase" in result.columns
        # All non-ground points should be climb
        airborne = result[result["altitude"] >= 2000]
        assert (airborne["phase"] == "climb").all()

    def test_steady_descent(self):
        df = _make_descent_df(n=50, start_alt=30000, end_alt=5000)
        flight = _FakeFlight(df)
        result = label_phases(flight)
        airborne = result[result["altitude"] >= 2000]
        assert (airborne["phase"] == "descent").all()

    def test_level_cruise(self):
        df = _make_cruise_df(n=50, altitude=35000)
        # Force vertical_rate to near-zero
        df["vertical_rate"] = np.random.normal(0, 10, 50)
        flight = _FakeFlight(df)
        result = label_phases(flight)
        assert (result["phase"] == "cruise").all()

    def test_ground_phase(self):
        df = _make_cruise_df(n=20, altitude=500, speed=30)
        df["vertical_rate"] = 0.0
        flight = _FakeFlight(df)
        result = label_phases(flight, ground_altitude_ft=2000)
        assert (result["phase"] == "ground").all()

    def test_mixed_flight(self):
        df = _make_full_flight_df(n_climb=40, n_cruise=60, n_descent=40)
        flight = _FakeFlight(df)
        result = label_phases(flight)
        phases = set(result["phase"].unique())
        # Should contain at least climb, cruise, and descent
        assert "climb" in phases
        assert "cruise" in phases
        assert "descent" in phases

    def test_short_phase_merged(self):
        """A 30-second cruise blip in a climb should be merged."""
        n = 60
        ts = _make_timestamps(n)
        alts = np.linspace(5000, 35000, n)
        vs = np.full(n, 3000.0)  # climbing at 3000 fpm
        # Insert a 6-point (30s) near-zero VS blip
        vs[25:31] = 50.0
        df = pd.DataFrame({
            "timestamp": ts,
            "latitude": np.linspace(34, 36, n),
            "longitude": np.linspace(-118, -116, n),
            "altitude": alts,
            "groundspeed": np.full(n, 350),
            "track": np.full(n, 90.0),
            "vertical_rate": vs,
        })
        flight = _FakeFlight(df)
        result = label_phases(flight, min_phase_seconds=60)
        # The blip should be absorbed — no cruise phase should appear
        airborne = result[result["altitude"] >= 2000]
        assert "cruise" not in airborne["phase"].values

    def test_lstm_backend_raises(self):
        df = _make_cruise_df(n=10)
        flight = _FakeFlight(df)
        with pytest.raises(NotImplementedError, match="LSTM"):
            label_phases(flight, backend="lstm")

    def test_unknown_backend_raises(self):
        df = _make_cruise_df(n=10)
        flight = _FakeFlight(df)
        with pytest.raises(ValueError, match="Unknown"):
            label_phases(flight, backend="magic")


# ===================================================================
# TestPhaseHelpers
# ===================================================================


class TestPhaseHelpers:
    """Test low-level phase labeling utilities."""

    def test_rolling_median_constant(self):
        arr = np.full(20, 5.0)
        result = _rolling_median(arr, window=5)
        np.testing.assert_array_almost_equal(result, 5.0)

    def test_rolling_median_short_array(self):
        arr = np.array([1.0, 3.0])
        result = _rolling_median(arr, window=5)
        assert len(result) == 2
        assert result[0] == pytest.approx(2.0)

    def test_find_runs(self):
        arr = np.array(["a", "a", "b", "b", "b", "a"])
        runs = _find_runs(arr)
        assert len(runs) == 3
        assert runs[0] == (0, 2, "a")
        assert runs[1] == (2, 5, "b")
        assert runs[2] == (5, 6, "a")

    def test_find_runs_empty(self):
        assert _find_runs(np.array([])) == []

    def test_merge_short_phases(self):
        # climb(20s), cruise(5s), climb(20s) — cruise is too short
        phases = np.array([
            "climb", "climb", "climb", "climb", "climb",  # 0-20s
            "cruise",                                       # 25s
            "climb", "climb", "climb", "climb", "climb",  # 30-50s
        ])
        ts = np.arange(len(phases), dtype=float) * 5.0
        result = _merge_short_phases(phases.copy(), ts, min_seconds=10)
        # The short "cruise" (5s) should be merged into "climb"
        assert "cruise" not in result


# ===================================================================
# TestReconstructAirdata
# ===================================================================


class TestReconstructAirdata:
    """Test wind-corrected TAS reconstruction."""

    def test_still_air_identity(self):
        """With no wind, TAS should equal groundspeed."""
        from hyplan.aircraft.adsb.airdata import reconstruct_airdata

        df = _make_cruise_df(n=20, speed=450)
        df["phase"] = "cruise"
        wf = StillAirField()
        result = reconstruct_airdata(df, wf)

        np.testing.assert_array_almost_equal(
            result["tas_kt"].values, 450.0, decimal=1
        )
        np.testing.assert_array_almost_equal(
            result["heading_true_deg"].values, 45.0, decimal=1
        )
        np.testing.assert_array_almost_equal(result["wind_u_mps"].values, 0.0)
        np.testing.assert_array_almost_equal(result["wind_v_mps"].values, 0.0)

    def test_headwind_reduces_groundspeed(self):
        """With a headwind, TAS > groundspeed (aircraft flies faster than ground track)."""
        from hyplan.aircraft.adsb.airdata import reconstruct_airdata

        df = _make_cruise_df(n=20, speed=400)
        df["phase"] = "cruise"
        df["track"] = 0.0  # flying north

        # 30-knot wind from the north (headwind)
        wf = ConstantWindField(30 * ureg.knot, 0.0)
        result = reconstruct_airdata(df, wf)

        # TAS should be greater than GS because wind opposes motion
        mean_tas = result["tas_kt"].mean()
        assert mean_tas > 400.0
        assert mean_tas == pytest.approx(430.0, abs=2.0)

    def test_tailwind_increases_groundspeed(self):
        """With a tailwind, TAS < groundspeed."""
        from hyplan.aircraft.adsb.airdata import reconstruct_airdata

        df = _make_cruise_df(n=20, speed=400)
        df["phase"] = "cruise"
        df["track"] = 0.0  # flying north

        # 30-knot wind from the south (tailwind)
        wf = ConstantWindField(30 * ureg.knot, 180.0)
        result = reconstruct_airdata(df, wf)

        mean_tas = result["tas_kt"].mean()
        assert mean_tas < 400.0
        assert mean_tas == pytest.approx(370.0, abs=2.0)

    def test_crosswind_changes_heading(self):
        """With a 90-degree crosswind, heading should differ from track."""
        from hyplan.aircraft.adsb.airdata import reconstruct_airdata

        df = _make_cruise_df(n=20, speed=400)
        df["phase"] = "cruise"
        df["track"] = 0.0  # flying north

        # 50-knot wind from the west (crosswind from left)
        wf = ConstantWindField(50 * ureg.knot, 270.0)
        result = reconstruct_airdata(df, wf)

        # Heading should be offset to the west to compensate
        mean_heading = result["heading_true_deg"].mean()
        assert mean_heading != pytest.approx(0.0, abs=1.0)

    def test_wind_columns_present(self):
        """All expected output columns should exist."""
        from hyplan.aircraft.adsb.airdata import reconstruct_airdata

        df = _make_cruise_df(n=10)
        df["phase"] = "cruise"
        result = reconstruct_airdata(df, StillAirField())
        expected = {"wind_u_mps", "wind_v_mps", "tas_kt", "heading_true_deg",
                    "wind_speed_kt", "wind_from_deg"}
        assert expected.issubset(set(result.columns))


# ===================================================================
# TestFitSchedules
# ===================================================================


class TestFitSchedules:
    """Test schedule and profile fitting."""

    def _make_airdata_df(self):
        """Build a combined phase-labeled + wind-corrected DataFrame."""
        climb = _make_climb_df(n=100, start_alt=5000, end_alt=35000)
        climb["phase"] = "climb"
        climb["tas_kt"] = climb["groundspeed"]

        cruise = _make_cruise_df(n=200, altitude=35000, speed=450)
        cruise["phase"] = "cruise"
        cruise["tas_kt"] = cruise["groundspeed"]

        descent = _make_descent_df(n=100, start_alt=35000, end_alt=3000)
        descent["phase"] = "descent"
        descent["tas_kt"] = descent["groundspeed"]

        return pd.concat([climb, cruise, descent], ignore_index=True)

    def test_returns_fit_result(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        assert isinstance(result, FitResult)

    def test_schedules_are_tas_schedule(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        assert isinstance(result.climb_schedule, TasSchedule)
        assert isinstance(result.cruise_schedule, TasSchedule)
        assert isinstance(result.descent_schedule, TasSchedule)

    def test_profiles_are_vertical_profile(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        assert isinstance(result.climb_profile, VerticalProfile)
        assert isinstance(result.descent_profile, VerticalProfile)

    def test_service_ceiling_inferred(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        # Max altitude is 35000, ceiling should be rounded up
        assert result.service_ceiling_ft >= 35000

    def test_service_ceiling_override(self):
        df = self._make_airdata_df()
        result = fit_schedules(df, service_ceiling_ft=41000)
        assert result.service_ceiling_ft == 41000

    def test_approach_speed_estimated(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        assert result.approach_speed_kt > 0

    def test_max_schedule_points_respected(self):
        df = self._make_airdata_df()
        result = fit_schedules(df, altitude_bin_ft=1000, max_schedule_points=4)
        assert len(result.cruise_schedule.points) <= 4

    def test_metrics_present(self):
        df = self._make_airdata_df()
        result = fit_schedules(df)
        assert "cruise_speed" in result.metrics
        sm = result.metrics["cruise_speed"]
        assert isinstance(sm, ScheduleFitMetrics)
        assert 0.0 <= sm.r_squared <= 1.0
        assert sm.n_observations > 0

    def test_single_altitude_constant(self):
        """Data at one altitude should produce a 1-point schedule."""
        df = _make_cruise_df(n=50, altitude=30000, speed=400)
        df["phase"] = "cruise"
        df["tas_kt"] = df["groundspeed"]
        result = fit_schedules(df)
        assert len(result.cruise_schedule.points) == 1

    def test_outlier_rejection(self):
        df = self._make_airdata_df()
        # Inject extreme outliers in cruise
        cruise_mask = df["phase"] == "cruise"
        outlier_idx = df[cruise_mask].index[:5]
        df.loc[outlier_idx, "tas_kt"] = 900.0  # absurd
        result = fit_schedules(df, outlier_sigma=2.0)
        # Cruise speed should not be pulled to 900
        cruise_tas = result.cruise_schedule.tas_at(35000 * ureg.feet).m_as(ureg.knot)
        assert cruise_tas < 600

    def test_missing_phase_fallback(self):
        """If no climb data exists, fallback schedule should be used."""
        cruise = _make_cruise_df(n=100, altitude=35000, speed=450)
        cruise["phase"] = "cruise"
        cruise["tas_kt"] = cruise["groundspeed"]
        result = fit_schedules(cruise)
        # Should still have a climb schedule (fallback)
        assert isinstance(result.climb_schedule, TasSchedule)
        assert isinstance(result.climb_profile, VerticalProfile)


# ===================================================================
# TestRDP
# ===================================================================


class TestRDPSimplify:
    """Test Ramer-Douglas-Peucker simplification."""

    def test_identity_below_max(self):
        pts = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
        result = _rdp_simplify(pts, max_points=5)
        assert len(result) <= 5

    def test_reduces_to_max(self):
        pts = np.column_stack([np.arange(20), np.sin(np.arange(20))])
        result = _rdp_simplify(pts, max_points=4)
        assert len(result) <= 4

    def test_preserves_endpoints(self):
        pts = np.column_stack([np.arange(10, dtype=float), np.random.randn(10)])
        result = _rdp_simplify(pts, max_points=3)
        np.testing.assert_array_almost_equal(result[0], pts[0])
        np.testing.assert_array_almost_equal(result[-1], pts[-1])

    def test_collinear_points(self):
        """Perfectly linear data should simplify to 2 endpoints."""
        pts = np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float)])
        result = _rdp_simplify(pts, max_points=5)
        assert len(result) == 2


class TestRejectOutliers:
    def test_no_outliers(self):
        arr = np.array([10.0, 11.0, 10.5, 10.2, 10.8])
        result = _reject_outliers(arr, sigma=2.5)
        assert len(result) == len(arr)

    def test_removes_extreme(self):
        arr = np.array([10.0, 10.1, 10.2, 100.0, 10.3])
        result = _reject_outliers(arr, sigma=2.0)
        assert 100.0 not in result


# ===================================================================
# TestFitResult
# ===================================================================


class TestFitResult:
    """Test FitResult metadata methods."""

    def _make_fit_result(self) -> FitResult:
        return FitResult(
            climb_schedule=TasSchedule(points=[
                (0 * ureg.feet, 250 * ureg.knot),
                (35000 * ureg.feet, 400 * ureg.knot),
            ]),
            cruise_schedule=TasSchedule(points=[
                (35000 * ureg.feet, 450 * ureg.knot),
            ]),
            descent_schedule=TasSchedule(points=[
                (0 * ureg.feet, 220 * ureg.knot),
                (35000 * ureg.feet, 370 * ureg.knot),
            ]),
            climb_profile=VerticalProfile(points=[
                (0 * ureg.feet, 2500 * ureg.feet / ureg.minute),
                (35000 * ureg.feet, 500 * ureg.feet / ureg.minute),
            ]),
            descent_profile=VerticalProfile(points=[
                (0 * ureg.feet, 1500 * ureg.feet / ureg.minute),
            ]),
            service_ceiling_ft=36000.0,
            approach_speed_kt=140.0,
            metrics={
                "climb_speed": ScheduleFitMetrics(0.95, 5.0, 200, 0.9, 3),
                "cruise_speed": ScheduleFitMetrics(0.98, 3.0, 500, 1.0, 1),
                "descent_speed": ScheduleFitMetrics(0.90, 8.0, 150, 0.8, 3),
                "climb_vertical": ScheduleFitMetrics(0.85, 100.0, 200, 0.9, 3),
                "descent_vertical": ScheduleFitMetrics(0.80, 120.0, 150, 0.8, 2),
            },
            icao24="abc123",
            n_flights=3,
            wind_source="merra2",
        )

    def test_confidence_range(self):
        fit = self._make_fit_result()
        pc = fit.overall_confidence()
        assert isinstance(pc, PerformanceConfidence)
        assert 0.0 <= pc.climb <= 1.0
        assert 0.0 <= pc.cruise <= 1.0
        assert 0.0 <= pc.descent <= 1.0
        assert pc.turns == 0.3  # v1 fixed

    def test_high_r_squared_gives_high_confidence(self):
        fit = self._make_fit_result()
        pc = fit.overall_confidence()
        # With R^2 > 0.85 and good coverage, confidence should be > 0.5
        assert pc.climb > 0.5
        assert pc.cruise > 0.5

    def test_source_records(self):
        fit = self._make_fit_result()
        records = fit.source_records()
        assert len(records) == 1
        assert isinstance(records[0], SourceRecord)
        assert records[0].source_type == "adsb"
        assert "abc123" in records[0].reference
        assert "merra2" in records[0].reference

    def test_confidence_with_no_metrics(self):
        fit = self._make_fit_result()
        fit.metrics = {}
        pc = fit.overall_confidence()
        # Should fall back to 0.2 for all phases
        assert pc.climb == 0.2
        assert pc.cruise == 0.2
        assert pc.descent == 0.2


# ===================================================================
# TestPriors
# ===================================================================


class TestPriors:
    """Test the v1 stub functions."""

    def test_apply_prior_noop(self):
        from hyplan.aircraft.adsb.priors import apply_prior

        fit = TestFitResult()._make_fit_result()
        result = apply_prior(fit)
        assert result is fit  # same object returned

    def test_score_fit_empty(self):
        from hyplan.aircraft.adsb.priors import score_fit

        fit = TestFitResult()._make_fit_result()
        result = score_fit(fit)
        assert result == {}
