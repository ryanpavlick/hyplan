"""Tests for the v1.10 dropsonde object model.

Covers the locked contracts from ``plans/dropsonde-object-model.md``:

- frozen identity-equal dataclasses; defensive waypoint copy
- ``WindField.is_time_dependent`` early-fail contract
- tri-state ``qc_release_ok`` aggregation
- ``DropsondePlan.simulate`` is functional
- trajectory ↔ release identity invariant (ensembles + skips)
- ``qc_splash_in_target_polygon`` evaluated against ensemble-mean splash
- ``from_pattern`` cadence resets per line + multi-line timestamp warning
- ``solve_release_for_target`` requires a ``flight_track``
- ``FlightPlanTrack.from_compute_flight_plan`` preserves labelled indices
- ``create_sensor`` returns the singleton (``is``-identity)
- golden-section search recovers a quadratic minimum
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
from shapely.geometry import LineString, Point

from hyplan import ureg
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_line import FlightLine
from hyplan.instruments import create_sensor
from hyplan.instruments.dropsondes import (
    AVAPS_NRD41,
    AircraftTrackSample,
    DropsondePlan,
    DropsondeRelease,
    DropsondeSystem,
    DropsondeTrajectory,
    FlightPlanTrack,
    PlannedSegment,
    RD94,
    golden_section_search,
    releases_along_flight_line,
    simulate_descent_trajectory,
    simulate_release,
    solve_release_for_target,
    summarize_trajectories,
    terminal_velocity_nrd41,
)
from hyplan.pattern import Pattern
from hyplan.waypoint import Waypoint
from hyplan.winds import ConstantWindField, StillAirField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flight_line_demo() -> FlightLine:
    wp1 = Waypoint(
        latitude=30.0, longitude=-80.0, heading=90.0,
        altitude_msl=25000 * 0.3048 * ureg.meter,
        name="demo_start",
    )
    wp2 = Waypoint(
        latitude=30.0, longitude=-76.0, heading=270.0,
        altitude_msl=25000 * 0.3048 * ureg.meter,
        name="demo_end",
    )
    return FlightLine(wp1, wp2, site_name="demo")


def _simple_plan_gdf(
    *,
    altitude_ft: float = 25000.0,
    leg_length_km: float = 200.0,
    groundspeed_kts: float = 200.0,
    duration_min: float | None = None,
) -> gpd.GeoDataFrame:
    """One flight-line segment, geodesic length-consistent."""
    if duration_min is None:
        # 1 nm = 1.852 km; gs_kts * 1.852 km/h.
        duration_min = (leg_length_km / (groundspeed_kts * 1.852)) * 60.0
    deg_per_km_lon = 1.0 / (111.32 * float(np.cos(np.radians(30.0))))
    end_lon = -80.0 + leg_length_km * deg_per_km_lon
    return gpd.GeoDataFrame(
        [
            {
                "geometry": LineString([(-80.0, 30.0), (end_lon, 30.0)]),
                "segment_type": "flight_line",
                "segment_name": "Leg 1",
                "site_name": "Leg 1",
                "start_lat": 30.0, "start_lon": -80.0,
                "end_lat": 30.0, "end_lon": end_lon,
                "start_altitude": altitude_ft, "end_altitude": altitude_ft,
                "time_to_segment": duration_min,
                "distance": leg_length_km / 1.852,
                "groundspeed_kts": groundspeed_kts,
                "planned_track": 90.0,
            }
        ],
        geometry="geometry", crs="EPSG:4326",
    )


def _takeoff() -> _dt.datetime:
    return _dt.datetime(2026, 5, 16, 12, 0, 0, tzinfo=_dt.timezone.utc)


# ---------------------------------------------------------------------------
# Sensor + registry
# ---------------------------------------------------------------------------


class TestDropsondeSystem:
    def test_default_nrd41_curve_clamps(self):
        # ~22 m/s near FL420 region; ~11 m/s at sea level.
        w_sea = terminal_velocity_nrd41(0 * ureg.meter)
        w_high = terminal_velocity_nrd41(13_000 * ureg.meter)
        assert 10.5 <= float(w_sea.m_as("meter / second")) <= 11.5
        assert 21.0 <= float(w_high.m_as("meter / second")) <= 23.0

    def test_create_sensor_returns_singleton(self):
        # The factory must return the *same object*, not a fresh
        # configured copy.
        assert create_sensor("AVAPS_NRD41") is AVAPS_NRD41
        assert create_sensor("Vaisala NRD41") is AVAPS_NRD41
        assert create_sensor("NRD41") is AVAPS_NRD41
        assert create_sensor("RD94") is RD94
        assert create_sensor("Vaisala RD94") is RD94

    def test_fall_time_monotone_increasing(self):
        t1 = AVAPS_NRD41.fall_time(5000 * ureg.meter)
        t2 = AVAPS_NRD41.fall_time(10000 * ureg.meter)
        assert float(t1.m_as("second")) < float(t2.m_as("second"))

    def test_axctd_fall_time_reasonable(self):
        # AXCTD air phase from 6 km MSL: at ~11 m/s mean, ~9-10 min total.
        # Use generous bounds — the table is a planning approximation,
        # not a calibrated probe model.
        from hyplan.instruments.dropsondes import AXCTD
        t = AXCTD.fall_time(6000 * ureg.meter)
        assert 400.0 < float(t.m_as("second")) < 700.0

    def test_axctd_singleton_via_factory(self):
        from hyplan.instruments.dropsondes import AXCTD
        assert create_sensor("AXCTD") is AXCTD
        assert create_sensor("Sippican AXCTD") is AXCTD
        assert create_sensor("SIPPICAN_AXCTD") is AXCTD

    def test_axctd_simulate_release_works(self):
        # End-to-end: build a release from AXCTD and simulate it under
        # still air; verify a sensible splash result.
        from hyplan.instruments.dropsondes import AXCTD
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=6000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(
            wp, sensor=AXCTD, release_time=_takeoff(),
        )
        traj = simulate_release(rel, wind_field=StillAirField())
        assert traj.qc_terminated_at_ground is True
        assert float(traj.splash_waypoint.altitude_msl.m_as("meter")) <= 1e-3
        # Still air + no aircraft velocity = essentially vertical drop.
        assert float(traj.drift_distance.m_as("meter")) < 5.0


# ---------------------------------------------------------------------------
# Waypoint copy + frozen + identity
# ---------------------------------------------------------------------------


class TestDropsondeReleaseModel:
    def test_from_waypoint_defensive_copy(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        # Mutate the source waypoint after construction.
        wp.latitude = 99.0
        assert rel.waypoint.latitude == 30.0  # release unchanged

    def test_to_waypoint_defensive_copy(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        out = rel.to_waypoint()
        out.latitude = 99.0
        assert rel.waypoint.latitude == 30.0  # release still unchanged

    def test_identity_equality(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        r1 = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        r2 = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        assert r1 != r2
        assert hash(r1) != hash(r2)
        assert r1 == r1
        # Hash must succeed even when source could be a Pattern (it's None here).
        assert isinstance(hash(r1), int)

    def test_frozen_no_assignment(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rel.release_id = 99  # type: ignore[misc]

    def test_qc_release_ok_tri_state(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        # All True → True.
        r = DropsondeRelease.from_waypoint(
            wp, sensor=AVAPS_NRD41,
            qc_min_alt_ok=True, qc_aircraft_envelope_ok=True, qc_segment_allowed=True,
        )
        assert r.qc_release_ok is True
        # Any False → False.
        r = DropsondeRelease.from_waypoint(
            wp, sensor=AVAPS_NRD41,
            qc_min_alt_ok=False, qc_aircraft_envelope_ok=None, qc_segment_allowed=None,
        )
        assert r.qc_release_ok is False
        # Mixed True/None → None.
        r = DropsondeRelease.from_waypoint(
            wp, sensor=AVAPS_NRD41,
            qc_min_alt_ok=True, qc_aircraft_envelope_ok=None, qc_segment_allowed=True,
        )
        assert r.qc_release_ok is None
        # All None → None.
        r = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41)
        assert r.qc_release_ok is None

    def test_qc_polygon_not_in_gate(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        r = DropsondeRelease.from_waypoint(
            wp, sensor=AVAPS_NRD41,
            qc_min_alt_ok=True, qc_aircraft_envelope_ok=True, qc_segment_allowed=True,
            qc_splash_in_target_polygon=False,
        )
        # Polygon flag is False but it's NOT a gate — qc_release_ok stays True.
        assert r.qc_release_ok is True


# ---------------------------------------------------------------------------
# Wind-field is_time_dependent contract
# ---------------------------------------------------------------------------


class TestWindFieldTimeDependence:
    def test_simple_winds_are_time_independent(self):
        assert StillAirField.is_time_dependent is False
        assert ConstantWindField.is_time_dependent is False

    def test_base_default_is_time_dependent(self):
        # Anything not explicitly overriding is treated as time-dependent.
        from hyplan.winds.base import WindField
        assert WindField.is_time_dependent is True

    def test_simulate_release_without_time_against_still_air_works(self):
        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41, release_time=None)
        traj = simulate_release(rel, wind_field=StillAirField())
        # Drift in still air with no aircraft velocity must be ~0.
        assert float(traj.drift_distance.m_as("meter")) < 5.0

    def test_simulate_release_without_time_raises_for_time_dependent(self):
        class FakeTimeDep(StillAirField):
            is_time_dependent = True

        wp = Waypoint(30.0, -80.0, heading=90.0, altitude_msl=8000 * ureg.meter)
        rel = DropsondeRelease.from_waypoint(wp, sensor=AVAPS_NRD41, release_time=None)
        with pytest.raises(HyPlanValueError, match="release_time"):
            simulate_release(rel, wind_field=FakeTimeDep())


# ---------------------------------------------------------------------------
# Descent simulator (kernel)
# ---------------------------------------------------------------------------


class TestDescentSimulator:
    def test_still_air_vertical_descent(self):
        traj = simulate_descent_trajectory(
            release_lat=30.0, release_lon=-80.0,
            release_altitude_msl=8000 * ureg.meter,
            release_time_utc=_takeoff(),
            sensor=AVAPS_NRD41, wind_field=StillAirField(),
        )
        first = traj.iloc[0]
        last = traj.iloc[-1]
        assert abs(last["latitude"] - first["latitude"]) < 1e-5
        assert abs(last["longitude"] - first["longitude"]) < 1e-5
        assert last["altitude_msl_m"] <= 1e-3

    def test_constant_wind_drift_east(self):
        traj = simulate_descent_trajectory(
            release_lat=30.0, release_lon=-80.0,
            release_altitude_msl=8000 * ureg.meter,
            release_time_utc=_takeoff(),
            sensor=AVAPS_NRD41,
            wind_field=ConstantWindField(10 * ureg.meter / ureg.second, wind_from_deg=270.0),
        )
        last = traj.iloc[-1]
        # Constant 10 m/s east wind → drift east, lat ≈ unchanged.
        assert last["longitude"] > -80.0
        assert abs(last["latitude"] - 30.0) < 1e-3

    def test_surface_elevation_termination(self):
        traj = simulate_descent_trajectory(
            release_lat=30.0, release_lon=-80.0,
            release_altitude_msl=8000 * ureg.meter,
            release_time_utc=_takeoff(),
            sensor=AVAPS_NRD41, wind_field=StillAirField(),
            surface_elevation_msl=500 * ureg.meter,
        )
        last = traj.iloc[-1]
        assert abs(last["altitude_msl_m"] - 500.0) < 1.0

    def test_deployment_transient_adds_drift(self):
        # 100 m/s east aircraft inheritance over 5 s deploy ≈ 250 m east.
        traj = simulate_descent_trajectory(
            release_lat=30.0, release_lon=-80.0,
            release_altitude_msl=8000 * ureg.meter,
            release_time_utc=_takeoff(),
            sensor=AVAPS_NRD41, wind_field=StillAirField(),
            aircraft_velocity_mps=(100.0, 0.0),
        )
        last = traj.iloc[-1]
        # Eastward drift ≈ 250 m (within 30 m tolerance).
        m_per_deg_lon = 111_320.0 * float(np.cos(np.radians(30.0)))
        dx = (last["longitude"] - (-80.0)) * m_per_deg_lon
        assert 200.0 < dx < 320.0


# ---------------------------------------------------------------------------
# FlightPlanTrack adapter
# ---------------------------------------------------------------------------


class TestFlightPlanTrack:
    def test_from_compute_flight_plan_basic(self):
        plan = _simple_plan_gdf()
        track = FlightPlanTrack.from_compute_flight_plan(plan)
        assert len(track.segments) == 1
        seg = track.segments[0]
        assert seg.segment_type == "flight_line"
        assert seg.start_elapsed_s == 0.0
        # 200 km / (200 kts * 0.5144) ≈ 1944 s.
        assert 1800.0 <= seg.duration_s <= 2100.0
        assert seg.end_elapsed_s == seg.start_elapsed_s + seg.duration_s

    def test_preserves_labelled_index(self):
        plan = _simple_plan_gdf()
        plan.index = ["leg-a"]
        track = FlightPlanTrack.from_compute_flight_plan(plan)
        assert track.segments[0].index == "leg-a"

    def test_filter_preserves_cumulative_timing(self):
        # Two segments, only the second is a flight_line.
        plan = _simple_plan_gdf()
        # Add a fake "climb" segment ahead of the flight_line.
        climb = plan.iloc[0].copy()
        climb["segment_type"] = "climb"
        plan2 = gpd.GeoDataFrame(
            pd.concat([pd.DataFrame([climb]), pd.DataFrame(plan.iloc[[0]])], ignore_index=True),
            geometry="geometry", crs="EPSG:4326",
        )
        track = FlightPlanTrack.from_compute_flight_plan(plan2)
        filtered = track.filter(("flight_line",))
        assert len(filtered.segments) == 1
        # The flight line's start_elapsed_s must equal the climb's duration.
        climb_dur = track.segments[0].duration_s
        assert abs(filtered.segments[0].start_elapsed_s - climb_dur) < 1e-6

    def test_sample_at_elapsed_midpoint(self):
        plan = _simple_plan_gdf()
        track = FlightPlanTrack.from_compute_flight_plan(plan)
        mid = track.segments[0].end_elapsed_s / 2.0
        sample = track.sample_at_elapsed(mid * ureg.second)
        # Midpoint of a 200 km east-bound leg at 30N starts at lon -80;
        # 100 km east of that is roughly lon -78.96.
        assert -79.5 < sample.longitude < -78.5
        assert sample.segment_type == "flight_line"


# ---------------------------------------------------------------------------
# releases_along_flight_line
# ---------------------------------------------------------------------------


class TestReleasesAlongFlightLine:
    def test_count(self):
        fl = _flight_line_demo()  # ~385 km
        rels = releases_along_flight_line(
            fl,
            spacing=50 * ureg.kilometer,
            takeoff_time=_takeoff(),
            groundspeed=100 * ureg.meter / ureg.second,
        )
        # 385 km / 50 km ≈ 7-8 releases.
        assert 7 <= len(rels) <= 9

    def test_requires_groundspeed_for_distance_spacing(self):
        fl = _flight_line_demo()
        with pytest.raises(HyPlanValueError, match="groundspeed"):
            releases_along_flight_line(
                fl,
                spacing=20 * ureg.nautical_mile,
                takeoff_time=_takeoff(),
                # no groundspeed, no aircraft
            )

    def test_aircraft_speed_fallback(self):
        # When groundspeed is not supplied but aircraft is, fall back
        # to aircraft.cruise_speed_at(altitude).
        from hyplan.aircraft import NASA_P3
        fl = _flight_line_demo()
        ac = NASA_P3()
        rels = releases_along_flight_line(
            fl,
            spacing=50 * ureg.kilometer,
            takeoff_time=_takeoff(),
            aircraft=ac,
        )
        assert len(rels) > 1

    def test_release_time_populated_when_takeoff_supplied(self):
        fl = _flight_line_demo()
        rels = releases_along_flight_line(
            fl, spacing=100 * ureg.kilometer,
            takeoff_time=_takeoff(),
            groundspeed=100 * ureg.meter / ureg.second,
        )
        assert rels[0].release_time == _takeoff()
        assert rels[-1].release_time > _takeoff()


# ---------------------------------------------------------------------------
# DropsondePlan.from_flight_plan + simulate
# ---------------------------------------------------------------------------


class TestDropsondePlan:
    def test_from_flight_plan_sets_flight_track(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        assert plan.flight_track is not None
        assert len(plan.releases) > 0

    def test_simulate_is_functional(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        plan2 = plan.simulate(wind_field=StillAirField())
        # New plan; original is untouched.
        assert plan2 is not plan
        assert plan.trajectories == ()
        assert len(plan2.trajectories) == len(plan2.releases)
        assert all(r.qc_splash_in_target_polygon is None for r in plan.releases)

    def test_trajectory_release_identity_invariant_deterministic(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        plan2 = plan.simulate(wind_field=StillAirField())
        release_by_id = {r.release_id: r for r in plan2.releases}
        for t in plan2.trajectories:
            assert t.release is release_by_id[t.release.release_id]
        # Length: one trajectory per non-skipped release.
        expected = sum(
            1 for r in plan2.releases if r.qc_release_ok is not False
        )
        assert len(plan2.trajectories) == expected

    def test_trajectory_release_identity_invariant_ensemble(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=80 * ureg.nautical_mile,  # fewer releases for speed
        )
        plan2 = plan.simulate(
            wind_field=StillAirField(),
            n_ensemble=3,
            rng_seed=42,
        )
        release_by_id = {r.release_id: r for r in plan2.releases}
        for t in plan2.trajectories:
            assert t.release is release_by_id[t.release.release_id]
        # Total trajectories = n_ensemble per non-skipped release.
        non_skipped = sum(
            1 for r in plan2.releases if r.qc_release_ok is not False
        )
        assert len(plan2.trajectories) == 3 * non_skipped

    def test_simulate_skips_failed_releases(self):
        # Low-altitude plan with a surface forces qc_min_alt_ok=False.
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(altitude_ft=200.0),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
            surface_elevation_msl=0 * ureg.meter,
        )
        # 200 ft = 60.96 m AGL < default 300 m min → all fail.
        assert all(r.qc_min_alt_ok is False for r in plan.releases)
        with pytest.warns(UserWarning, match="qc_release_ok=False"):
            plan2 = plan.simulate(wind_field=StillAirField())
        assert len(plan2.trajectories) == 0
        # Releases are still present, unchanged.
        assert len(plan2.releases) == len(plan.releases)

    def test_labelled_index_works(self):
        plan_gdf = _simple_plan_gdf()
        plan_gdf.index = ["leg-a"]
        plan = DropsondePlan.from_flight_plan(
            plan_gdf,
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        plan2 = plan.simulate(wind_field=StillAirField())
        assert len(plan2.trajectories) > 0
        assert all(r.source_id == "leg-a" for r in plan2.releases)

    def test_target_polygon_lifecycle_with_ensemble(self):
        # Polygon centred on splash mean → True; far away → False.
        plan_gdf = _simple_plan_gdf()
        plan_centred = DropsondePlan.from_flight_plan(
            plan_gdf,
            takeoff_time=_takeoff(),
            spacing=80 * ureg.nautical_mile,
            target_polygon=shapely.geometry.box(-80.5, 29.5, -75.5, 30.5),
        )
        # Pre-sim: all NA.
        assert all(r.qc_splash_in_target_polygon is None for r in plan_centred.releases)
        plan2 = plan_centred.simulate(
            wind_field=StillAirField(),
            n_ensemble=5, rng_seed=7,
        )
        # Polygon spans the flight line; every splash falls inside.
        assert all(
            r.qc_splash_in_target_polygon is True for r in plan2.releases
        )
        # Polygon far away → all False.
        plan_far = DropsondePlan.from_flight_plan(
            plan_gdf,
            takeoff_time=_takeoff(),
            spacing=80 * ureg.nautical_mile,
            target_polygon=shapely.geometry.box(0, 0, 1, 1),
        )
        plan_far_sim = plan_far.simulate(
            wind_field=StillAirField(), n_ensemble=5, rng_seed=7,
        )
        assert all(
            r.qc_splash_in_target_polygon is False for r in plan_far_sim.releases
        )

    def test_manifest_includes_provenance_columns(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        df = plan.to_manifest_gdf()
        for col in (
            "release_id", "source_id", "source_pattern_id",
            "qc_min_alt_ok", "qc_aircraft_envelope_ok",
            "qc_segment_allowed", "qc_release_ok",
            "qc_splash_in_target_polygon",
        ):
            assert col in df.columns

    def test_summary_ensemble_polygon_columns(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=80 * ureg.nautical_mile,
            target_polygon=shapely.geometry.box(-80.5, 29.5, -75.5, 30.5),
        ).simulate(wind_field=StillAirField(), n_ensemble=4, rng_seed=1)
        summary = plan.summary()
        assert "n_members_in_polygon" in summary.columns
        assert "fraction_in_polygon" in summary.columns

    def test_plan_is_frozen(self):
        plan = DropsondePlan.from_flight_plan(
            _simple_plan_gdf(),
            takeoff_time=_takeoff(),
            spacing=40 * ureg.nautical_mile,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.releases = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DropsondePlan.from_pattern
# ---------------------------------------------------------------------------


class TestDropsondePlanFromPattern:
    def _two_line_pattern(self) -> Pattern:
        from hyplan.flight_patterns import racetrack
        return racetrack(
            center=(30.0, -80.0),
            heading=90.0,
            altitude=25000 * 0.3048 * ureg.meter,
            leg_length=100 * ureg.kilometer,
            n_legs=2,
            offset=20 * ureg.kilometer,
        )

    def test_line_based_basic(self):
        pat = self._two_line_pattern()
        # spacing_time avoids needing a groundspeed.
        plan = DropsondePlan.from_pattern(pat, spacing_time=120 * ureg.second)
        assert plan.flight_track is None
        assert len(plan.releases) > 0
        # pattern_id may be empty string for racetrack default; check
        # at least one release carries it.
        assert any(r.source_id is not None for r in plan.releases)

    def test_waypoint_based_returns_empty_with_warning(self):
        from hyplan.flight_patterns import sawtooth
        pat = sawtooth(
            center=(30.0, -80.0),
            heading=90.0,
            altitude_min=10000 * ureg.meter,
            altitude_max=12000 * ureg.meter,
            leg_length=100 * ureg.kilometer,
            n_cycles=2,
        )
        with pytest.warns(UserWarning, match="line-based"):
            plan = DropsondePlan.from_pattern(pat, spacing_time=60 * ureg.second)
        assert len(plan.releases) == 0

    def test_multi_line_duplicate_timestamp_warning(self):
        # release_time is only populated when groundspeed is resolvable;
        # supply an aircraft so the warning has timestamps to inspect.
        from hyplan.aircraft import NASA_P3
        pat = self._two_line_pattern()
        with pytest.warns(UserWarning, match="line-to-line transit"):
            plan = DropsondePlan.from_pattern(
                pat,
                spacing_time=180 * ureg.second,
                takeoff_time=_takeoff(),
                aircraft=NASA_P3(),
            )
        # First release on each line shares takeoff_time + start_elapsed=0.
        times = [r.release_time for r in plan.releases if r.release_time]
        assert len(set(times)) < len(times)  # at least one duplicate


# ---------------------------------------------------------------------------
# Inverse targeting
# ---------------------------------------------------------------------------


class TestInverseTargeting:
    def test_requires_flight_track_on_plan(self):
        from hyplan.flight_patterns import racetrack
        pat = racetrack(
            center=(30.0, -80.0),
            heading=90.0,
            altitude=25000 * 0.3048 * ureg.meter,
            leg_length=100 * ureg.kilometer,
            n_legs=2,
            offset=20 * ureg.kilometer,
        )
        plan = DropsondePlan.from_pattern(pat, spacing_time=120 * ureg.second)
        with pytest.raises(HyPlanValueError, match="flight_track"):
            solve_release_for_target(
                target=Waypoint(30.0, -78.0, heading=0.0),
                flight_plan=plan,
                takeoff_time=_takeoff(),
                wind_field=StillAirField(),
            )

    def test_solver_with_flight_track(self):
        track = FlightPlanTrack.from_compute_flight_plan(_simple_plan_gdf())
        # Target in the middle of the flight line, slightly offset for wind drift.
        target = Waypoint(30.0, -78.0, heading=0.0)
        sol = solve_release_for_target(
            target=target,
            flight_plan=track,
            takeoff_time=_takeoff(),
            wind_field=StillAirField(),
            tolerance=10_000 * ureg.meter,  # generous (deployment-transient drift)
            coarse_step=30 * ureg.second,
        )
        assert sol.feasible is True or float(sol.miss_distance.m_as("meter")) < 10_000

    def test_solver_reports_infeasible_for_distant_target(self):
        track = FlightPlanTrack.from_compute_flight_plan(_simple_plan_gdf())
        target = Waypoint(0.0, 0.0, heading=0.0)  # nowhere near the line
        sol = solve_release_for_target(
            target=target,
            flight_plan=track,
            takeoff_time=_takeoff(),
            wind_field=StillAirField(),
            tolerance=100 * ureg.meter,
            coarse_step=60 * ureg.second,
        )
        assert sol.feasible is False
        assert sol.reason is not None


# ---------------------------------------------------------------------------
# Golden-section helper
# ---------------------------------------------------------------------------


class TestGoldenSection:
    def test_recovers_quadratic_minimum(self):
        x = golden_section_search(lambda v: (v - 0.37) ** 2, 0.0, 1.0, tol=1e-6)
        assert abs(x - 0.37) < 1e-3

    def test_bounded(self):
        # Minimum at 1.5 (outside [-1, 1]); should clamp to right edge.
        x = golden_section_search(lambda v: (v - 1.5) ** 2, -1.0, 1.0, tol=1e-6)
        assert abs(x - 1.0) < 1e-3
