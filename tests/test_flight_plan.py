"""Tests for hyplan.flight_plan."""
# ruff: noqa: E402

import numpy as np
import pytest
import geopandas as gpd
from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.aircraft import KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_plan import (
    compute_flight_plan,
    _track_hold_solution_from_uv,
)


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture
def b200():
    return KingAirB200()


@pytest.fixture
def flight_line():
    return FlightLine.start_length_azimuth(
        lat1=34.05, lon1=-118.25,
        length=ureg.Quantity(50000, "meter"),
        az=45.0,
        altitude_msl=ureg.Quantity(20000, "feet"),
        site_name="Test Line",
    )


class TestComputeFlightPlan:
    def test_returns_geodataframe(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0

    def test_has_required_columns(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        for col in ["segment_type", "segment_name", "distance", "time_to_segment"]:
            assert col in plan.columns

    def test_segment_types(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        segment_types = plan["segment_type"].unique()
        # Should contain at least takeoff, data collection, and landing segments
        assert len(segment_types) >= 2

    def test_no_airports(self, b200, flight_line):
        # Should work without airports (just flight lines)
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
        )
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0

    def test_waypoint_loiter(self, b200):
        """Waypoint with delay produces a loiter segment."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                       delay=ureg.Quantity(5, "minute"))
        wp2 = Waypoint(34.1, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        loiter = plan[plan["segment_type"] == "loiter"]
        assert len(loiter) == 1
        assert loiter.iloc[0]["time_to_segment"] == pytest.approx(5.0)
        assert loiter.iloc[0]["distance"] == 0.0

    def test_waypoint_speed_override(self, b200):
        """Per-waypoint speed override is used for the departing leg."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1")
        wp2 = Waypoint(34.1, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")

        # Without speed override
        plan_default = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        time_default = plan_default["time_to_segment"].sum()

        # With slower speed override
        wp1_slow = Waypoint(34.0, -118.0, 0.0,
                            altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                            speed=ureg.Quantity(50, "knot"))
        plan_slow = compute_flight_plan(aircraft=b200, flight_sequence=[wp1_slow, wp2])
        time_slow = plan_slow["time_to_segment"].sum()

        # Slower speed should take longer
        assert time_slow > time_default

    def test_waypoint_segment_type(self, b200):
        """Waypoint segment_type overrides 'transit' label."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                       segment_type="pattern")
        wp2 = Waypoint(34.01, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        # The level leg from WP1 to WP2 should be labeled "pattern"
        pattern_segs = plan[plan["segment_type"] == "pattern"]
        assert len(pattern_segs) >= 1


class TestWindCorrectedTransit:
    """Wind-aware transit time via compute_flight_plan wind_speed/wind_direction."""

    def _north_leg(self):
        wp1 = Waypoint(34.00, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1")
        wp2 = Waypoint(34.50, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        return wp1, wp2

    def test_zero_wind_regression(self, b200):
        """Explicit zero wind must reproduce the default (wind=None) result."""
        wp1, wp2 = self._north_leg()
        plan_default = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_zero = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(0, "knot"),
            wind_direction=0.0,
        )
        assert plan_zero["time_to_segment"].sum() == pytest.approx(
            plan_default["time_to_segment"].sum()
        )

    def test_headwind_slows_transit(self, b200):
        """Northbound leg into a due-north headwind should take longer."""
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_hw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(50, "knot"),
            wind_direction=0.0,  # wind FROM the north → headwind on a northbound leg
        )
        assert plan_hw["time_to_segment"].sum() > plan_calm["time_to_segment"].sum()

    def test_tailwind_speeds_transit(self, b200):
        """Northbound leg with a due-south tailwind should take less time."""
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_tw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(50, "knot"),
            wind_direction=180.0,  # wind FROM the south → tailwind northbound
        )
        assert plan_tw["time_to_segment"].sum() < plan_calm["time_to_segment"].sum()

    def test_crosswind_small_effect(self, b200):
        """Pure crosswind has a small but non-zero effect on total time.

        With trochoidal Dubins path planning, crosswind affects the turn
        geometry slightly, so we allow up to 2% deviation from still air.
        """
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_cw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(30, "knot"),
            wind_direction=90.0,  # wind FROM the east → crosswind on northbound
        )
        assert plan_cw["time_to_segment"].sum() == pytest.approx(
            plan_calm["time_to_segment"].sum(), rel=0.02
        )

    def test_headwind_magnitude_matches_hand_calc(self, b200):
        """With a 30 kt pure headwind, flight-line time should grow by
        TAS/(TAS-30)."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,  # due north
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="North Line",
        )
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[fl])
        plan_hw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[fl],
            wind_speed=ureg.Quantity(30, "knot"),
            wind_direction=0.0,
        )
        row_calm = plan_calm[plan_calm["segment_type"] == "flight_line"].iloc[0]
        row_hw = plan_hw[plan_hw["segment_type"] == "flight_line"].iloc[0]
        tas_kt = b200.cruise_speed_at(ureg.Quantity(20000, "feet")).m_as("knot")
        expected_factor = tas_kt / (tas_kt - 30.0)
        assert row_hw["time_to_segment"] / row_calm["time_to_segment"] == pytest.approx(
            expected_factor, rel=1e-3
        )

    def test_wind_direction_required_with_nonzero_wind(self, b200):
        wp1, wp2 = self._north_leg()
        with pytest.raises(Exception):
            compute_flight_plan(
                aircraft=b200,
                flight_sequence=[wp1, wp2],
                wind_speed=ureg.Quantity(30, "knot"),
                # wind_direction omitted on purpose
            )

    def test_unflyable_headwind_raises(self, b200):
        """A headwind larger than TAS must raise — ground speed would go negative."""
        wp1, wp2 = self._north_leg()
        with pytest.raises(Exception):
            compute_flight_plan(
                aircraft=b200,
                flight_sequence=[wp1, wp2],
                wind_speed=ureg.Quantity(1000, "knot"),
                wind_direction=0.0,
            )


class TestTrackHoldSolution:
    """Test the crab-angle-aware track-hold wind solver."""

    def _zero_wind(self):
        return 0.0 * ureg.meter / ureg.second

    def test_no_wind(self):
        """No wind: crab=0, heading=track, groundspeed=TAS."""
        tas = 250 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 90.0, self._zero_wind(), self._zero_wind(),
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["heading_deg"] == pytest.approx(90.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(250.0, rel=0.01)

    def test_pure_tailwind(self):
        """Tailwind along track: crab=0, GS > TAS."""
        tas = 250 * ureg.knot
        # Track north, wind from south (tailwind)
        # Wind from south = v positive (northward component)
        v_tail = 30 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, self._zero_wind(), v_tail,
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) > 250.0
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(280.0, rel=0.01)

    def test_pure_headwind(self):
        """Headwind along track: crab=0, GS < TAS."""
        tas = 250 * ureg.knot
        # Track north, wind from north (headwind) = v negative
        v_head = -30 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, self._zero_wind(), v_head,
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) < 250.0
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(220.0, rel=0.01)

    def test_pure_crosswind(self):
        """Crosswind: nonzero crab, GS = TAS*cos(crab)."""
        tas = 250 * ureg.knot
        # Track north, wind from west (eastward u component)
        u_cross = 50 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, u_cross, self._zero_wind(),
        )
        assert abs(sol["crab_angle_deg"]) > 1.0
        # GS should be TAS*cos(crab) (no along-track wind component)
        expected_gs = 250.0 * np.cos(np.radians(sol["crab_angle_deg"]))
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(expected_gs, rel=0.01)

    def test_crosswind_exceeds_tas_raises(self):
        """Crosswind > TAS: cannot hold track."""
        tas = 100 * ureg.knot
        u_huge = 200 * ureg.knot
        with pytest.raises(HyPlanValueError, match="Crosswind"):
            _track_hold_solution_from_uv(
                tas, 0.0, u_huge, self._zero_wind(),
            )

    def test_crab_sign_convention(self):
        """Crosswind from the right requires left (negative) crab."""
        tas = 250 * ureg.knot
        # Track north, wind from east (u negative = westward)
        # Crosswind = u*cos(track) - v*sin(track) = u*cos(0) = u (negative)
        # crab = asin(-crosswind/TAS) = asin(positive) = positive
        # Actually: wind from east means u < 0 (westward component)
        u_from_east = -50 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, u_from_east, self._zero_wind(),
        )
        # Aircraft must crab right (positive) to compensate westward drift
        # crosswind = u*cos(0) - v*sin(0) = u = -50 kt (negative)
        # crab = asin(-crosswind/TAS) = asin(50/250) > 0
        assert sol["crab_angle_deg"] > 0


# ---------------------------------------------------------------------------
# End-to-end planner regression tests
# ---------------------------------------------------------------------------

from hyplan.winds import ConstantWindField
from hyplan.flight_patterns import racetrack


class TestPlannerRegression:
    """End-to-end regression tests for compute_flight_plan."""

    def test_single_line_no_wind(self, b200):
        """Single flight line, no wind: basic sanity."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="North Line",
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[fl])
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0
        assert "flight_line" in plan["segment_type"].values
        assert plan["time_to_segment"].sum() > 0
        assert plan["distance"].dropna().sum() > 0

    def test_multi_line_racetrack(self, b200):
        """Racetrack pattern: all flight lines present with transitions."""
        pat = racetrack(
            center=(34.05, -118.25),
            heading=0.0,
            altitude=ureg.Quantity(20000, "feet"),
            leg_length=ureg.Quantity(50000, "meter"),
            n_legs=4,
            offset=ureg.Quantity(5000, "meter"),
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[pat])
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) >= 4
        seg_types = set(plan["segment_type"].values)
        assert "pattern" in seg_types or "flight_line" in seg_types

    def test_airport_departure_and_return(self, b200):
        """Takeoff and approach phases appear with correct ordering."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.45, lon1=-119.85,
            length=ureg.Quantity(30000, "meter"),
            az=90.0,
            altitude_msl=ureg.Quantity(15000, "feet"),
            site_name="SBA Line",
        )
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[fl],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        seg_types = list(plan["segment_type"].values)
        # First segment should be takeoff or climb
        assert seg_types[0] in ("takeoff", "climb")
        # Last segment should be approach or descent
        assert seg_types[-1] in ("approach", "descent")
        # Flight line should be somewhere in the middle
        assert "flight_line" in seg_types

    def test_constant_wind_populates_fields(self, b200):
        """Constant wind: crab angle, groundspeed, tailwind fields populated."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Wind Test Line",
        )
        wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=270.0)
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[fl], wind_source=wf,
        )
        fl_row = plan[plan["segment_type"] == "flight_line"].iloc[0]
        # Wind fields should be present and non-NaN
        assert "crab_angle_deg" in fl_row.index
        assert "groundspeed_kts" in fl_row.index
        assert "tailwind_kts" in fl_row.index
        assert "crosswind_kts" in fl_row.index
        # With westerly wind on a northbound line, crosswind should be nonzero
        assert abs(fl_row["crosswind_kts"]) > 0.1
        # Groundspeed should be positive and reasonable
        assert fl_row["groundspeed_kts"] > 50
