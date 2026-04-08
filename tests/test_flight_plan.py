"""Tests for hyplan.flight_plan."""

import pytest
import geopandas as gpd
from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.aircraft import DynamicAviation_B200
from hyplan.airports import Airport, initialize_data
from hyplan.flight_plan import compute_flight_plan


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture
def b200():
    return DynamicAviation_B200()


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

    def test_crosswind_negligible(self, b200):
        """Pure crosswind should not materially change ground speed under
        the small-crosswind-angle approximation used by _wind_factor."""
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_cw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(30, "knot"),
            wind_direction=90.0,  # wind FROM the east → crosswind on northbound
        )
        assert plan_cw["time_to_segment"].sum() == pytest.approx(
            plan_calm["time_to_segment"].sum(), rel=1e-6
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
