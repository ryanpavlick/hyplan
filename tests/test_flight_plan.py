"""Tests for hyplan.flight_plan."""

import pytest
import geopandas as gpd
from hyplan.units import ureg
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
