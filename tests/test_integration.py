"""End-to-end integration test for the HyPlan flight planning workflow.

Exercises the full pipeline: flight box generation -> optimization ->
flight plan computation -> multi-format export.
"""

import os

import pytest
from shapely.geometry import Polygon

from hyplan import (
    AVIRIS3,
    KingAirB200,
    box_around_polygon,
    compute_flight_plan,
    greedy_optimize,
    to_foreflight_csv,
    to_gpx,
    to_kml,
    ureg,
)
from hyplan.airports import Airport, initialize_data


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture(scope="module")
def aircraft():
    return KingAirB200()


@pytest.fixture(scope="module")
def sensor():
    return AVIRIS3()


@pytest.fixture(scope="module")
def study_polygon():
    """A small rectangular study area near Santa Barbara, CA."""
    return Polygon([
        (-119.85, 34.40),
        (-119.75, 34.40),
        (-119.75, 34.48),
        (-119.85, 34.48),
        (-119.85, 34.40),
    ])


@pytest.fixture(scope="module")
def flight_lines(sensor, study_polygon):
    """Generate flight lines from the study polygon."""
    altitude = ureg.Quantity(20000, "feet")
    lines = box_around_polygon(
        instrument=sensor,
        altitude_msl=altitude,
        polygon=study_polygon,
        azimuth=0.0,
        box_name="INT",
        overlap=20,
    )
    return lines


@pytest.fixture(scope="module")
def airports():
    """Airports for optimization and flight plan."""
    ksba = Airport("KSBA")
    return ksba


@pytest.fixture(scope="module")
def optimized_result(aircraft, flight_lines, airports):
    """Run greedy optimization on the generated flight lines."""
    result = greedy_optimize(
        aircraft=aircraft,
        flight_lines=flight_lines,
        airports=[airports],
        takeoff_airport=airports,
        return_airport=airports,
    )
    return result


@pytest.fixture(scope="module")
def flight_plan(aircraft, optimized_result, airports):
    """Compute a full flight plan from the optimized sequence."""
    plan = compute_flight_plan(
        aircraft=aircraft,
        flight_sequence=optimized_result["flight_sequence"],
        takeoff_airport=airports,
        return_airport=airports,
    )
    return plan


class TestFlightBoxGeneration:
    """Verify that box_around_polygon produces valid flight lines."""

    def test_produces_flight_lines(self, flight_lines):
        assert len(flight_lines) > 0

    def test_flight_lines_have_site_names(self, flight_lines):
        for fl in flight_lines:
            assert fl.site_name is not None
            assert fl.site_name.startswith("INT")


class TestFlightOptimization:
    """Verify that the optimizer produces a valid result."""

    def test_all_lines_covered(self, optimized_result, flight_lines):
        assert optimized_result["lines_covered"] == len(flight_lines)

    def test_no_lines_skipped(self, optimized_result):
        assert len(optimized_result["lines_skipped"]) == 0

    def test_flight_sequence_non_empty(self, optimized_result):
        assert len(optimized_result["flight_sequence"]) > 0

    def test_total_time_positive(self, optimized_result):
        assert optimized_result["total_time"] > 0


class TestFlightPlan:
    """Verify that compute_flight_plan produces a well-formed GeoDataFrame."""

    def test_is_geodataframe(self, flight_plan):
        import geopandas as gpd
        assert isinstance(flight_plan, gpd.GeoDataFrame)

    def test_non_empty(self, flight_plan):
        assert len(flight_plan) > 0

    def test_has_required_columns(self, flight_plan):
        for col in ["segment_type", "segment_name", "distance", "time_to_segment"]:
            assert col in flight_plan.columns

    def test_has_data_collection_segments(self, flight_plan):
        segment_types = flight_plan["segment_type"].unique()
        assert "flight_line" in segment_types

    def test_total_distance_positive(self, flight_plan):
        assert flight_plan["distance"].sum() > 0

    def test_total_time_positive(self, flight_plan):
        assert flight_plan["time_to_segment"].sum() > 0


class TestExports:
    """Verify that exports produce non-empty, well-formed files."""

    def test_to_kml(self, flight_plan, tmp_path):
        path = str(tmp_path / "integration.kml")
        to_kml(flight_plan, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        with open(path) as f:
            content = f.read()
        assert "<kml" in content or "<Placemark" in content

    def test_to_gpx(self, flight_plan, tmp_path):
        path = str(tmp_path / "integration.gpx")
        to_gpx(flight_plan, path, mission_name="INTEGRATION_TEST")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        with open(path) as f:
            content = f.read()
        assert "<gpx" in content
        assert "<rte>" in content

    def test_to_foreflight_csv(self, flight_plan, tmp_path):
        path = str(tmp_path / "integration_FF.csv")
        to_foreflight_csv(flight_plan, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "Waypoint,Description,LAT,LONG"
        assert len(lines) > 1
