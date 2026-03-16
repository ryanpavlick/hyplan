"""Tests for hyplan.flight_optimizer."""

import pytest
import networkx as nx
from hyplan.units import ureg
from hyplan.flight_line import FlightLine
from hyplan.aircraft import DynamicAviation_B200
from hyplan.airports import Airport, initialize_data
from hyplan.flight_optimizer import build_graph, greedy_optimize


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture
def b200():
    return DynamicAviation_B200()


@pytest.fixture
def flight_lines():
    """Three short flight lines near Santa Barbara."""
    lines = []
    for i, az in enumerate([0, 45, 90]):
        fl = FlightLine.start_length_azimuth(
            lat1=34.4 + i * 0.05,
            lon1=-119.8,
            length=ureg.Quantity(20000, "meter"),
            az=az,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name=f"Line_{i}",
        )
        lines.append(fl)
    return lines


@pytest.fixture
def airports():
    return [Airport("KSBA"), Airport("KBUR")]


class TestBuildGraph:
    def test_returns_digraph(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_contains_airport_nodes(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for apt in airports:
            assert apt.icao_code in G.nodes

    def test_edges_have_weight(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert data["weight"] > 0


class TestGreedyOptimize:
    def test_basic_optimization(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
        )
        assert isinstance(result, dict)
        assert "lines_covered" in result
        assert "total_time" in result
        assert result["lines_covered"] > 0

    def test_with_endurance(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        assert result["total_time"] <= 4.0 or result["refuel_stops"] > 0

    def test_multi_day(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
            max_daily_flight_time=8.0,
            max_days=3,
        )
        assert "days_used" in result
        assert "daily_times" in result
        assert result["days_used"] >= 1

    def test_result_keys(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
        )
        expected_keys = {
            "flight_sequence", "route", "total_time", "daily_times",
            "lines_covered", "lines_skipped", "refuel_stops",
            "days_used", "takeoff_airport", "return_airport", "graph",
        }
        assert expected_keys.issubset(result.keys())
