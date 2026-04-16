"""Tests for hyplan.flight_optimizer."""

import pytest
import networkx as nx
from hyplan.units import ureg
from hyplan.flight_line import FlightLine
from hyplan.aircraft import KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.flight_optimizer import build_graph, greedy_optimize, _opposite_endpoint
from hyplan.exceptions import HyPlanValueError


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture
def b200():
    return KingAirB200()


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


class TestOppositeEndpoint:
    def test_start_to_end(self):
        assert _opposite_endpoint("Line_0_start") == "Line_0_end"

    def test_end_to_start(self):
        assert _opposite_endpoint("Line_0_end") == "Line_0_start"

    def test_invalid_raises(self):
        with pytest.raises(HyPlanValueError, match="not a flight line endpoint"):
            _opposite_endpoint("Line_0_middle")


class TestBuildGraphStructure:
    """Verify graph structure details: node types, edge types, and line_keys."""

    def test_flight_line_endpoint_nodes(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for fl in flight_lines:
            start_node = f"{fl.site_name}_start"
            end_node = f"{fl.site_name}_end"
            assert start_node in G.nodes, f"Missing node {start_node}"
            assert end_node in G.nodes, f"Missing node {end_node}"
            assert G.nodes[start_node]["nodetype"] == "flight_line_endpoint"
            assert G.nodes[end_node]["nodetype"] == "flight_line_endpoint"

    def test_flight_line_edges_both_directions(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for fl in flight_lines:
            start_node = f"{fl.site_name}_start"
            end_node = f"{fl.site_name}_end"
            assert G.has_edge(start_node, end_node)
            assert G.has_edge(end_node, start_node)
            assert G[start_node][end_node]["edgetype"] == "flight_line"
            assert G[end_node][start_node]["edgetype"] == "flight_line"

    def test_departure_edges_from_airports(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for apt in airports:
            departure_edges = [
                (u, v) for u, v, d in G.edges(data=True)
                if u == apt.icao_code and d.get("edgetype") == "departure"
            ]
            assert len(departure_edges) > 0, f"No departure edges from {apt.icao_code}"

    def test_return_edges_to_airports(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        for apt in airports:
            return_edges = [
                (u, v) for u, v, d in G.edges(data=True)
                if v == apt.icao_code and d.get("edgetype") == "return"
            ]
            assert len(return_edges) > 0, f"No return edges to {apt.icao_code}"

    def test_line_keys_stored_on_graph(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        assert "line_keys" in G.graph
        line_keys = G.graph["line_keys"]
        assert len(line_keys) == len(flight_lines)

    def test_transit_edges_between_airports(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        # With 2 airports, there should be transit edges in both directions
        icao1, icao2 = airports[0].icao_code, airports[1].icao_code
        transit_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edgetype") == "transit" and u in (icao1, icao2) and v in (icao1, icao2)
        ]
        assert len(transit_edges) == 2  # one in each direction


class TestGreedyOptimizeSingleLine:
    """Edge case: optimize with a single flight line."""

    def test_single_line_covered(self, b200, airports):
        single_line = FlightLine.start_length_azimuth(
            lat1=34.4, lon1=-119.8,
            length=ureg.Quantity(10000, "meter"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Single",
        )
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[single_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
        )
        assert result["lines_covered"] == 1
        assert len(result["flight_sequence"]) == 1
        assert result["total_time"] > 0

    def test_single_line_route_starts_and_ends_at_airport(self, b200, airports):
        single_line = FlightLine.start_length_azimuth(
            lat1=34.4, lon1=-119.8,
            length=ureg.Quantity(10000, "meter"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Single",
        )
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[single_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
        )
        route = result["route"]
        assert route[0] == airports[0].icao_code
        assert route[-1] == airports[0].icao_code


class TestGreedyOptimizeEndurance:
    """Test endurance constraints and refueling behavior."""

    def test_tight_endurance_skips_lines(self, b200, airports):
        """With impossibly tight endurance, lines should be skipped."""
        long_lines = []
        for i in range(3):
            fl = FlightLine.start_length_azimuth(
                lat1=35.5 + i * 0.2, lon1=-119.8,
                length=ureg.Quantity(50000, "meter"),
                az=90,
                altitude_msl=ureg.Quantity(20000, "feet"),
                site_name=f"LongLine_{i}",
            )
            long_lines.append(fl)
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=long_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=0.5,
            max_daily_flight_time=10.0,
            max_days=3,
        )
        # With 0.5h max endurance, lines far away should be unreachable
        assert len(result["lines_skipped"]) > 0
        assert len(result["flight_sequence"]) < len(long_lines)

    def test_return_airport_defaults_to_takeoff(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
        )
        assert result["return_airport"] is airports[0]

    def test_all_lines_covered_with_generous_endurance(self, b200, flight_lines, airports):
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=10.0,
        )
        assert result["lines_covered"] == len(flight_lines)
        assert len(result["lines_skipped"]) == 0


class TestBuildGraphDuplicateNames:
    """Test that build_graph handles duplicate site_name values."""

    def test_duplicate_site_names_disambiguated(self, b200, airports):
        lines = []
        for i in range(3):
            fl = FlightLine.start_length_azimuth(
                lat1=34.4 + i * 0.05, lon1=-119.8,
                length=ureg.Quantity(10000, "meter"),
                az=90,
                altitude_msl=ureg.Quantity(20000, "feet"),
                site_name="DupName",
            )
            lines.append(fl)
        G = build_graph(b200, lines, airports)
        line_keys = G.graph["line_keys"]
        # All keys should be unique even though site_name is the same
        keys = list(line_keys.values())
        assert len(set(keys)) == 3
