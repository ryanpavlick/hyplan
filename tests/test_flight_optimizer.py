"""Tests for hyplan.flight_optimizer."""

import itertools

import networkx as nx
import pytest

from hyplan.aircraft import KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_line import FlightLine
from hyplan.flight_optimizer import (
    _opposite_endpoint,
    _pattern_internal_time,
    _transit_time,
    build_graph,
    greedy_optimize,
)
from hyplan.flight_patterns import racetrack, sawtooth, spiral
from hyplan.pattern import Pattern
from hyplan.units import ureg
from hyplan.waypoint import Waypoint


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
        for _u, _v, data in G.edges(data=True):
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
            "lines_covered", "lines_skipped",
            "items_covered", "items_skipped",
            "refuel_stops",
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
    """Verify graph structure details: node types, edge types, and item_keys."""

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

    def test_item_keys_stored_on_graph(self, b200, flight_lines, airports):
        G = build_graph(b200, flight_lines, airports)
        assert "item_keys" in G.graph
        item_keys = G.graph["item_keys"]
        # item_keys is a list[(item, key)] — see build_graph for rationale.
        assert len(item_keys) == len(flight_lines)
        assert all(item is fl for (item, _key), fl in zip(item_keys, flight_lines, strict=False))

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


class TestRefuelDailyBudget:
    """Refueling is only inserted when it enables an item within BOTH the
    endurance and daily flight time budgets."""

    def test_no_phantom_refuels_when_daily_budget_blocks_everything(self, b200, airports):
        """Regression: a line that fits endurance but not the daily budget
        must not trigger repeated in-place refueling that burns whole days."""
        far_line = FlightLine.start_length_azimuth(
            lat1=39.5, lon1=-122.0,
            length=ureg.Quantity(150, "kilometer"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="FAR_150KM",
        )
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[far_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=6.0,
            max_daily_flight_time=1.0,
            max_days=3,
        )
        # Sanity: the sortie fits within endurance, so only the daily
        # budget makes it infeasible.
        G = result["graph"]
        sortie = (
            G["KSBA"]["FAR_150KM_start"]["weight"]
            + G["FAR_150KM_start"]["FAR_150KM_end"]["weight"]
            + G["FAR_150KM_end"]["KSBA"]["weight"]
            + 0.25
        )
        assert sortie <= 6.0
        assert result["refuel_stops"] == []
        assert result["items_covered"] == 0
        assert result["items_skipped"] == ["FAR_150KM"]
        # The day loop terminates after the first zero-progress day instead
        # of padding the result with max_days of phantom refueling.
        assert result["days_used"] < 3
        assert result["daily_times"] == []
        assert result["total_time"] == 0.0

    def test_legitimate_refueling_still_occurs(self, b200, flight_lines, airports):
        """With endurance tight enough to need a refuel but a generous daily
        budget, the optimizer still refuels and covers all lines."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=1.0,
            max_daily_flight_time=8.0,
            max_days=1,
        )
        assert result["items_covered"] == len(flight_lines)
        assert len(result["refuel_stops"]) >= 1
        assert result["days_used"] == 1


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
        item_keys = G.graph["item_keys"]
        # All keys should be unique even though site_name is the same.
        # item_keys is a list[(item, key)].
        keys = [k for (_item, k) in item_keys]
        assert len(set(keys)) == 3


# ---------------------------------------------------------------------------
# Pattern atomicity tests
# ---------------------------------------------------------------------------


@pytest.fixture
def racetrack_pattern():
    """A short three-leg racetrack near Santa Barbara at 8000 ft."""
    return racetrack(
        center=(34.4, -119.8),
        heading=90.0,
        altitude=ureg.Quantity(8_000, "foot"),
        leg_length=ureg.Quantity(15, "kilometer"),
        n_legs=3,
        offset=ureg.Quantity(2, "kilometer"),
        name="SBA_RT",
    )


@pytest.fixture
def free_lines_far_and_near():
    """Two free FlightLines: one near KSBA, one further south near KBUR."""
    fl_near = FlightLine.start_length_azimuth(
        lat1=34.5, lon1=-119.9,
        length=ureg.Quantity(15, "kilometer"), az=0.0,
        altitude_msl=ureg.Quantity(8_000, "foot"),
        site_name="FL_NEAR",
    )
    fl_far = FlightLine.start_length_azimuth(
        lat1=34.2, lon1=-118.5,
        length=ureg.Quantity(15, "kilometer"), az=0.0,
        altitude_msl=ureg.Quantity(8_000, "foot"),
        site_name="FL_FAR",
    )
    return fl_near, fl_far


class TestPatternAtomicity:
    """Pattern is atomic in greedy_optimize — never split apart."""

    def test_pattern_stays_atomic_in_output_order(self, b200, racetrack_pattern, free_lines_far_and_near, airports):
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, racetrack_pattern, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        seq = result["flight_sequence"]
        # Pattern appears exactly once in the output, as a Pattern instance —
        # the racetrack's three internal lines are NOT in the output as
        # separate FlightLines.
        patterns_in_seq = [x for x in seq if isinstance(x, Pattern)]
        assert len(patterns_in_seq) == 1
        assert patterns_in_seq[0] is racetrack_pattern
        # Free flight lines appear separately, not interleaved with pattern internals.
        free_lines = [x for x in seq if isinstance(x, FlightLine)]
        free_names = {x.site_name for x in free_lines}
        assert "FL_NEAR" in free_names
        assert "FL_FAR" in free_names
        # No FlightLine in the output should have come from inside the racetrack.
        racetrack_line_ids = set(racetrack_pattern.line_ids)
        for fl in free_lines:
            assert fl.site_name not in racetrack_line_ids

    def test_pattern_cost_uses_full_internal_time(self, b200, racetrack_pattern, airports):
        """The graph's pattern along-edge weight equals _pattern_internal_time."""
        G = build_graph(b200, [racetrack_pattern], airports)
        item_keys = G.graph["item_keys"]
        _, key = item_keys[0]
        edge_weight = G[f"{key}_start"][f"{key}_end"]["weight"]
        expected = _pattern_internal_time(b200, racetrack_pattern)
        # Weight is the full internal traversal time, not a great-circle
        # distance between entry and exit — sanity-check by confirming the
        # weight is well above the entry->exit transit estimate alone.
        endpoint_only = _transit_time(b200, racetrack_pattern.entry_waypoint, racetrack_pattern.exit_waypoint)
        assert edge_weight == pytest.approx(expected)
        assert edge_weight > endpoint_only * 1.5  # internal traversal is much longer

    def test_pattern_has_no_reverse_along_edge(self, b200, racetrack_pattern, airports):
        """Pattern atomicity is enforced structurally: no reverse along-edge."""
        G = build_graph(b200, [racetrack_pattern], airports)
        item_keys = G.graph["item_keys"]
        _, key = item_keys[0]
        assert G.has_edge(f"{key}_start", f"{key}_end")  # forward exists
        assert not G.has_edge(f"{key}_end", f"{key}_start")  # reverse does NOT

    def test_pattern_position_can_change_relative_to_lines(self, b200, racetrack_pattern, free_lines_far_and_near, airports):
        """The optimizer reorders patterns vs free flight lines by transit cost.

        Input order is [FL_FAR, pattern_near_origin, FL_NEAR] but starting
        from KSBA the optimizer should visit the near pattern (or FL_NEAR)
        before the far flight line. The point is that the input order is
        not necessarily the output order.
        """
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, racetrack_pattern, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        seq = result["flight_sequence"]
        names = [x.name if isinstance(x, Pattern) else x.site_name for x in seq]
        # FL_FAR should NOT be visited first — pattern or FL_NEAR is closer to KSBA.
        assert names[0] != "FL_FAR"

    def test_pattern_forces_pre_refuel_when_does_not_fit(self, b200, racetrack_pattern, airports):
        """A pattern too big to fit in remaining endurance forces a refuel BEFORE entry, not inside.

        Construct a scenario with a tiny endurance budget such that the
        pattern (~25-30 min internal time) plus surrounding transit can
        only complete after a refuel. Atomicity guarantees the refuel
        appears before the pattern, not in the middle of it: the
        flight_sequence still contains the pattern intact.
        """
        # Modest endurance — long enough to reach the pattern but too short
        # to fly the pattern + return without refueling.
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[racetrack_pattern],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=1.5,
        )
        # The pattern is in the output exactly once and intact.
        seq = result["flight_sequence"]
        patterns_in_seq = [x for x in seq if isinstance(x, Pattern)]
        assert len(patterns_in_seq) == 1
        assert patterns_in_seq[0] is racetrack_pattern

    def test_flightline_only_input_unchanged(self, b200, flight_lines, airports):
        """Passing only FlightLines (no Pattern) yields a flight-line-only output.

        This pins the backward-compat invariant: existing callers that pass
        list[FlightLine] see no Patterns in the output and can keep their
        existing iteration / typing assumptions at runtime.
        """
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        assert all(isinstance(x, FlightLine) for x in result["flight_sequence"])

    def test_waypoint_pattern_stays_atomic(self, b200, free_lines_far_and_near, airports):
        """A waypoint-based pattern (sawtooth) is also atomic in the optimizer."""
        st = sawtooth(
            center=(34.4, -119.8),
            heading=0.0,
            leg_length=ureg.Quantity(15, "kilometer"),
            altitude_min=ureg.Quantity(3_000, "meter"),
            altitude_max=ureg.Quantity(6_000, "meter"),
            n_cycles=2,
        )
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, st, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        seq = result["flight_sequence"]
        patterns_in_seq = [x for x in seq if isinstance(x, Pattern)]
        assert len(patterns_in_seq) == 1
        assert patterns_in_seq[0] is st


class TestPatternEndpoints:
    """Pattern.entry_waypoint and Pattern.exit_waypoint structural API."""

    def test_line_based_entry_is_first_line_waypoint1(self, racetrack_pattern):
        first_line = next(iter(racetrack_pattern.lines.values()))
        assert racetrack_pattern.entry_waypoint is first_line.waypoint1

    def test_line_based_exit_is_last_line_waypoint2(self, racetrack_pattern):
        last_line = next(reversed(racetrack_pattern.lines.values()))
        assert racetrack_pattern.exit_waypoint is last_line.waypoint2

    def test_waypoint_based_entry_is_first_waypoint(self):
        st = sawtooth(
            center=(34.0, -118.0),
            heading=0.0,
            leg_length=ureg.Quantity(20, "kilometer"),
            altitude_min=ureg.Quantity(3_000, "meter"),
            altitude_max=ureg.Quantity(6_000, "meter"),
            n_cycles=2,
        )
        assert st.entry_waypoint is st.waypoints[0]

    def test_waypoint_based_exit_is_last_waypoint(self):
        st = sawtooth(
            center=(34.0, -118.0),
            heading=0.0,
            leg_length=ureg.Quantity(20, "kilometer"),
            altitude_min=ureg.Quantity(3_000, "meter"),
            altitude_max=ureg.Quantity(6_000, "meter"),
            n_cycles=2,
        )
        assert st.exit_waypoint is st.waypoints[-1]


# ---------------------------------------------------------------------------
# Bare Waypoint atomicity tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_waypoint():
    """A bare Waypoint near Santa Barbara at 8000 ft with a 5-minute loiter."""
    return Waypoint(
        latitude=34.45,
        longitude=-119.85,
        heading=0.0,
        altitude_msl=ureg.Quantity(8_000, "foot"),
        delay=ureg.Quantity(5, "minute"),
        name="LOITER_SBA",
    )


class TestWaypointAtomicity:
    """Bare Waypoint is atomic in greedy_optimize — its delay never splits."""

    def test_bare_waypoint_orderable(self, b200, bare_waypoint, free_lines_far_and_near, airports):
        """A bare Waypoint is reorderable relative to free flight lines."""
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, bare_waypoint, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        seq = result["flight_sequence"]
        # The bare Waypoint appears in the output exactly once, as a Waypoint.
        wps_in_seq = [x for x in seq if isinstance(x, Waypoint)]
        assert len(wps_in_seq) == 1
        assert wps_in_seq[0] is bare_waypoint
        # Free flight lines appear separately.
        free_lines = [x for x in seq if isinstance(x, FlightLine)]
        free_names = {x.site_name for x in free_lines}
        assert "FL_NEAR" in free_names
        assert "FL_FAR" in free_names
        # Starting from KSBA, the optimizer should not pick FL_FAR first —
        # both bare_waypoint and FL_NEAR are closer.
        first = seq[0]
        first_name = first.name if isinstance(first, Waypoint) else first.site_name
        assert first_name != "FL_FAR"

    def test_bare_waypoint_zero_internal_time_when_no_delay(self, b200, airports):
        """A Waypoint with delay=None gets a zero-weight forward along-edge."""
        wp = Waypoint(
            latitude=34.45,
            longitude=-119.85,
            heading=0.0,
            altitude_msl=ureg.Quantity(8_000, "foot"),
            name="NO_DELAY_WP",
        )
        assert wp.delay is None
        G = build_graph(b200, [wp], airports)
        item_keys = G.graph["item_keys"]
        _, key = item_keys[0]
        assert G[f"{key}_start"][f"{key}_end"]["weight"] == 0.0

    def test_bare_waypoint_internal_time_equals_delay(self, b200, airports):
        """A Waypoint's along-edge weight equals delay converted to hours."""
        wp = Waypoint(
            latitude=34.45,
            longitude=-119.85,
            heading=0.0,
            altitude_msl=ureg.Quantity(8_000, "foot"),
            delay=ureg.Quantity(10, "minute"),
            name="DELAY_WP",
        )
        G = build_graph(b200, [wp], airports)
        item_keys = G.graph["item_keys"]
        _, key = item_keys[0]
        weight = G[f"{key}_start"][f"{key}_end"]["weight"]
        assert weight == pytest.approx(10.0 / 60.0)

    def test_bare_waypoint_no_reverse_along_edge(self, b200, bare_waypoint, airports):
        """Atomicity is structural: forward along-edge exists, reverse does NOT."""
        G = build_graph(b200, [bare_waypoint], airports)
        item_keys = G.graph["item_keys"]
        _, key = item_keys[0]
        assert G.has_edge(f"{key}_start", f"{key}_end")  # forward exists
        assert not G.has_edge(f"{key}_end", f"{key}_start")  # reverse does NOT

    def test_bare_waypoint_with_long_delay_remains_atomic(self, b200, airports):
        """A Waypoint with a delay long relative to endurance stays intact.

        Mirrors the Pattern atomicity-under-tight-endurance test: regardless
        of when the optimizer schedules the loiter, it cannot be split — the
        single forward along-edge of weight = delay is atomic.
        """
        wp = Waypoint(
            latitude=34.45,
            longitude=-119.85,
            heading=0.0,
            altitude_msl=ureg.Quantity(8_000, "foot"),
            delay=ureg.Quantity(45, "minute"),
            name="LONG_LOITER",
        )
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[wp],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=2.0,
        )
        seq = result["flight_sequence"]
        wps_in_seq = [x for x in seq if isinstance(x, Waypoint)]
        assert len(wps_in_seq) == 1
        assert wps_in_seq[0] is wp

    def test_mixed_sequence_flightline_pattern_waypoint(
        self, b200, racetrack_pattern, free_lines_far_and_near, bare_waypoint, airports
    ):
        """Input combining all three item kinds preserves all atomicity invariants."""
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, racetrack_pattern, bare_waypoint, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        seq = result["flight_sequence"]
        patterns = [x for x in seq if isinstance(x, Pattern)]
        wps = [x for x in seq if isinstance(x, Waypoint)]
        flines = [x for x in seq if isinstance(x, FlightLine)]
        # Each kind appears with the expected count.
        assert len(patterns) == 1
        assert len(wps) == 1
        assert len(flines) == 2
        # Identity is preserved (same object instances flow through).
        assert patterns[0] is racetrack_pattern
        assert wps[0] is bare_waypoint
        free_names = {x.site_name for x in flines}
        assert free_names == {"FL_NEAR", "FL_FAR"}
        # The pattern's internal lines are NOT broken out as separate items.
        racetrack_line_ids = set(racetrack_pattern.line_ids)
        for fl in flines:
            assert fl.site_name not in racetrack_line_ids


# ---------------------------------------------------------------------------
# Result schema: items_covered vs lines_covered semantics
# ---------------------------------------------------------------------------


class TestCoverageCounts:
    """``lines_covered`` keeps its pre-v1.2 line-leg meaning; ``items_covered`` is new."""

    def test_flightline_only_lines_equals_items(self, b200, flight_lines, airports):
        """For all-FlightLine input the two pairs of fields agree."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
        )
        assert result["items_covered"] == len(flight_lines)
        assert result["lines_covered"] == len(flight_lines)
        assert result["items_skipped"] == result["lines_skipped"] == []

    def test_pattern_lines_covered_counts_all_legs(self, b200, racetrack_pattern, airports):
        """A 3-leg line-based Pattern contributes 3 to lines_covered, 1 to items_covered."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[racetrack_pattern],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        # Sanity: the racetrack fixture has 3 internal legs.
        assert len(racetrack_pattern.lines) == 3
        assert result["items_covered"] == 1
        assert result["lines_covered"] == 3

    def test_waypoint_contributes_zero_lines(self, b200, bare_waypoint, airports):
        """A bare Waypoint counts as one item but zero flight-line legs."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[bare_waypoint],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        assert result["items_covered"] == 1
        assert result["lines_covered"] == 0

    def test_mixed_input_counts(self, b200, racetrack_pattern, free_lines_far_and_near, bare_waypoint, airports):
        """Mixed input: items_covered counts each visit item; lines_covered counts legs."""
        fl_near, fl_far = free_lines_far_and_near
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_far, racetrack_pattern, bare_waypoint, fl_near],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
        )
        # 4 visit items: 1 pattern + 1 waypoint + 2 free lines.
        assert result["items_covered"] == 4
        # 5 actual flight-line legs: 3 racetrack legs + 2 free FlightLines.
        assert result["lines_covered"] == 5

    def test_items_covered_never_negative_when_all_items_infeasible(
        self, b200, racetrack_pattern, airports
    ):
        """Regression: items_covered must be 0, not negative, when nothing was flown.

        The previous implementation computed
        ``items_covered = len(visited_items) - len(skipped_items)``, which
        went negative whenever any item was skipped without anything else
        flown.
        """
        # Endurance below takeoff_landing_overhead — nothing can launch.
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[racetrack_pattern],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=0.1,
            takeoff_landing_overhead=0.25,
        )
        assert result["items_covered"] == 0
        assert result["lines_covered"] == 0
        # Pattern is the sole input, so its key is the only entry in items_skipped.
        assert len(result["items_skipped"]) == 1
        assert result["flight_sequence"] == []

    def test_lines_skipped_expands_skipped_line_based_pattern(
        self, b200, racetrack_pattern, airports
    ):
        """A skipped line-based Pattern expands to one ``lines_skipped`` entry per leg."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[racetrack_pattern],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=0.1,
            takeoff_landing_overhead=0.25,
        )
        assert result["lines_covered"] == 0
        # The racetrack has 3 internal legs — each should appear in lines_skipped
        # as "{item_key}:{line_id}".
        assert len(result["lines_skipped"]) == 3
        item_key = result["items_skipped"][0]
        for leg_key, line_id in zip(result["lines_skipped"], racetrack_pattern.line_ids, strict=False):
            assert leg_key.startswith(f"{item_key}:")
            assert leg_key == f"{item_key}:{line_id}"

    def test_waypoint_skipped_contributes_zero_lines_skipped(
        self, b200, bare_waypoint, airports
    ):
        """A skipped bare Waypoint shows up in items_skipped but not in lines_skipped."""
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[bare_waypoint],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=0.1,
            takeoff_landing_overhead=0.25,
        )
        assert result["items_covered"] == 0
        assert result["items_skipped"] == [bare_waypoint.name]
        # No along-line collection -> no lines_skipped contribution.
        assert result["lines_skipped"] == []

    def test_mixed_skips_keep_item_and_line_counts_consistent(
        self, b200, free_lines_far_and_near, airports
    ):
        """Mixed scheduled + skipped input keeps the four counts mutually consistent.

        One short FlightLine near KSBA flies; one pattern with very long legs
        and one Waypoint with a multi-hour delay are infeasible under the
        chosen endurance and end up skipped.
        """
        fl_near, _fl_far = free_lines_far_and_near
        # A line-based pattern whose internal traversal alone exceeds endurance.
        big_rt = racetrack(
            center=(34.4, -119.8),
            heading=90.0,
            altitude=ureg.Quantity(8_000, "foot"),
            leg_length=ureg.Quantity(1_500, "kilometer"),
            n_legs=2,
            offset=ureg.Quantity(2, "kilometer"),
            name="HUGE_RT",
        )
        # A Waypoint whose loiter delay alone exceeds endurance.
        long_loiter = Waypoint(
            latitude=34.45, longitude=-119.85, heading=0.0,
            altitude_msl=ureg.Quantity(8_000, "foot"),
            delay=ureg.Quantity(10, "hour"),
            name="ENDLESS_LOITER",
        )

        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[fl_near, big_rt, long_loiter],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=1.5,
        )

        # Item-level: exactly one item flew, two were skipped.
        assert result["items_covered"] == 1
        assert len(result["items_skipped"]) == 2
        assert set(result["items_skipped"]) == {"HUGE_RT", "ENDLESS_LOITER"}

        # Line-level: 1 leg flew (the FlightLine); the skipped pattern's
        # 2 legs appear in lines_skipped, and the skipped Waypoint adds nothing.
        assert result["lines_covered"] == 1
        assert len(result["lines_skipped"]) == 2
        for leg_key in result["lines_skipped"]:
            assert leg_key.startswith("HUGE_RT:")
        assert all("ENDLESS_LOITER" not in leg_key for leg_key in result["lines_skipped"])


# ---------------------------------------------------------------------------
# Takeoff/landing overhead accounting
# ---------------------------------------------------------------------------


def _replay_route_times(result, airports, *, takeoff_landing_overhead,
                        refuel_time, max_endurance):
    """Re-walk a single-day route with the feasibility arithmetic.

    Sums the graph's leg weights, charging the takeoff/landing overhead at
    every airport arrival (matching how the feasibility checks reserve it)
    plus the refuel time at each refuel stop, and asserts the endurance
    limit is honored between refuels.
    """
    G = result["graph"]
    icaos = {a.icao_code for a in airports}
    refuels = list(result["refuel_stops"])
    total = 0.0
    since_refuel = 0.0
    for u, v in itertools.pairwise(result["route"]):
        leg = G[u][v]["weight"]
        total += leg
        since_refuel += leg
        if v in icaos:
            total += takeoff_landing_overhead
            since_refuel += takeoff_landing_overhead
            assert since_refuel <= max_endurance + 1e-9
            if refuels and refuels[0] == v:
                refuels.pop(0)
                total += refuel_time
                since_refuel = 0.0
    assert refuels == []
    return total


class TestOverheadAccounting:
    """takeoff_landing_overhead is charged to reported times, mirroring the
    arithmetic the feasibility checks use (one overhead per airborne cycle)."""

    def test_single_line_daily_time_includes_overhead(self, b200, airports):
        line = FlightLine.start_length_azimuth(
            lat1=34.4, lon1=-119.8,
            length=ureg.Quantity(10000, "meter"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Single",
        )
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=4.0,
            takeoff_landing_overhead=0.25,
        )
        G = result["graph"]
        legs = sum(
            G[u][v]["weight"] for u, v in itertools.pairwise(result["route"])
        )
        # One airborne cycle (takeoff -> landing) -> exactly one overhead.
        assert result["daily_times"][0] == pytest.approx(legs + 0.25)
        assert result["total_time"] == pytest.approx(legs + 0.25)
        assert result["issues"] == []

    def test_refuel_day_replay_matches_reported_times(self, b200, flight_lines, airports):
        """Replaying the route with feasibility arithmetic reproduces the
        reported daily time and honors both endurance and daily budgets."""
        max_endurance = 1.0
        max_daily = 8.0
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=flight_lines,
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[0],
            max_endurance=max_endurance,
            max_daily_flight_time=max_daily,
            max_days=1,
        )
        assert result["items_covered"] == len(flight_lines)
        assert len(result["refuel_stops"]) >= 1
        replayed = _replay_route_times(
            result, airports,
            takeoff_landing_overhead=0.25,
            refuel_time=0.5,
            max_endurance=max_endurance,
        )
        assert result["daily_times"][0] == pytest.approx(replayed)
        assert result["daily_times"][0] <= max_daily
        assert result["total_time"] == pytest.approx(sum(result["daily_times"]))
        assert result["issues"] == []


# ---------------------------------------------------------------------------
# Day-end return feasibility
# ---------------------------------------------------------------------------


class TestDayEndReturn:
    """The end-of-day return leg is checked against endurance, refueling
    en route when needed and recording issues when no feasible route exists."""

    @pytest.fixture
    def sba_line(self):
        return FlightLine.start_length_azimuth(
            lat1=34.45, lon1=-119.8,
            length=ureg.Quantity(10, "kilometer"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="SBA_LINE",
        )

    def test_far_return_refuels_en_route(self, b200, sba_line):
        """Direct return to a distant airport exceeds remaining endurance, so
        the optimizer inserts a refuel stop on the way home."""
        airports = [Airport("KSBA"), Airport("KMRY"), Airport("KSFO")]
        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[sba_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[2],
            max_endurance=1.4,
            max_daily_flight_time=8.0,
        )
        assert result["items_covered"] == 1
        # Refueling proves the direct return was infeasible — the optimizer
        # prefers the direct leg whenever endurance allows it. KSBA is the
        # closest airport from which KSFO fits in a fresh tank.
        assert result["refuel_stops"] == ["KSBA"]
        assert result["route"][-1] == "KSFO"
        assert result["issues"] == []
        replayed = _replay_route_times(
            result, airports,
            takeoff_landing_overhead=0.25,
            refuel_time=0.5,
            max_endurance=1.4,
        )
        assert result["daily_times"][0] == pytest.approx(replayed)

    def test_infeasible_return_records_issue(self, b200, sba_line, caplog):
        """No refuel stop can make the return reachable: the schedule is
        emitted with a prominent warning and a recorded issue."""
        import logging

        airports = [Airport("KSBA"), Airport("KBUR"), Airport("KJFK")]
        with caplog.at_level(logging.WARNING, logger="hyplan.flight_optimizer"):
            result = greedy_optimize(
                aircraft=b200,
                flight_lines=[sba_line],
                airports=airports,
                takeoff_airport=airports[0],
                return_airport=airports[2],
                max_endurance=1.5,
            )
        assert result["items_covered"] == 1
        assert result["route"][-1] == "KJFK"
        assert any("infeasible return leg" in msg for msg in result["issues"])
        assert any("infeasible return leg" in rec.message for rec in caplog.records)

    def test_missing_return_edge_records_issue(self, b200, sba_line, caplog):
        """A return airport absent from the graph triggers a loud warning."""
        import logging

        airports = [Airport("KSBA"), Airport("KBUR")]
        with caplog.at_level(logging.WARNING, logger="hyplan.flight_optimizer"):
            result = greedy_optimize(
                aircraft=b200,
                flight_lines=[sba_line],
                airports=airports,
                takeoff_airport=airports[0],
                return_airport=Airport("KSFO"),
                max_endurance=4.0,
            )
        assert result["items_covered"] == 1
        assert result["route"][-1] != "KSFO"
        assert any("no return edge" in msg for msg in result["issues"])
        assert any("no return edge" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Multi-day continuity
# ---------------------------------------------------------------------------


class TestMultiDayContinuity:
    """Each day after the first starts where the previous day ended."""

    def test_day_two_starts_at_previous_day_end(self, b200):
        airports = [Airport("KSBA"), Airport("KBUR")]
        sba_line = FlightLine.start_length_azimuth(
            lat1=34.45, lon1=-119.8,
            length=ureg.Quantity(10, "kilometer"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="SBA_LINE",
        )
        bur_line = FlightLine.start_length_azimuth(
            lat1=34.25, lon1=-118.4,
            length=ureg.Quantity(10, "kilometer"),
            az=90,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="BUR_LINE",
        )
        # Daily budget sized from a single-line day so day 1 can fly only
        # the SBA line before returning to KBUR for the night.
        single = greedy_optimize(
            aircraft=b200,
            flight_lines=[sba_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[1],
            max_endurance=4.0,
        )
        budget = single["daily_times"][0] + 0.05

        result = greedy_optimize(
            aircraft=b200,
            flight_lines=[sba_line, bur_line],
            airports=airports,
            takeoff_airport=airports[0],
            return_airport=airports[1],
            max_endurance=4.0,
            max_daily_flight_time=budget,
            max_days=2,
        )
        assert result["items_covered"] == 2
        assert result["days_used"] == 2
        route = result["route"]
        # Day 2 departs KBUR (where day 1 ended) — the takeoff airport
        # appears exactly once, at the very start of the mission.
        assert route[0] == "KSBA"
        assert route.count("KSBA") == 1
        # KBUR closes both days; the day-2 start is not re-appended.
        assert route[-1] == "KBUR"
        assert route.count("KBUR") == 2
        assert all(t <= budget + 1e-9 for t in result["daily_times"])
        assert result["issues"] == []


# ---------------------------------------------------------------------------
# Waypoint-based pattern cost model vs the planning engine
# ---------------------------------------------------------------------------


class TestPatternInternalTimeMatchesEngine:
    """_pattern_internal_time mirrors compute_flight_plan's intra-pattern
    direct-segment cost model for densely spaced waypoint patterns."""

    def test_spiral_internal_time_matches_engine(self, b200):
        from hyplan.flight_plan import compute_flight_plan

        sp = spiral(
            center=(34.4, -119.8),
            heading=0.0,
            altitude_start=ureg.Quantity(3000, "meter"),
            altitude_end=ureg.Quantity(5000, "meter"),
            radius=ureg.Quantity(3, "kilometer"),
            n_turns=1.0,
            points_per_turn=12,
        )
        internal_hours = _pattern_internal_time(b200, sp)
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[sp])
        engine_hours = plan["time_to_segment"].sum() / 60.0
        assert internal_hours == pytest.approx(engine_hours, rel=0.05)
