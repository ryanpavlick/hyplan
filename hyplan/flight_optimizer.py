"""Graph-based visit-item ordering and multi-day scheduling.

Builds a weighted directed graph where nodes are visit-item endpoints and
airports, and edge weights are transit times computed from aircraft
performance. :func:`greedy_optimize` traverses this graph with a
nearest-neighbour heuristic, respecting endurance limits, daily flight-time
caps, and refueling constraints to produce a feasible multi-day schedule.

Visit items may be free-standing :class:`~hyplan.flight_line.FlightLine`
objects, atomic :class:`~hyplan.pattern.Pattern` objects, or bare
:class:`~hyplan.waypoint.Waypoint` objects with optional loiter delay.
"""

from __future__ import annotations

import itertools
import logging

import networkx as nx

from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .waypoint import Waypoint
from .flight_line import FlightLine
from .pattern import Pattern
from .exceptions import HyPlanValueError, HyPlanRuntimeError

logger = logging.getLogger(__name__)

__all__ = [
    "build_graph",
    "greedy_optimize",
]


def _waypoint_from_airport(airport: Airport) -> Waypoint:
    """Create a Waypoint from an Airport (heading=0, used as placeholder).

    Args:
        airport: Airport to convert.

    Returns:
        Waypoint at the airport's location with heading=0 and MSL elevation.
    """
    return Waypoint(
        latitude=airport.latitude,
        longitude=airport.longitude,
        heading=0.0,
        altitude_msl=airport.elevation,
        name=airport.name
    )


def _transit_time(aircraft: Aircraft, start_wp: Waypoint, end_wp: Waypoint) -> float:
    """Compute transit time in hours between two waypoints.

    Uses the aircraft's cruise performance model (includes Dubins path
    and climb/descent).

    Args:
        aircraft: Aircraft performance model.
        start_wp: Origin waypoint.
        end_wp: Destination waypoint.

    Returns:
        Transit time in hours.
    """
    info = aircraft.time_to_cruise(start_wp, end_wp)
    return info["total_time"].m_as(ureg.hour)  # type: ignore[no-any-return]


def _departure_time(aircraft: Aircraft, airport: Airport, wp: Waypoint) -> float:
    """Compute takeoff + climb + cruise time in hours from airport to waypoint.

    Args:
        aircraft: Aircraft performance model.
        airport: Departure airport.
        wp: Target waypoint.

    Returns:
        Total departure time in hours.
    """
    info = aircraft.time_to_takeoff(airport, wp)
    return info["total_time"].m_as(ureg.hour)  # type: ignore[no-any-return]


def _return_time(aircraft: Aircraft, wp: Waypoint, airport: Airport) -> float:
    """Compute cruise + descent + approach time in hours from waypoint to airport.

    Args:
        aircraft: Aircraft performance model.
        wp: Current waypoint.
        airport: Destination airport.

    Returns:
        Total return time in hours.
    """
    info = aircraft.time_to_return(wp, airport)
    return info["total_time"].m_as(ureg.hour)  # type: ignore[no-any-return]


def _flight_line_time(aircraft: Aircraft, flight_line: FlightLine, cruise_speed=None) -> float:
    """Compute time in hours to fly along a flight line at cruise speed.

    Args:
        aircraft: Aircraft performance model.
        flight_line: Flight line to traverse.
        cruise_speed: Optional precomputed cruise speed at the flight line's
            altitude. When supplied, skips the per-call ``cruise_speed_at``
            lookup (useful in batch graph construction).

    Returns:
        Flight line traversal time in hours.
    """
    if cruise_speed is None:
        cruise_speed = aircraft.cruise_speed_at(flight_line.altitude_msl)
    return (flight_line.length / cruise_speed).m_as(ureg.hour)  # type: ignore[no-any-return]


def _pattern_internal_time(aircraft: Aircraft, pattern: Pattern) -> float:
    """Compute total in-pattern traversal time in hours.

    Sums internal element traversal times plus inter-element transitions,
    using the same cost helpers (``_flight_line_time``, ``_transit_time``)
    the optimizer already trusts for free flight lines and transit between
    them. This keeps the cost model used for ordering identical to the one
    used for endurance/refueling feasibility.

    For a **line-based** pattern, the total is:

    - sum of along-line traversal times for each leg, plus
    - sum of transit times between consecutive legs (waypoint2 of leg N
      to waypoint1 of leg N+1).

    For a **waypoint-based** pattern, the total is the sum of transit
    times between consecutive waypoints.

    Args:
        aircraft: Aircraft performance model.
        pattern: Pattern whose internal cost is being evaluated.

    Returns:
        Total in-pattern traversal time in hours. Returns ``0.0`` for an
        empty pattern (no flight lines and no waypoints).
    """
    if pattern.is_line_based:
        lines = list(pattern.lines.values())
        if not lines:
            return 0.0
        total = 0.0
        for line in lines:
            total += _flight_line_time(aircraft, line)
        for prev_line, next_line in zip(lines, lines[1:]):
            total += _transit_time(aircraft, prev_line.waypoint2, next_line.waypoint1)
        return total

    waypoints = list(pattern.waypoints)
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for prev_wp, next_wp in zip(waypoints, waypoints[1:]):
        total += _transit_time(aircraft, prev_wp, next_wp)
    return total


def _item_endpoints(item) -> tuple[Waypoint, Waypoint]:
    """Return (entry, exit) Waypoints for a visit item.

    Visit items are FlightLine, Pattern, or bare Waypoint. A bare Waypoint
    has identical entry and exit (a single point in space).
    """
    if isinstance(item, Pattern):
        return item.entry_waypoint, item.exit_waypoint
    if isinstance(item, Waypoint):
        return item, item
    return item.waypoint1, item.waypoint2


def _item_internal_time(aircraft: Aircraft, item, cruise_speed=None) -> float:
    """Internal traversal time in hours for a visit item.

    For a bare Waypoint this is the loiter ``delay`` (or 0.0 if unset);
    the optimizer uses this both for ordering cost and for endurance
    feasibility, matching how ``compute_flight_plan`` accounts for the
    waypoint's loiter segment.
    """
    if isinstance(item, Pattern):
        return _pattern_internal_time(aircraft, item)
    if isinstance(item, Waypoint):
        if item.delay is None:
            return 0.0
        return item.delay.m_as(ureg.hour)  # type: ignore[no-any-return]
    return _flight_line_time(aircraft, item, cruise_speed=cruise_speed)


def _item_supports_reverse(item) -> bool:
    """Whether this visit item can be traversed exit -> entry as well as forward.

    FlightLines can be flown in either direction; Patterns are atomic and
    direction-locked (entry -> exit only) in this release. Bare Waypoints
    have entry == exit, so reversal is structurally meaningless and they
    behave like direction-locked Patterns in the graph.
    """
    return isinstance(item, FlightLine)


def _item_line_count(item) -> int:
    """Number of actual flight-line legs contributed by a visit item.

    Used for the backward-compatible ``lines_covered`` summary in
    :func:`greedy_optimize` results. A line-based Pattern contributes one
    line per internal leg; a waypoint-based Pattern and a bare Waypoint
    contribute zero (no along-line data collection).
    """
    if isinstance(item, Pattern):
        if item.is_line_based:
            return len(item.lines)
        return 0
    if isinstance(item, Waypoint):
        return 0
    return 1


def _item_skipped_line_keys(item, key: str) -> list:
    """Stable line-leg identifiers for an item that the optimizer skipped.

    Used to expand the per-item ``skipped_items`` set into the line-leg
    granularity historically reported by ``lines_skipped``. The returned
    keys are namespaced under the item's optimizer key so they remain
    unique across the result.

    - FlightLine: ``[key]``
    - Line-based Pattern: ``[f"{key}:{line_id}", ...]`` for each internal leg
    - Waypoint-based Pattern: ``[]`` (no along-line collection)
    - Bare Waypoint: ``[]`` (no along-line collection)
    """
    if isinstance(item, Pattern):
        if item.is_line_based:
            return [f"{key}:{line_id}" for line_id in item.line_ids]
        return []
    if isinstance(item, Waypoint):
        return []
    return [key]


def _item_base_key(item, fallback_index: int) -> str:
    """Derive a stable graph-node base key for a visit item.

    FlightLine -> ``site_name``; Pattern -> ``pattern_id`` or ``name``;
    bare Waypoint -> ``name``. Falls back to a positional key if no
    user-supplied identifier is available.
    """
    if isinstance(item, Pattern):
        return item.pattern_id or item.name or f"pattern_{fallback_index}"
    if isinstance(item, Waypoint):
        return item.name or f"waypoint_{fallback_index}"
    return item.site_name or f"line_{fallback_index}"


def build_graph(
    aircraft: Aircraft,
    flight_lines: list,
    airports: list,
) -> nx.DiGraph:
    """
    Build a directed graph connecting airports and visit-item endpoints.

    Each input item is one of:

    - :class:`~hyplan.flight_line.FlightLine` — a single bidirectional
      flight line.
    - :class:`~hyplan.pattern.Pattern` — an atomic, direction-locked
      composite visit item.
    - :class:`~hyplan.waypoint.Waypoint` — a single point in space, with
      optional ``delay`` for in-place loiter time.

    Patterns and bare Waypoints are represented as a single pair of
    endpoint nodes (entry/exit) with one forward along-edge — there is no
    individual graph node for an internal pattern leg, and a Waypoint's
    two endpoint nodes share the same waypoint reference. This is what
    enforces atomicity in ``greedy_optimize``.

    Nodes:
        - Airport nodes keyed by ICAO code
        - Visit-item endpoint nodes keyed by ``"{key}_start"`` and ``"{key}_end"``,
          where ``key`` is derived from the item's ``site_name`` (FlightLine),
          ``pattern_id``/``name`` (Pattern), or ``name`` (Waypoint).

    Edges:
        - flight_line: along each FlightLine (both directions)
        - pattern: along each Pattern (forward only — entry -> exit)
        - waypoint: along each bare Waypoint (forward only; weight equals
          ``waypoint.delay`` in hours, or 0)
        - departure: airport -> visit-item endpoint
        - transit: between visit-item endpoints (via Dubins path)
        - return: visit-item endpoint -> airport

    All edge weights are transit time in hours.

    Args:
        aircraft: Aircraft to use for performance calculations.
        flight_lines: List of ``FlightLine | Pattern | Waypoint`` objects
            to schedule. (Parameter name retained for backward
            compatibility; the optimizer now also accepts ``Pattern`` and
            bare ``Waypoint`` objects in this list.)
        airports: List of Airport objects (potential departure/return/refuel points).

    Returns:
        nx.DiGraph with time-weighted edges.
    """
    G = nx.DiGraph()

    # Assign unique keys to visit items (FlightLine | Pattern). Stored as
    # a list[(item, key)] rather than dict[item, key] because Pattern is
    # a default-mutable @dataclass and therefore unhashable; FlightLine is
    # hashable by identity but we use a list uniformly for both kinds.
    item_keys: list = []
    seen_keys: set = set()
    collision_counter: dict = {}
    for item in flight_lines:
        base = _item_base_key(item, fallback_index=len(item_keys))
        key = base
        while key in seen_keys:
            collision_counter[base] = collision_counter.get(base, 0) + 1
            key = f"{base}_{collision_counter[base]}"
        seen_keys.add(key)
        item_keys.append((item, key))

    # Cache cruise speed per altitude — most missions reuse a small set of MSLs.
    cruise_speed_cache: dict = {}
    def _cruise_speed_for(alt):
        key_alt = round(alt.m_as(ureg.feet), 3)
        if key_alt not in cruise_speed_cache:
            cruise_speed_cache[key_alt] = aircraft.cruise_speed_at(alt)
        return cruise_speed_cache[key_alt]

    # Memoize transit time for repeated waypoint pairs (airports reused N times
    # against every flight-line endpoint). Key on rounded coords + altitude +
    # heading; use a simple dict closed over locals.
    def _wp_key(wp):
        alt_m = wp.altitude_msl.m_as(ureg.meter) if wp.altitude_msl is not None else 0.0
        return (
            round(wp.latitude, 6),
            round(wp.longitude, 6),
            round(alt_m, 1),
            round(wp.heading, 3),
        )

    transit_cache: dict = {}
    def _cached_transit(wp_a, wp_b):
        k = (_wp_key(wp_a), _wp_key(wp_b))
        if k not in transit_cache:
            transit_cache[k] = _transit_time(aircraft, wp_a, wp_b)
        return transit_cache[k]

    # --- Add airport nodes ---
    for airport in airports:
        G.add_node(airport.icao_code, nodetype="airport", obj=airport)

    # --- Add visit-item endpoint nodes and along-item edges ---
    for item, key in item_keys:
        start_node = f"{key}_start"
        end_node = f"{key}_end"
        entry_wp, exit_wp = _item_endpoints(item)

        if isinstance(item, Pattern):
            G.add_node(start_node, nodetype="pattern_endpoint",
                       waypoint=entry_wp, pattern=item, endpoint="start")
            G.add_node(end_node, nodetype="pattern_endpoint",
                       waypoint=exit_wp, pattern=item, endpoint="end")
            internal_time = _pattern_internal_time(aircraft, item)
            # Forward along-edge only — Patterns are direction-locked.
            G.add_edge(start_node, end_node, weight=internal_time,
                       edgetype="pattern", pattern=item, direction="forward")
        elif isinstance(item, Waypoint):
            # Bare Waypoint: entry == exit (single point). Both endpoint
            # nodes carry the same waypoint reference; the forward
            # along-edge weight is the loiter delay (0 if unset).
            G.add_node(start_node, nodetype="waypoint_endpoint",
                       waypoint=entry_wp, waypoint_item=item, endpoint="start")
            G.add_node(end_node, nodetype="waypoint_endpoint",
                       waypoint=exit_wp, waypoint_item=item, endpoint="end")
            internal_time = _item_internal_time(aircraft, item)
            # Forward along-edge only — single-point items are direction-locked.
            G.add_edge(start_node, end_node, weight=internal_time,
                       edgetype="waypoint", waypoint_item=item, direction="forward")
        else:
            G.add_node(start_node, nodetype="flight_line_endpoint",
                       waypoint=entry_wp, flight_line=item, endpoint="start")
            G.add_node(end_node, nodetype="flight_line_endpoint",
                       waypoint=exit_wp, flight_line=item, endpoint="end")
            line_time = _flight_line_time(aircraft, item, cruise_speed=_cruise_speed_for(item.altitude_msl))
            # Along-line edges in both directions — FlightLines are reversible.
            G.add_edge(start_node, end_node, weight=line_time,
                       edgetype="flight_line", flight_line=item, direction="forward")
            G.add_edge(end_node, start_node, weight=line_time,
                       edgetype="flight_line", flight_line=item, direction="reverse")

    # --- Add departure edges: airport -> visit-item endpoints ---
    for airport in airports:
        for item, key in item_keys:
            entry_wp, exit_wp = _item_endpoints(item)
            for endpoint, wp in [("start", entry_wp), ("end", exit_wp)]:
                node = f"{key}_{endpoint}"
                try:
                    t = _departure_time(aircraft, airport, wp)
                    G.add_edge(airport.icao_code, node, weight=t, edgetype="departure")
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute departure {airport.icao_code} -> {node}: {e}")

    # --- Add return edges: visit-item endpoints -> airport ---
    for airport in airports:
        for item, key in item_keys:
            entry_wp, exit_wp = _item_endpoints(item)
            for endpoint, wp in [("start", entry_wp), ("end", exit_wp)]:
                node = f"{key}_{endpoint}"
                try:
                    t = _return_time(aircraft, wp, airport)
                    G.add_edge(node, airport.icao_code, weight=t, edgetype="return")
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute return {node} -> {airport.icao_code}: {e}")

    # --- Add transit edges between airports ---
    for a1, a2 in itertools.combinations(airports, 2):
        wp1 = _waypoint_from_airport(a1)
        wp2 = _waypoint_from_airport(a2)
        try:
            t = _cached_transit(wp1, wp2)
            G.add_edge(a1.icao_code, a2.icao_code, weight=t, edgetype="transit")
        except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
            logger.warning(f"Could not compute transit {a1.icao_code} -> {a2.icao_code}: {e}")
        try:
            t = _cached_transit(wp2, wp1)
            G.add_edge(a2.icao_code, a1.icao_code, weight=t, edgetype="transit")
        except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
            logger.warning(f"Could not compute transit {a2.icao_code} -> {a1.icao_code}: {e}")

    # --- Add transit edges between visit-item endpoints ---
    fl_items = list(item_keys)
    for (item1, key1), (item2, key2) in itertools.combinations(fl_items, 2):
        entry1, exit1 = _item_endpoints(item1)
        entry2, exit2 = _item_endpoints(item2)
        endpoints1 = [("start", entry1), ("end", exit1)]
        endpoints2 = [("start", entry2), ("end", exit2)]

        for ep1, wp1 in endpoints1:
            for ep2, wp2 in endpoints2:
                node1 = f"{key1}_{ep1}"
                node2 = f"{key2}_{ep2}"
                try:
                    t = _cached_transit(wp1, wp2)
                    G.add_edge(node1, node2, weight=t, edgetype="transit")
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute transit {node1} -> {node2}: {e}")
                try:
                    t = _cached_transit(wp2, wp1)
                    G.add_edge(node2, node1, weight=t, edgetype="transit")
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute transit {node2} -> {node1}: {e}")

    # Stash item_keys on the graph so greedy_optimize can reuse it without
    # rebuilding (and without risking inconsistency with build_graph's keying).
    G.graph["item_keys"] = item_keys
    return G


def _find_closest_unvisited_item(
    G, current_node, visited_items, item_keys,
    airports=None, time_since_refuel=0.0, time_elapsed=0.0,
    max_endurance=float("inf"), max_daily_flight_time=float("inf"),
    takeoff_landing_overhead=0.0,
) -> tuple[str | None, str | None, float | None]:
    """
    Find the closest unvisited visit item (FlightLine, Pattern, or bare
    Waypoint) that is feasible within constraints.

    A visit item is feasible if the aircraft can transit to it, traverse it,
    and return to the closest airport, all within both endurance and daily
    time limits. For FlightLines, both endpoints are considered as candidate
    entries; for Patterns and bare Waypoints, only the entry node
    (``_start``) is a candidate because they are direction-locked. The
    graph topology — Patterns and Waypoints have a forward along-edge but
    no reverse along-edge — automatically excludes reverse traversal via
    the ``has_edge`` guard below.

    Returns:
        (item_key, entry_node, time_to_entry) or (None, None, None) if none feasible.
    """
    best_key = None
    best_node = None
    best_time = float("inf")

    for item, key in item_keys:
        if key in visited_items:
            continue
        for endpoint in ["start", "end"]:
            node = f"{key}_{endpoint}"
            exit_node = _opposite_endpoint(node)
            if not G.has_edge(current_node, node):
                continue
            # For Patterns, only the forward along-edge exists, so entering
            # from `_end` would have no along-edge to traverse — skip those.
            if not G.has_edge(node, exit_node):
                continue

            t_entry = G[current_node][node]["weight"]
            t_line = G[node][exit_node]["weight"]

            # Find time to closest airport from exit
            t_return = float("inf")
            if airports is not None:
                for airport in airports:
                    icao = airport.icao_code
                    if G.has_edge(exit_node, icao):
                        t_return = min(t_return, G[exit_node][icao]["weight"])

            total_leg = t_entry + t_line + t_return + takeoff_landing_overhead

            # Check endurance constraint
            if time_since_refuel + total_leg > max_endurance:
                continue
            # Check daily flight time constraint
            if time_elapsed + t_entry + t_line + t_return + takeoff_landing_overhead > max_daily_flight_time:
                continue

            if t_entry < best_time:
                best_time = t_entry
                best_key = key
                best_node = node

    if best_key is None:
        return None, None, None
    return best_key, best_node, best_time


def _find_closest_airport(G, current_node, airports) -> tuple[str | None, float]:
    """
    Find the closest airport from the current node.

    Returns:
        (airport_icao, time_to_airport) or (None, inf) if unreachable.
    """
    best_icao = None
    best_time = float("inf")

    for airport in airports:
        icao = airport.icao_code
        if G.has_edge(current_node, icao):
            t = G[current_node][icao]["weight"]
            if t < best_time:
                best_time = t
                best_icao = icao

    return best_icao, best_time


def _find_best_refuel_airport(
    G, current_node, airports, visited_items, item_keys,
    time_since_refuel, time_elapsed, max_endurance, max_daily_flight_time,
    refuel_time, takeoff_landing_overhead,
) -> tuple[str | None, float]:
    """
    Find the best airport to refuel at, ensuring that refueling there
    actually enables reaching at least one more unvisited visit item
    (FlightLine or Pattern).

    Returns:
        (airport_icao, time_to_airport) or (None, inf) if no useful refuel exists.
    """
    best_icao = None
    best_time = float("inf")

    for airport in airports:
        icao = airport.icao_code
        if not G.has_edge(current_node, icao):
            continue
        t_to_airport = G[current_node][icao]["weight"]

        # Check we can reach this airport within current endurance
        if time_since_refuel + t_to_airport + takeoff_landing_overhead > max_endurance:
            continue

        # Check that after refueling here, at least one unvisited item is reachable.
        can_continue = False
        for item, key in item_keys:
            if key in visited_items:
                continue
            for endpoint in ["start", "end"]:
                node = f"{key}_{endpoint}"
                exit_node = _opposite_endpoint(node)
                if not G.has_edge(icao, node):
                    continue
                # Skip pattern reverse traversal (no along-edge in that direction).
                if not G.has_edge(node, exit_node):
                    continue
                t_depart = G[icao][node]["weight"]
                t_line = G[node][exit_node]["weight"]

                # Find return time from exit to any airport
                t_return = float("inf")
                for ret_airport in airports:
                    ret_icao = ret_airport.icao_code
                    if G.has_edge(exit_node, ret_icao):
                        t_return = min(t_return, G[exit_node][ret_icao]["weight"])

                total_sortie = t_depart + t_line + t_return + takeoff_landing_overhead
                if total_sortie <= max_endurance:
                    can_continue = True
                    break
            if can_continue:
                break

        if can_continue and t_to_airport < best_time:
            best_time = t_to_airport
            best_icao = icao

    return best_icao, best_time


def _opposite_endpoint(node: str) -> str:
    """Given 'key_start', return 'key_end' and vice versa.

    Args:
        node: Graph node name ending in '_start' or '_end'.

    Returns:
        The complementary endpoint node name.

    Raises:
        ValueError: If node does not end with '_start' or '_end'.
    """
    if node.endswith("_start"):
        return node[:-6] + "_end"
    elif node.endswith("_end"):
        return node[:-4] + "_start"
    raise HyPlanValueError(f"Node {node} is not a flight line endpoint")


def greedy_optimize(
    aircraft: Aircraft,
    flight_lines: list,
    airports: list,
    takeoff_airport: Airport,
    return_airport: Airport | None = None,
    max_endurance: float | None = None,
    refuel_time: float = 0.5,
    max_daily_flight_time: float | None = None,
    takeoff_landing_overhead: float = 0.25,
    max_days: int = 1,
) -> dict:
    """
    Greedy nearest-neighbor optimization of visit-item ordering.

    Builds a graph of all visit items and airports, then iteratively
    selects the closest feasible unvisited item, inserting refuel stops
    when endurance limits would be exceeded. Supports multi-day missions
    where daily flight time resets each day.

    Args:
        aircraft: Aircraft performing the mission.
        flight_lines: List of ``FlightLine | Pattern | Waypoint`` objects
            to cover. Patterns and bare Waypoints are treated as **atomic**
            visit items: the optimizer may reorder a Pattern or Waypoint
            relative to other items but never splits it apart. Pattern
            traversal is direction-locked (entry -> exit). A bare Waypoint
            has identical entry/exit (a single point); its internal time
            equals ``waypoint.delay`` (loiter), or 0 if unset, and a
            ``delay`` too large to fit in remaining endurance forces a
            refuel **before** the waypoint, never inside the loiter.
        airports: List of Airport objects available for refueling.
        takeoff_airport: Departure airport.
        return_airport: Return airport (defaults to takeoff_airport).
        max_endurance: Maximum flight time in hours before refueling.
            Defaults to aircraft.endurance.
        refuel_time: Time in hours for refueling stop (default 0.5).
        max_daily_flight_time: Maximum flying hours per day.
            Defaults to aircraft.endurance (no daily limit beyond endurance).
        takeoff_landing_overhead: Time in hours for takeoff/landing procedures
            not captured in route calculations (default 0.25).
        max_days: Maximum number of flight days (default 1).

    Returns:
        dict with:
            - "flight_sequence": list of ``FlightLine | Pattern | Waypoint``
              objects in the order they were scheduled. FlightLines may be
              reversed from their original orientation; Patterns and bare
              Waypoints appear unchanged (direction-locked entry -> exit).
            - "items_covered": ``int`` — number of completed visit items
              (each Pattern, FlightLine, or Waypoint counts as 1).
            - "items_skipped": ``list[str]`` — keys of visit items the
              optimizer was unable to schedule.
            - "lines_covered": ``int`` — number of actual flight-line legs
              completed. A line-based Pattern contributes one per internal
              leg; a waypoint-based Pattern or bare Waypoint contributes 0.
              For all-FlightLine input this equals ``items_covered``.
            - "lines_skipped": ``list[str]`` — keys of the actual
              flight-line legs that were skipped. A skipped line-based
              Pattern is expanded to ``"{item_key}:{line_id}"`` for each
              of its internal legs; skipped waypoint-based Patterns and
              bare Waypoints contribute nothing.
            - "route": list of node names traversed
            - "total_time": total mission time in hours (across all days)
            - "daily_times": list of flight time per day
            - "lines_covered": number of visit items completed
            - "lines_skipped": list of item keys that were infeasible
            - "refuel_stops": list of airport ICAO codes where refueling occurred
            - "days_used": number of days used
            - "takeoff_airport": Airport object
            - "return_airport": Airport object
            - "graph": the constructed DiGraph
    """
    if return_airport is None:
        return_airport = takeoff_airport

    if max_endurance is None:
        if aircraft.endurance is None:
            raise HyPlanValueError(
                "max_endurance must be specified when aircraft.endurance is None."
            )
        max_endurance = aircraft.endurance.m_as(ureg.hour)

    if max_daily_flight_time is None:
        max_daily_flight_time = max_endurance

    logger.info(f"Building flight graph for {len(flight_lines)} items and {len(airports)} airports...")
    G = build_graph(aircraft, flight_lines, airports)
    item_keys = G.graph["item_keys"]

    # Reverse lookup: key -> FlightLine | Pattern
    key_to_item = {key: item for item, key in item_keys}

    visited_items: set[str] = set()
    skipped_items: set[str] = set()

    route = []
    flight_sequence = []
    refuel_stops = []
    daily_times = []
    total_time = 0.0

    logger.info(f"Starting greedy optimization from {takeoff_airport.icao_code}")

    for day in range(1, max_days + 1):
        if len(visited_items) >= len(flight_lines):
            break

        logger.info(f"--- Day {day} ---")
        daily_time = 0.0
        time_since_refuel = 0.0
        current_node = takeoff_airport.icao_code
        route.append(current_node)

        while len(visited_items) < len(flight_lines):
            # Find closest feasible unvisited visit item (FlightLine or Pattern)
            item_key, entry_node, time_to_entry = _find_closest_unvisited_item(
                G, current_node, visited_items, item_keys,
                airports=airports,
                time_since_refuel=time_since_refuel,
                time_elapsed=daily_time,
                max_endurance=max_endurance,
                max_daily_flight_time=max_daily_flight_time,
                takeoff_landing_overhead=takeoff_landing_overhead,
            )

            if item_key is not None:
                # Fly to the item and across it (along-line for FlightLine,
                # entry -> exit for Pattern). Atomicity for Patterns is
                # enforced by the graph topology: only the forward along-edge
                # exists for Pattern items, so the optimizer cannot enter
                # from `_end` or split the pattern apart.
                exit_node = _opposite_endpoint(entry_node)  # type: ignore[arg-type]
                time_along_line = G[entry_node][exit_node]["weight"]

                daily_time += time_to_entry  # type: ignore[operator]
                time_since_refuel += time_to_entry  # type: ignore[operator]
                route.append(entry_node)

                daily_time += time_along_line
                time_since_refuel += time_along_line
                route.append(exit_node)

                # Record the visit item in the order/orientation flown.
                # Patterns and bare Waypoints are direction-locked (always forward);
                # FlightLines may be reversed if the optimizer entered from `_end`.
                item = key_to_item[item_key]
                if isinstance(item, (Pattern, Waypoint)):
                    flight_sequence.append(item)
                elif entry_node.endswith("_end"):  # type: ignore[union-attr]
                    flight_sequence.append(item.reverse())
                else:
                    flight_sequence.append(item)

                visited_items.add(item_key)
                current_node = exit_node

                logger.info(
                    f"  Flew {item_key} ({len(visited_items)}/{len(flight_lines)}), "
                    f"day time: {daily_time:.2f}h, since refuel: {time_since_refuel:.2f}h"
                )
            else:
                # No feasible item from current position — try refueling
                is_at_airport = G.nodes[current_node].get("nodetype") == "airport"
                if is_at_airport:
                    refuel_icao = current_node
                    time_to_refuel_airport = 0.0
                    # Check if refueling here enables any further items
                    _, _ = _find_best_refuel_airport(
                        G, current_node, airports, visited_items, item_keys,
                        time_since_refuel, daily_time, max_endurance,
                        max_daily_flight_time, refuel_time, takeoff_landing_overhead,
                    )
                    # Even at an airport, verify refueling is useful. The
                    # "edge exists in both directions" guard implicitly
                    # excludes pattern reverse traversal (no along-edge).
                    can_refuel_help = any(
                        G.has_edge(refuel_icao, f"{key}_{ep}")
                        and G.has_edge(f"{key}_{ep}", _opposite_endpoint(f"{key}_{ep}"))
                        and (G[refuel_icao][f"{key}_{ep}"]["weight"]
                             + G[f"{key}_{ep}"][_opposite_endpoint(f"{key}_{ep}")]["weight"]
                             + takeoff_landing_overhead) <= max_endurance
                        for key in (k for it, k in item_keys if k not in visited_items)
                        for ep in ["start", "end"]
                    )
                    if not can_refuel_help:
                        logger.info(f"Day {day}: No feasible items remain from {current_node}. Ending day.")
                        break
                else:
                    refuel_icao, time_to_refuel_airport = _find_best_refuel_airport(
                        G, current_node, airports, visited_items, item_keys,
                        time_since_refuel, daily_time, max_endurance,
                        max_daily_flight_time, refuel_time, takeoff_landing_overhead,
                    )

                if refuel_icao is None:
                    logger.info(f"Day {day}: No useful refueling option. Ending day.")
                    break

                # Check daily time allows transit to airport + refuel
                if daily_time + time_to_refuel_airport + refuel_time > max_daily_flight_time:
                    logger.info(f"Day {day}: Not enough daily time to refuel. Ending day.")
                    break

                logger.info(f"Refueling at {refuel_icao} (time since last refuel: {time_since_refuel:.2f}h)")
                daily_time += time_to_refuel_airport
                time_since_refuel = 0.0
                if refuel_icao != current_node:
                    route.append(refuel_icao)
                current_node = refuel_icao
                refuel_stops.append(refuel_icao)
                daily_time += refuel_time

        # Return to airport at end of day
        if current_node != return_airport.icao_code:
            if G.has_edge(current_node, return_airport.icao_code):
                return_t = G[current_node][return_airport.icao_code]["weight"]
                daily_time += return_t
                route.append(return_airport.icao_code)
                current_node = return_airport.icao_code

        daily_times.append(daily_time)
        total_time += daily_time
        logger.info(f"Day {day} complete: {daily_time:.2f}h flown")

    # Check for items that were never reachable
    for item, key in item_keys:
        if key not in visited_items and key not in skipped_items:
            skipped_items.add(key)

    # items_covered is just the count of items the optimizer actually flew.
    # skipped_items is a disjoint set of items it never reached, so the two
    # sum to len(flight_lines); they must not be combined arithmetically.
    items_flown = len(visited_items)
    items_skipped_list = list(skipped_items)

    # Count actual flight-line legs covered: each line-based Pattern
    # contributes len(pattern.lines), each FlightLine contributes 1, and
    # waypoint-based Patterns and bare Waypoints contribute 0. For
    # all-FlightLine input this matches the pre-v1.2 lines_covered
    # semantics (and equals items_covered).
    lines_flown = sum(_item_line_count(item) for item in flight_sequence)

    # Symmetrically, expand skipped items down to line-leg keys: a skipped
    # line-based Pattern's legs each appear in lines_skipped, while a
    # skipped Waypoint or waypoint-based Pattern contributes nothing.
    skipped_key_to_item = {key: item for item, key in item_keys if key in skipped_items}
    lines_skipped_list = [
        leg_key
        for key, item in skipped_key_to_item.items()
        for leg_key in _item_skipped_line_keys(item, key)
    ]

    logger.info(
        f"Optimization complete: {items_flown}/{len(flight_lines)} items covered "
        f"over {len(daily_times)} day(s)"
        + (f", {len(skipped_items)} skipped" if skipped_items else "")
        + f", total time: {total_time:.2f}h"
    )

    return {
        "flight_sequence": flight_sequence,
        "route": route,
        "total_time": total_time,
        "daily_times": daily_times,
        "items_covered": items_flown,
        "items_skipped": items_skipped_list,
        "lines_covered": lines_flown,
        "lines_skipped": lines_skipped_list,
        "refuel_stops": refuel_stops,
        "days_used": len(daily_times),
        "takeoff_airport": takeoff_airport,
        "return_airport": return_airport,
        "graph": G,
    }
