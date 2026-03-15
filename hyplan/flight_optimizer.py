import itertools
import logging

import networkx as nx
import pymap3d.vincenty

from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .dubins_path import Waypoint, DubinsPath
from .flight_line import FlightLine

logger = logging.getLogger(__name__)


def _waypoint_from_airport(airport: Airport) -> Waypoint:
    """Create a Waypoint from an Airport (heading=0, used as placeholder)."""
    return Waypoint(
        latitude=airport.latitude,
        longitude=airport.longitude,
        heading=0.0,
        altitude=airport.elevation,
        name=airport.name
    )


def _transit_time(aircraft: Aircraft, start_wp: Waypoint, end_wp: Waypoint) -> float:
    """
    Compute transit time in hours between two waypoints using the aircraft's
    cruise performance model (includes Dubins path and climb/descent).
    """
    info = aircraft.time_to_cruise(start_wp, end_wp)
    return info["total_time"].to(ureg.hour).magnitude


def _departure_time(aircraft: Aircraft, airport: Airport, wp: Waypoint) -> float:
    """Compute takeoff + climb + cruise time in hours from airport to waypoint."""
    info = aircraft.time_to_takeoff(airport, wp)
    return info["total_time"].to(ureg.hour).magnitude


def _return_time(aircraft: Aircraft, wp: Waypoint, airport: Airport) -> float:
    """Compute cruise + descent + approach time in hours from waypoint to airport."""
    info = aircraft.time_to_return(wp, airport)
    return info["total_time"].to(ureg.hour).magnitude


def _flight_line_time(aircraft: Aircraft, flight_line: FlightLine) -> float:
    """Compute time in hours to fly along a flight line at cruise speed."""
    return (flight_line.length / aircraft.cruise_speed_at(flight_line.altitude)).to(ureg.hour).magnitude


def build_graph(
    aircraft: Aircraft,
    flight_lines: list,
    airports: list,
) -> nx.DiGraph:
    """
    Build a directed graph connecting airports and flight line endpoints.

    Nodes:
        - Airport nodes keyed by ICAO code
        - Flight line endpoint nodes keyed by "{site_name}_start" and "{site_name}_end"

    Edges:
        - flight_line: along each flight line (both directions)
        - departure: airport -> flight line endpoint
        - transit: between flight line endpoints (via Dubins path)
        - return: flight line endpoint -> airport

    All edge weights are transit time in hours.

    Args:
        aircraft: Aircraft to use for performance calculations.
        flight_lines: List of FlightLine objects.
        airports: List of Airport objects (potential departure/return/refuel points).

    Returns:
        nx.DiGraph with time-weighted edges.
    """
    G = nx.DiGraph()

    # Assign unique keys to flight lines
    line_keys = {}
    for fl in flight_lines:
        key = fl.site_name or f"line_{id(fl)}"
        # Handle duplicate names
        if key in line_keys.values():
            key = f"{key}_{id(fl)}"
        line_keys[fl] = key

    # --- Add airport nodes ---
    for airport in airports:
        G.add_node(airport.icao_code, nodetype="airport", obj=airport)

    # --- Add flight line endpoint nodes and along-line edges ---
    for fl, key in line_keys.items():
        start_node = f"{key}_start"
        end_node = f"{key}_end"

        G.add_node(start_node, nodetype="flight_line_endpoint",
                   waypoint=fl.waypoint1, flight_line=fl, endpoint="start")
        G.add_node(end_node, nodetype="flight_line_endpoint",
                   waypoint=fl.waypoint2, flight_line=fl, endpoint="end")

        # Along-line edges (both directions)
        line_time = _flight_line_time(aircraft, fl)
        G.add_edge(start_node, end_node, weight=line_time,
                   edgetype="flight_line", flight_line=fl, direction="forward")
        G.add_edge(end_node, start_node, weight=line_time,
                   edgetype="flight_line", flight_line=fl, direction="reverse")

    # --- Add departure edges: airport -> flight line endpoints ---
    for airport in airports:
        for fl, key in line_keys.items():
            for endpoint, wp in [("start", fl.waypoint1), ("end", fl.waypoint2)]:
                node = f"{key}_{endpoint}"
                try:
                    t = _departure_time(aircraft, airport, wp)
                    G.add_edge(airport.icao_code, node, weight=t, edgetype="departure")
                except Exception as e:
                    logger.warning(f"Could not compute departure {airport.icao_code} -> {node}: {e}")

    # --- Add return edges: flight line endpoints -> airport ---
    for airport in airports:
        for fl, key in line_keys.items():
            for endpoint, wp in [("start", fl.waypoint1), ("end", fl.waypoint2)]:
                node = f"{key}_{endpoint}"
                try:
                    t = _return_time(aircraft, wp, airport)
                    G.add_edge(node, airport.icao_code, weight=t, edgetype="return")
                except Exception as e:
                    logger.warning(f"Could not compute return {node} -> {airport.icao_code}: {e}")

    # --- Add transit edges between flight line endpoints ---
    fl_items = list(line_keys.items())
    for (fl1, key1), (fl2, key2) in itertools.combinations(fl_items, 2):
        endpoints1 = [("start", fl1.waypoint1), ("end", fl1.waypoint2)]
        endpoints2 = [("start", fl2.waypoint1), ("end", fl2.waypoint2)]

        for ep1, wp1 in endpoints1:
            for ep2, wp2 in endpoints2:
                node1 = f"{key1}_{ep1}"
                node2 = f"{key2}_{ep2}"
                try:
                    t = _transit_time(aircraft, wp1, wp2)
                    G.add_edge(node1, node2, weight=t, edgetype="transit")
                except Exception as e:
                    logger.warning(f"Could not compute transit {node1} -> {node2}: {e}")
                try:
                    t = _transit_time(aircraft, wp2, wp1)
                    G.add_edge(node2, node1, weight=t, edgetype="transit")
                except Exception as e:
                    logger.warning(f"Could not compute transit {node2} -> {node1}: {e}")

    return G


def _find_closest_unvisited_line(G, current_node, visited_lines, line_keys):
    """
    Find the closest unvisited flight line from the current node.

    Returns:
        (line_key, entry_node, time_to_entry) or (None, None, None) if all visited.
    """
    best_key = None
    best_node = None
    best_time = float("inf")

    for fl, key in line_keys.items():
        if key in visited_lines:
            continue
        for endpoint in ["start", "end"]:
            node = f"{key}_{endpoint}"
            if G.has_edge(current_node, node):
                t = G[current_node][node]["weight"]
                if t < best_time:
                    best_time = t
                    best_key = key
                    best_node = node

    if best_key is None:
        return None, None, None
    return best_key, best_node, best_time


def _find_closest_airport(G, current_node, airports):
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


def _opposite_endpoint(node):
    """Given 'key_start', return 'key_end' and vice versa."""
    if node.endswith("_start"):
        return node[:-6] + "_end"
    elif node.endswith("_end"):
        return node[:-4] + "_start"
    raise ValueError(f"Node {node} is not a flight line endpoint")


def greedy_optimize(
    aircraft: Aircraft,
    flight_lines: list,
    airports: list,
    takeoff_airport: Airport,
    return_airport: Airport = None,
    max_endurance: float = None,
    refuel_time: float = 0.5,
    max_daily_flight_time: float = None,
    takeoff_landing_overhead: float = 0.25,
) -> dict:
    """
    Greedy nearest-neighbor optimization of flight line ordering.

    Builds a graph of all flight lines and airports, then iteratively
    selects the closest unvisited flight line, inserting refuel stops
    when endurance limits would be exceeded.

    Args:
        aircraft: Aircraft performing the mission.
        flight_lines: List of FlightLine objects to cover.
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

    Returns:
        dict with:
            - "flight_sequence": list of FlightLine and Waypoint objects
                suitable for compute_flight_plan()
            - "route": list of node names traversed
            - "total_time": total mission time in hours
            - "lines_covered": number of flight lines flown
            - "refuel_stops": list of airport ICAO codes where refueling occurred
            - "takeoff_airport": Airport object
            - "return_airport": Airport object
    """
    if return_airport is None:
        return_airport = takeoff_airport

    if max_endurance is None:
        max_endurance = aircraft.endurance.to(ureg.hour).magnitude

    if max_daily_flight_time is None:
        max_daily_flight_time = max_endurance

    logger.info(f"Building flight graph for {len(flight_lines)} lines and {len(airports)} airports...")
    G = build_graph(aircraft, flight_lines, airports)

    # Build line_keys mapping (must match what build_graph used)
    line_keys = {}
    for fl in flight_lines:
        key = fl.site_name or f"line_{id(fl)}"
        if key in line_keys.values():
            key = f"{key}_{id(fl)}"
        line_keys[fl] = key

    # Reverse lookup: key -> FlightLine
    key_to_line = {v: k for k, v in line_keys.items()}

    visited_lines = set()
    skipped_lines = set()
    current_node = takeoff_airport.icao_code
    time_elapsed = 0.0
    time_since_refuel = 0.0

    route = [current_node]
    flight_sequence = []
    refuel_stops = []

    logger.info(f"Starting greedy optimization from {takeoff_airport.icao_code}")

    while len(visited_lines) < len(flight_lines):
        # Find closest unvisited flight line
        line_key, entry_node, time_to_entry = _find_closest_unvisited_line(
            G, current_node, visited_lines, line_keys
        )

        if line_key is None:
            logger.warning("No reachable unvisited flight lines remaining.")
            break

        # Determine the exit node (opposite end of the flight line)
        exit_node = _opposite_endpoint(entry_node)

        # Time to fly the line itself
        time_along_line = G[entry_node][exit_node]["weight"]

        # Time to get back to closest airport from the exit node
        closest_airport_icao, time_to_airport = _find_closest_airport(G, exit_node, airports)

        # Check endurance: can we fly to the line, along it, and back to an airport?
        total_leg_time = time_to_entry + time_along_line + time_to_airport
        if time_since_refuel + total_leg_time + takeoff_landing_overhead > max_endurance:
            # Check if this leg is feasible even with full fuel
            if total_leg_time + takeoff_landing_overhead > max_endurance:
                logger.warning(
                    f"Flight line {line_key} unreachable within endurance "
                    f"({total_leg_time:.2f}h + {takeoff_landing_overhead:.2f}h overhead "
                    f"> {max_endurance:.2f}h). Skipping."
                )
                skipped_lines.add(line_key)
                visited_lines.add(line_key)  # Mark to avoid infinite loop
                continue

            # Need to refuel first - go to closest airport from current position
            # If already at an airport, just refuel in place
            is_at_airport = G.nodes[current_node].get("nodetype") == "airport"
            if is_at_airport:
                refuel_icao = current_node
                time_to_refuel_airport = 0.0
            else:
                refuel_icao, time_to_refuel_airport = _find_closest_airport(
                    G, current_node, airports
                )
            if refuel_icao is None:
                logger.warning("Cannot reach any airport for refueling. Stopping.")
                break

            logger.info(f"Refueling at {refuel_icao} (time since last refuel: {time_since_refuel:.2f}h)")
            time_elapsed += time_to_refuel_airport + refuel_time
            time_since_refuel = 0.0
            current_node = refuel_icao
            route.append(refuel_icao)
            refuel_stops.append(refuel_icao)

            # Recompute after refueling
            line_key, entry_node, time_to_entry = _find_closest_unvisited_line(
                G, current_node, visited_lines, line_keys
            )
            if line_key is None:
                break
            exit_node = _opposite_endpoint(entry_node)
            time_along_line = G[entry_node][exit_node]["weight"]
            closest_airport_icao, time_to_airport = _find_closest_airport(G, exit_node, airports)
            total_leg_time = time_to_entry + time_along_line + time_to_airport

        # Check daily flight time limit
        if time_elapsed + total_leg_time + takeoff_landing_overhead > max_daily_flight_time:
            logger.info(f"Daily flight time limit reached ({time_elapsed:.2f}h elapsed). Stopping.")
            break

        # Fly to the entry point, along the line, then update position
        time_elapsed += time_to_entry
        time_since_refuel += time_to_entry
        route.append(entry_node)

        time_elapsed += time_along_line
        time_since_refuel += time_along_line
        route.append(exit_node)

        # Record the flight line (in the direction flown)
        fl = key_to_line[line_key]
        if entry_node.endswith("_end"):
            # Flying in reverse direction
            flight_sequence.append(fl.reverse())
        else:
            flight_sequence.append(fl)

        visited_lines.add(line_key)
        current_node = exit_node

        lines_flown = len(visited_lines) - len(skipped_lines)
        logger.info(
            f"  Flew {line_key} ({lines_flown}/{len(flight_lines)}), "
            f"elapsed: {time_elapsed:.2f}h, since refuel: {time_since_refuel:.2f}h"
        )

    # Return to airport
    if current_node != return_airport.icao_code:
        if G.has_edge(current_node, return_airport.icao_code):
            return_time = G[current_node][return_airport.icao_code]["weight"]
            time_elapsed += return_time
            time_since_refuel += return_time
            route.append(return_airport.icao_code)

    lines_flown = len(visited_lines) - len(skipped_lines)
    logger.info(
        f"Optimization complete: {lines_flown}/{len(flight_lines)} lines covered"
        + (f", {len(skipped_lines)} skipped" if skipped_lines else "")
        + f", total time: {time_elapsed:.2f}h"
    )

    return {
        "flight_sequence": flight_sequence,
        "route": route,
        "total_time": time_elapsed,
        "lines_covered": lines_flown,
        "lines_skipped": list(skipped_lines),
        "refuel_stops": refuel_stops,
        "takeoff_airport": takeoff_airport,
        "return_airport": return_airport,
        "graph": G,
    }
