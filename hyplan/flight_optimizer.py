import itertools
import logging
from typing import Optional, Tuple

import networkx as nx
import pymap3d.vincenty

from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .waypoint import Waypoint
from .flight_line import FlightLine
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
    return info["total_time"].to(ureg.hour).magnitude


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
    return info["total_time"].to(ureg.hour).magnitude


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
    return info["total_time"].to(ureg.hour).magnitude


def _flight_line_time(aircraft: Aircraft, flight_line: FlightLine) -> float:
    """Compute time in hours to fly along a flight line at cruise speed.

    Args:
        aircraft: Aircraft performance model.
        flight_line: Flight line to traverse.

    Returns:
        Flight line traversal time in hours.
    """
    return (flight_line.length / aircraft.cruise_speed_at(flight_line.altitude_msl)).to(ureg.hour).magnitude


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
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute departure {airport.icao_code} -> {node}: {e}")

    # --- Add return edges: flight line endpoints -> airport ---
    for airport in airports:
        for fl, key in line_keys.items():
            for endpoint, wp in [("start", fl.waypoint1), ("end", fl.waypoint2)]:
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
            t = _transit_time(aircraft, wp1, wp2)
            G.add_edge(a1.icao_code, a2.icao_code, weight=t, edgetype="transit")
        except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
            logger.warning(f"Could not compute transit {a1.icao_code} -> {a2.icao_code}: {e}")
        try:
            t = _transit_time(aircraft, wp2, wp1)
            G.add_edge(a2.icao_code, a1.icao_code, weight=t, edgetype="transit")
        except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
            logger.warning(f"Could not compute transit {a2.icao_code} -> {a1.icao_code}: {e}")

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
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute transit {node1} -> {node2}: {e}")
                try:
                    t = _transit_time(aircraft, wp2, wp1)
                    G.add_edge(node2, node1, weight=t, edgetype="transit")
                except (HyPlanValueError, HyPlanRuntimeError, ValueError) as e:
                    logger.warning(f"Could not compute transit {node2} -> {node1}: {e}")

    return G


def _find_closest_unvisited_line(
    G, current_node, visited_lines, line_keys,
    airports=None, time_since_refuel=0.0, time_elapsed=0.0,
    max_endurance=float("inf"), max_daily_flight_time=float("inf"),
    takeoff_landing_overhead=0.0,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Find the closest unvisited flight line that is feasible within constraints.

    A line is feasible if the aircraft can transit to it, fly it, and return
    to the closest airport, all within both endurance and daily time limits.

    Returns:
        (line_key, entry_node, time_to_entry) or (None, None, None) if none feasible.
    """
    best_key = None
    best_node = None
    best_time = float("inf")

    for fl, key in line_keys.items():
        if key in visited_lines:
            continue
        for endpoint in ["start", "end"]:
            node = f"{key}_{endpoint}"
            exit_node = _opposite_endpoint(node)
            if not G.has_edge(current_node, node):
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


def _find_closest_airport(G, current_node, airports) -> Tuple[Optional[str], float]:
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
    G, current_node, airports, visited_lines, line_keys,
    time_since_refuel, time_elapsed, max_endurance, max_daily_flight_time,
    refuel_time, takeoff_landing_overhead,
) -> Tuple[Optional[str], float]:
    """
    Find the best airport to refuel at, ensuring that refueling there
    actually enables reaching at least one more unvisited flight line.

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

        # Check that after refueling here, at least one unvisited line is reachable
        can_continue = False
        for fl, key in line_keys.items():
            if key in visited_lines:
                continue
            for endpoint in ["start", "end"]:
                node = f"{key}_{endpoint}"
                exit_node = _opposite_endpoint(node)
                if not G.has_edge(icao, node):
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
    return_airport: Airport = None,
    max_endurance: float = None,
    refuel_time: float = 0.5,
    max_daily_flight_time: float = None,
    takeoff_landing_overhead: float = 0.25,
    max_days: int = 1,
) -> dict:
    """
    Greedy nearest-neighbor optimization of flight line ordering.

    Builds a graph of all flight lines and airports, then iteratively
    selects the closest feasible unvisited flight line, inserting refuel
    stops when endurance limits would be exceeded. Supports multi-day
    missions where daily flight time resets each day.

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
        max_days: Maximum number of flight days (default 1).

    Returns:
        dict with:
            - "flight_sequence": list of FlightLine objects in order flown
            - "route": list of node names traversed
            - "total_time": total mission time in hours (across all days)
            - "daily_times": list of flight time per day
            - "lines_covered": number of flight lines flown
            - "lines_skipped": list of line keys that were infeasible
            - "refuel_stops": list of airport ICAO codes where refueling occurred
            - "days_used": number of days used
            - "takeoff_airport": Airport object
            - "return_airport": Airport object
            - "graph": the constructed DiGraph
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

    route = []
    flight_sequence = []
    refuel_stops = []
    daily_times = []
    total_time = 0.0

    logger.info(f"Starting greedy optimization from {takeoff_airport.icao_code}")

    for day in range(1, max_days + 1):
        if len(visited_lines) >= len(flight_lines):
            break

        logger.info(f"--- Day {day} ---")
        daily_time = 0.0
        time_since_refuel = 0.0
        current_node = takeoff_airport.icao_code
        route.append(current_node)

        while len(visited_lines) < len(flight_lines):
            # Find closest feasible unvisited flight line
            line_key, entry_node, time_to_entry = _find_closest_unvisited_line(
                G, current_node, visited_lines, line_keys,
                airports=airports,
                time_since_refuel=time_since_refuel,
                time_elapsed=daily_time,
                max_endurance=max_endurance,
                max_daily_flight_time=max_daily_flight_time,
                takeoff_landing_overhead=takeoff_landing_overhead,
            )

            if line_key is not None:
                # Fly to the line and along it
                exit_node = _opposite_endpoint(entry_node)
                time_along_line = G[entry_node][exit_node]["weight"]

                daily_time += time_to_entry
                time_since_refuel += time_to_entry
                route.append(entry_node)

                daily_time += time_along_line
                time_since_refuel += time_along_line
                route.append(exit_node)

                # Record the flight line (in the direction flown)
                fl = key_to_line[line_key]
                if entry_node.endswith("_end"):
                    flight_sequence.append(fl.reverse())
                else:
                    flight_sequence.append(fl)

                visited_lines.add(line_key)
                current_node = exit_node

                lines_flown = len(visited_lines) - len(skipped_lines)
                logger.info(
                    f"  Flew {line_key} ({lines_flown}/{len(flight_lines)}), "
                    f"day time: {daily_time:.2f}h, since refuel: {time_since_refuel:.2f}h"
                )
            else:
                # No feasible line from current position — try refueling
                is_at_airport = G.nodes[current_node].get("nodetype") == "airport"
                if is_at_airport:
                    refuel_icao = current_node
                    time_to_refuel_airport = 0.0
                    # Check if refueling here enables any further lines
                    _, _ = _find_best_refuel_airport(
                        G, current_node, airports, visited_lines, line_keys,
                        time_since_refuel, daily_time, max_endurance,
                        max_daily_flight_time, refuel_time, takeoff_landing_overhead,
                    )
                    # Even at an airport, verify refueling is useful
                    can_refuel_help = any(
                        G.has_edge(refuel_icao, f"{key}_{ep}")
                        and (G[refuel_icao][f"{key}_{ep}"]["weight"]
                             + G[f"{key}_{ep}"][_opposite_endpoint(f"{key}_{ep}")]["weight"]
                             + takeoff_landing_overhead) <= max_endurance
                        for key in (k for fl, k in line_keys.items() if k not in visited_lines)
                        for ep in ["start", "end"]
                    )
                    if not can_refuel_help:
                        logger.info(f"Day {day}: No feasible lines remain from {current_node}. Ending day.")
                        break
                else:
                    refuel_icao, time_to_refuel_airport = _find_best_refuel_airport(
                        G, current_node, airports, visited_lines, line_keys,
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

    # Check for lines that were never reachable
    for fl, key in line_keys.items():
        if key not in visited_lines and key not in skipped_lines:
            skipped_lines.add(key)

    lines_flown = len(visited_lines) - len(skipped_lines)
    logger.info(
        f"Optimization complete: {lines_flown}/{len(flight_lines)} lines covered "
        f"over {len(daily_times)} day(s)"
        + (f", {len(skipped_lines)} skipped" if skipped_lines else "")
        + f", total time: {total_time:.2f}h"
    )

    return {
        "flight_sequence": flight_sequence,
        "route": route,
        "total_time": total_time,
        "daily_times": daily_times,
        "lines_covered": lines_flown,
        "lines_skipped": list(skipped_lines),
        "refuel_stops": refuel_stops,
        "days_used": len(daily_times),
        "takeoff_airport": takeoff_airport,
        "return_airport": return_airport,
        "graph": G,
    }
