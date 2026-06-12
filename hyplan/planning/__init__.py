"""Flight planning engine and segment construction.

Core entry point is :func:`compute_flight_plan`, which assembles a sequence
of flight lines and waypoints into a complete mission plan with takeoff,
transit, data-collection, and landing phases.
"""

from .engine import compute_flight_plan, expand_sequence, flag_below_min_safe_speed
from .isochrone import (
    compute_concentric_isochrones,
    compute_isochrone,
    compute_multi_base_isochrone,
    compute_multi_refuel_isochrone,
    compute_refuel_isochrone,
    evaluate_target_reachability,
    isochrone_polygon,
    plot_isochrone,
)
from .segments import create_flight_line_record, process_flight_phase

__all__ = [
    "compute_concentric_isochrones",
    "compute_flight_plan",
    "compute_isochrone",
    "compute_multi_base_isochrone",
    "compute_multi_refuel_isochrone",
    "compute_refuel_isochrone",
    "create_flight_line_record",
    "evaluate_target_reachability",
    "expand_sequence",
    "flag_below_min_safe_speed",
    "isochrone_polygon",
    "plot_isochrone",
    "process_flight_phase",
]
