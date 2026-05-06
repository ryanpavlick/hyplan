"""Flight planning engine and segment construction.

Core entry point is :func:`compute_flight_plan`, which assembles a sequence
of flight lines and waypoints into a complete mission plan with takeoff,
transit, data-collection, and landing phases.
"""

from .engine import compute_flight_plan  # noqa: F401
from .isochrone import (  # noqa: F401
    compute_isochrone,
    compute_refuel_isochrone,
    isochrone_polygon,
    plot_isochrone,
)
from .segments import create_flight_line_record, process_flight_phase  # noqa: F401

__all__ = [
    "compute_flight_plan",
    "compute_isochrone",
    "compute_refuel_isochrone",
    "isochrone_polygon",
    "plot_isochrone",
    "create_flight_line_record",
    "process_flight_phase",
]
