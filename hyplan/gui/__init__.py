"""Interactive Jupyter widgets for flight planning.

Provides two focused widgets and a shared observable state object:

* :class:`PlannerState` -- traitlets-based state shared across widgets.
* :class:`WaypointEditor` -- place, drag, and edit waypoints on an
  ipyleaflet map with a synchronized ipydatagrid table.
* :class:`FlightLineManager` -- add, select, deselect, and reorder
  flight lines for flight plan sequencing.

These widgets require the ``gui`` optional dependencies::

    pip install hyplan[gui]
"""

__all__ = [
    "PlannerState",
    "WaypointEditor",
    "FlightLineManager",
]


def __getattr__(name: str):
    if name == "PlannerState":
        from ._state import PlannerState
        return PlannerState
    if name == "WaypointEditor":
        from .waypoint_editor import WaypointEditor
        return WaypointEditor
    if name == "FlightLineManager":
        from .flight_line_manager import FlightLineManager
        return FlightLineManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
