"""Observable shared state for interactive flight-planning widgets.

:class:`PlannerState` uses traitlets so that any widget observing
the state is notified automatically when waypoints, flight lines,
or the selection order change.
"""

from traitlets import HasTraits, Instance, List, Int

from ..waypoint import Waypoint
from ..flight_line import FlightLine
from ..aircraft import Aircraft


class PlannerState(HasTraits):
    """Shared observable state consumed by interactive widgets.

    Attributes:
        aircraft: The :class:`~hyplan.aircraft.Aircraft` used for flight
            planning calculations (optional).
        waypoints: Ordered list of :class:`~hyplan.waypoint.Waypoint` objects.
        flight_lines: List of :class:`~hyplan.flight_line.FlightLine` objects.
        selected_indices: Ordered list of indices into *flight_lines*
            representing the user's chosen sequencing for the flight plan.
    """

    aircraft = Instance(Aircraft, allow_none=True)
    waypoints = List()  # type: ignore[var-annotated]
    flight_lines = List()  # type: ignore[var-annotated]
    selected_indices = List(Int())

    def append_waypoint(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the end of the list."""
        self.waypoints = list(self.waypoints) + [waypoint]

    def remove_waypoint(self, index: int) -> None:
        """Remove the waypoint at *index*."""
        wps = list(self.waypoints)
        wps.pop(index)
        self.waypoints = wps

    def add_flight_line(self, flight_line: FlightLine) -> None:
        """Add a flight line to the end of the list."""
        self.flight_lines = list(self.flight_lines) + [flight_line]

    def remove_flight_line(self, index: int) -> None:
        """Remove the flight line at *index* and update selected_indices."""
        fls = list(self.flight_lines)
        fls.pop(index)
        self.flight_lines = fls
        # Remove the deleted index and shift higher indices down
        self.selected_indices = [
            i if i < index else i - 1
            for i in self.selected_indices
            if i != index
        ]

    def reset(self) -> None:
        """Clear all waypoints, flight lines, and selection."""
        self.waypoints = []
        self.flight_lines = []
        self.selected_indices = []
