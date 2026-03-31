"""Tests for hyplan.interactive widgets and state."""

import pytest

from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.interactive._state import PlannerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wp(lat=34.0, lon=-118.0, heading=0.0, alt_m=1000.0, name=None):
    return Waypoint(lat, lon, heading, altitude_msl=alt_m * ureg.meter, name=name)


def _fl(lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-118.0, name="test"):
    wp1 = Waypoint(lat1, lon1, 0.0, altitude_msl=1000.0 * ureg.meter, name=f"{name}_s")
    wp2 = Waypoint(lat2, lon2, 0.0, altitude_msl=1000.0 * ureg.meter, name=f"{name}_e")
    return FlightLine(wp1, wp2, site_name=name)


# ---------------------------------------------------------------------------
# PlannerState
# ---------------------------------------------------------------------------

class TestPlannerState:
    def test_initial_state(self):
        state = PlannerState()
        assert state.waypoints == []
        assert state.flight_lines == []
        assert state.selected_indices == []
        assert state.aircraft is None

    def test_append_waypoint(self):
        state = PlannerState()
        wp = _wp(name="WP1")
        state.append_waypoint(wp)
        assert len(state.waypoints) == 1
        assert state.waypoints[0].name == "WP1"

    def test_remove_waypoint(self):
        state = PlannerState()
        state.append_waypoint(_wp(name="A"))
        state.append_waypoint(_wp(name="B"))
        state.append_waypoint(_wp(name="C"))
        state.remove_waypoint(1)
        assert len(state.waypoints) == 2
        assert state.waypoints[0].name == "A"
        assert state.waypoints[1].name == "C"

    def test_add_flight_line(self):
        state = PlannerState()
        fl = _fl(name="L1")
        state.add_flight_line(fl)
        assert len(state.flight_lines) == 1
        assert state.flight_lines[0].site_name == "L1"

    def test_remove_flight_line_updates_selection(self):
        state = PlannerState()
        state.add_flight_line(_fl(name="L0"))
        state.add_flight_line(_fl(name="L1"))
        state.add_flight_line(_fl(name="L2"))
        state.selected_indices = [0, 1, 2]

        state.remove_flight_line(1)

        assert len(state.flight_lines) == 2
        assert state.flight_lines[0].site_name == "L0"
        assert state.flight_lines[1].site_name == "L2"
        # Index 1 removed; index 2 shifted to 1
        assert state.selected_indices == [0, 1]

    def test_reset(self):
        state = PlannerState()
        state.append_waypoint(_wp())
        state.add_flight_line(_fl())
        state.selected_indices = [0]
        state.reset()
        assert state.waypoints == []
        assert state.flight_lines == []
        assert state.selected_indices == []

    def test_observer_fires(self):
        state = PlannerState()
        changes = []
        state.observe(lambda change: changes.append(change["name"]), names=["waypoints"])
        state.append_waypoint(_wp())
        assert "waypoints" in changes

    def test_selected_indices_ordering(self):
        state = PlannerState()
        for i in range(5):
            state.add_flight_line(_fl(name=f"L{i}"))
        state.selected_indices = [3, 1, 4]
        assert state.selected_indices == [3, 1, 4]


# ---------------------------------------------------------------------------
# Widget instantiation (headless — no display)
# ---------------------------------------------------------------------------

class TestWaypointEditorInstantiation:
    def test_creates_without_error(self):
        import ipyleaflet
        from hyplan.interactive.waypoint_editor import WaypointEditor

        state = PlannerState()
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        editor = WaypointEditor(state, m)
        assert editor is not None
        assert len(state.waypoints) == 0

    def test_prepopulated_waypoints(self):
        import ipyleaflet
        from hyplan.interactive.waypoint_editor import WaypointEditor

        state = PlannerState()
        state.append_waypoint(_wp(name="WP1"))
        state.append_waypoint(_wp(lat=34.1, name="WP2"))
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        editor = WaypointEditor(state, m)
        assert len(editor._markers) == 2


class TestFlightLineManagerInstantiation:
    def test_creates_without_error(self):
        import ipyleaflet
        from hyplan.interactive.flight_line_manager import FlightLineManager

        state = PlannerState()
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        manager = FlightLineManager(state, m)
        assert manager is not None
        assert len(state.flight_lines) == 0

    def test_initial_flight_lines(self):
        import ipyleaflet
        from hyplan.interactive.flight_line_manager import FlightLineManager

        state = PlannerState()
        lines = [_fl(name="A"), _fl(lat2=34.2, name="B")]
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        manager = FlightLineManager(state, m, flight_lines=lines)
        assert len(state.flight_lines) == 2
        assert len(state.selected_indices) == 2
        assert manager.selected_lines[0].site_name == "A"

    def test_toggle_selection(self):
        import ipyleaflet
        from hyplan.interactive.flight_line_manager import FlightLineManager

        state = PlannerState()
        lines = [_fl(name="A"), _fl(lat2=34.2, name="B")]
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        manager = FlightLineManager(state, m, flight_lines=lines)

        # Deselect first line
        manager._toggle_selection(0)
        assert state.selected_indices == [1]

        # Re-select first line
        manager._toggle_selection(0)
        assert state.selected_indices == [1, 0]

    def test_reorder(self):
        import ipyleaflet
        from hyplan.interactive.flight_line_manager import FlightLineManager

        state = PlannerState()
        lines = [_fl(name="A"), _fl(lat2=34.2, name="B"), _fl(lat2=34.3, name="C")]
        m = ipyleaflet.Map(center=(34.0, -118.0), zoom=10)
        manager = FlightLineManager(state, m, flight_lines=lines)

        # Move second selected item up
        manager._move_up(1)
        assert state.selected_indices == [1, 0, 2]
