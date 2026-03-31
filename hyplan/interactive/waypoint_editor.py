"""Interactive waypoint editor widget.

Click on an ipyleaflet map to place waypoints, drag markers to reposition,
and edit properties in a synchronized ipydatagrid table.
"""

import ipywidgets as widgets
import ipyleaflet
from ipydatagrid import DataGrid
import pandas as pd
import numpy as np

from ..waypoint import Waypoint
from ..units import ureg
from ._state import PlannerState

_SELECTED_COLOR = "#e74c3c"
_DEFAULT_COLOR = "#3388ff"


class WaypointEditor(widgets.VBox):
    """Place and edit waypoints on an ipyleaflet map with a data table.

    Args:
        state: Shared :class:`PlannerState` instance.
        leaflet_map: An existing :class:`ipyleaflet.Map` to draw on.
        default_altitude: Default altitude for new waypoints (meters or
            pint Quantity with length units).  Defaults to ``1000 * ureg.meter``.
        default_heading: Default heading for new waypoints (degrees).
    """

    def __init__(
        self,
        state: PlannerState,
        leaflet_map: ipyleaflet.Map,
        *,
        default_altitude=None,
        default_heading: float = 0.0,
    ):
        super().__init__()
        self._state = state
        self._map = leaflet_map
        self._default_altitude = default_altitude or 1000.0 * ureg.meter
        self._default_heading = default_heading
        self._markers: list[ipyleaflet.Marker] = []
        self._adding = False
        self._selected_index: int | None = None

        # --- Buttons ---
        self._add_btn = widgets.ToggleButton(
            value=False, description="Add Waypoint",
            icon="plus", tooltip="Click map to place a waypoint",
        )
        self._delete_btn = widgets.Button(
            description="Delete", icon="trash",
            tooltip="Delete selected waypoint",
            button_style="danger",
        )
        self._delete_btn.disabled = True
        toolbar = widgets.HBox([self._add_btn, self._delete_btn])

        # --- Data grid ---
        self._grid = DataGrid(
            pd.DataFrame(columns=["name", "lat", "lon", "heading", "alt_m", "speed_m_s", "delay_s"]),
            editable=True,
            selection_mode="row",
            layout=widgets.Layout(height="200px"),
        )

        self.children = [toolbar, self._grid]

        # --- Wiring ---
        self._add_btn.observe(self._on_add_toggle, names="value")
        self._delete_btn.on_click(self._on_delete)
        self._grid.on_cell_change(self._on_cell_edited)
        self._state.observe(self._on_waypoints_changed, names=["waypoints"])

        # Render existing waypoints
        self._sync_from_state()

    # ------------------------------------------------------------------
    # Map interaction
    # ------------------------------------------------------------------

    def _on_add_toggle(self, change):
        if change["new"]:
            self._map.on_interaction(self._on_map_click)
            self._map.style.cursor = "crosshair"
        else:
            self._map.on_interaction(self._on_map_click, remove=True)
            self._map.style.cursor = "grab"

    def _on_map_click(self, **kwargs):
        if kwargs.get("type") != "click":
            return
        coords = kwargs["coordinates"]
        lat, lon = coords[0], coords[1]
        wp = Waypoint(
            latitude=lat, longitude=lon,
            heading=self._default_heading,
            altitude_msl=self._default_altitude,
            name=f"WP{len(self._state.waypoints) + 1}",
        )
        self._state.append_waypoint(wp)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------

    def _add_marker(self, index: int, wp: Waypoint):
        marker = ipyleaflet.Marker(
            location=(wp.latitude, wp.longitude),
            draggable=True,
            title=wp.name or "",
        )
        marker._wp_index = index

        def on_drag(change, idx=index):
            loc = change["new"]
            self._update_waypoint_position(idx, loc[0], loc[1])

        marker.observe(on_drag, names=["location"])
        self._map.add(marker)
        self._markers.append(marker)

    def _clear_markers(self):
        for m in self._markers:
            try:
                self._map.remove(m)
            except Exception:
                pass
        self._markers.clear()

    def _update_waypoint_position(self, index: int, lat: float, lon: float):
        """Update a waypoint's lat/lon from a marker drag."""
        wps = list(self._state.waypoints)
        if index >= len(wps):
            return
        old = wps[index]
        wps[index] = Waypoint(
            latitude=lat, longitude=lon,
            heading=old.heading,
            altitude_msl=old.altitude_msl,
            name=old.name,
            speed=old.speed,
            delay=old.delay,
            headwind=old.headwind,
            segment_type=old.segment_type,
        )
        # Temporarily disconnect observer to avoid loop
        self._state.unobserve(self._on_waypoints_changed, names=["waypoints"])
        self._state.waypoints = wps
        self._refresh_grid()
        self._state.observe(self._on_waypoints_changed, names=["waypoints"])

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _refresh_grid(self):
        wps = self._state.waypoints
        if not wps:
            self._grid.data = pd.DataFrame(
                columns=["name", "lat", "lon", "heading", "alt_m", "speed_m_s", "delay_s"]
            )
            return

        rows = []
        for wp in wps:
            alt = wp.altitude_msl.to(ureg.meter).magnitude if wp.altitude_msl is not None else np.nan
            spd = wp.speed.to(ureg.meter / ureg.second).magnitude if wp.speed is not None else np.nan
            dly = wp.delay.to(ureg.second).magnitude if wp.delay is not None else np.nan
            rows.append({
                "name": wp.name or "",
                "lat": round(wp.latitude, 6),
                "lon": round(wp.longitude, 6),
                "heading": round(wp.heading, 1),
                "alt_m": round(alt, 1) if not np.isnan(alt) else np.nan,
                "speed_m_s": round(spd, 2) if not np.isnan(spd) else np.nan,
                "delay_s": round(dly, 1) if not np.isnan(dly) else np.nan,
            })
        self._grid.data = pd.DataFrame(rows)

    def _on_cell_edited(self, cell):
        row = cell["row"]
        col = cell["column"]
        value = cell["value"]
        wps = list(self._state.waypoints)
        if row >= len(wps):
            return
        old = wps[row]

        kwargs = dict(
            latitude=old.latitude,
            longitude=old.longitude,
            heading=old.heading,
            altitude_msl=old.altitude_msl,
            name=old.name,
            speed=old.speed,
            delay=old.delay,
            headwind=old.headwind,
            segment_type=old.segment_type,
        )

        if col == "name":
            kwargs["name"] = str(value)
        elif col == "lat":
            kwargs["latitude"] = float(value)
        elif col == "lon":
            kwargs["longitude"] = float(value)
        elif col == "heading":
            kwargs["heading"] = float(value)
        elif col == "alt_m":
            kwargs["altitude_msl"] = float(value) * ureg.meter if not np.isnan(float(value)) else None
        elif col == "speed_m_s":
            kwargs["speed"] = float(value) * (ureg.meter / ureg.second) if not np.isnan(float(value)) else None
        elif col == "delay_s":
            kwargs["delay"] = float(value) * ureg.second if not np.isnan(float(value)) else None

        wps[row] = Waypoint(**kwargs)
        self._state.unobserve(self._on_waypoints_changed, names=["waypoints"])
        self._state.waypoints = wps
        self._sync_markers()
        self._state.observe(self._on_waypoints_changed, names=["waypoints"])

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _on_delete(self, _btn):
        sel = self._grid.selections
        if not sel:
            return
        # ipydatagrid selections: list of dicts with r1, r2, c1, c2
        row = sel[0].get("r1", None)
        if row is not None and row < len(self._state.waypoints):
            self._state.remove_waypoint(row)

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def _on_waypoints_changed(self, change):
        self._sync_from_state()

    def _sync_from_state(self):
        self._clear_markers()
        self._sync_markers()
        self._refresh_grid()
        self._delete_btn.disabled = len(self._state.waypoints) == 0

    def _sync_markers(self):
        self._clear_markers()
        for i, wp in enumerate(self._state.waypoints):
            self._add_marker(i, wp)
