"""Interactive flight line manager widget.

Add, select/deselect, and reorder flight lines on an ipyleaflet map
with a sidebar list for sequencing.
"""

from __future__ import annotations

import ipywidgets as widgets
import ipyleaflet

from ..flight_line import FlightLine
from ..waypoint import Waypoint
from ..units import ureg
from ._state import PlannerState

_COLOR_SELECTED = "#3388ff"
_COLOR_DESELECTED = "#aaaaaa"
_COLOR_DRAWING = "#e74c3c"


class FlightLineManager(widgets.VBox):
    """Manage flight lines for flight plan sequencing.

    Renders flight lines on an ipyleaflet map. Click a line to toggle its
    selection; use the sidebar controls to reorder selected lines.

    Args:
        state: Shared :class:`PlannerState` instance.
        leaflet_map: An existing :class:`ipyleaflet.Map` to draw on.
        flight_lines: Optional initial list of
            :class:`~hyplan.flight_line.FlightLine` objects to load.
    """

    def __init__(
        self,
        state: PlannerState,
        leaflet_map: ipyleaflet.Map,
        *,
        flight_lines: list[FlightLine] | None = None,
    ):
        super().__init__()
        self._state = state
        self._map = leaflet_map
        self._polylines: list[ipyleaflet.Polyline] = []

        # Load initial flight lines if provided
        if flight_lines:
            self._state.flight_lines = list(flight_lines)
            self._state.selected_indices = list(range(len(flight_lines)))

        # --- "Add Line" mode controls ---
        self._add_btn = widgets.ToggleButton(
            value=False, description="Add Line",
            icon="plus", tooltip="Click two points on map to create a line",
        )
        self._name_input = widgets.Text(
            value="", placeholder="Line name",
            layout=widgets.Layout(width="120px"),
        )
        self._alt_input = widgets.FloatText(
            value=1000.0, description="Alt (m):",
            layout=widgets.Layout(width="180px"),
        )
        self._delete_btn = widgets.Button(
            description="Delete Selected", icon="trash",
            button_style="danger",
        )
        self._select_all_btn = widgets.Button(
            description="Select All", icon="check-square",
        )
        self._deselect_all_btn = widgets.Button(
            description="Deselect All", icon="square",
        )
        toolbar = widgets.HBox([
            self._add_btn, self._name_input, self._alt_input,
        ])
        action_bar = widgets.HBox([
            self._select_all_btn, self._deselect_all_btn, self._delete_btn,
        ])

        # --- Sidebar: ordered list of selected lines ---
        self._sidebar_label = widgets.HTML("<b>Selected lines (flight order):</b>")
        self._sidebar_box = widgets.VBox()
        sidebar = widgets.VBox(
            [self._sidebar_label, self._sidebar_box],
            layout=widgets.Layout(
                min_width="250px", max_height="300px",
                overflow_y="auto", border="1px solid #ccc",
                padding="4px",
            ),
        )

        self.children = [toolbar, action_bar, sidebar]

        # --- Draw state ---
        self._draw_start = None  # first click lat/lon

        # --- Wiring ---
        self._add_btn.observe(self._on_add_toggle, names="value")
        self._delete_btn.on_click(self._on_delete_selected)
        self._select_all_btn.on_click(self._on_select_all)
        self._deselect_all_btn.on_click(self._on_deselect_all)
        self._state.observe(self._on_state_changed, names=["flight_lines", "selected_indices"])

        # Initial render
        self._sync_from_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_lines(self) -> list[FlightLine]:
        """Return selected flight lines in the user-chosen order."""
        fls = self._state.flight_lines
        return [fls[i] for i in self._state.selected_indices if i < len(fls)]

    # ------------------------------------------------------------------
    # "Add Line" mode
    # ------------------------------------------------------------------

    def _on_add_toggle(self, change):
        if change["new"]:
            self._draw_start = None
            self._map.on_interaction(self._on_map_click_draw)
            self._map.style.cursor = "crosshair"
        else:
            self._draw_start = None
            self._map.on_interaction(self._on_map_click_draw, remove=True)
            self._map.style.cursor = "grab"

    def _on_map_click_draw(self, **kwargs):
        if kwargs.get("type") != "click":
            return
        coords = kwargs["coordinates"]
        lat, lon = coords[0], coords[1]

        if self._draw_start is None:
            # First click — store start point
            self._draw_start = (lat, lon)
        else:
            # Second click — create the flight line
            lat1, lon1 = self._draw_start
            name = self._name_input.value or f"L{len(self._state.flight_lines) + 1}"
            alt = self._alt_input.value * ureg.meter

            wp1 = Waypoint(latitude=lat1, longitude=lon1, heading=0.0,
                           altitude_msl=alt, name=f"{name}_start")
            wp2 = Waypoint(latitude=lat, longitude=lon, heading=0.0,
                           altitude_msl=alt, name=f"{name}_end")
            fl = FlightLine(wp1, wp2, site_name=name)
            new_idx = len(self._state.flight_lines)
            self._state.add_flight_line(fl)
            # Auto-select the new line
            self._state.selected_indices = list(self._state.selected_indices) + [new_idx]

            self._draw_start = None
            self._add_btn.value = False

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _toggle_selection(self, index: int):
        sel = list(self._state.selected_indices)
        if index in sel:
            sel.remove(index)
        else:
            sel.append(index)
        self._state.selected_indices = sel

    def _on_select_all(self, _btn):
        self._state.selected_indices = list(range(len(self._state.flight_lines)))

    def _on_deselect_all(self, _btn):
        self._state.selected_indices = []

    def _on_delete_selected(self, _btn):
        # Delete selected lines in reverse order to preserve indices
        for idx in sorted(self._state.selected_indices, reverse=True):
            self._state.remove_flight_line(idx)

    # ------------------------------------------------------------------
    # Reorder
    # ------------------------------------------------------------------

    def _move_up(self, index_in_selection: int):
        sel = list(self._state.selected_indices)
        if index_in_selection <= 0:
            return
        sel[index_in_selection - 1], sel[index_in_selection] = (
            sel[index_in_selection], sel[index_in_selection - 1]
        )
        self._state.selected_indices = sel

    def _move_down(self, index_in_selection: int):
        sel = list(self._state.selected_indices)
        if index_in_selection >= len(sel) - 1:
            return
        sel[index_in_selection], sel[index_in_selection + 1] = (
            sel[index_in_selection + 1], sel[index_in_selection]
        )
        self._state.selected_indices = sel

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_polylines(self):
        for pl in self._polylines:
            try:
                self._map.remove(pl)
            except Exception:
                pass
        self._polylines.clear()

    def _sync_from_state(self):
        self._clear_polylines()
        fls = self._state.flight_lines
        sel = set(self._state.selected_indices)

        for i, fl in enumerate(fls):
            color = _COLOR_SELECTED if i in sel else _COLOR_DESELECTED
            weight = 3 if i in sel else 2
            opacity = 1.0 if i in sel else 0.5

            pl = ipyleaflet.Polyline(
                locations=[(fl.lat1, fl.lon1), (fl.lat2, fl.lon2)],
                color=color,
                weight=weight,
                opacity=opacity,
            )
            pl._fl_index = i

            def make_click_handler(idx):
                def handler(**kwargs):
                    self._toggle_selection(idx)
                return handler

            pl.on_click(make_click_handler(i))
            self._map.add(pl)
            self._polylines.append(pl)

        self._refresh_sidebar()

    def _refresh_sidebar(self):
        fls = self._state.flight_lines
        sel = self._state.selected_indices
        items = []
        for pos, fl_idx in enumerate(sel):
            if fl_idx >= len(fls):
                continue
            fl = fls[fl_idx]
            label = widgets.HTML(
                f"<b>{pos + 1}.</b> {fl.site_name or f'Line {fl_idx + 1}'}",
                layout=widgets.Layout(width="140px"),
            )
            up_btn = widgets.Button(icon="arrow-up", layout=widgets.Layout(width="32px"))
            down_btn = widgets.Button(icon="arrow-down", layout=widgets.Layout(width="32px"))

            def make_up(p):
                def handler(_):
                    self._move_up(p)
                return handler

            def make_down(p):
                def handler(_):
                    self._move_down(p)
                return handler

            up_btn.on_click(make_up(pos))
            down_btn.on_click(make_down(pos))
            items.append(widgets.HBox([label, up_btn, down_btn]))

        if not items:
            items = [widgets.HTML("<i>No lines selected</i>")]
        self._sidebar_box.children = items

    def _on_state_changed(self, change):
        self._sync_from_state()
