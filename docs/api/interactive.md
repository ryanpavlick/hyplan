# Interactive Widgets

Lightweight Jupyter widgets for interactive flight planning in notebooks.
Requires the `interactive` optional dependencies:

```bash
pip install hyplan[interactive]
```

## Shared state

```{eval-rst}
.. autoclass:: hyplan.interactive.PlannerState
   :members:
   :show-inheritance:
```

## WaypointEditor

Click on an ipyleaflet map to place waypoints, drag markers to reposition,
and edit properties (name, heading, altitude, speed) in a synchronized
ipydatagrid table.

```{eval-rst}
.. autoclass:: hyplan.interactive.WaypointEditor
   :members:
   :show-inheritance:
```

## FlightLineManager

Render flight lines on an ipyleaflet map. Click a line to toggle its
selection; use the sidebar controls to reorder selected lines for
flight plan sequencing.

```{eval-rst}
.. autoclass:: hyplan.interactive.FlightLineManager
   :members: selected_lines
   :show-inheritance:
```
