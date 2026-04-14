# Developer Guide

This page describes the internal layout of HyPlan and the conventions
to follow when adding new modules or extending existing ones.

## Package layout

HyPlan uses a mix of multi-file packages and single-file modules. The
rule of thumb: a module becomes a package when it exceeds ~500 lines or
when it has clearly separable concerns (e.g. base class vs providers).

| Package | Contents |
|---------|----------|
| `aircraft/` | `_base.py` (base class), `_models.py` (15 aircraft), `adsb/` (internal) |
| `instruments/` | `_base.py`, `line_scanner.py`, `lvis.py`, `frame_camera.py`, `radar.py` |
| `winds/` | `base.py`, `simple.py`, `gridded.py`, `factory.py`, `utils.py`, `providers/` |
| `planning/` | `engine.py` (orchestrator), `segments.py` (record builders) |
| `exports/` | `_common.py` (shared), one file per format (`excel.py`, `csv.py`, ...) |
| `gui/` | `_state.py` (shared state), `waypoint_editor.py`, `flight_line_manager.py` |

Single-file modules (`terrain.py`, `flight_line.py`, `dubins3d.py`, etc.)
stay as single files until they outgrow their scope.

### Naming conventions

- `_base.py`, `_common.py` — internal implementation, not imported directly
- `_models.py` — concrete instances (aircraft definitions, sensor presets)
- Leading underscore on classes (`_GriddedWindField`) — internal base, not
  part of the public API but re-exported for `isinstance` checks

## Re-export pattern

Every package has an `__init__.py` that re-exports its public API:

```python
from ._base import Aircraft  # noqa: F401
from ._models import NASA_GV  # noqa: F401

__all__ = ["Aircraft", "NASA_GV"]
```

The top-level `hyplan/__init__.py` then re-exports key names from each
package so that `from hyplan import Aircraft, FlightLine` works. When
adding a new public name, add it to both the package `__init__.py` and
the top-level `__init__.py` `__all__` list.

Backward-compatible shims (e.g. `flight_plan.py` re-exporting from
`planning/`) preserve old import paths after refactors. These are thin
files with only re-imports.

## Extending HyPlan

### Adding a wind provider

1. Create `hyplan/winds/providers/mydata.py`
2. Subclass `_GriddedWindField` from `hyplan.winds.gridded`
3. Implement the required hooks:
   - `_build_urls()` — return OPeNDAP/download URLs for the time range
   - `_open_dataset(url)` — open a single dataset (override for auth)
   - Override `_var_names()`, `_dim_names()`, `_decode_time()`,
     `_time_slice()` if the data source uses non-standard conventions
   - Or override `_fetch_slab()` entirely for non-OPeNDAP sources (see `GFSWindField`)
4. Re-export from `hyplan/winds/providers/__init__.py`
5. Add a branch in `hyplan/winds/factory.py` → `wind_field_from_plan()`
6. Add tests in `tests/test_winds.py`

### Adding a sensor

1. Subclass `Sensor` (or `LineScanner`, `SidelookingRadar`, etc.) in the
   appropriate file under `hyplan/instruments/`
2. Implement `half_angle` and `swath_width(altitude_agl)` at minimum
3. Add the class to `hyplan/instruments/__init__.py` re-exports and `__all__`
4. Add to `SENSOR_REGISTRY` in `_base.py` if it should be discoverable
   via `create_sensor(name)`

### Adding an export format

1. Create `hyplan/exports/myformat.py` with a `to_myformat(flight_plan_gdf, path)` function
2. Re-export from `hyplan/exports/__init__.py`
3. Add to `hyplan/__init__.py` if it should be a top-level import

## Where future refactors should land

| If you are adding... | Put it in... |
|----------------------|-------------|
| A new data source for winds | `winds/providers/` |
| Wind vector math or heading/track solvers | `winds/utils.py` |
| A new flight pattern generator | `flight_patterns.py` |
| A new segment type or record builder | `planning/segments.py` |
| Changes to the planning orchestrator | `planning/engine.py` |
| A new aircraft model | `aircraft/_models.py` |
| A new sensor class | `instruments/<type>.py` |
| A new output format | `exports/<format>.py` |
| Coordinate math or projection helpers | `geometry.py` |
| DEM or elevation helpers | `terrain.py` |

If a single-file module grows past ~500 lines with separable concerns,
follow the existing pattern: create a package directory, split into
focused files, add `__init__.py` re-exports, and leave a shim at the
old path for backward compatibility.

## Testing conventions

- All tests live in a flat `tests/` directory (no mirroring of package structure)
- One test file per top-level module or package: `test_winds.py`, `test_flight_plan.py`, etc.
- Import from the public API: `from hyplan.winds import ConstantWindField`
- Private functions can be tested by importing from canonical locations:
  `from hyplan.winds.utils import _wind_factor`
- Shared fixtures go in `tests/conftest.py`
- Airport data initialization uses `@pytest.fixture(scope="module")` to
  avoid repeated downloads

## CI workflows

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `tests.yml` | Push/PR to main | Lint (ruff), type check (mypy, non-blocking), pytest with coverage on Python 3.9/3.11/3.12 |
| `docs.yml` | Push to main | Build Sphinx docs and deploy to GitHub Pages |
| `notebooks.yml` | Nightly + PR (if notebooks changed) | Execute tutorial, exports, and aircraft notebooks via papermill |
