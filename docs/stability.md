# API Stability

HyPlan uses stability levels to set expectations about which modules
have settled APIs and which may still change.

## What the levels mean

**Stable** — The public API (function signatures, return types, class
constructors) will not change in backward-incompatible ways within a
major version (1.x).  Bug fixes and new optional parameters are
permitted.  If a breaking change becomes necessary, the old interface
will be deprecated for at least one minor release before removal.

**Experimental** — The module works and is tested, but the API may
change between minor releases based on user feedback.  Pin to a
specific version if you depend on exact signatures.

## Module stability levels

### Stable

| Module | Description |
|--------|-------------|
| `hyplan.FlightLine` | Flight line creation, splitting, clipping, offsetting |
| `hyplan.Waypoint` | Navigation waypoints with heading, altitude, speed |
| `hyplan.planning` | `compute_flight_plan` and segment record builders |
| `hyplan.terrain` | DEM download, elevation lookup, ray-terrain intersection |
| `hyplan.swath` | Swath polygon generation and gap/overlap analysis |
| `hyplan.winds` | Wind field abstractions, MERRA-2, GMAO, GFS providers |
| `hyplan.aircraft` | Aircraft base class and pre-configured models |
| `hyplan.atmosphere` | ISA model and airspeed conversions (CAS/TAS/Mach) |
| `hyplan.exports` | Flight plan output formats (Excel, CSV, KML, GPX, FMS, ICARTT) |
| `hyplan.airports` | Airport database lookup and nearest-airport search |
| `hyplan.flight_box` | Flight box generation from center lines and polygons |
| `hyplan.flight_optimizer` | Graph-based flight line ordering |
| `hyplan.sun` | Solar position and illumination windows |
| `hyplan.geometry` | Geodesic math and coordinate utilities |
| `hyplan.units` | Pint unit registry and conversion helpers |

```{note}
Aircraft models are stable in their API, but individual aircraft
parameters (cruise speed, climb rate, endurance) will continue to be
refined as ADS-B calibration data becomes available.
```

### Experimental

| Module | Description |
|--------|-------------|
| `hyplan.instruments.LVIS` | Full-waveform lidar sensor model |
| `hyplan.instruments.SidelookingRadar` | SAR sensor model (UAVSAR variants) |
| `hyplan.instruments.FrameCamera` | Frame camera and multi-camera rig |
| `hyplan.gui` | Interactive Jupyter widgets (waypoint editor, flight line manager) |
| `hyplan.campaign` | Campaign management and airspace conflict detection |
| `hyplan.clouds` | Cloud fraction climatology and forecasts |
| `hyplan.phenology` | Vegetation phenology from MODIS |
| `hyplan.satellites` | Satellite overpass prediction |
| `hyplan.glint` | Specular reflection (sun glint) prediction |
| `hyplan.airspace` | Airspace conflict detection (OpenAIP, FAA TFR/NASR) |
| `hyplan.dubins3d` | 3D minimum-turn path planning |
| `hyplan.flight_patterns` | Pattern generators (racetrack, rosette, spiral, etc.) |
| `hyplan.plotting` | Folium maps, altitude profiles, terrain cross-sections |

## Deprecation policy

When a stable API must change:

1. The old interface is preserved and emits a `DeprecationWarning` for
   at least one minor release.
2. The deprecation message names the replacement.
3. The old interface is removed in the next major version.

Experimental modules may change without a deprecation cycle, but
changes will be documented in the release notes.
