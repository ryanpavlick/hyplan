# HyPlan

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-green.svg)](https://ryanpavlick.github.io/hyplan/)

An open-source Python library for planning airborne remote sensing campaigns.

HyPlan helps scientists and engineers design remote sensing flight missions. It handles flight line generation, sensor modeling, swath coverage, solar glint prediction, cloud analysis, terrain-aware calculations, and mission logistics including airport selection and aircraft performance.

```
 Study Area          Flight Lines      Sensor Swaths         Mission Plan
 ┌─────────┐        ┌─────────┐         ┌─────────┐          ┌─────────┐
 │ ▓▓▓▓▓▓▓ │        │ ──────► │         │▒──────►▒│          │ ✈ ─ ─ ► │
 │ ▓▓▓▓▓▓▓ │  ───►  │ ◄────── │  ───►   │▒◄──────▒│  ───►    │ ──────► │
 │ ▓▓▓▓▓▓▓ │        │ ──────► │         │▒──────►▒│          │ ◄────── │
 └─────────┘        └─────────┘         └─────────┘          │ ─ ─ ✈ ◄ │
  define area      generate lines     compute coverage       └─────────┘
                                                             plan & optimize
```

## Features

- **Flight planning** — Define flight lines, generate multi-line coverage patterns, and compute complete mission plans with takeoff, transit, data collection, and landing segments
- **Flight optimization** — Automatically order flight lines with multi-day scheduling, endurance constraints, and refueling stops
- **Sensor modeling** — Pre-configured NASA instruments (AVIRIS-3, AVIRIS-5, HyTES, PRISM, MASTER, and more) with ground sample distance and swath calculations
- **Lidar & radar** — LVIS full-waveform lidar and UAVSAR L/P/Ka-band SAR sensor models
- **Solar glint prediction** — Predict glint angles across flight lines for water observation missions
- **Solar illumination** — Compute solar position and daily data-collection windows for any site and date
- **Terrain-aware analysis** — Download DEM data and compute where the sensor field of view intersects the ground
- **Cloud cover analysis** — Estimate clear-sky probability from MODIS imagery via Google Earth Engine
- **Aircraft performance** — 14 pre-configured aircraft models (NASA ER-2, WB-57, G-III, B200, Twin Otter, and others) with climb/cruise/descent profiles
- **Airport logistics** — Search and filter airports by location, runway length, surface type, and country
- **Satellite coordination** — Predict satellite overpasses and compute ground-track swaths for 14+ satellites
- **Dubins path planning** — Minimum-radius turning trajectories between waypoints for realistic aircraft maneuvering
- **Flight patterns** — Generate racetrack, rosette, spiral, sawtooth, polygon, and coordinated dual-aircraft (five-point line) flight patterns for profiling, survey, and multi-platform missions
- **Geospatial export** — Output to Excel, KML, GPX, ForeFlight CSV, Honeywell FMS, ER-2, ICARTT, and interactive Folium maps
- **Interactive planning** — Jupyter widget for map-based waypoint placement, pattern generation, and flight box creation

---

## Installation

### Requirements

- Python 3.9+
- GDAL (must be installed at the system level before `pip install`)

On macOS:
```bash
brew install gdal
```

On Ubuntu/Debian:
```bash
sudo apt-get install gdal-bin libgdal-dev
```

With conda/mamba (recommended — handles GDAL automatically):
```bash
mamba install gdal
```

### Option 1: pip

```bash
git clone https://github.com/ryanpavlick/hyplan
cd hyplan
pip install -e .
```

### Option 2: conda/mamba

```bash
mamba env create --name hyplan --file environment.yml
mamba activate hyplan
pip install -e .
```

### Optional dependencies

- **Google Earth Engine** (`earthengine-api`) — required for `hyplan.clouds`
- **Interactive planning** (`ipyleaflet`, `ipywidgets`, `ipydatagrid`) — install with `pip install hyplan[interactive]`

---

## Quick Start

### Define a flight line

```python
from hyplan import FlightLine, ureg

flight_line = FlightLine.start_length_azimuth(
    lat1=34.05, lon1=-118.25,
    length=ureg.Quantity(50, "km"),
    az=45.0,
    altitude_msl=ureg.Quantity(20000, "feet"),
    site_name="LA Northeast",
)
```

### Compute a flight plan

```python
from hyplan import DynamicAviation_B200, Airport, compute_flight_plan, plot_flight_plan

aircraft = DynamicAviation_B200()
departure = Airport("KSBA")
destination = Airport("KBUR")

plan = compute_flight_plan(aircraft, [flight_line], departure, destination)
plot_flight_plan(plan, departure, destination, [flight_line])
```

### Optimize flight line ordering

```python
from hyplan import greedy_optimize

result = greedy_optimize(
    aircraft=aircraft,
    flight_lines=[flight_line],
    airports=[departure, destination],
    takeoff_airport=departure,
    return_airport=destination,
    max_endurance=4.0,
    max_daily_flight_time=8.0,
    max_days=3,
)
print(f"Lines covered: {result['lines_covered']}, Days: {result['days_used']}")
```

### Predict solar glint

```python
from datetime import datetime, timezone
from hyplan import AVIRIS3
from hyplan.glint import compute_glint_vectorized

sensor = AVIRIS3()
obs_time = datetime(2025, 2, 17, 18, 0, tzinfo=timezone.utc)

gdf = compute_glint_vectorized(flight_line, sensor, obs_time)
gdf.to_file("glint_results.geojson", driver="GeoJSON")
```

---

## Modules

```
                          ┌──────────────────────┐
                          │     Flight Planning  │
                          ├──────────────────────┤
                          │  flight_line         │
                          │  flight_box          │
                          │  flight_plan         │
                          │  flight_optimizer    │
                          │  flight_patterns     │
                          │  dubins3d            │
                          │  waypoint            │
                          └────────┬─────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │   Instruments    │  │    Environment  │  │    Logistics    │
   ├──────────────────┤  ├─────────────────┤  ├─────────────────┤
   │  instruments/    │  │  sun            │  │  aircraft       │
   │   line_scanner   │  │  glint          │  │  airports       │
   │   lvis           │  │  terrain        │  │  satellites     │
   │   radar          │  │  clouds         │  │  units          │
   │   frame_camera   │  │  geometry       │  │  plotting       │
   │  swath           │  │                 │  │                 │
   └──────────────────┘  └─────────────────┘  ├─────────────────┤
                                              │  exports        │
                                              │  interactive    │
                                              └─────────────────┘
```

| Module | Description |
|--------|-------------|
| `flight_line` | Create, modify, split, clip, and export individual flight lines |
| `flight_box` | Generate parallel flight lines covering a geographic area |
| `flight_plan` | Compute complete mission plans with timing and altitude profiles |
| `flight_optimizer` | Graph-based flight line ordering with multi-day and refueling support |
| `aircraft` | Aircraft performance models (14 pre-configured research aircraft) |
| `instruments` | All sensor models — line scanners (AVIRIS-3, AVIRIS-5, HyTES, PRISM, MASTER, etc.), LVIS lidar, UAVSAR SAR, and frame cameras |
| `glint` | Solar glint angle prediction for water observations |
| `swath` | Sensor swath coverage with terrain integration |
| `terrain` | DEM data acquisition and ray-terrain intersection |
| `sun` | Solar position and timing calculations |
| `clouds` | Cloud cover analysis and clear-sky probability from MODIS |
| `satellites` | Satellite overpass prediction and swath modeling |
| `airports` | Airport database with search, filtering, and runway data |
| `flight_patterns` | Flight pattern generators (racetrack, rosette, spiral, sawtooth, polygon, coordinated line) |
| `waypoint` | Waypoint class for flight planning with altitude, heading, and speed |
| `dubins3d` | 3D Dubins path planning with pitch constraints (Vana et al., ICRA 2020) |
| `exports` | Export flight plans to Excel, KML, GPX, ForeFlight, Honeywell FMS, ER-2, ICARTT |
| `interactive` | Interactive Jupyter map-based flight planning with ipyleaflet |
| `geometry` | Geospatial utilities (haversine, coordinate transforms, polygons) |
| `units` | Unit conversions using Pint (meters, feet, knots, etc.) |
| `plotting` | Interactive Folium map generation and altitude profiles |

---

## Notebooks

The [`notebooks/`](notebooks/) directory contains Jupyter notebooks with interactive tutorials and visualizations covering every HyPlan module:

### Getting Started

| Notebook | Description |
|----------|-------------|
| [tutorial.ipynb](notebooks/tutorial.ipynb) | End-to-end workflow: sensor setup, flight box generation, solar checks, airport selection, optimization, flight planning, and map visualization |
| [validation.ipynb](notebooks/validation.ipynb) | Validates HyPlan calculations against reference values (Vincenty, NOAA solar, analytical swath/GSD) |

### Flight Planning

| Notebook | Description |
|----------|-------------|
| [flight_line_operations.ipynb](notebooks/flight_line_operations.ipynb) | Creating, clipping, splitting, offsetting, rotating, and exporting flight lines |
| [flight_box_generation.ipynb](notebooks/flight_box_generation.ipynb) | Generating parallel flight lines over study areas with swath overlap control |
| [flight_plan_computation.ipynb](notebooks/flight_plan_computation.ipynb) | Segment-by-segment flight plans with altitude profiles and map visualization |
| [flight_optimizer_demo.ipynb](notebooks/flight_optimizer_demo.ipynb) | Greedy line ordering with endurance constraints, refueling, and multi-day scheduling |
| [dubins_path_planning.ipynb](notebooks/dubins_path_planning.ipynb) | Dubins path basics: turn radius, speed/bank effects, and flight line integration |
| [airport_selection.ipynb](notebooks/airport_selection.ipynb) | Finding, filtering, and comparing airports by location, runway, and aircraft requirements |
| [flight_patterns.ipynb](notebooks/flight_patterns.ipynb) | Racetrack, rosette, spiral, sawtooth, and polygon flight patterns |
| [interactive_planning.ipynb](notebooks/interactive_planning.ipynb) | Interactive map-based flight planning with waypoint editing and pattern generation |

### Instruments & Sensors

| Notebook | Description |
|----------|-------------|
| [sensor_comparison.ipynb](notebooks/sensor_comparison.ipynb) | Comparing GSD, swath width, and critical speed across imaging spectrometers |
| [frame_camera_planning.ipynb](notebooks/frame_camera_planning.ipynb) | Frame camera FOV, footprints, GSD, and along-track sampling |
| [lidar_lvis_planning.ipynb](notebooks/lidar_lvis_planning.ipynb) | LVIS lens options, swath geometry, contiguous coverage, and coverage rates |
| [radar_sar_missions.ipynb](notebooks/radar_sar_missions.ipynb) | UAVSAR L/P/Ka-band swath geometry, resolution, and InSAR line spacing |

### Environment & Conditions

| Notebook | Description |
|----------|-------------|
| [solar_planning.ipynb](notebooks/solar_planning.ipynb) | Solar azimuth/elevation, daily collection windows, seasonal and cross-site comparisons |
| [glint_analysis.ipynb](notebooks/glint_analysis.ipynb) | Glint angle prediction, heading optimization, and time-of-day effects for aquatic missions |
| [glint_arc_planning.ipynb](notebooks/glint_arc_planning.ipynb) | GlintArc geometry for specular reflection flight paths over water |
| [terrain_aware_planning.ipynb](notebooks/terrain_aware_planning.ipynb) | DEM-based terrain profiles, AGL variation effects on GSD and swath |
| [cloud_analysis.ipynb](notebooks/cloud_analysis.ipynb) | MODIS cloud cover from Google Earth Engine, visit simulation, campaign duration planning |

### Aircraft & Satellites

| Notebook | Description |
|----------|-------------|
| [aircraft_performance.ipynb](notebooks/aircraft_performance.ipynb) | Fleet comparison, speed profiles, climb/descent performance, range/endurance, custom aircraft |
| [satellite_coordination.ipynb](notebooks/satellite_coordination.ipynb) | Satellite ground tracks, overpass prediction, and multi-satellite search |

### Export & Integration

| Notebook | Description |
|----------|-------------|
| [export_formats.ipynb](notebooks/export_formats.ipynb) | Export flight plans to Excel, KML, GPX, ForeFlight, Honeywell FMS, ER-2, ICARTT, and text formats |

---

## Documentation

Full API documentation is available at **[ryanpavlick.github.io/hyplan](https://ryanpavlick.github.io/hyplan/)**.

To build the documentation locally:

```bash
pip install sphinx myst-parser furo sphinx-autodoc-typehints
cd docs
make html
```

---

## Known Limitations

- **Flight optimizer** — Uses a greedy nearest-neighbor heuristic; does not guarantee globally optimal ordering; still experimental
- **Terrain module** — DEM downloads require internet access and may be slow for large areas
- **Google Earth Engine** — The `clouds` module requires a Google Earth Engine account and authentication

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch
2. Install in development mode: `pip install -e .`
3. Run the notebooks in `notebooks/` to verify your changes
4. Submit a pull request

Please open an [issue](https://github.com/ryanpavlick/hyplan/issues) for bug reports, feature requests, or questions.

---

<!-- ## Citation

If you use HyPlan in your research, please cite it as:

```bibtex
@software{hyplan,
  author = {Pavlick, Ryan},
  title = {HyPlan: Planning Software for Airborne Remote Sensing Campaigns},
  url = {https://github.com/ryanpavlick/hyplan},
  license = {Apache-2.0}
}
``` -->

## License

HyPlan is licensed under the Apache License, Version 2.0. See [`LICENSE.md`](LICENSE.md) for details.

## Contact

For inquiries or further information, please contact Ryan Pavlick (ryan.p.pavlick@nasa.gov).
