# HyPlan

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-green.svg)](https://ryanpavlick.github.io/hyplan/)

An open-source Python library for planning airborne remote sensing campaigns.

HyPlan helps scientists and engineers design remote sensing flight missions. It handles flight line generation, sensor modeling, swath coverage, solar glint prediction, cloud analysis, terrain-aware calculations, and mission logistics including airport selection and aircraft performance.

## Features

- **Flight planning** — Define flight lines, generate multi-line coverage patterns, and compute complete mission plans with takeoff, transit, data collection, and landing segments
- **Flight optimization** — Automatically order flight lines with multi-day scheduling, endurance constraints, and refueling stops
- **Sensor modeling** — Pre-configured NASA instruments (AVIRIS-3, AVIRIS-NG, HyTES, PRISM, MASTER, and more) with ground sample distance and swath calculations
- **Solar glint prediction** — Predict glint angles across flight lines for water observation missions
- **Terrain-aware analysis** — Download DEM data and compute where the sensor field of view intersects the ground
- **Cloud cover analysis** — Estimate clear-sky probability from MODIS imagery via Google Earth Engine
- **Aircraft performance** — Pre-configured aircraft models (NASA ER-2, G-III, B200, and others) with climb/cruise/descent calculations
- **Airport logistics** — Search and filter airports by location, runway length, and other criteria
- **Geospatial export** — Output to GeoJSON, KML, and interactive Folium maps

---

## Installation

### Requirements

- Python 3.7+
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
mamba env create --file hyplan/hyplan.yml
mamba activate hyplan-env
pip install -e hyplan
```

### Optional dependencies

- **Google Earth Engine** (`earthengine-api`) — required for `hyplan.clouds`
- **Ray** — required for parallelized graph construction in `hyplan.flight_optimizer` (optional)

---

## Quick Start

### Define a flight line

```python
from hyplan import FlightLine, ureg

flight_line = FlightLine.start_length_azimuth(
    lat1=34.05, lon1=-118.25,
    length=ureg.Quantity(50000, "meter"),
    az=45.0,
    altitude_msl=ureg.Quantity(1000, "meter"),
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

| Module | Description |
|--------|-------------|
| `flight_line` | Create, modify, split, clip, and export individual flight lines |
| `flight_box` | Generate parallel flight lines covering a geographic area |
| `flight_plan` | Compute complete mission plans with timing and altitude profiles |
| `flight_optimizer` | Graph-based flight line ordering with multi-day and refueling support |
| `aircraft` | Aircraft performance models (NASA ER-2, G-III, B200, and others) |
| `sensors` | Sensor definitions (AVIRIS-3, AVIRIS-NG, HyTES, PRISM, MASTER, etc.) |
| `frame_camera` | Frame camera modeling with ground footprint calculations |
| `lvis` | LVIS full-waveform scanning lidar sensor model |
| `radar` | Side-looking SAR sensor models (UAVSAR L/P/Ka-band) |
| `glint` | Solar glint angle prediction for water observations |
| `swath` | Sensor swath coverage with terrain integration |
| `terrain` | DEM data acquisition and ray-terrain intersection |
| `sun` | Solar position and timing calculations |
| `clouds` | Cloud cover analysis and clear-sky probability from MODIS |
| `satellites` | Satellite overpass prediction and swath modeling |
| `airports` | Airport database with search, filtering, and runway data |
| `dubins_path` | Minimum-radius turning trajectories between waypoints |
| `geometry` | Geospatial utilities (haversine, coordinate transforms, polygons) |
| `units` | Unit conversions using Pint (meters, feet, knots, etc.) |
| `plotting` | Interactive Folium map generation |
| `download` | File download with caching and retry |

---

## Examples

The [`examples/`](examples/) directory contains runnable scripts demonstrating each module:

- [`test_flight_line.py`](examples/test_flight_line.py) — Flight line creation, clipping, rotation, splitting
- [`test_flight_box.py`](examples/test_flight_box.py) — Multi-line coverage patterns
- [`test_flight_plan.py`](examples/test_flight_plan.py) — Full mission planning with airports
- [`test_flight_optimizer.py`](examples/test_flight_optimizer.py) — Multi-day flight line optimization with refueling
- [`test_aircraft.py`](examples/test_aircraft.py) — Aircraft performance calculations
- [`test_sensors.py`](examples/test_sensors.py) — Sensor instantiation and GSD calculations
- [`test_glint.py`](examples/test_glint.py) — Solar glint prediction and visualization
- [`test_swath.py`](examples/test_swath.py) — Swath polygon generation
- [`test_terrain.py`](examples/test_terrain.py) — DEM handling and ray-terrain intersection
- [`test_sun.py`](examples/test_sun.py) — Solar position calculations
- [`test_clouds.py`](examples/test_clouds.py) — Cloud cover analysis (requires Google Earth Engine)
- [`test_airports.py`](examples/test_airports.py) — Airport search and filtering
- [`test_dubins.py`](examples/test_dubins.py) — Dubins path trajectories
- [`test_frame_camera.py`](examples/test_frame_camera.py) — Frame camera footprint calculations
- [`test_geometry.py`](examples/test_geometry.py) — Geometric utilities
- [`test_units.py`](examples/test_units.py) — Unit conversions
- [`test_satellites.py`](examples/test_satellites.py) — Satellite overpass prediction

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

- **Cloud analysis** — Campaign date ranges cannot span a year boundary (e.g., December to January)
- **Flight optimizer** — Uses a greedy nearest-neighbor heuristic; does not guarantee globally optimal ordering; still experimental
- **Terrain module** — DEM downloads require internet access and may be slow for large areas
- **Google Earth Engine** — The `clouds` module requires a Google Earth Engine account and authentication

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch
2. Install in development mode: `pip install -e .`
3. Run the example scripts in `examples/` to verify your changes
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
