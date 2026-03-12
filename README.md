# HyPlan

An open-source Python library for planning airborne remote sensing campaigns.

HyPlan helps scientists and engineers design flight missions for hyperspectral and multispectral remote sensing. It handles flight line generation, sensor modeling, swath coverage, solar glint prediction, cloud analysis, terrain-aware calculations, and mission logistics including airport selection and aircraft performance.

## Features

- **Flight planning** — Define flight lines, generate multi-line coverage patterns, and compute complete mission plans with takeoff, transit, data collection, and landing segments
- **Sensor modeling** — Pre-configured NASA instruments (AVIRIS-3, AVIRIS-NG, HyTES, PRISM, MASTER, and more) with ground sample distance and swath calculations
- **Solar glint prediction** — Predict glint angles across flight lines for water observation missions
- **Terrain-aware analysis** — Download DEM data and compute where the sensor field of view intersects the ground
- **Cloud cover analysis** — Estimate clear-sky probability from MODIS imagery via Google Earth Engine
- **Aircraft performance** — Pre-configured aircraft models (NASA ER-2, G-III, B200, and others) with climb/cruise/descent calculations
- **Airport logistics** — Search and filter airports by location, runway length, and other criteria
- **Geospatial export** — Output to GeoJSON, KML, and interactive Folium maps

---

## Installation

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

---

## Quick Start

### Define a flight line

```python
from hyplan.flight_line import FlightLine
from hyplan.units import ureg

flight_line = FlightLine.start_length_azimuth(
    lat1=34.05, lon1=-118.25,
    length=ureg.Quantity(50000, "meter"),
    az=45.0,
    altitude=ureg.Quantity(1000, "meter"),
    site_name="LA Northeast",
)
```

### Compute a flight plan

```python
from hyplan.aircraft import DynamicAviation_B200
from hyplan.airports import Airport
from hyplan.flight_plan import compute_flight_plan, plot_flight_plan

aircraft = DynamicAviation_B200()
departure = Airport("KSBA")
destination = Airport("KBUR")

plan = compute_flight_plan(aircraft, [flight_line], departure, destination)
plot_flight_plan(plan, departure, destination, [flight_line])
```

### Predict solar glint

```python
from datetime import datetime, timezone
from hyplan.sensors import AVIRIS3
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
| `aircraft` | Aircraft performance models (NASA ER-2, G-III, B200, and others) |
| `sensors` | Sensor definitions (AVIRIS-3, AVIRIS-NG, HyTES, PRISM, MASTER, etc.) |
| `frame_camera` | Frame camera modeling with ground footprint calculations |
| `glint` | Solar glint angle prediction for water observations |
| `swath` | Sensor swath coverage with terrain integration |
| `terrain` | DEM data acquisition and ray-terrain intersection |
| `sun` | Solar position and timing calculations |
| `clouds` | Cloud cover analysis and clear-sky probability from MODIS |
| `airports` | Airport database with search, filtering, and runway data |
| `dubins_path` | Minimum-radius turning trajectories between waypoints |
| `geometry` | Geospatial utilities (haversine, coordinate transforms, polygons) |
| `units` | Unit conversions using Pint (meters, feet, knots, etc.) |
| `plotting` | Interactive Folium map generation |
| `download` | File download with caching and retry |

---

## Examples

The `examples/` directory contains runnable scripts demonstrating each module:

- `test_flight_line.py` — Flight line creation, clipping, rotation, splitting
- `test_flight_box.py` — Multi-line coverage patterns
- `test_flight_plan.py` — Full mission planning with airports
- `test_aircraft.py` — Aircraft performance calculations
- `test_sensors.py` — Sensor instantiation and GSD calculations
- `test_glint.py` — Solar glint prediction and visualization
- `test_swath.py` — Swath polygon generation
- `test_terrain.py` — DEM handling and ray-terrain intersection
- `test_sun.py` — Solar position calculations
- `test_clouds.py` — Cloud cover analysis
- `test_airports.py` — Airport search and filtering
- `test_dubins.py` — Dubins path trajectories
- `test_frame_camera.py` — Frame camera footprint calculations
- `test_geometry.py` — Geometric utilities
- `test_units.py` — Unit conversions

---

## Contributing

Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request.

## License

HyPlan is licensed under the Apache License, Version 2.0. See `LICENSE.md` for details.

## Contact

For inquiries or further information, please contact Ryan Pavlick (ryan.p.pavlick@nasa.gov).
