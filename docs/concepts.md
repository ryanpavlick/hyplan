# Concepts

## Altitude: MSL vs AGL

HyPlan distinguishes between two altitude references:

- **MSL (Mean Sea Level)** — The altitude above the geoid, as used in aviation.
  Flight lines and waypoints store `altitude_msl` because this is what pilots
  fly and what air traffic control assigns.

- **AGL (Above Ground Level)** — The altitude above the local terrain surface.
  Sensor geometry methods (GSD, swath width, FOV) use `altitude_agl` because
  the sensor's ground footprint depends on height above the target, not above
  sea level.

Variable and parameter names throughout HyPlan include `_msl` or `_agl` to
make the reference explicit. Over flat terrain near sea level the two are
nearly identical, but over mountainous terrain or high-elevation sites the
distinction matters.

## Sensor hierarchy

All sensors derive from the {class}`~hyplan.instruments.Sensor` base class:

- {class}`~hyplan.instruments.LineScanner` — Whiskbroom/pushbroom imagers
  (AVIRIS-3, HyTES, PRISM, MASTER, G-LiHT, GCAS, eMAS, PICARD).
  Defined by FOV, across-track pixels, and frame rate.

- {class}`~hyplan.instruments.SidelookingRadar` — Side-looking SAR instruments
  (UAVSAR L-band, P-band, Ka-band). Defined by frequency, bandwidth,
  near/far range angles, and look direction.

- {class}`~hyplan.instruments.LVIS` — Full-waveform scanning lidar. The scanner
  geometry defines a fixed maximum swath ($\text{swath} = 0.2 \times
  \text{altitude}$), but the effective swath depends on laser repetition
  rate, lens divergence (footprint size), and aircraft speed.

- {class}`~hyplan.instruments.FrameCamera` — Frame cameras defined by
  sensor dimensions, focal length, resolution, and frame rate.

All sensor types provide `half_angle` and `swath_width(altitude_agl)` so they
work with {func}`~hyplan.swath.generate_swath_polygon`,
{func}`~hyplan.flight_box.generate_flight_lines`, and other planning tools.

## Flight planning workflow

A typical mission planning workflow:

1. **Define flight lines** — Use {class}`~hyplan.flight_line.FlightLine` to
   create individual lines by start/end coordinates, start/length/azimuth,
   or from GeoJSON.

2. **Generate coverage patterns** — Use
   {func}`~hyplan.flight_box.box_around_center_line` or
   {func}`~hyplan.flight_box.box_around_polygon` to create parallel flight
   lines covering a target area, spaced by the sensor's swath width.

3. **Compute the mission plan** — Use
   {func}`~hyplan.flight_plan.compute_flight_plan` with an aircraft and
   departure/destination airports to generate a complete flight plan with
   climb, transit, data collection, descent, and landing segments.

4. **Analyze constraints** — Check solar glint angles
   ({mod}`hyplan.glint`), cloud climatology ({mod}`hyplan.clouds`),
   and terrain interactions ({mod}`hyplan.terrain`).

## Aircraft speed profiles

Aircraft performance is modeled with piecewise-linear speed profiles — a
list of `(altitude, speed)` breakpoints. The aircraft's speed at any altitude
is interpolated from these breakpoints using {func}`numpy.interp`. This
captures the variation between low-altitude maneuvering speeds and
high-altitude cruise speeds.

## Imports

Core classes and functions are re-exported from the top-level `hyplan` package
for convenience:

```python
from hyplan import FlightLine, Airport, AVIRIS3, DynamicAviation_B200, ureg
```

Specialized modules with heavier or optional dependencies should be imported
directly from their submodules:

```python
from hyplan.clouds import create_cloud_data_array_with_limit
from hyplan.terrain import download_dem
from hyplan.satellites import get_satellite_overpass_times
from hyplan.glint import compute_glint_vectorized
from hyplan.sun import solar_threshold_times
```

See the API Reference for the full list of available classes and functions.

## Units

HyPlan uses [Pint](https://pint.readthedocs.io/) for physical quantities.
All public APIs accept and return `pint.Quantity` objects with explicit units.
The shared unit registry is available as `hyplan.units.ureg`.
