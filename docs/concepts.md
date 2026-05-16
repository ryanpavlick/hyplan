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

- {class}`~hyplan.instruments.ALSLidar` — Generic scanning-mirror discrete-return
  topographic Airborne Laser Scanner. Models rotating-polygon scanners
  (RIEGL VQ-series, Leica TerrainMapper, Optech Galaxy). Parametrised by
  PRF, scan rate, beam divergence, scan half-angle, wavelength, max range,
  MTA zones, and scan geometry. Pre-configured reference instance:
  {data}`~hyplan.instruments.RIEGL_VQ_480II` (1200 kHz operating point of
  the RIEGL VQ-480 II). Provides nominal-density / contiguity / inverse
  solvers (`solve_for_altitude`, `solve_for_groundspeed` — both with a
  `ContiguityError` guard) plus terrain-aware methods
  (`footprint_on_terrain`, `effective_swath_on_terrain`,
  `terrain_summary`) modelled on the LVIS pattern.

- {class}`~hyplan.instruments.MultiALSLidarRig` — Multi-lidar rig of
  identical {class}`~hyplan.instruments.ALSLidar` units with known mount
  orientations (pitch and roll tilts). Supports the two common
  configurations: forward/backward pitch tilt for multi-angle returns
  over the same swath (NASA G-LiHT pattern) and left/right roll tilt for
  extended cross-track swath. Conforms to the same `ScanningSensor`
  protocol as a single sensor. Pre-configured reference instance:
  {data}`~hyplan.instruments.GLIHT_DUAL_VQ_480I` (G-LiHT 2017+ dual
  VQ-480i, ±7° forward/backward pitch).

- {class}`~hyplan.instruments.FrameCamera` — Frame cameras defined by
  sensor dimensions, focal length, resolution, and frame rate.

- {class}`~hyplan.instruments.ProfilingLidar` — Nadir-pointing single-beam
  atmospheric profiling lidars ({class}`~hyplan.instruments.HSRL2`,
  {class}`~hyplan.instruments.HALO`, {class}`~hyplan.instruments.CPL`).
  Defined by laser wavelengths, pulse rate, telescope/beam optics, vertical
  bin resolution, and FPGA-averaged sampling rate. These instruments record
  a vertical column directly beneath the platform and have no cross-track
  swath; horizontal resolution is set by post-processing averaging.

- {class}`~hyplan.instruments.AerosolWindProfiler` — Doppler wind lidar
  with dual line-of-sight geometry (NASA Langley AWP). Modeled separately
  from `ProfilingLidar` because the dual-LOS vector-retrieval geometry
  needs a different abstraction.

- {class}`~hyplan.instruments.DropsondeSystem` — Event-based sampler
  (experimental in v1.9).  Unlike the swath and profile sensors above,
  a dropsonde is a single release at a (lat, lon, alt, time) tuple;
  the science footprint is the slant column from release to splash,
  drifting through wind.  The class does not provide `half_angle` /
  `swath_width` (no swath geometry); release events and splash
  predictions are produced via the first-class
  {class}`~hyplan.instruments.DropsondeRelease`,
  {class}`~hyplan.instruments.DropsondeTrajectory`, and
  {class}`~hyplan.instruments.DropsondePlan` objects, which consume a
  computed flight plan (or a bare {class}`~hyplan.flight_line.FlightLine`
  / {class}`~hyplan.pattern.Pattern`) and any
  {class}`~hyplan.winds.WindField` provider.

Imaging and scanning sensors (`LineScanner`, `SidelookingRadar`, `LVIS`,
`ALSLidar`, `MultiALSLidarRig`, `FrameCamera`) provide `half_angle` and
`swath_width(altitude_agl)` so they
work with {func}`~hyplan.swath.generate_swath_polygon`,
{func}`~hyplan.flight_box.generate_flight_lines`, and other planning tools.
Profiling lidars do not have a cross-track swath and use their own geometry
helpers (e.g., `footprint_diameter`, `horizontal_resolution`).

## Flight planning workflow

A typical mission planning workflow:

1. **Define flight lines** — Use {class}`~hyplan.flight_line.FlightLine` to
   create individual lines by start/end coordinates, start/length/azimuth,
   or from GeoJSON.

2. **Generate coverage patterns** — Use
   {func}`~hyplan.flight_box.box_around_center_line` or
   {func}`~hyplan.flight_box.box_around_polygon` to create parallel flight
   lines covering a target area, spaced by the sensor's swath width, or
   use generators in {mod}`hyplan.flight_patterns` to create reusable
   {class}`~hyplan.pattern.Pattern` objects such as racetracks,
   polygons, spirals, and glint arcs.

3. **Organize a campaign** — Use {class}`~hyplan.campaign.Campaign` to
   store free-standing lines, reusable patterns, and reference data with
   stable IDs and revision metadata.

4. **Compute the mission plan** — Use
   {func}`~hyplan.planning.compute_flight_plan` with an aircraft and
   departure/destination airports to generate a complete flight plan with
   climb, transit, data collection, descent, and landing segments.

5. **Analyze constraints** — Check solar glint angles
   ({mod}`hyplan.glint`), cloud climatology ({mod}`hyplan.clouds`),
   and terrain interactions ({mod}`hyplan.terrain`).

## Pattern model

HyPlan pattern generators return {class}`~hyplan.pattern.Pattern`
objects rather than plain waypoint lists.

- **Line-based patterns** contain ordered
  {class}`~hyplan.flight_line.FlightLine` legs, such as racetracks and
  rosettes.
- **Waypoint-based patterns** contain ordered
  {class}`~hyplan.waypoint.Waypoint` elements, such as spirals,
  sawtooths, polygons, and glint arcs.

This design makes patterns reusable and editable:

- `Pattern.regenerate()` recreates a pattern from its stored generator
  parameters with selective overrides.
- `compute_flight_plan()` can expand a `Pattern` directly.
- `Campaign` can persist patterns with stable `pattern_id` and `line_id`
  values for interactive tools.

### Patterns are atomic in the flight-line optimizer

When a `Pattern` appears in
{func}`~hyplan.flight_optimizer.greedy_optimize`'s input sequence alongside
free-standing {class}`~hyplan.flight_line.FlightLine` items, the optimizer
treats it as a single indivisible visit item. Concretely:

- The optimizer **may** reorder whole patterns relative to free
  flight lines.
- The optimizer **may not** split a pattern apart: once it begins a
  pattern, it visits every element of that pattern in pattern definition
  order before moving on to any other item.
- Pattern traversal direction is fixed (entry → exit). Reversal is not
  supported in this release.

Endurance and refueling feasibility evaluate the pattern as one chunk:
if `transit_in + pattern_internal_time + transit_out` would exceed
remaining endurance, the optimizer schedules a refuel **before** the
pattern, never inside it.

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
from hyplan import FlightLine, Airport, AVIRIS3, KingAirB200, ureg
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
