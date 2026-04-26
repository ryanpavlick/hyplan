# Architecture

HyPlan is organized around six subsystems that compose into a
flight-planning pipeline. Each subsystem is a self-contained package
or module with a well-defined interface.

## Data flow

```text
FlightLine / Waypoint / Pattern
        │
        ▼
FlightBox / FlightPatterns   ──▶   Campaign / flight sequence
                                        │
                          Aircraft ──▶   │   ◀── WindField
                                        ▼
                              compute_flight_plan
                                        │
                                        ▼
                                  GeoDataFrame
                                   ╱         ╲
                             exports        plotting
```

## Subsystems

### Geometry and flight lines

The atomic unit of data collection is a
{class}`~hyplan.flight_line.FlightLine` — a straight-and-level segment
defined by two endpoints, an altitude, and metadata. Flight lines are
generated individually or in bulk via
{func}`~hyplan.flight_box.box_around_center_line` (parallel coverage of a
study area) and the pattern generators in {mod}`hyplan.flight_patterns`
(racetracks, rosettes, spirals, polygons). Pattern generators now return
{class}`~hyplan.pattern.Pattern`, which preserves generator parameters and
element ordering for reuse, editing, and persistence. Intermediate route
points use {class}`~hyplan.waypoint.Waypoint`.

See: [Flight Lines](api/flight_line.md),
[Waypoints](api/waypoint.md),
[Flight Box](api/flight_box.md),
[Flight Patterns](api/flight_patterns.md)

### Planning

{func}`~hyplan.planning.compute_flight_plan` is the main orchestrator. It
takes an ordered sequence of flight lines, waypoints, and patterns,
connects them with 3-D Dubins paths ({mod}`hyplan.dubins3d`), classifies
each segment (takeoff, climb, transit, flight line, descent, approach,
pattern), and returns a {class}`~geopandas.GeoDataFrame` with timing,
distance, altitude, and geometry for every segment. The
{mod}`hyplan.flight_optimizer` provides graph-based line ordering with
endurance constraints and refueling stops.

See: [Flight Plan](api/flight_plan.md),
[Flight Optimizer](api/flight_optimizer.md),
[Dubins 3D](api/dubins3d.md)

### Winds and atmosphere

The {class}`~hyplan.winds.WindField` abstract base defines a single
method — `wind_at(lat, lon, altitude, time)` — that returns eastward and
northward wind components. Concrete implementations range from
{class}`~hyplan.winds.StillAirField` and
{class}`~hyplan.winds.ConstantWindField` to gridded providers that fetch
data from MERRA-2, NOAA GFS, and GMAO GEOS-FP via OPeNDAP or GRIB.
The planner uses these to compute crab angles, groundspeeds, and
wind-corrected segment times.

The {mod}`hyplan.atmosphere` module provides the International Standard
Atmosphere (ISA) model for pressure-altitude conversions and airspeed
calculations (CAS/TAS/Mach).

See: [Winds](api/winds.md), [Atmosphere](api/atmosphere.md)

### Aircraft

{class}`~hyplan.aircraft.Aircraft` models an aircraft's performance
envelope: speed schedules (CAS/Mach vs altitude), climb and descent rates,
bank angles, turn radii, and endurance. Fifteen pre-configured research
aircraft are included (NASA ER-2, GV, P-3, etc.). The planner calls
`time_to_cruise`, `time_to_takeoff`, and `time_to_return` to generate
Dubins paths with realistic vertical profiles.

See: [Aircraft](api/aircraft.md)

### Terrain and swath

{mod}`hyplan.terrain` downloads and caches 30-meter Copernicus DEM tiles,
merges them with rasterio, and provides bulk elevation lookup and a
vectorized ray-terrain intersection algorithm for off-nadir sensors.
{mod}`hyplan.swath` computes ground footprint polygons for any sensor type
using the sensor's half-angle and the flight line geometry.

See: [Terrain](api/terrain.md), [Swath](api/swath.md)

### Exports and visualization

Flight plans are exported to pilot-facing formats (ForeFlight CSV, Honeywell
FMS, Excel briefing sheets), archival formats (ICARTT, KML, GPX), and
plain text. {mod}`hyplan.plotting` provides Folium maps, altitude profiles,
and terrain cross-sections.

See: [Exports](api/exports.md), [Plotting](api/plotting.md)

## Supporting modules

| Module | Purpose |
|--------|---------|
| {mod}`hyplan.sun` | Solar position and data-collection windows |
| {mod}`hyplan.glint` | Specular reflection (glint) angle prediction |
| {mod}`hyplan.clouds` | Cloud climatology via Open-Meteo or Google Earth Engine |
| {mod}`hyplan.satellites` | Satellite overpass prediction and temporal coincidence |
| {mod}`hyplan.airspace` | Airspace conflict detection (OpenAIP, FAA TFR/NASR) |
| {mod}`hyplan.airports` | Airport database and runway queries |
| {mod}`hyplan.campaign` | Persistent folder structure, revision metadata, and mutation API for a study area |
| {mod}`hyplan.pattern` | Serializable reusable pattern object returned by pattern generators |

## Units

All public APIs accept and return {class}`pint.Quantity` objects with
explicit units. The shared unit registry is
{data}`hyplan.units.ureg`. Parameter names include suffixes like `_msl`,
`_agl`, `_cas`, `_tas` to make the reference frame explicit.
