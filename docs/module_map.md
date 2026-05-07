# Module Map

A top-down tour of HyPlan's package layout, aimed at new
contributors.  For each module: a one-paragraph purpose, a few
key entry points with file:line anchors, and a "use it when"
note.  Public-API stability classifications live in
[stability](stability.md); this document is about *where things
live and how they fit together*.

## Decision tree: which function do I want?

* **"Where can the aircraft observe and recover within a time
  budget?"** → [`hyplan.compute_isochrone`](api/isochrone.md).
* **"...with optional refuel stops?"** →
  [`hyplan.compute_refuel_isochrone`](api/isochrone.md).
* **"...at multiple budgets at once (1/2/3/4 hours)?"** →
  [`hyplan.compute_concentric_isochrones`](api/isochrone.md).
* **"Can I reach *this specific* point?"** →
  [`hyplan.evaluate_target_reachability`](api/isochrone.md).
* **"How long does this sortie take, segment by segment?"** →
  [`hyplan.compute_flight_plan`](api/flight_plan.md).
* **"What flight lines should I order across multiple days, given
  endurance + refuel constraints?"** →
  [`hyplan.greedy_optimize`](api/flight_optimizer.md).
* **"What sensor footprint / GSD / swath do I get at this altitude
  and speed?"** → instrument-specific modules under
  [`hyplan.instruments`](api/sensors.md).

## Package layout

### Aircraft and physics ([`hyplan/aircraft/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/aircraft))

The performance model.  Aircraft classes carry calibrated speed
schedules (CAS/Mach for jets, TAS-vs-altitude for turboprops),
climb / descent / cruise profiles, turn models, approach
profiles, and provenance metadata.  v1.4 ships eight calibrated
platforms (ER-2, G-III, G-V, WB-57, P-3, B-200, Twin Otter,
C-130) plus four brochure-derived platforms (G-IV, C-20A, B-777,
A-90).

* `Aircraft` ABC — [hyplan/aircraft/_base.py:553](hyplan/aircraft/_base.py#L553)
* Concrete classes — [hyplan/aircraft/_models.py](hyplan/aircraft/_models.py)
* `Aircraft._climb` / `_descend` / `step_climb` — vertical-phase
  integrators that take optional `wind_along_track`.
* `Aircraft._hybrid_path` — 2D Dubins (trochoidal under wind) +
  integrated vertical profile; the core leg-time solver used by
  `compute_flight_plan`.
* `climb_with_wind_field` / `descend_with_wind_field` —
  [hyplan/aircraft/wind_path.py](hyplan/aircraft/wind_path.py).
  Wrap `_climb`/`_descend` with `WindField` sampling at phase-mid
  altitude, used by `compute_isochrone(wind_sampling="phase_midpoint")`
  and `compute_flight_plan(wind_sampling="phase_midpoint")`.
* Calibration loaders for IWG1 / ICARTT / ADS-B —
  [hyplan/aircraft/iwg1.py](hyplan/aircraft/iwg1.py),
  [hyplan/aircraft/icartt.py](hyplan/aircraft/icartt.py),
  [hyplan/aircraft/adsb/](hyplan/aircraft/adsb/).

**Use it when:** computing leg times / climb performance / cruise
TAS for any aircraft; wiring wind into trajectory math; loading
post-flight calibration data.

### Planning ([`hyplan/planning/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/planning))

The mission-level entry points.  Two families:

**Trajectory engine** —
[hyplan/planning/engine.py](hyplan/planning/engine.py).
* `compute_flight_plan(aircraft, flight_sequence, ...)` — the
  primary planner: takes a list of `FlightLine` / `Waypoint` /
  `Pattern` objects and produces a per-segment
  `GeoDataFrame` with timing, altitude, ground track, and segment
  classification.  Wind-aware (constant or `WindField` provider);
  per-phase wind sampling via `wind_sampling="phase_midpoint"`.
* `expand_sequence(seq)` — utility: unwraps `Pattern` objects in
  the input list to flat `FlightLine | Waypoint`.

**Reachability engine** —
[hyplan/planning/isochrone.py](hyplan/planning/isochrone.py).
* `compute_isochrone(...)` — wind-aware reachability boundary in
  three modes (`one_way`, `round_trip`, `return_safe`).
* `compute_refuel_isochrone(...)` — extends reach with a single
  optional refuel stop; tracks `sortie_budget` (per fuel cycle)
  and `flight_day_budget` (total wall-clock).
* `compute_concentric_isochrones(...)` — sweep multiple budgets
  with seeded bracket searches.
* `evaluate_target_reachability(...)` — single-point spot check.
* `_leg_time(...)` — internal leg-time helper that powers the
  ray sweeps; same physics as `_hybrid_path` but with a simpler
  scalar return shape.  Wind-sampling kwargs:
  `wind_sampling`, `wind_sample_spacing`, `max_wind_samples_per_leg`.

**Use it when:** building a mission plan (`compute_flight_plan`)
or asking a "where can I observe / can I get to here?" question
(`compute_isochrone` and friends).

### Wind and atmosphere ([`hyplan/winds/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/winds), [`hyplan/atmosphere.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/atmosphere.py))

* `WindField` — abstract provider returning `(u, v)` at a given
  `(lat, lon, altitude, time)`.  Concrete: `StillAirField`,
  `ConstantWindField`, `MERRA2WindField` (NASA reanalysis),
  `GFSWindField` (NOAA forecast), `GMAOWindField` (GEOS-FP
  near-real-time), `IWG1TraceWindField` (replay from a flown
  IWG1 trace).
* `wind_field_from_plan(...)` — factory that builds the right
  provider from a `WindPlan` config.
* `_track_hold_solution_from_uv(tas, track_deg, u, v)` —
  [hyplan/winds/utils.py](hyplan/winds/utils.py).  Solves crab +
  groundspeed for a given track and wind; reused by `_leg_time`
  and `_hybrid_path` for cruise integration.
* `Atmosphere` — ISA model and CAS/TAS/Mach conversions in
  [hyplan/atmosphere.py](hyplan/atmosphere.py).

**Use it when:** integrating a real wind field into a flight
plan or isochrone; converting between airspeed conventions.

### Geometry ([`hyplan/flight_line.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/flight_line.py), [`hyplan/waypoint.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/waypoint.py), [`hyplan/flight_box.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/flight_box.py), [`hyplan/pattern.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/pattern.py), [`hyplan/flight_patterns.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/flight_patterns.py))

* `FlightLine` — a parameterized 2D line with start/end altitude,
  heading, and per-line metadata.  Splittable, clippable,
  rotatable, exportable.
* `Waypoint` — a 4D state (lat, lon, altitude, heading) with
  optional `delay`, segment-type override, and dwell time.
* `FlightBox` — a polygon of parallel flight lines covering a
  study area.
* `Pattern` — first-class container for racetrack / rosette /
  spiral / sawtooth / polygon / glint-arc / coordinated-line
  geometries.  Generators in
  [hyplan/flight_patterns.py](hyplan/flight_patterns.py).

**Use it when:** defining mission geometry; combining or
modifying flight lines; building a multi-line survey or named
pattern.

### Instruments and swath ([`hyplan/instruments/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/instruments), [`hyplan/swath.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/swath.py))

* Sensor models: AVIRIS-3 / AVIRIS-5 / HyTES / PRISM / MASTER
  (line-scanners), LVIS (full-waveform lidar), UAVSAR
  L/P/Ka-band (SAR), frame cameras, profiling lidar, AWP.
* `generate_swath_polygon(...)` — terrain-aware swath polygon via
  ray-DEM intersection.
* `calculate_swath_widths(...)` — per-line median swath width.

**Use it when:** computing GSD, swath coverage, or sensor
footprints over real terrain.

### Environment and timing ([`hyplan/sun.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/sun.py), [`hyplan/glint.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/glint.py), [`hyplan/clouds/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/clouds), [`hyplan/phenology/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/phenology))

* Solar position and illumination windows.
* Sun-glint angle prediction for water observation missions.
* Cloud climatology (ERA5 via Open-Meteo) and analysis (MODIS via
  Earth Engine).
* Vegetation phenology (MODIS NDVI / EVI / LAI / FPAR).

**Use it when:** picking *when* to fly, not *where*.

### Logistics ([`hyplan/airports.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/airports.py), [`hyplan/airspace.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/airspace.py), [`hyplan/satellites.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/satellites.py), [`hyplan/flight_optimizer.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/flight_optimizer.py))

* Airport database (OurAirports), runway/length/surface filters,
  nearest-airport search.
* Airspace conflict detection (FAA NASR + OpenAIP).
* Satellite overpass prediction (Skyfield + CelesTrak TLEs).
* `greedy_optimize(...)` — graph-based flight-line ordering with
  endurance, max-daily-flight-time, and refuel constraints.

**Use it when:** picking *which* airport to base out of,
checking *which* airspaces to avoid, or scheduling line ordering
across multiple days.

### Outputs ([`hyplan/exports/`](https://github.com/ryanpavlick/hyplan/tree/main/hyplan/exports), [`hyplan/plotting.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/plotting.py))

* Plain-format exports: Excel, CSV (ForeFlight), KML, GPX, FMS
  (Honeywell), ER-2 nav file, ICARTT.
* Plotting: Folium maps (`map_flight_lines`, `plot_isochrone`,
  `map_airspace`); Cartopy static figures (`plot_airspace_map`,
  `plot_isochrone_static`); altitude-trajectory profiles.

**Use it when:** delivering a flight plan to pilots, creating
publication figures, or producing a campaign artifact.

### Campaign management ([`hyplan/campaign.py`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/campaign.py))

* `Campaign` — multi-mission container with stable IDs, revision
  metadata, and plain-folder persistence.

**Use it when:** organizing a multi-day or multi-flight survey
into a coherent, version-controlled artifact.

### Internals

* `hyplan/geometry.py` — geodesic math (Vincenty via `pymap3d`),
  bearing/distance, polygon utilities, longitude wrapping.
* `hyplan/units.py` — Pint registry + airspeed/altitude/distance
  conversion helpers.
* `hyplan/download.py` — chunked HTTP downloader for DEMs and
  similar.
* `hyplan/exceptions.py` — `HyPlanError` hierarchy
  (`HyPlanValueError`, `HyPlanRuntimeError`, etc.).

## Cross-cutting concerns

### Wind, three places

Today three code paths consume `WindField` providers:

1. **`compute_flight_plan`** ([engine.py](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/planning/engine.py))
   — resolves wind to a single `(u, v)` per leg at the leg
   midpoint at cruise altitude (default), then forwards to
   `Aircraft._hybrid_path(wind=...)`.  Opt into per-phase
   sampling with `wind_sampling="phase_midpoint"`.
2. **`compute_isochrone`** ([isochrone.py](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/planning/isochrone.py))
   — its `_leg_time` helper has its own dispatch on
   `wind_sampling` ∈ {`cruise_midpoint`, `phase_midpoint`,
   `segmented_cruise`}.  Phase-aware modes use
   `aircraft.wind_path.{climb,descend}_with_wind_field`.
3. **`Aircraft._hybrid_path`** itself can sample a wind field
   directly when given `wind_source` + `t_anchor` (used by
   `compute_flight_plan` in `phase_midpoint` mode).

A future `LegSolver` unification could collapse these to one
call site; for now the three coexist and share the
`hyplan.aircraft.wind_path` primitives for the hard part.

### Time, two clocks

`compute_refuel_isochrone` tracks two budgets simultaneously:

* `sortie_budget` — per-fuel-cycle endurance, *resets* after each
  refuel.
* `flight_day_budget` — wall-clock budget covering *everything*
  (flying + refueling + on-station), does *not* reset.

`reserve` applies per fuel cycle only.  This dual-clock model
also lives in `flight_optimizer.greedy_optimize` for line
ordering across multiple days.

### Calibration provenance

Every aircraft carries a `SourceRecord` documenting where its
performance numbers came from (data fit / brochure / inferred
from sibling airframe) plus a `PerformanceConfidence` (0.7 for
brochure, 0.85 for data-fit).  Methodology is in
[docs/calibration.md](calibration.md); per-aircraft notebooks
live under [`notebooks/calibration/`](https://github.com/ryanpavlick/hyplan/tree/main/notebooks/calibration).

## See also

* [Concepts](concepts.md) — domain language and acronyms.
* [Architecture](architecture.md) — high-level design choices
  (programmatic vs. interactive, physics vs. geometric, etc.).
* [API reference](api/) — autodoc per module.
* [Calibration](calibration.md) — how aircraft performance fits
  are produced and validated.
