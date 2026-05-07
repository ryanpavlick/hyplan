# Changelog

## v1.5.1 — 2026-05-07

Maintenance release: aircraft calibration provenance, smoother
plotted Dubins arcs, and paper / docs polish.  No breaking changes.

### Aircraft calibration status (`Aircraft.calibration_status`)

Every shipped aircraft now carries an explicit
`calibration_status` attribute, surfaced in the docstring and on
the instance.  Three values:

* `"calibrated"` — climb / cruise / descent profiles fit to in-situ
  IWG1 / ICARTT data.  Eight aircraft today: `NASA_ER2`,
  `NASA_GIII`, `NASA_GV`, `NASA_WB57`, `NASA_P3`, `KingAirB200`,
  `C130`, `TwinOtter`.
* `"inferred"` — performance mirrored from a calibrated cousin
  airframe of the same type certificate.  `NCAR_GV` (mirrors
  `NASA_GV`) and `NASA_C20A` (mirrors `NASA_GIII`).
* `"uncalibrated"` — performance from manufacturer brochures with
  no in-situ flight-data fit.  `NASA_GIV`, `NASA_B777`,
  `KingAirA90`.

Uncalibrated and inferred classes carry an explicit `.. warning::`
or `.. note::` block in their docstrings flagging the provenance
limitation.  The status field is the default mechanism for
querying provenance programmatically; `confidence` and `sources`
remain available for finer-grained inspection.

### Plotting: `n_samples` for transit Dubins arcs

`compute_flight_plan(...)` gains an `n_samples: int = 20` kwarg
that plumbs through `time_to_takeoff` / `time_to_cruise` /
`time_to_return` to `_hybrid_path`'s sublinestring sampling.
Higher values densify the per-phase Dubins-arc geometry so
transits render as smooth curves rather than visible polylines at
typical figure scales.  Default preserves prior behaviour.
Timing is computed analytically and is unchanged.

### Paper / figures

* fig2 wind narrative: 80 kt south wind (173°) demonstrates the
  "wind aligned with mission saves time" case — the planner
  reroutes the relocation strip to its wind-favorable traversal
  direction and the mission completes 8.5 min faster than still
  air on a 93.0 min baseline (~9.1%).  Caption rewritten to
  reflect this.
* fig2 PDF / PNG re-rendered at `n_samples=80`; transit Dubins
  arcs are now visibly smooth.
* paper.pdf rebuilt via `openjournals/inara`.

### Other housekeeping

* `notebooks/README.md` — wording refresh to reflect v1.5.0
  capabilities (wind-aware reachability, calibration scope,
  optimizer + sortie grouping).
* `paper/talks/` is now gitignored (drafted slide decks live
  there; not part of the published artifact).

## v1.5.0 — 2026-05-06

This release introduces **wind-aware isochrones** — reachability
boundaries that answer "where can the aircraft observe and recover
within a given time budget?".  Four public functions (`compute_isochrone`,
`compute_concentric_isochrones`, `compute_refuel_isochrone`,
`evaluate_target_reachability`) plus two plotters (Folium for
interactive use, Cartopy for static figures) cover round-trip,
one-way, and return-safe modes; refuel-extended reach with two-clock
budget tracking; multi-budget concentric contours; and single-target
spot checks.  No breaking changes to existing public APIs.

This release also prunes three aircraft (Learjet, BAe-146, Dash-8)
that shipped with brochure-only models in v1.4 but never received
calibration.  The fleet count drops from 15 to 12.

### Why this release

Pre-v1.5, deciding "is this study area reachable?" required either
hand calculation or running `compute_flight_plan` against multiple
candidate targets.  Real campaigns ask the question constantly —
during pre-flight site selection, in-flight re-tasking after weather,
and when negotiating refuel logistics with hosts.  v1.5 turns this
into a one-call query whose output (a GeoDataFrame of boundary
points) drops directly into existing flight-line workflows.

The isochrone solver uses the same wind-correction machinery as
`compute_flight_plan` (cruise track-hold via `_track_hold_solution_from_uv`,
calibrated climb / cruise / descent profiles per aircraft, and the
existing `WindField` provider stack), so results are consistent
between the two functions.

### Wind-aware isochrones — `compute_isochrone(...)`

`compute_isochrone(aircraft, start, budget, ...)` returns a
`GeoDataFrame` of boundary points around `start` reachable within
`budget`, accounting for climb / descent overhead, on-station dwell,
mandatory reserve, and a wind field.

* **Three modes**:
  - `"one_way"` — single-leg reach.
  - `"round_trip"` — out-and-back from the same place (default).
  - `"return_safe"` — out, observe at target, recover at a *different*
    airport within `budget`.
* **Wind correction** — cruise wind sampled at the cruise-segment
  midpoint with up to three fixed-point iterations on cruise time.
  Climb and descent are still-air in v1.5 (a known simplification
  documented in the module's "Limitations" section; configurable
  segmented sampling is queued for a follow-up).
* **Diagnostics per ray** — `outbound_time_min`, `return_time_min`,
  `outbound_headwind_kt`, `return_headwind_kt`, `net_headwind_kt`,
  `headwind_asymmetry_kt`, `time_slack_min`, `limiting_leg`.
* **`gdf.attrs`** stashes invocation context (mode, budget, reserve,
  on-station time, cruise altitude, return destination, wind source,
  start time) so a saved boundary remains self-describing.

Notebook tutorial: [`notebooks/isochrone.ipynb`](notebooks/isochrone.ipynb)
walks through all three modes with a B-200 from KEFD, fleet
comparison (G-V vs. G-III vs. B-200), MERRA-2 reanalysis on a real
date, and the Folium / Cartopy plotters.

### Refuel-extended reach — `compute_refuel_isochrone(...)`

For campaigns that allow a mid-mission refuel stop at a pre-cleared
field, `compute_refuel_isochrone` extends each ray's reach by trying
three itinerary templates per azimuth — `direct`, `outbound_refuel(R)`,
`return_refuel(R)` — and picks the one that goes farthest while
honoring both a **per-fuel-cycle `sortie_budget`** (resets after each
refuel) and a **total wall-clock `flight_day_budget`** (does not
reset, absorbs `refuel_time`).  `reserve` applies per fuel cycle only.

* `max_refuel_stops=1` in v1 — a single sortie touches at most two
  tanks.  Chained refuels deferred.
* Per-template independent bracket-and-search per ray; the winning
  template's diagnostics populate the row, with explicit per-leg
  time columns (`start_to_refuel_time_min`,
  `refuel_to_target_time_min`, etc.) so consumers don't have to
  guess which legs are populated for which itinerary.
* `gdf.attrs` reports `refuel_airports_evaluated`,
  `refuel_airports_unreachable` (with reason), and
  `refuel_airports_used`.
* Refuel-leg caching — fixed legs (`start → R` and, for
  anchor-invariant winds, `R → recovery`) are computed once in the
  prefilter and reused per probe instead of being recomputed on
  every binary-search iteration.

### Single-target spot checks — `evaluate_target_reachability(...)`

The complement of `compute_refuel_isochrone`: rather than sweeping
azimuths to find the boundary, this asks *given this target*, which
itineraries reach it?  Returns a structured dict with the best route
plus all feasible alternatives, sorted by ascending day-total time.

```python
result = evaluate_target_reachability(aircraft, start, target, ...)
# {"reachable": True,
#  "best": {"itinerary": "outbound_refuel", "refuel_airport": "KLBB", ...},
#  "alternatives": [{"itinerary": "return_refuel", ...}, ...],
#  "unreachable_reason": None, ...}
```

Useful for pre-flight site go/no-go decisions and flight-following
re-tasking ("can we still catch this overpass?").

### Concentric reach — `compute_concentric_isochrones(...)`

Sweep multiple budgets (e.g., 1 / 2 / 3 / 4 hr) in one call.  Each
budget seeds the next larger budget's lower bound, so total cost is
roughly O(M+N) instead of O(M·N) for M budgets and N rays.  Returns
one stacked GeoDataFrame tagged with `budget_min` / `budget_hr`
columns; `plot_isochrone_static` auto-renders nested contours with
a viridis color ramp.

### Plotters

* `plot_isochrone(gdf, ...)` — interactive Folium map (in
  `hyplan.planning.isochrone`).  Recognizes refuel results and
  decorates them with refuel-airport markers (used / evaluated /
  unreachable) and per-itinerary dot coloring; recognizes concentric
  results and labels the popup accordingly.
* `plot_isochrone_static(layers, ...)` — publication-quality Cartopy
  plotter (in `hyplan.plotting`, alongside `plot_airspace_map`).
  Accepts either a single GeoDataFrame or a list of
  `(gdf, color, label)` tuples for layered comparison; auto-handles
  concentric (viridis ramp), refuel markers, and optional wind-barb
  overlay sampled from a `WindField`.

### Fleet pruning: Learjet, BAe-146, Dash-8 removed

The v1.4 release shipped 15 aircraft; three (Learjet 35,
British Aerospace BAe-146, de Havilland Dash-8) carried only
brochure-derived performance values that never made it onto the
calibration roadmap.  Rather than letting brochure-only entries
linger and silently drift in mission timing, v1.5 removes them.
Calibration data for these airframes (notably the FAAM BAe-146)
remains accessible behind the same auth-walled paths documented in
[`docs/calibration.md`](docs/calibration.md); when a campaign
requires one of these platforms, the relevant calibration loader
skeleton is in place to ingest a delivered IWG1/ICARTT archive.

The remaining fleet is **12 aircraft**: NASA ER-2, G-III, G-IV, G-V,
C-20A, P-3, WB-57, B-777, King Air B-200, King Air A-90, C-130, and
Twin Otter.  Eight (ER-2, G-III, G-V, WB-57, P-3, B-200, Twin Otter,
C-130) are data-fit calibrated; the rest carry brochure-derived
performance with `confidence=0.7` until calibration data lands.

### Notebook polish

* `notebooks/aircraft_performance.ipynb` — climb-rate plot now
  splits cleanly by VerticalProfile mode; the stale "analytical
  exponential" chart title is gone.
* `notebooks/calibration/er2/calibration.ipynb` — narrative no
  longer hard-codes a sortie count that drifted post-BlueFlux.

### API additions (all backward-compatible)

* New: `hyplan.compute_isochrone`,
  `hyplan.compute_concentric_isochrones`,
  `hyplan.compute_refuel_isochrone`,
  `hyplan.evaluate_target_reachability`,
  `hyplan.isochrone_polygon`, `hyplan.plot_isochrone`,
  `hyplan.plot_isochrone_static`.
* Stability: the `hyplan.planning.isochrone` module is **Stable** —
  see [`docs/stability.md`](docs/stability.md).

### Removed

* `hyplan.Learjet`, `hyplan.BAe146`, `hyplan.Dash8` aircraft classes.
  Callers using these names will get an `ImportError`; replace with
  a calibrated alternative or load via the campaign's own
  calibration archive.


## v1.4.1 — 2026-05-05

Patch release.  No public-API changes; ships post-release CI
hygiene plus a notebook re-execution sweep so the user-facing
notebooks reflect the v1.4 calibrations.

### Notebook refresh

Bulk papermill re-execution of all 38 user-facing notebooks
against the v1.4 multi-aircraft calibration.  Visible changes:
the tutorial's B-200 example now shows cruise 238.5 kt and
ceiling 30000 ft (vs the pre-v1.4 brochure 233.75 kt / 35000 ft),
and the per-aircraft calibration notebooks pick up the post-
BlueFlux sortie counts in their summary tables and paste-ready
cells.

### CI hygiene

The v1.4.0 tag was green locally on Python 3.11 but failed CI on
Python 3.9 / 3.11 / 3.12 against newer numpy 2.4 stubs and a
recent timezonefinder upgrade.  This release fixes that without
changing the runtime.

* Reverts the v1.4.0 mypy strict-mode pass (the 121 stripped
  `# type: ignore` comments were load-bearing on CI's older pint
  stubs).  The four substantive type fixes from that pass
  (`airports.py` Optional, `winds/utils.py` Optional[float],
  `units.py` float() wrap, `atmosphere.py` Quantity annotation)
  are also reverted; if desired they can be re-applied once
  tested against the CI matrix.
* `pyproject.toml` mypy overrides:
  - `timezonefinder.*` skipped — the package added match-statement
    syntax that mypy can't parse under `python_version=3.9`.
  - `pint.*` skipped — pint's stubs interact differently with
    newer numpy stubs (PlainQuantity vs Quantity); skipping
    treats Quantity as Any, matching the project's de-facto
    pre-existing posture.
  - `disable_error_code = ["var-annotated"]` — newer numpy stubs
    require annotations on `np.empty(...)` assignments that
    pre-existing code doesn't carry.
* Spot fixes: `np.trapezoid` mypy stub gap (3×), several
  `np.array` reassignment + `np.ma.masked_array` return-type
  ignores, `Figure.colorbar` union-attr ignore.

### Bug fix

`Aircraft._climb` and `Aircraft._descend` short-circuit when
`wind_along_track=0 * ureg.knot` so the result is exactly equal
to `wind_along_track=None` rather than ULP-level equal.  Pre-v1.4.1
the tests `test_climb_still_air_unchanged` /
`test_descent_still_air_unchanged` happened to pass on local
numpy/pint and fail on CI's; both now hold by construction.


## v1.4.0 — 2026-05-05

This release expands the data-fit aircraft fleet from one platform (the NASA ER-2) to **eight** (ER-2, G-III, G-V, WB-57, C-130H, P-3, King Air B-200, Twin Otter), introduces the `ClimbOutPolicy` abstraction so per-aircraft pre-cruise level-offs are explicit rather than absorbed into `climb_profile`, and removes the legacy `DubinsPath3D` solver that the v1.3 hybrid path superseded. Public APIs are unchanged for callers; **mission timing for any aircraft other than the ER-2 will shift** because seven previously-brochure platforms now carry calibrated climb / descent / cruise schedules.

### Why this release

Pre-v1.4 only the ER-2 had a calibrated performance model; the rest of the fleet ran on manufacturer brochures. With the v1.3 hybrid planner consuming the calibrated curves directly, the gap between ER-2 and the rest of the fleet became the dominant source of mission-timing error for non-ER-2 platforms. v1.4 closes that gap by data-fitting seven additional aircraft from public NASA ASP archive IWG1 logs and per-campaign ICARTT deliveries, and by making the ER-2's pre-cruise step climb / hold structure expressible through a first-class `ClimbOutPolicy` rather than smuggled into `climb_profile`.

### Aircraft calibration: seven new platforms

`NASA_GIII`, `NASA_GV`, `NASA_WB57`, `C130`, `NASA_P3`, `KingAirB200`, and `TwinOtter` now ship with per-altitude-bin medians for `climb_profile` / `descent_profile`, independent climb / cruise / descent `TasSchedule`s, data-derived `approach_speed`, and operational-p99 `service_ceiling`. `turn_model.max_bank_deg` is `max(AFM normal-ops 30°, data p90)` so the planner uses a bank the aircraft is actually flown at, not the typical-mix median. `confidence=0.85` for data-fit calibrations; brochure / sibling-airframe-inferred values keep `confidence=0.7`.

Sources by class:

* **NASA ASP archive IWG1** — G-III (153 sorties), G-V (101), WB-57 (100), C-130H (87), P-3 (252).
* **NASA ICARTT (multi-campaign)** — King Air B-200 (250 sorties across ACTAMERICA, DISCOVER-AQ California / Colorado / Texas, KORUS-AQ, LMOS).
* **NOAA ICARTT (FIREX-AQ)** — N48RF Twin Otter.

Each platform has a per-aircraft `notebooks/calibration/<aircraft>/calibration.ipynb` notebook (renamed from `iwg1_calibration.ipynb` since two of them ingest ICARTT, not IWG1) and a paste-ready constructor block emitting calibrated constants + `SourceRecord`. See [`docs/calibration.md`](docs/calibration.md) for the methodology overview, the operational-vs-aircraft-intrinsic caveats, and the deferred-access plan for `NCAR_GV` (HIAPER), `NASA_C20A`, `BAe146`, and `KingAirA90`.

### `ClimbOutPolicy` for explicit pre-cruise structure

`Aircraft.typical_climb_out: ClimbOutPolicy` names what `climb_profile` is calibrated to absorb. The ER-2's 19–21 kft fuel-management hold and 23 kft post-step recovery are now encoded as a separate policy rather than deformed into the active-climb profile. `compute_flight_plan` accepts `climb_plan='auto'` to resolve to the aircraft's `typical_climb_out`. The ER-2 `climb_profile` is refit to active-climb-only medians at 5-kft bins, and the monotone-from-SL clamp is dropped (jets peak ROC near FL050–FL100, not at SL). Same posture applies to `descent_profile`: monotonicity is no longer enforced because real descent VS peaks near FL150–FL200 (CAS-limited) and declines in the upper levels (Mach-limited).

### `ICARTT` loader + `IWG1TraceWindField` provider

`hyplan.aircraft.icartt.load_icartt` reads NASA AMES FFI 1001 ICARTT (.ict) files into the canonical IWG1-style schema, with flexible `_COLUMN_PATTERNS` covering the per-campaign variable-name conventions (DISCOVER-AQ, KORUS-AQ, ACTAMERICA, LMOS, FIREX-AQ). `detect_platform()` reads `PLATFORM:` from the ICT free-text block. `hyplan.winds.providers.IWG1TraceWindField` is promoted from a notebook helper to a first-class wind provider so the planner can pull per-segment wind from a flight's own trace. New skeleton loaders at `hyplan/aircraft/eol_ncar.py` (NCAR HIAPER LRT name table) and `hyplan/aircraft/faam_netcdf.py` (FAAM Core variable map) are in place pending NCAR EOL ORDER and CEDA registration respectively.

### Inferred classes + loader prep for FAAM / EOL

`NCAR_GV` (HIAPER, N677F) is registered as a separate class mirroring `NASA_GV`'s calibrated values (confidence 0.7) so HIAPER-specific values can replace these without affecting `NASA_GV`. `NASA_C20A` (NASA 502, AFRC G-III variant) inherits from the calibrated `NASA_GIII` (same airframe, same type certificate; confidence 0.7).

### Calibration notebook infrastructure

`notebooks/calibration/_common.py` extracts the helpers every per-aircraft builder used to inline: `label_phases`, `apply_sortie_filters`, `per_bin`, `tas_per_bin`, `schedule_pts`, `evaluate_profile`, `summary_table`. Aircraft-specific knobs (active-VS threshold, target altitudes, rotation TAS, ER-2 hold bands) stay in the per-aircraft builder. `notebooks/calibration/_asp_fetch.py` provides a tail-keyed crawler for the NASA ASP archive. Every calibration notebook gains a §1 `summary_table` cell and a §9 operational-vs-aircraft-intrinsic markdown block so reviewers can tell which numbers describe airframe performance vs. mission-mix behavior.

### Other additions

`Aircraft.stall_speed_cas` + `min_safe_speed_at(altitude, margin=1.3)` for caller-side bounds checking, using compressible CAS-to-TAS conversion. `compute_flight_plan` accepts a `ClimbPlan` parameter (`'auto'` / explicit / `None`). Per-aircraft `bank_by_phase` is consumed end-to-end. `split_iwg1_alltracks` autodetects deliveries with missing `HEADER` rows. `load_iwg1` filters implausible GPS positions and isolated GPS spikes (rolling-median outlier detection), and `trim_ground_taxi` falls back to altitude-only airborne detection when groundspeed is missing from a delivery.

### Behavior changes & migration

Mission timing and ground-track geometry from `compute_flight_plan` shift for the seven newly-calibrated aircraft. Tests or scripts pinning exact times or top-of-climb coordinates for `NASA_GIII`, `NASA_GV`, `NASA_WB57`, `C130`, `NASA_P3`, `KingAirB200`, or `TwinOtter` need their expected values regenerated. `NASA_ER2` timing is also slightly affected by the active-climb refit + `ClimbOutPolicy` migration; ER-2 reference sorties stay within ±5 % of v1.3.

`service_ceiling` and `approach_speed` on the data-fit classes are now operational p99 / median rather than airframe brochure ceiling / typical AFM approach. The B-200 ships `service_ceiling=30000 ft` (vs 35000 ft brochure); the Twin Otter ships `service_ceiling=15000 ft` (vs 25000 ft brochure). Each class explicitly comments which value is shipped and why.

### Removed / deprecated

`DubinsPath3D` and `Aircraft.pitch_limits()` are removed (the v1.3 hybrid planner already superseded both). `notebooks/calibration/<aircraft>/iwg1_calibration.ipynb` is renamed to `calibration.ipynb` for every aircraft.

### Code quality

121 unused `# type: ignore` comments stripped across 24 modules. Strict-mode mypy (`--warn-unused-ignores --warn-redundant-casts --warn-return-any --warn-unreachable`) is now clean. Three latent bugs surfaced + fixed: `_wind_factor` / `_wind_factor_from_uv` heading_deg annotation (was `float`, callers pass `None`); `convert_speed` returning `Any`; `mach_to_tas` intermediate Quantity annotation. `airports.py` lazy-loaded fields are now properly `Optional` with accesses routed through `require_airports()` / `require_runways()`.


## v1.3.0 — 2026-04-30

This release recalibrates the flight planner against real-world telemetry. Public APIs are unchanged, but **mission timing and ground-track geometry shift** for any aircraft with a non-trivial vertical profile or under non-zero wind.

### Why this release

Pre-v1.3 the planner used `DubinsPath3D`, a constant-pitch 3D Dubins solver whose pitch came from each aircraft's sea-level rate of climb. That collapsed every climb into a single linear ramp regardless of the actual rate-of-climb curve. For the ER-2 — with its 19–21 kft step-climb plateau and 23 kft post-step recovery — top-of-climb fell tens of nautical miles closer to departure than reality, skewing on-station entry and total mission time. The hybrid path fixes this by decoupling horizontal geometry (Dubins) from the vertical profile (integrated point-by-point against the calibrated rate curve).

### Hybrid 2D + integrated-vertical planner

`Aircraft._hybrid_path` replaces the 3D Dubins solver behind `time_to_takeoff` / `time_to_cruise` / `time_to_return`. Horizontal geometry is solved by 2D Dubins with turn radius from the per-phase bank; vertical profile is integrated against horizontal distance from `climb_profile` / `descent_profile`. Top-of-climb and top-of-descent now land at physically realistic positions. Short legs that cannot reach the requested cruise altitude get a spiral-up (or spiral-down) orbit at the phase-appropriate bank and midpoint altitude, rather than a collapsed cruise. `bank_by_phase` is consumed end-to-end by both `_hybrid_path` and `loiter_orbit_geometry`.

### Trochoidal wind support

`_TrochoidDubins2D` dispatches between BSB, the proper Sachdev/Moon (2023) trochoidal CCC solver (1-D Newton on the middle-arc half-angle, with k₄ wrap enumeration), and an iterative air-drift fallback by total time. Wind-bent racetrack turns now find their true time-optimal solution. `DubinsPath2D` is promoted to a public class.

### NASA ER-2 calibration

`NASA_ER2()` is calibrated against 17 NASA AFRC IWG1 in-situ sorties (~64 000 cruise fixes): distinct climb / cruise / descent TAS schedules (was a single brochure curve); 8-anchor `climb_profile` resolving the 19–21 kft step climb and 23 kft recovery (was 2-point linear); 6-anchor `descent_profile` (was 3-point); `ApproachProfile` with empirical 2.51° glideslope; calibrated `bank_by_phase` (climb 11°, cruise 20°, descent 16°, approach 9°). Modeled-vs-flown total duration: NM17 B +2.1 %, CO07v4 +7.9 %, CO06 +1.4 %; multi-sortie time-to-cruise residual ≤ 5 % across the n=17 set (vs. ~36 % pre-v1.3).

New supporting tooling: the `hyplan.aircraft.iwg1` loader (with `trim_ground_taxi`), `IWG1TraceWindField`, a planned-sortie parser (Green Card XLSX/PDF via `pdfplumber` + KML), and three calibration notebooks under `notebooks/calibration/er2/` (`iwg1_calibration`, `sortie_replay`, `planned_vs_flown`).

### Other additions

`Aircraft.climb_speed_at`, `step_climb` (climb-out pauses for fuel burn — distinct from cross-survey altitude drift), `climb_gradient_at` / `descent_gradient_at`, `max_bank_under_budget` (defensive ceiling from `TurnModel.max_load_factor`, default 2.5 g), and a service-ceiling warning when `_hybrid_path` is asked for a cruise altitude above the published ceiling.

### Behavior changes & migration

Mission timing and geometry from `compute_flight_plan` differ from v1.2.0 (typically ±5–15 % on total time; on-station entry shifts by tens of nmi for the ER-2). Tests or scripts pinning exact times or top-of-climb coordinates need their expected values regenerated; the updated `tests/` in this release demonstrate the pattern.

### Deprecation

`Aircraft.pitch_limits()` is legacy — only `DubinsPath3D` consumes it, which the planner no longer uses. Slated for removal.


## v1.2.0 — 2026-04-26

Backwards-compatible feature release. The flight-line optimizer now accepts heterogeneous visit-item input — `FlightLine`, `Pattern`, and bare `Waypoint` objects can all be mixed in a single call to `greedy_optimize`.

### New features

- **Heterogeneous visit-item optimizer** (`hyplan.flight_optimizer.greedy_optimize`): the `flight_lines` argument now accepts a mixed list of `FlightLine | Pattern | Waypoint` objects. The output `flight_sequence` is a list of the same heterogeneous kinds in scheduled order.
  - **Atomic Patterns**: a `Pattern` in the input is treated as a single indivisible visit item. The optimizer may reorder a Pattern relative to other items but never splits it apart, and pattern traversal is direction-locked (entry → exit). Endurance and refueling feasibility evaluate the pattern as one chunk: if `transit_in + pattern_internal_time + transit_out` would exceed remaining endurance, the optimizer schedules a refuel **before** the pattern, never inside it. New `Pattern.entry_waypoint` / `Pattern.exit_waypoint` properties expose the structural endpoints.
  - **Bare Waypoints**: a `Waypoint` in the input is treated as an atomic single-point visit item. Internal time equals `waypoint.delay` (loiter time) if set, else 0. Like Patterns, bare Waypoints are direction-locked, and a long `delay` that doesn't fit in remaining endurance forces a refuel **before** the waypoint, never inside the loiter. `compute_flight_plan` already emits the corresponding `"loiter"` segment.
  - **Result schema**: the `greedy_optimize` result dict now distinguishes visit-item counts from line-leg counts:
    - `items_covered` (`int`) and `items_skipped` (`list[str]`) report whole visit items — each Pattern, FlightLine, or Waypoint is one item.
    - `lines_covered` (`int`) and `lines_skipped` (`list[str]`) keep the pre-v1.2 line-leg semantics: a line-based Pattern contributes one per internal leg (legs of a skipped Pattern appear in `lines_skipped` as `"{item_key}:{line_id}"`); waypoint-based Patterns and bare Waypoints contribute zero on either side.
    - For all-FlightLine input the item and line pairs are equivalent.
- **`hyplan.plotting.plot_flight_plan`**: now renders `Pattern` objects in the `flight_sequence` overlay by drawing each constituent leg (line-based) or waypoint (waypoint-based), labeling the first child with the pattern name. Previously, Patterns were silently omitted from the overlay.
- **Loiter hold-orbit geometry** (`hyplan.planning.segments.loiter_orbit_geometry`): `compute_flight_plan` now renders a real circular hold orbit for `loiter` segments (right-hand turn at the aircraft's `cruise_deg` bank angle, radius from `v² / (g · tan φ)`) instead of a single-point geometry. The loiter row's `distance` becomes the actual ground covered during the loiter (cruise speed × delay) rather than zero. With no `altitude_msl` on the Waypoint, the planner falls back to the prior Point geometry / zero distance.
- **`Waypoint` round-trip serialization**: `Waypoint.to_dict()` now includes all eight fields (`speed`, `delay`, `segment_type` were missing). New companion `Waypoint.from_dict(d)` classmethod reconstructs a Waypoint from a `to_dict` dictionary, supporting full round-trip identity.

## v1.1.0 — 2026-04-26

Backwards-compatible feature release. New atmospheric profiling-lidar instrument family, AWP planning helpers, the Pattern abstraction, public campaign mutation API, and a top-to-bottom documentation polish pass. No v1.0.0 stable APIs change.

### New features

- **ProfilingLidar family** (`hyplan.instruments.profilinglidar`): new `ProfilingLidar(Sensor)` base class for nadir-pointing single-beam atmospheric profiling lidars, with three pre-configured instruments:
  - `HSRL2` — NASA Langley High Spectral Resolution Lidar (3 wavelengths, 200 Hz, 40 cm telescope; defaults from Müller et al. 2014 + Hair et al. 2008 heritage).
  - `HALO` — NASA Langley High Altitude Lidar Observatory (4 wavelengths including methane DIAL at 1645 nm; defaults from Carroll et al. 2022).
  - `CPL` — NASA Goddard Cloud Physics Lidar (3 wavelengths, 5 kHz photon-counting; defaults from McGill et al. 2002).
  - Helpers: `footprint_diameter`, `horizontal_resolution`, `pulses_per_profile`. New tutorial notebook [`notebooks/profiling_lidar_planning.ipynb`](notebooks/profiling_lidar_planning.ipynb) demonstrating the family.
- **AWP instrument planning** (`hyplan.instruments.awp`): NASA Langley Aerosol Wind Profiler (Doppler dual-LOS) instrument model and planning helpers (`flag_awp_stable_segments`, `awp_profile_locations_for_flight_line`, `awp_profile_locations_for_plan`). Supports terrain-aware LOS placement via DEM ray-tracing. New notebook [`notebooks/awp_planning.ipynb`](notebooks/awp_planning.ipynb).
- **Pattern abstraction** (`hyplan.pattern`): first-class `Pattern` class sitting between flight-pattern generators and campaign/planning workflows. New `glint_arc` generator for solar-glint observation patterns.
- **Public campaign mutation API** + `FlightLine.from_geojson` + revision metadata on plan records.

### Documentation

- 100% of `hyplan.__all__` symbols rendered in the API docs (closed 22 coverage gaps in airspace, flight_box, plotting, sensors, swath, and added `hyplan.setup_logging`).
- Sphinx build is now clean under `-W` (warnings-as-errors) — down from 86 warnings to zero.
- New API page [`docs/api/profiling_lidar.md`](docs/api/profiling_lidar.md); new prose page documenting the `Pattern` class.
- Module docstring fixes in `aircraft/_models.py`, `clouds/sources.py`, `flight_patterns.py`, `geometry.py`, `glint.py`, `phenology/plotting.py`, `planning/segments.py`, `satellites.py`, `sun.py`, and `terrain/__init__.py`.

### Community

- New `CODE_OF_CONDUCT.md` and `CONTRIBUTORS.md`.
- New `SECURITY.md` documenting the vulnerability reporting process (GitHub Private Vulnerability Reporting + email fallback) and supported-versions policy.
- New `.github/ISSUE_TEMPLATE/` directory with structured **bug report** and **feature request** YAML forms, plus a `config.yml` routing usage questions to docs and Discussions.
- New `.github/PULL_REQUEST_TEMPLATE.md` with summary, test-plan checklist, CHANGELOG checkbox, and backward-compatibility prompt. GitHub community-profile health is now 100%.
- `LICENSE.md` reformatted to embed the canonical Apache 2.0 license text verbatim so GitHub's licensee tool detects it as Apache-2.0 (was previously reported as "Other / NOASSERTION" because the file was a brief summary). The legal license is unchanged — has been Apache 2.0 since v1.0.0.

### Bug fixes

- Fixed CI lint step that was blocking `tests.yml` on Python 3.9 / 3.11 / 3.12 (7 ruff errors across 5 files).
- `compute_overpass_overlap`: harden geometry-emptiness check to handle non-`None` non-Geometry sentinels (e.g. NaN) that GeoPandas can produce in `geometry=[None]` rows.

### Dependencies

- Dropped the `seaborn` dependency from the `[clouds]` extra. The two `sns.heatmap()` call sites in `hyplan.clouds.plotting` now use a small private matplotlib-only helper instead.

### Cleanup

- Removed empty `hyplan.gui` subpackage.
- AWP instrument code moved from `hyplan/awp.py` into `hyplan/instruments/awp.py`; the public `from hyplan.instruments import AerosolWindProfiler` import path is unchanged.

## v1.0.0

HyPlan v1.0.0 is the first stable release — core flight planning workflows are production-ready and covered by API stability guarantees for the 1.x series.

### Highlights

- **API stability**: 22 modules promoted to **Stable** (flight lines, terrain, swath, planning, winds, aircraft, exports, airports, atmosphere, flight box, flight optimizer, sun, glint, clouds, phenology, satellites, airspace, Dubins paths, flight patterns, plotting, geometry, units). Stable APIs will not break within the 1.x series. See [`docs/stability.md`](docs/stability.md) for the full listing and deprecation policy.
- **Notebook overhaul**: All 28 notebooks refactored with standardized structure — header blocks, conceptual framing, result interpretation, operational takeaways, and common pitfalls. A new [`notebooks/README.md`](notebooks/README.md) organizes them into a guided learning path.
- **Code quality**: Zero `mypy` and `ruff` errors across the entire codebase. +3,500 lines of new tests, including end-to-end workflow regression tests.
- **JOSS paper**: Revised figures and text submitted for review.

### New features

- **Vegetation phenology module** (`hyplan.phenology`): retrieve historical NDVI/EVI, LAI/FPAR, and phenological transition dates from MODIS products via NASA EarthData. Includes seasonal profile plots, phenology calendar, year-over-year heatmaps, and combined cloud + phenology visualizations.
- **Shared EarthData authentication** (`hyplan._auth`): extracted from the winds module so both `winds` and `phenology` can authenticate without cross-package coupling.
- **Terrain module refactored** into a package with `DEMGrid` dataclass. Assumptions and limitations now documented.
- **Clouds module refactored** into a package with separate source, analysis, forecast, and plotting submodules.
- **BSB trochoidal Dubins solver**: ported from castacks/trochoids for wind-corrected transit paths.
- **pyhdf-based reader** for MODIS HDF4-EOS phenology files.

### Breaking changes

- `DynamicAviation_*` aircraft classes renamed to generic platform names.
- Experimental GUI module moved to `dev/gui-widgets` branch (removed from main).

### Bug fixes

- Fixed `_validate_quantity` in sensor base class to accept both `Quantity` and `Unit` arguments.
- MERRA-2 wind provider now uses `dap2://` scheme to avoid pydap protocol detection warnings.
- Fixed variable name bug in `winds.ipynb`.
- Updated Twin Otter service ceiling and max bank angle.
- Small gaps are now merged when clipping flight lines to polygons.
- Resolved all `mypy` type errors in `_trochoid_solver`, `dubins3d`, and `phenology` modules.
- Resolved all `ruff` lint errors (unused imports/variables in `phenology`).
- Fixed `solar_planning.ipynb` cell incorrectly typed as code instead of markdown.
- Fixed `dubins_path_planning.ipynb` missing imports for `compute_flight_plan` integration example.

### Known limitations

- Aircraft performance parameters are approximate; ADS-B calibration infrastructure exists in `hyplan.aircraft.adsb` but calibration is ongoing.
- Cloud fraction sources (GEE vs Open-Meteo/ERA5) differ in spatial resolution and interpretation — see module docstrings.
- Terrain intersection uses fixed-step ray marching, not root-finding.
- Flight optimizer does not yet incorporate environmental constraints (solar windows, cloud forecasts, airspace conflicts); these are applied as separate filtering steps.
- CCC (RLR/LRL) Dubins path types are disabled under wind — only BSB paths are solved for trochoidal cases.
