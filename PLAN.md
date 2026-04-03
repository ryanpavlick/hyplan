
# HyPlan Improvement Plan

---

## Recently Completed

- **FAA L-Band radar exclusion zone check** (`radar.py`) — `check_lband_radar_exclusions()` checks UAVSAR swath polygons against bundled 10 NMI exclusion zones; `RadarExclusionConflict` dataclass; GeoJSON bundled at `hyplan/data/faa_radar_exclusion_zones.geojson`
- **Terrain-aware flight box — Mode 3** (`flight_box.py`) — `box_around_polygon_terrain` and `box_around_center_terrain` now accept `target_agl`; each flight line receives an individual altitude from mean nadir terrain elevation + target AGL (Zhao et al. 2021)
- **Terrain utility functions** (`terrain.py`) — `terrain_elevation_along_track()` (min/mean/max along nadir track); `terrain_aspect_azimuth()` (dominant downslope direction for optimal flight azimuth)
- **Notebook updates** — `terrain_aware_planning.ipynb` sections 6–7: terrain aspect azimuth and Mode 2 vs Mode 3 altitude comparison; `radar_sar_missions.ipynb` section 10: exclusion zone check demo

---

## Not Yet Scoped

### Bugs

#### Trivial (1–2 lines each)

- **`flight_line.py:477`** — `wrap_to_180(180+az21)` in `offset_along` corrupts azimuth semantics. `wrap_to_180` maps 350° → −10°, flipping the direction. Fix: `(az21 + 180) % 360` without wrapping.

- **`flight_line.py:~360`** — `track(precision=0)` causes `ZeroDivisionError` via `ceil(length / 0)`. Fix: `if precision_m <= 0: raise HyPlanValueError("precision must be positive")`.

- **`flight_plan.py:~123`** — `latitudes, longitudes` unpacked from `process_linestring()` are never used. Fix: replace with `_, _` or `*_`.

- **`geometry.py:37–61`** — `wrap_to_180` applies `np.squeeze()`, `wrap_to_360` does not. Scalar inputs return different types. Fix: add `np.squeeze()` to `wrap_to_360` for consistency.

- **`terrain.py:266–267`** — The bug note is misleading: `np.round()` is already called before `.astype(int)`, so truncation is not actually occurring. Fix: add a comment clarifying pixel-center rounding semantics to prevent future regression.

- **`lvis.py:257`** — `0.999` contiguity tolerance is unexplained. Fix: define as class constant `_CONTIGUITY_TOLERANCE = 0.999` with a docstring explaining it accounts for floating-point error in footprint/speed calculations.

- **`plotting.py`** — Not a real bug; `return m` is present. Remove from list.

#### Small (3–10 lines each)

- **`campaign.py:464–466`** — `pymap3d.vincenty.vdist()` returns `NaN` azimuth when start == end (distance = 0). Fix: check for equal coordinates before calling `vdist` and use a default azimuth (e.g., 0°) or raise a descriptive error.

- **`flight_optimizer.py:142–148`** — Duplicate site-name detection uses `if key in line_keys.values()` (O(n)) with `id(fl)` fallback (unreliable across GC). Fix: maintain a `seen_keys: set` for O(1) lookup; use an integer counter for collision suffixes instead of `id()`.

- **`flight_plan.py:~123`** — `distances[-1]` on a 1-point LineString silently returns 0 instead of raising. Fix: validate segment length > 0 before processing, or propagate the `track(precision=0)` fix above.

- **`clouds.py:66–94`** — `_ee_initialized` is never set to `True` on failure (correct), but `_ee` is left as a partially-initialized module if `ee.Initialize()` fails after `import ee` succeeds, causing `AttributeError` on retry. Fix: only assign `_ee = _ee_mod` after `Initialize()` succeeds.

- **`airspace.py:405–414`** — Dedup fallback uses `id(it)` when dict has no `"_id"`/`"id"` field; a re-fetched item produces a different `id()` and slips through. Fix: build a deterministic key from content fields (name + geometry hash), or log a warning and skip items with no stable ID.

#### Medium (>10 lines)

- **`aircraft.py:251–256, 314–315`** — ROC linear interpolation model is duplicated between `_climb()` and `climb_altitude_profile()` with inconsistent unit handling. Fix: extract a shared `_roc_at_altitude_m(altitude_m: float) -> float` helper returning ft/min as a float; both methods call it.

### Package-wide
- Consistent logging configuration across all modules (some still use basicConfig)
- Add type stubs or `py.typed` marker for downstream IDE support
- Publish to PyPI / conda-forge
- Add CLI entry point for common tasks (generate flight lines, export plan)

### New modules
- **weather.py** — GMAO forecast API, wind barbs, real-time cloud/visibility, time-on-deck predictions

### Module improvements

**units.py:** angular units (degrees, radians); time unit conversions

**geometry.py:** regionate geometries by UTM zone for large areas; get time zone from lat/lon
- Pattern generators for grids and concentric circles
**flight_line.py:** curved flight lines for glint/CH4 mapping; multi-point flight lines (>2 vertices); merge adjacent collinear lines

**flight_box.py:** accept target GSD instead of raw overlap %; return daily_lines grouping from optimizer-aware box generation

**flight_plan.py:** wind in transit time estimates; fuel burn estimation per segment; multiple altitude levels in single plan; pilot-ready briefing sheet export

**flight_optimizer.py:** convex hull + cheapest insertion route optimizer (monotone chain hull around flight line endpoints to build outer loop, then greedy insertion of interior lines — fast, non-crossing routes; can seed 2-opt; inspired by SynthFlight); Google OR-Tools or 2-opt/simulated annealing; return daily_lines grouping in result dict; time-window constraints (solar, satellite overpass); multi-aircraft scheduling; weight edges by fuel burn

**dubins3d.py:** wind-corrected Dubins paths

**aircraft.py:** altitude-dependent fuel burn model; weight-dependent performance (payload vs range); turboprop vs jet engine modeling

**airports.py:** cache downloaded CSV data; fuel availability and FBO information; filter by services (hangars, maintenance)

**sensors.py:** spectral band metadata (wavelength, bit depth); more sensor implementations; data rate estimation for storage planning

**frame_camera.py:** oblique viewing angles; stereo coverage planning (forward/nadir/backward)

**lvis.py:** shot density vs ground speed model; swath width variation with scan pattern

**radar.py:** interferometric SAR baseline planning; near/far range incidence angle maps; UAVSAR line length constraints (`check_uavsar_line_lengths`); processed swath margins (`processed_swath_margins`, `generate_swath_polygon(imaging_mode=...)`)

**swath.py:** ground pixel size along flight line; plot swath colored by pixel size; 2D cross-section with terrain; gap/overlap analysis between adjacent lines

**sun.py:** integrate with flight_optimizer for solar-constrained scheduling

**glint.py:** percentage of line exceeding glint angle threshold; map predicted glint intensity; glint avoidance routing; vectorize nested Python loops

**clouds.py:** non-GEE cloud sources (ERA5, MERRA-2); probabilistic clear-sky forecasting

**terrain.py:** additional DEM sources (Copernicus GLO-30, ASTER GDEM); obstacle clearance analysis; dominant terrain aspect azimuth for optimal flight line orientation (`terrain_aspect_azimuth`)

**satellites.py:** multi-satellite coordinated overpass planning; revisit frequency maps; underfly timing optimization; vectorize `_compute_headings()` loop

**plotting.py:** 3D flight trajectory visualization; animated flight plan playback; side-by-side multi-day comparison; publication-quality export

**exports.py:** TrackAir format; import flight lines from KML/shapefile; `--strict` mode that raises on missing data

**campaign.py:** bounds validation (flight lines within campaign area)

**airspace.py:** filtering by type (show only RESTRICTED, etc.); handle "no ceiling" SUA special case

**interactive:** hover to reveal solar elevation times; copy/delete/rotate/nudge individual lines; aircraft icons on map; wind barbs overlay
