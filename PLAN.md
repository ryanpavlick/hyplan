
# HyPlan Improvement Plan

---

## Not Yet Scoped

### Bugs (medium/low — not yet planned)
- `campaign.py`: vincenty azimuth undefined when start == end point
- `flight_line.py`: `wrap_to_180(180+az21)` in `offset_along` breaks azimuth semantics
- `flight_optimizer.py`: duplicate key detection is O(n) with brittle `id()` fallback
- `flight_plan.py`: `distances[-1]` on 1-point LineString defaults to 0 silently
- `terrain.py`: `.astype(int)` truncation instead of rounding at pixel boundaries
- `clouds.py`: `_ee_initialized` never resets after failure
- `flight_line.py`: `track(precision=0)` causes division issues — add validation
- `flight_plan.py`: unused `latitudes`/`longitudes` variables in segment processing
- `geometry.py`: `wrap_to_180` and `wrap_to_360` return type inconsistency (squeeze vs not)
- `aircraft.py`: duplicated ROC calculation in `_climb()` and `climb_altitude_profile()`
- `airspace.py`: dedup fallback uses `id()` — unreliable across API calls
- `plotting.py`: missing explicit `return m` in `map_flight_lines`
- `lvis.py`: hardcoded tolerance `0.999` lacks justification comment

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
;L .`,.
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

**radar.py:** interferometric SAR baseline planning; near/far range incidence angle maps

**swath.py:** ground pixel size along flight line; plot swath colored by pixel size; 2D cross-section with terrain; gap/overlap analysis between adjacent lines

**sun.py:** integrate with flight_optimizer for solar-constrained scheduling

**glint.py:** percentage of line exceeding glint angle threshold; map predicted glint intensity; glint avoidance routing; vectorize nested Python loops

**clouds.py:** non-GEE cloud sources (ERA5, MERRA-2); probabilistic clear-sky forecasting

**terrain.py:** additional DEM sources (Copernicus GLO-30, ASTER GDEM); obstacle clearance analysis

**satellites.py:** multi-satellite coordinated overpass planning; revisit frequency maps; underfly timing optimization; vectorize `_compute_headings()` loop

**plotting.py:** 3D flight trajectory visualization; animated flight plan playback; side-by-side multi-day comparison; publication-quality export

**exports.py:** TrackAir format; import flight lines from KML/shapefile; `--strict` mode that raises on missing data

**campaign.py:** bounds validation (flight lines within campaign area)

**airspace.py:** filtering by type (show only RESTRICTED, etc.); handle "no ceiling" SUA special case

**interactive:** hover to reveal solar elevation times; copy/delete/rotate/nudge individual lines; aircraft icons on map; wind barbs overlay
