
# HyPlan Improvement Plan

---

## Not Yet Scoped

### Package-wide
- Add CLI entry point for common tasks (generate flight lines, export plan)

### Module improvements

**geometry.py:** regionate geometries by UTM zone for large areas
- Pattern generators for grids and concentric circles
**flight_line.py:** curved flight lines for glint/CH4 mapping; multi-point flight lines (>2 vertices); merge adjacent collinear lines

**flight_box.py:** accept target GSD instead of raw overlap %; return daily_lines grouping from optimizer-aware box generation

**flight_plan.py:** wind in transit time estimates; fuel burn estimation per segment; multiple altitude levels in single plan; pilot-ready briefing sheet export

**flight_optimizer.py:** convex hull + cheapest insertion route optimizer (monotone chain hull around flight line endpoints to build outer loop, then greedy insertion of interior lines — fast, non-crossing routes; can seed 2-opt; inspired by SynthFlight); Google OR-Tools or 2-opt/simulated annealing; return daily_lines grouping in result dict; time-window constraints (solar, satellite overpass); multi-aircraft scheduling; weight edges by fuel burn

**dubins3d.py:** wind-corrected Dubins paths

**aircraft.py:** altitude-dependent fuel burn model; weight-dependent performance (payload vs range); turboprop vs jet engine modeling

**sensors.py:** spectral band metadata (wavelength, bit depth); more sensor implementations; data rate estimation for storage planning

**frame_camera.py:** oblique viewing angles; stereo coverage planning (forward/nadir/backward)

**lvis.py:** shot density vs ground speed model; swath width variation with scan pattern

**radar.py:** interferometric SAR baseline planning; near/far range incidence angle maps; UAVSAR line length constraints (`check_uavsar_line_lengths`); processed swath margins (`processed_swath_margins`, `generate_swath_polygon(imaging_mode=...)`)

**swath.py:** ground pixel size along flight line; plot swath colored by pixel size; 2D cross-section with terrain

**sun.py:** integrate with flight_optimizer for solar-constrained scheduling

**glint.py:** map predicted glint intensity; glint avoidance routing

**clouds.py:** probabilistic clear-sky forecasting

**terrain.py:** obstacle clearance analysis; dominant terrain aspect azimuth for optimal flight line orientation (`terrain_aspect_azimuth`)

**satellites.py:** multi-satellite coordinated overpass planning; revisit frequency maps; underfly timing optimization

**plotting.py:** 3D flight trajectory visualization; animated flight plan playback; side-by-side multi-day comparison; publication-quality export

**exports.py:** TrackAir format; import flight lines from KML/shapefile; `--strict` mode that raises on missing data

**airspace.py:** filtering by type (show only RESTRICTED, etc.); handle "no ceiling" SUA special case

**gui:** hover to reveal solar elevation times; copy/delete/rotate/nudge individual lines; aircraft icons on map; wind barbs overlay
