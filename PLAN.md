
# HyPlan Improvement Plan

---

## Not Yet Scoped

### Package-wide
- Add CLI entry point for common tasks (generate flight lines, export plan)

### Module improvements

**geometry.py:** regionate geometries by UTM zone for large areas
- Pattern generators for grids and concentric circles
**flight_line.py:** merge adjacent collinear lines

**flight_box.py:** accept target GSD instead of raw overlap %; return daily_lines grouping from optimizer-aware box generation

**flight_plan.py:** fuel burn estimation per segment; multiple altitude levels in single plan; pilot-ready briefing sheet export; 1-second path interpolation for satellite coordination timing

**flight_optimizer.py:** convex hull + cheapest insertion route optimizer (monotone chain hull around flight line endpoints to build outer loop, then greedy insertion of interior lines — fast, non-crossing routes; can seed 2-opt; inspired by SynthFlight); Google OR-Tools or 2-opt/simulated annealing; return daily_lines grouping in result dict; time-window constraints (solar, satellite overpass); multi-aircraft scheduling; weight edges by fuel burn

**aircraft.py:** altitude-dependent fuel burn model; weight-dependent performance (payload vs range); turboprop vs jet engine modeling

**sensors.py:** spectral band metadata (wavelength, bit depth); more sensor implementations; data rate estimation for storage planning

**lvis.py:** shot density vs ground speed model; swath width variation with scan pattern

**radar.py:** UAVSAR line length constraints (`check_uavsar_line_lengths`); processed swath margins (`processed_swath_margins`, `generate_swath_polygon(imaging_mode=...)`)

**swath.py:** ground pixel size along flight line; plot swath colored by pixel size; 2D cross-section with terrain

**sun.py:** integrate with flight_optimizer for solar-constrained scheduling

**glint.py:** map predicted glint intensity; glint avoidance routing

**clouds.py:** probabilistic clear-sky forecasting; ~~DOY-averaged cloud fraction summary across years per polygon ("best weeks historically")~~; ~~morning vs afternoon cloud discrimination (Terra vs Aqua separately)~~; ~~spatial cloud fraction maps within flight boxes (not just polygon mean)~~; ~~cloud cover forecasts via Open-Meteo (up to 16-day, multi-model)~~

**terrain.py:** obstacle clearance analysis

**satellites.py:** multi-satellite coordinated overpass planning; revisit frequency maps; underfly timing optimization

**plotting.py:** 3D flight trajectory visualization; animated flight plan playback; side-by-side multi-day comparison; publication-quality export; WMS weather imagery layers; AERONET AOD real-time overlay

**exports.py:** import flight plans from Excel/ICARTT/KML/shapefile; PowerPoint briefing generation (python-pptx); Word mission summary (python-docx); `--strict` mode that raises on missing data

**airspace.py:**
- **P1 — Quick wins:**
  - ~~Type filtering: add `type_filter` param to `fetch_airspaces()`/`fetch_and_check()` to show only RESTRICTED, PROHIBITED, etc.~~
  - ~~Conflict severity classification: tag each `AirspaceConflict` as HARD/ADVISORY/INFO based on airspace class (enables color-coded display)~~
  - ~~Handle "no ceiling" SUA special case (distinguish unlimited ceiling from 60000 ft default)~~
  - ~~Multi-country bounding box: auto-include airspaces from neighboring countries when bbox spans borders~~
- **P2 — Moderate effort, high value:**
  - Entry/exit point extraction: compute lat/lon where flight lines cross airspace boundaries (from existing intersection geometry)
  - Lateral buffer / near-miss warning: optional buffer around airspace polygons to flag proximity without penetration
  - AGL-to-MSL floor conversion: use `terrain.get_elevations()` to convert GND-referenced floors to MSL at conflict points
  - Airspace summary table: `airspace_summary_df()` returning a DataFrame for quick pilot reference
- **P3 — Significant effort, high value:**
  - ~~FAA SUA/NASR data source: free GeoJSON from 28-day NASR cycle (no API key needed for US ops)~~
  - ~~TFR (Temporary Flight Restriction) data retrieval from FAA TFR feed~~
  - Airspace visualization in `plotting.py`: integrate airspace overlay into `plot_flight_plan()` with severity color-coding
- **P4 — High effort or niche:**
  - ~~NOTAM integration (FAA or ICAO API) for temporary airspace changes~~
  - Active schedule / time-of-day filtering: parse MOA/restricted area schedules to filter out inactive airspaces
  - Vertical profile view: distance-vs-altitude cross-section showing airspace floors/ceilings along route
  - NAT/POCAT oceanic track display (via FlightPlanDB API)
  - FIR (Flight Information Region) boundaries (largely covered by type filtering + plotting)

**gui:** hover to reveal solar elevation times; copy/rotate/nudge individual lines; aircraft icons on map; wind barbs overlay; distance measurement tool

