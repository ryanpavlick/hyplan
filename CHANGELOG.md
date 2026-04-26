# Changelog

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
