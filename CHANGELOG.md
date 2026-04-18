# Changelog

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
