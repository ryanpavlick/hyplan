# Changelog

## v1.0.0 (unreleased)

### Highlights

HyPlan v1.0 marks the first stable release. Core flight planning
workflows are production-ready and covered by the API stability
guarantees described in the [stability guide](docs/stability.md).

### API stability

Modules are now classified as **Stable** or **Experimental**.
Stable APIs (flight lines, terrain, swath, planning, winds, aircraft,
exports, airports, atmosphere) will not break within the 1.x series.
Experimental modules (LVIS, radar, frame camera, GUI, campaign, clouds,
phenology, satellites, glint, airspace, Dubins paths, flight patterns,
plotting) work and are tested but may change based on user feedback.

See `docs/stability.md` for the full listing and deprecation policy.

### New features

- **Vegetation phenology module** (`hyplan.phenology`): retrieve
  historical NDVI/EVI, LAI/FPAR, and phenological transition dates
  from MODIS products via NASA EarthData. Includes seasonal profile
  plots, phenology calendar, year-over-year heatmaps, and combined
  cloud + phenology visualizations.

- **Shared EarthData authentication** (`hyplan._auth`): extracted
  from the winds module so both `winds` and `phenology` can
  authenticate without cross-package coupling.

- **Terrain module refactored** into a package with `DEMGrid`
  dataclass. Assumptions and limitations now documented.

- **Clouds module refactored** into a package with separate source,
  analysis, forecast, and plotting submodules.

### Improvements

- All modules pass `mypy --ignore-missing-imports` with zero errors.
- Added `from __future__ import annotations` across the codebase for
  consistent Python 3.9+ type annotation support.
- Fixed `_validate_quantity` in sensor base class to accept both
  `Quantity` and `Unit` arguments.
- MERRA-2 wind provider now uses `dap2://` scheme to avoid pydap
  protocol detection warnings.

### Known limitations

- Aircraft performance parameters are approximate and will be refined
  with ADS-B calibration data in future releases.
- Cloud fraction sources (GEE vs Open-Meteo/ERA5) differ in spatial
  resolution and interpretation — see module docstrings.
- Terrain intersection uses fixed-step ray marching, not root-finding.
