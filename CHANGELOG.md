# Changelog

## v1.0.0 (unreleased)

## v0.99 — Release Candidate for v1.0

### Highlights

This release transforms the HyPlan notebook collection from
developer-oriented examples into a **cohesive, user-facing learning
system and operational reference**. All 28 notebooks have been
refactored for clarity, completeness, and usability. The codebase is
also substantially hardened: 10 modules promoted to Stable, all linting
and type-checking errors resolved, and test coverage expanded across
nine modules.

### Notebook overhaul

Every notebook now follows a standard structure:

- **Header block** — title, purpose paragraph, metadata table (audience
  level, runtime, internet/credential requirements, example data), and
  "What You Will Learn" bullets.
- **Conceptual framing** — mental-model explanations before major code
  sections so readers understand *what problem is being solved* before
  seeing code.
- **Result interpretation** — after every major output (plot, table,
  map), a short paragraph explains what the result shows, why it
  matters, and what the user should conclude.
- **Operational Takeaways** — 3–6 bullets at the end summarizing
  real-world implications.
- **Common Pitfalls** — warnings about units confusion (feet vs meters),
  MSL vs AGL, heading vs ground track, wind direction conventions, and
  other frequent mistakes.

Notebook-specific additions include:

- **`tutorial.ipynb`**: high-level workflow diagram, "What This Notebook
  Produces" section, final outputs summary, and "Next Steps" linking to
  deeper notebooks.
- **`flight_line_operations.ipynb`**: visual glossary of azimuth,
  start/end, midpoint, and along-track/across-track offsets; geodetic
  assumptions section.
- **`flight_plan_computation.ipynb`**: text-based conceptual pipeline
  diagram, segment type definitions, summary metrics discussion.
- **`wind_effects.ipynb`**: three new analysis sections — crab angle vs
  wind speed curves, swath footprint distortion visualization under
  crosswind, and multi-line coverage gap/overlap analysis quantifying
  the error when wind is ignored in flight planning.
- **`winds.ipynb`**: wind convention explanations, decision guidance for
  constant wind vs MERRA-2.
- **`terrain_aware_planning.ipynb`**: MSL vs AGL pitfalls, DEM source
  documentation, terrain profile interpretation.
- **`frame_camera_planning.ipynb`**: frame vs line scanner comparison,
  overlap requirements.
- **`stereo_oblique_planning.ipynb`**: stereo constraints, nadir vs
  oblique tradeoffs.
- **`radar_sar_missions.ipynb`**: SAR geometry explanation.
- **`lidar_lvis_planning.ipynb`**: LVIS mission geometry, comparison
  with imaging spectroscopy planning.

A new **`notebooks/README.md`** organizes all 28 notebooks into a
guided learning path with categories (Start Here, Core Geometry,
Environmental Constraints, Mission Types, Terrain & Airspace, Campaign
Management, Export, Validation), descriptions, suggested order, and a
quick-reference prerequisites table.

### New features

- **Vegetation phenology module** (`hyplan.phenology`): retrieve
  historical NDVI/EVI, LAI/FPAR, and phenological transition dates
  from MODIS products via NASA EarthData.
- **Shared EarthData authentication** (`hyplan._auth`).
- **Terrain module refactored** into a package with `DEMGrid` dataclass.
- **Clouds module refactored** into a package with separate submodules.
- **BSB trochoidal Dubins solver** ported from castacks/trochoids for
  wind-corrected transit paths.
- **pyhdf-based reader** for MODIS HDF4-EOS phenology files.

### Code quality

- 10 modules promoted from Experimental to **Stable** for the 1.x
  series.
- All modules pass `mypy --ignore-missing-imports` with zero errors.
- All modules pass `ruff` linting with zero errors.
- Test coverage expanded across nine modules (+3,500 lines of tests).
- End-to-end workflow regression tests added.
- API stability levels, changelog, and deprecation policy documented.

### Breaking changes

- `DynamicAviation_*` aircraft classes renamed to generic platform names.
- Experimental GUI module moved to `dev/gui-widgets` branch (removed
  from main).

### Bug fixes

- Fixed `_validate_quantity` in sensor base class to accept both
  `Quantity` and `Unit` arguments.
- MERRA-2 wind provider now uses `dap2://` scheme to avoid pydap
  protocol detection warnings.
- Fixed variable name bug in `winds.ipynb`.
- Updated Twin Otter service ceiling and max bank angle.
- Small gaps are now merged when clipping flight lines to polygons.

### Highlights

HyPlan v1.0 marks the first stable release. Core flight planning
workflows are production-ready and covered by the API stability
guarantees described in the [stability guide](docs/stability.md).

### API stability

Modules are now classified as **Stable** or **Experimental**.
Stable APIs will not break within the 1.x series; algorithms, defaults,
and datasets may still be improved. Stable modules include: flight lines,
terrain, swath, planning, winds, aircraft, exports, airports, atmosphere,
flight box, flight optimizer, sun, glint, clouds, phenology, satellites,
airspace, Dubins paths, flight patterns, plotting, geometry, and units.
Experimental modules (LVIS, radar, frame camera, campaign) work and
are tested but their signatures may still change based on user feedback.

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
  with ADS-B calibration data in future releases. Infrastructure for
  fitting models to ADS-B tracks exists in `hyplan.aircraft.adsb` but
  that calibration work is ongoing.
- Cloud fraction sources (GEE vs Open-Meteo/ERA5) differ in spatial
  resolution and interpretation — see module docstrings.
- Terrain intersection uses fixed-step ray marching, not root-finding.
- Flight optimizer does not yet incorporate environmental constraints
  (solar windows, cloud forecasts, airspace conflicts); these are
  applied as separate filtering steps outside the optimizer.
- CCC (RLR/LRL) Dubins path types are disabled under wind — only
  BSB (bang-straight-bang) paths are solved for trochoidal cases
  to avoid degenerate multi-loop solutions. This may produce slightly
  longer paths in rare geometries where a CCC solution would be shorter.
