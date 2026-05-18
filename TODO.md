# TODO

Deferred work items from prior releases.  Each item documents the
issue, why it was deferred, and where it's discussed in code or docs.
Roughly grouped by category and rough target release.

When an item ships, move it into the relevant `## vX.Y.Z` section in
[CHANGELOG.md](CHANGELOG.md).

---

## Maintenance backlog (v1.6.2+)

### Calibration data-quality fixes

* **`NERC_DO228` cruise schedule inversion above FL100** — fitted
  cruise TAS decreases from 176 kt @ FL100 → 146 kt @ FL200, which
  is unphysical for a turboprop.  Sample-size artifact: the
  FL150 / FL200 cruise bins are populated by < 200 fixes from a
  34-sortie sample across two CEDA campaigns.  Approach speed
  110 kt is also above the brochure 80-90 kt Vref.
  Fix: truncate cruise schedule at FL100 or add a sparse-bin
  caveat in `notebooks/calibration/NERC_DO228/calibrate.py`.
  Discussed in `docs/calibration.md` "Performance envelope
  cross-check" and the `NERC_DO228` constructor.

* **`DLR_HALO` flat cruise artifact FL200-FL300** — fitted cruise
  is implausibly flat at 457 kt through FL200-FL300; a G550
  climbing through these levels should see ~360 kt @ FL200 and
  ~430 kt @ FL300.  18-sortie HALO-AC3 sample is heavily biased
  toward high-altitude transit; sub-FL300 cruise bins are
  populated by descent fixes mislabeled as cruise.  Already
  ships with `confidence=0.7` reflecting low sample size.
  Fix: truncate cruise schedule below FL300 or add a
  campaign-specific caveat.

* **`KingAirA90` wind-triangle TAS** — current calibration uses
  groundspeed-as-TAS still-air baseline (`wind_source="still_air"`
  in `notebooks/calibration/KingAirA90/calibrate.py`).  Switch to
  MERRA-2 wind-triangle reconstruction via the existing
  `hyplan.aircraft.adsb.airdata.reconstruct_airdata`.  Requires
  Earthdata token; no new code beyond the wind-source argument.
  Should tighten the cruise schedule (currently has 5-10 kt of
  noise from neglected wind).

* **`KingAirA90` broader sample** — current calibration is from a
  30-day pull (2026-04-09 → 2026-05-08).  airplanes.live archive
  goes back to ~2025-01.  Re-run with a 12-month window once the
  daily-cron has been pulling for a few months.  Sample size
  should grow from 643 sorties to a few thousand.

* **`KingAir350` (UWKA-2) more sortie windows** — currently 22
  sorties from 18 active days (2025-01 through 2026-04, single
  tail).  As UW flies new science campaigns (CHEESEHEAD-followups,
  etc.), pull updated airplanes.live history and re-run.

### Calibration coverage gaps

* **`NASA_GIV`** — NASA AFRC G-IV (N817NA) was deregistered
  2024-07-18.  airplanes.live globe-history archive does have
  meaningful 2023-2024 coverage (cf. May 2024 pulls of
  ~465 KB / day).  An ADS-B-based calibration covering the final
  18 months of NASA AFRC operation would replace the current
  brochure values.

* **`NASA_C20A`** — inferred from `NASA_GIII` (same airframe + type
  certificate).  Still no public ICARTT/IWG1 nav data; AFRC
  mission ops contact required for NASDAT housekeeping logs.

* **`NASA_B777`** — generic placeholder for NASA experimental 777
  ops.  No specific tail or campaign in the public record.

* **USAF WC-130J Hurricane Hunters** — data on disk under
  `data/HRD/USAF_WC130J/` from the NOAA AOML hurricane archive
  (tail-letter `U` in HRD filenames).  Could be a future
  `USAF_WC130J` class, distinct from `NASA_C130` (different
  operator, different mission profile, four-engine WC-130J vs
  NASA's H-model).

### Documentation polish

* **Companion `calibration.ipynb` for the ADS-B classes**
  (`KingAirA90`, `KingAir350`).  Current state is `calibrate.py`
  only.  `notebooks/calibration/_make_notebook.py` is currently
  ICARTT/IWG1-shaped; would need a parallel ADS-B template.

* **`NOAA_GIV` docstring** — add a sentence explaining that the
  47,500 ft service ceiling exceeds the certificated G-IV-SP
  brochure ceiling of 45,000 ft because op-p99 over the
  93-sortie hurricane-surveillance sample sees the airframe at
  light fuel state above MTOW limits.

* **`NASA_WB57` / `NASA_GV` approach speed notes** — both ship
  with calibrated approach speeds below the brochure Vref (117 kt
  vs 130 kt for WB-57; 126 kt vs 130-140 for G-V).  Add docstring
  notes that this reflects light science fuel state at landing.

### Code quality polish

* **Long-function refactors (remaining)** — v1.6.1 split
  `fetch_phenology` (280 lines → 7 helpers) and
  `compute_flight_plan` (383 → 306 lines + 2 builders).
  The other long functions remain unsplit:
  * `Aircraft._hybrid_path` (`aircraft/_base.py`) — 505 lines.
    Hard refactor: single math computation with tight closure on
    phase state (climb / cruise / descent integration).
  * `greedy_optimize` (`flight_optimizer.py`) — 276 lines.
    Medium: graph traversal + result accounting.
  * `compute_refuel_isochrone` (`planning/isochrone.py`) — 244
    lines.  Hard: bisection algorithm with many parameters.
  * `_evaluate_refuel_at_d` (`planning/isochrone.py`) — 243
    lines.  Hard: internal isochrone evaluator.
  * `plot_airspace_map` (`plotting.py`) — 229 lines.  Medium;
    plotting code, low test coverage.
  * `_solve_rays` (`planning/isochrone.py`) — 228 lines.  Hard:
    vectorized ray-bisection inner loop.
  * `effective_swath_on_terrain` (`instruments/lvis.py`) — 222
    lines.  Medium: geometric computation.

* **Expanded ruff rule sets** — `B` (bugbear), `SIM115`
  (file-open without context manager), `UP` (pyupgrade), `RUF`
  (Ruff-specific with selective ignores), and `RET` (flake8-return)
  are enabled.  Remaining queued:
  * `SIM` (flake8-simplify): defers due to manual-fix volume.
    SIM117 nested with-stmts (15 sites — auto-fix refused even
    with --unsafe-fixes, needs case-by-case manual review),
    SIM102 collapsible-if (7), SIM108 if-else-as-expression (10
    — opt out, hurts readability for long expressions), SIM105
    suppressible-exception (1), SIM113 enumerate-for-loop (1).
  * `RUF001/002/003` (ambiguous-unicode-character): permanently
    ignored — HyPlan uses en-dash / em-dash / curly quotes
    intentionally for typography (~230 sites).
  * `RUF046` (unnecessary-cast-to-int): 73 sites, mostly
    defensive casts of pint magnitudes / floor/ceil results.
    Needs case-by-case review; deferred.
  * `RUF059` (unused-unpacked-variable): 49 sites in algorithmic
    code that unpacks tuples and uses a subset; defer.
  Pure cosmetic; no bug fixes; gradually opt in.

* **B905 zip-without-explicit-strict** — ~40 sites.  `zip(a, b)` →
  `zip(a, b, strict=True)` (length-mismatch detection) or
  `strict=False` (silent truncation, current behavior).  Pure
  defensive coding; would need per-site judgment of which.

* **`tests/test_radar.py`** uses `pytest.raises(Exception)` 19
  places (B017).  Would benefit from narrowing to specific
  exception types where possible.

* **Remaining `# type: ignore` (66 sites)** — all genuine
  library-boundary cases (no stubs for earthengine /
  earthaccess / rasterio / geomag; pint Quantity Any returns;
  numpy ndarray returns; pandas indexing).  No further
  reductions possible without upstream stub additions; current
  state is fully documented (every ignore has an inline reason).

---

## Investigated, deferred to later

These were probed during v1.6.0 development and aren't blocking but
worth tracking.

* **University of Wyoming UWKA-2 ICARTT data** at
  `flights.uwyo.edu/projects/<slug>/` (MONARK 2020, APART-LITE
  2019, SNOWIE 2017, etc.) and at NCAR EOL (CHEESEHEAD,
  TRANS2AM, CAESAR, WE-CAN).  Higher quality than ADS-B if
  reachable.  Currently behind ORDER request via
  `datahelp@eol.ucar.edu`.
