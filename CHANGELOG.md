# Changelog

## v1.7.0 — 2026-05-11

Relative-location DSL, whole-pattern movement, and per-aircraft profile
externalization.  Aircraft performance numbers now live in editable JSON
files at `hyplan/data/aircraft/<short_name>.json` instead of Python
literals — every calibration update becomes a single-file diff,
reviewable line-by-line.  No behaviour change for existing user code:
all 22 aircraft round-trip to byte-identical values, the 18 calibrate.py
scripts that drive `_models.py` reproduce identical fits, and the 30/31
canonical notebook smoke set still passes.

### New features

* **`Waypoint.relative_to(anchor, bearing, distance, …)`** — define a
  waypoint as a geodesic offset (Vincenty) from another waypoint or
  `(lat, lon)` tuple.  Inspired by Lait's GSFC flight-planner DSL.
  `distance` as float defaults to nautical miles (matching the
  planning convention); pass a pint Quantity for other units.
  Heading defaults to the bearing for the ergonomic "fly toward the
  new point" case.

* **`Pattern.translate / move_to / rotate / from_relative`** — four
  whole-pattern movement methods on `hyplan.pattern.Pattern`.
  Previously a built-in pattern (lawnmower, rosette, spiral, polygon,
  sawtooth, glint arc) could only be re-anchored by passing
  overrides through `Pattern.regenerate(...)` on its stored params.
  Now any pattern can be shifted by N/E offset, re-centred on a new
  lat/lon, rotated about an arbitrary pivot, or anchored at a
  geodesic offset from another waypoint.  All four return new
  immutable Pattern instances, matching the FlightLine
  in-place-vs-functional split.

* **Per-aircraft JSON profile system** — every aircraft's schedules,
  profiles, scalars, sources, and confidence now live in
  `hyplan/data/aircraft/<short_name>.json` (compact JSON with units
  in field names; 22 files, ~80–180 lines each).  Editing a JSON file
  updates the corresponding `Aircraft` subclass with no Python
  changes.  `hyplan/aircraft/_models.py` shrinks from 2,616 lines of
  inline literals to 240 lines of thin wrappers.  Calibration
  scripts (`notebooks/calibration/<aircraft>/calibrate.py`) now write
  JSON directly via `apply_calibration_to_profile`, eliminating the
  "PASTE-READY PERFORMANCE BLOCK" copy-paste step.

* **New public API for profile I/O** — `hyplan.aircraft` re-exports:
  `load_aircraft_profile(short_name)` (JSON → Aircraft kwargs),
  `dump_aircraft_profile(aircraft, path)` (Aircraft → JSON),
  `write_calibrated_profile(short_name, **overrides)` (partial,
  in-place update of a bundled profile), and `profile_path(short_name)`
  (resolve the bundled file location).

### Bug fixes

* **`NOAA_GIV/calibrate.py` crashed on `nan` roll** — NOAA G-IV ARWO
  files don't carry roll, so `roll_p90` is `nan`; the new
  direct-to-JSON path tripped over `round(nan)`.  The
  `apply_calibration_to_profile` helper now treats `max_bank_deg=nan`
  the same as `max_bank_deg=None` and preserves the existing
  `TurnModel` rather than clobbering it.  BAS_TwinOtter's
  ArcticCyclones segment has the same gap and benefits identically.

### Internal

* **`hyplan/aircraft/_profile_io.py`** (new) — the JSON ↔ Aircraft
  serializer.  Discriminated `{"type": "tas" | "cas_mach"}` for speed
  schedules; nested `approach_profile` / `typical_climb_out` /
  `explicit_climb_plan` blocks for the more complex aircraft (NASA
  ER-2).  Unit suffixes in field names (`service_ceiling_ft`,
  `climb_schedule.points_ft_kt`) — no in-band Quantity strings.

* **`notebooks/calibration/_common.py`** gains
  `apply_calibration_to_profile(short_name, **calibration_outputs)`:
  bridges the plain-tuple form produced by the calibration pipeline
  (`climb_pts=[(alt_ft, tas_kt), …]`, `climb_profile_pts=[(alt_ft,
  fpm), …]`) into the typed objects `Aircraft.__init__` expects.
  Partial overrides leave un-touched fields intact.

* **`tests/test_aircraft_profiles.py`** (new) — 132 parametrized
  tests covering JSON schema validation, dump → load round-trip for
  every aircraft, and class-vs-JSON drift guard.

* **Bug-fix latitude:** the helper's `max_bank_deg` parameter accepts
  `None` to mean "don't touch the existing `TurnModel`"; pass it for
  aircraft whose source data has no roll channel.

### Migration notes

End users using only the public API (`from hyplan.aircraft import
NASA_GV; NASA_GV()`) are unaffected.  Anyone who was monkey-patching
inside `hyplan/aircraft/_models.py` directly should switch to editing
the corresponding JSON file in `hyplan/data/aircraft/` (or call
`write_calibrated_profile(short_name, …)` at runtime).

The previous "PASTE-READY" workflow for refreshed calibrations is
retired.  See `hyplan/data/aircraft/README.md` for the new edit-JSON
and regenerate-from-calibrate.py recipes.

## v1.6.3 — 2026-05-11

Math review + calibration expansion release. In parallel the four ADS-B /
IWG1 / ICARTT-derived aircraft calibrations — KingAir 350, KingAir A90,
NASA ER-2, NASA WB-57 — were refreshed against the new code paths and,
where applicable, against significantly expanded data caches pulled
from public NASA archives.  No public-API breakage.

### Bug fixes

* **`hyplan.aircraft.adsb._reject_outliers` mixed median centre with
  std-based scale** — the function centred on the median but compared
  deviations against `np.std`, which is mean-based and inflated by
  the very outliers it was meant to reject.  Switched to MAD scale
  (`1.4826 · median(|x − median(x)|)`), calibrated to match σ for
  Gaussian data but uncontaminated by outliers.  `outlier_sigma` keeps
  its usual interpretation; no caller change required.

* **`hyplan.planning` wind sampling used a naïve arithmetic-mean
  midpoint** — `(lat1+lat2)/2, (lon1+lon2)/2` is geometrically wrong
  across the antimeridian (`170 + −170 → 0` instead of `±180`), near
  the poles, and accumulates curvature error on long legs at high
  latitude.  The Arctic refuel-isochrone work currently in this
  codebase routinely exercises those conditions.  Added a new public
  helper `geometry.geodesic_midpoint` (Vincenty-based: distance and
  initial bearing via `vdist`, then `vreckon` at half-distance) and
  switched all four wind-sample sites in `planning/engine.py` (3) and
  `planning/segments.py` (1) to use it.  Anchorage → Reykjavik shifts
  by ~1 600 km between the two methods — a meaningful difference in
  the wrong wind regime for high-latitude planning.

* **`planning.segments.process_flight_phase` sliced the Dubins arc by
  phase time, not distance** — the time-fraction split silently
  assumed uniform ground speed across the Dubins-backed phases, so
  for a typical climb (slow GS) + cruise (fast GS) + descent leg the
  reported climb-top coordinate fell several nautical miles short of
  where the climb actually ended.  When every Dubins-backed phase
  carries an explicit `distance` field — the standard climb / cruise
  / descent / transit case — slice by distance fraction instead.
  Phases with their own `geometry` (terminal IFR approach, etc.)
  remain excluded from the slicing pool so their over-length doesn't
  distort the others' fractions.  Mixed sets fall back to the legacy
  time-fraction behaviour.

* **`hyplan.aircraft.icartt.load_icartt` silently dropped data on
  files with non-trivial scale factors** — the ICARTT FFI 1001 spec
  reserves header line 11 for per-column scale factors and line 12
  for per-column missing-value markers (one entry per dependent
  variable).  The parser was ignoring both lines, which produced
  NaN-everywhere DataFrames for high-rate instrument files that store
  values as scaled integers — notably NASA's MMS (Meteorological
  Measurement System) on the WB-57, where TAS is stored as integer
  cm/s with scale 0.01, lat/lon as integer micro-degrees with scale
  1e-5, and per-column missing markers ranging from -999 to
  -99999999 depending on the column's dynamic range.  Without scale-
  factor application a raw TAS value of 8 916 was interpreted as
  8 916 m/s → 17 329 kt → filtered as out-of-range → NaN.  An ACCLIP
  2022 sortie loaded 13 999 rows with **zero** valid TAS / altitude /
  lat / lon fixes before this fix; 13 848 valid after.  New
  `_parse_n_floats` helper for line-11 / line-12 parsing; per-column
  meta dict carrying `(scale, missing)`; extended global fallback
  sentinel set with -999 / -9999999 / -99999999; added MMS-style
  column patterns (`G_LAT_MMS` / `G_LONG_MMS` / `G_ALT_MMS`).

### Numerical improvements

* **`hyplan.aircraft.adsb._compute_metrics`: weighted R² and RMSE** —
  previously every altitude bin contributed equally to the fit quality
  metric regardless of how many raw observations fed into the bin
  median, so a bin with 2 observations and one with 1 000 weighed the
  same.  Pass the per-bin counts (already collected upstream) into
  `_compute_metrics` and weight residuals accordingly.  `n_observations`
  is now derived from `sum(weights)`.  The metric still measures fit to
  the bin medians, not to raw observations — added a docstring paragraph
  spelling that out so future readers don't mistake it for a classical
  regression R².

### New public API

* **`geometry.geodesic_midpoint(lat1, lon1, lat2, lon2)`** — new
  helper for Vincenty-based midpoint computation (used internally
  by the planning module after the arithmetic-mean fix above).
  Returns `(mid_lat, mid_lon)` correct across the antimeridian,
  poles, and long high-latitude legs.

* **`PerformanceConfidence.summary`** — new property returning the
  mean of `climb` / `cruise` / `descent`, excluding `turns` (a
  different epistemic class — bank-angle envelope vs. schedule fit).

* **Paste-ready `__repr__` for `TasSchedule` and `VerticalProfile`** —
  the ADS-B calibration scripts in `notebooks/calibration/<aircraft>/`
  print a PASTE-READY PERFORMANCE BLOCK whose schedule lines come from
  `repr()` of the fitted objects.  With the auto-generated dataclass
  repr, each breakpoint came out as `<Quantity(4000.0, 'foot')>` —
  accurate but unusable as a direct paste into
  `hyplan/aircraft/_models.py`.  Override `__repr__` on both classes
  to emit the canonical `(4000 * ureg.feet, 165 * ureg.knot)` form,
  rounding magnitudes to int.  `VerticalProfile` includes
  `source=...` only when non-empty.  Anyone who consumed the prior
  `repr()` output programmatically will see the new format.

### Calibration data expansion

Two new fetcher scripts pull substantial additional IWG1 / ICARTT
data from public NASA archives, closing temporal gaps that previously
limited the per-aircraft empirical baselines.  The data directories
themselves remain gitignored; the scripts are the canonical
artefacts.

* **`notebooks/calibration/NASA_ER2/_fetch_asp.py`** — wraps the
  shared `_asp_fetch.fetch_tail` helper to pull every available
  fiscal-year IWG1 sortie for NASA 806 (FY2017 / 2018 / 2022) and
  NASA 809 (FY2019 / 2020 / 2021 / 2022) from the public NASA ASP
  archive at `asp-archive.arc.nasa.gov`.  Closes the 2017-2022 gap
  in `data/er2/` — the cache previously held only 2012-2016 +
  2023-2026.  Adds 232 sorties (411 → ~643 raw, 618 successfully
  loaded), giving continuous fiscal-year coverage 2012-2026.
* **`notebooks/calibration/NASA_WB57/_fetch_acclip.py`** — CMR-driven
  fetcher for the 27 daily MMS-1HZ ICARTT files from the ACCLIP 2022
  deployment at NASA LaRC ASDC (collection
  `ACCLIP_MetNav_AircraftInSitu_WB57_Data`).  ACCLIP is exactly the
  campaign Lait's GSFC flight planner tuned its WB-57 ascent
  characteristics against (his ChangeLog 2022-08-02 /
  2022-08-16: "improved wb57 tuning to acclip 2022").  Uses CMR for
  granule discovery (no auth) and `EARTHDATA_TOKEN` from `.env` for
  download Bearer auth.  Handles the 2022-07-21 sortie which is
  split into two ICARTT parts (preserves `-part1` / `-part2`
  suffixes so files don't collide on disk).

### Aircraft model refresh

Re-ran four ADS-B / IWG1 / ICARTT-derived calibrations against
the post-MAD-outlier / geodesic-midpoint / weighted-R² code paths and
— for ER-2 and WB-57 — against the expanded data caches from the new
fetchers.  The calibration pipeline already separates fetch from
compute (`calibrate.py` reads from `data/<aircraft>/` and never
reaches out to the network), so refreshes are a single `python -m
notebooks.calibration.<aircraft>.calibrate` away.

* **KingAir 350** (UWKA-2, n = 22 sorties): climb-schedule RDP knee
  moved 18 000 → 22 000 ft; top-of-climb TAS 299 → 303 kt;
  climb-profile 8 000 ft VS −32 fpm.  Cruise peak (33 000 ft / 318 kt)
  unchanged.  The MAD refit also surfaced a 188 kt level-off dip at
  12 000 ft that reflects a brief step-climb pause rather than the
  underlying schedule — dropped so the schedule stays monotone.
* **King Air A90** (n = 428 sorties, 25 tails): descent-schedule
  16 000 ft TAS 174 → 170 kt; descent-profile 22 000 ft VS 896 → 960
  fpm.  All other points within sub-kt / sub-fpm rounding of the
  prior fit.  POH cross-checks still hold (max cruise 222 kt @ FL160
  vs. POH 226; service ceiling op-p99 25 000 ft vs. POH 26 400 ft).
* **NASA ER-2** (n = 618 sorties, 2012-2026 continuous IWG1 cache —
  up from 199):  small ±10 kt shifts at every TAS-schedule
  breakpoint; 70 kft extrapolation 410 → 400 kt.  climb_profile
  FL050 4 301 → 3 553 fpm (the expanded sample including 2017-2022
  routine ops pulls the low-altitude active-climb median down;
  DCOTSS test flights Lait tuned against were envelope-chasing
  climbs).  descent_profile: 14-point per-altitude-bin median → 3-
  anchor TOC / mid-descent / ceiling construction with bottom anchor
  keyed to top_of_approach_msl.  approach_profile touchdown 65 → 72
  kt (n=89 sorties, was n=6).  turn_model.bank_by_phase: climb 11° →
  14°, descent 16° → 13°, approach 9° → 11°.  typical_climb_out
  rederived from IWG1 alone — the historical FL356 12-min weight-
  management hold appears in only 0.8 % of post-2016 sorties;
  replaced with a single 13-min FL550 representative pause where the
  climb-out overhead actually concentrates in the modern sample.
* **NASA WB-57F** (n = 127 sorties combined IWG1 + ACCLIP 2022
  ICARTT — up from 100):  service_ceiling 63 000 → 64 000 ft;
  climb_schedule gains an FL600 anchor (402 kt) that was sparse in
  the IWG1-only fit; cruise_schedule shifts +5-17 kt across FL450-
  600 (the new ACCLIP data is dominated by high-altitude cruise
  legs).  climb_profile FL050 2 137 → 2 274 fpm; FL500 anchor added
  at 1 616 fpm.  approach speed 117 → 120 kt (n=111 vs. n=84).
  max_bank 33 → 32°.

### Documentation tightenings (no behaviour change)

* **`atmosphere`** — clarified that the ISA pressure formula is
  derived under geopotential altitude `H`, not geometric `z`.  For
  aviation altitudes (pressure altitude / flight levels) the two are
  operationally equivalent so no numerical change is needed, but a
  caller passing WGS-84 / GPS altitude at FL510+ would see a ~0.5 %
  pressure bias.  Module docstring now includes the geometric →
  geopotential conversion (`H = R⊕·z / (R⊕+z)`) for callers who need
  it.
* **`sun.solar_threshold_times`** — flagged that rise / set times are
  quantized to the 1-minute sampling grid (mean bias ~30 s) so
  callers needing sub-minute accuracy bracket-interpolate via `sunpos`.
* **`glint.GlintArc`** — the turn-radius computation `R = v² / (g · tan φ)`
  is a still-air coordinated-turn result, so the input `speed` should
  be true airspeed; in wind the actual ground-track radius differs
  and the planned arc is a centreline that the autopilot will
  crab/wind-correct against.
* **`geometry.translate_polygon`** — only meaningful in a projected
  CRS (e.g. UTM) where `+y` is grid-north and `+x` is grid-east.
  Passing a WGS-84 polygon translates it by `distance` degrees of
  lat/lon, almost never what you want; spelled this out.
* **`_trochoid_solver`** — clarified the `t2 ∈ (-t2pi, t2pi]` gate in
  the BSB solver: a negative `t2` is a parametric phasing offset, not
  a multi-loop ground track.
* **`planning/isochrone._solve_rays`** — the d = 0 feasibility probe
  is intentionally liberal (ignores climb/descent overhead) and is
  paired with a post-convergence probe at `distance_tolerance_nmi`
  that catches rays whose phase overhead alone exceeds the budget.
  Added comments at both sites so a future refactor doesn't remove
  the second guard without also tightening the first.

### Internal refactors (no user-visible behaviour change)

* **`_trochoid_solver.sample_trochoid`** — replaced a placeholder
  identity `w = Va / (Va / (Va / 1.0))` (which collapsed to `Va`,
  the wrong units, but was immediately overwritten by the actual
  recovery `w = _M2PI / t2pi`) with a direct `w = sol["w"]` lookup,
  and dropped a `+ del2 * _M2PI` term inside the `sin` / `cos`
  arguments of `xt20` / `yt20` (a 2π-periodic identity that
  produced no numerical effect).  Parity to 3 × 10⁻¹² m position
  and 7 × 10⁻¹⁵ rad heading across 5 000 random sample times — a
  pure cleanup at the math level — with roughly an 8 % per-call
  speedup from dropping a redundant `atan2` and division.

* **`sun.sunpos`** — replaced the manual size-1-broadcast block
  with a single `np.broadcast_arrays` call.  Same observable
  behaviour on every input shape that worked before; the
  refactor's value is in code clarity and matches numpy idiom.

### Tooling and test coverage

* **Test coverage push** — three targeted gaps surfaced by
  `pytest --cov` got dedicated tests this release.
  `hyplan/aircraft/icartt.py` 0 % → 92 % (new `tests/test_icartt.py`,
  including regression tests for the scale-factor / per-column-missing-
  value / MMS-column-name handling).  `hyplan/aircraft/iwg1.py` 63 %
  → 95 % (new `TestSplitIwg1Alltracks` covering the multi-sortie
  splitter).  `hyplan/aircraft/adsb/io.py` 12 % → 17 % (new
  `TestRequireTraffic` confirming the shim raises HyPlanRuntimeError
  when the optional `traffic` library is missing).  Total test count
  1 710 → 1 757; overall package coverage 82 % → 83 %.
* **Notebooks ruff cleanup** — ran `ruff check --fix` across
  `notebooks/` (226 violations resolved automatically), then manually
  fixed 14 residual issues in `notebooks/calibration/**/*.py` (B007
  unused loop variables renamed to `_tail` / `_date`; E701/E702
  one-liners split; E741 ambiguous `l` renamed to `link`).  Added two
  targeted per-file-ignores in `pyproject.toml`:
  `notebooks/calibration/**/*.py = ["E402"]` (intentional
  sys.path-insert-then-import pattern) and
  `notebooks/**/*.ipynb = ["E402", "E701", "E702", "B007", "F811",
  "F841"]` (notebook style legitimately keeps cells self-contained).
  `ruff check hyplan tests notebooks` is now fully clean.

### Verification

* Full test suite: 1 757 passed, 1 skipped (pytest), up from 1 710
  at v1.6.2.
* `mypy` strict: 0 issues across 89 source files.
* `ruff check hyplan tests notebooks`: all checks passed.
* End-to-end notebook execution (no failures): 28 of 29 non-GEE
  notebooks pass.  The one failure (`winds.ipynb`) is environmental
  (NASA Earthdata login required), not a regression.
* All four refreshed aircraft calibrations re-run end-to-end against
  their local data caches.  KingAir 350, KingAir A90, NASA ER-2 used
  cached data only; NASA WB-57 calibration pulled 27 new ACCLIP
  MMS-1HZ files (~60 MB) via `_fetch_acclip.py`.  Pre-vs.-post diffs
  match expected magnitudes from the math-review code changes and
  the expanded data samples.

## v1.6.2 — 2026-05-10

CI-recovery release.  No public-API or behavioral changes; main
had been silently red on lint and mypy since v1.6.1 was tagged.
This release closes both, plus adds defenses so neither slips
again.

### Lint cleanup

The CI lint step (`ruff check hyplan tests`) was tripping 68
violations — 39 `E402` (module-level import not at top) and 29
`F401` (unused import) — that had accumulated since the test
reorganization.  Most were intentional patterns ruff couldn't
distinguish, addressed via per-file-ignores in `pyproject.toml`:

* **`hyplan/__init__.py`** — public re-exports via
  `from .x import y` (F401), with optional-dep blocks gated
  behind `try/except` (E402).
* **`hyplan/*/__init__.py` and `hyplan/*/*/__init__.py`** —
  sub-package re-exports (F401).
* **`hyplan/flight_plan.py`** — backward-compat shim that
  re-exports private helpers under the old module path (F401).
* **`tests/*.py`** — section-grouped imports under comment
  headers (E402) and optional-dep `try/except` blocks for
  test-skip plumbing (F401).

Two genuine fixes alongside:

* `hyplan/instruments/_base.py` — drop unused `ureg` import.
* `hyplan/winds/providers/gfs.py` — explicit `# noqa: F401`
  on `import cfgrib` (kept for xarray-engine registration
  side effect).

### Type-system regression on the Python 3.10 matrix entry

The v1.6.1 type-system cleanup was validated locally on
numpy 2.4 and missed numpy 2.2's stricter generic stubs.
On the CI 3.10 matrix entry (numpy 2.2 is the last version
supporting Python 3.10; numpy 2.3+ requires 3.11+), bare
`np.ndarray` annotations triggered 74 `[type-arg]` errors
plus 9 `[no-untyped-call]` errors on `np.ma.MaskedArray`
constructor calls.

* **84 errors → 0** across 16 files (`hyplan/aircraft/_base.py`,
  `hyplan/aircraft/adsb/{fitting,phases}.py`, `hyplan/geometry.py`,
  `hyplan/glint.py`, `hyplan/instruments/{awp,frame_camera}.py`,
  `hyplan/phenology/{_qa,sources}.py`,
  `hyplan/planning/isochrone.py`, `hyplan/plotting.py`,
  `hyplan/satellites.py`, `hyplan/swath.py`,
  `hyplan/terrain/intersection.py`, `hyplan/winds/gridded.py`,
  `hyplan/winds/providers/gmao.py`).
* Replaced bare `np.ndarray` with `npt.NDArray[T]` annotations
  with per-site dtype judgment from surrounding code:
  `np.float64` for numeric arrays (most common), `np.integer[Any]`
  for index / mask arrays, `np.datetime64` for time arrays,
  `Any` for genuinely polymorphic helpers.
* `np.ma.MaskedArray` constructor calls switched to
  `np.ma.masked_array(...)` factory or `# type: ignore[no-untyped-call]`
  where the explicit class was load-bearing.
* `hyplan/clouds/sources.py` — earthengine `Any`-return ignores
  switched to `# type: ignore[no-any-return, unused-ignore]`
  to absorb a Python-version-dependent inference difference.

### Defense in depth

Two-layer guard against the same incident pattern (red main +
release tag slipping through):

* **`.pre-commit-config.yaml`** — local `ruff-check` hook (pinned
  to ruff 0.15.12 to match CI exactly), plus standard
  whitespace / yaml / toml hygiene hooks.  Activate per clone
  with `pip install pre-commit && pre-commit install`.
* **`release.yml` and `post-release.yml`** — both gain a lint
  step / `verify-lint` job before the tag-bumping side effects
  run.  A tag pushed against a red main now fails fast at the
  workflow level instead of silently bumping bookkeeping.

### Verification

* `mypy hyplan` — Success: no issues found in 89 source files
  (Python 3.10 + numpy 2.2; Python 3.11 + numpy 2.4).
* `ruff check hyplan tests` — All checks passed.
* `pytest tests` — 1697 passed, 1 skipped, 1 warning.
* CI matrix (Python 3.10 / 3.11 / 3.12) green.

## v1.6.1 — 2026-05-09

Maintenance / quality release.  No public-API or behavioral
changes; everything below is type-system, lint, and structural
hygiene.

### Type-system hardening (mypy strict cleanup)

`hyplan.aircraft.wind_path` and `hyplan.planning.isochrone` are now
genuinely strict-mode-clean, including the entire transitive import
call graph (~50 modules).  The strict overrides on these two modules
were retained from v1.6.0 but had cascaded ~462 errors across the
codebase that were silently accepted; this release closes them.

* **462 strict-mode errors → 0**, across 56 source files:
  * 169 `[no-untyped-def]` — added function / method signatures.
  * 148 `[type-arg]` — concrete generic args
    (`dict[str, Any]`, `list[float]`, `npt.NDArray[np.float64]`, ...).
  * 120 `[no-untyped-call]` — resolved transitively as the called
    fns gained signatures.
  * 18 `[attr-defined]` + 7 misc — case-by-case (mostly
    `assert is not None` narrowing; lazy-import helpers annotated
    `-> Any`; one `_GriddedWindField` import path correction).
* The 4-agent parallel cleanup grouped files into roughly equal
  workloads (~110-130 errors each); cross-group cascade resolved
  naturally during merge.

### `# type: ignore` documentation + reduction (149 → 66, -56%)

* **All 149 pre-existing bare `# type: ignore[code]` comments now
  carry an inline `# reason`** explaining why the suppression
  exists — `# type: ignore[code]  # short reason`.  Three parallel
  agents annotated 149 sites across 34 files; every site got a
  specific reason (no fallback to defaults).
* **Two structural fixes eliminated 43 ignores at the source**:
  * `is_waypoint()` typed as `TypeGuard[Waypoint]` in
    `hyplan.waypoint`.  mypy now narrows through
    `if is_waypoint(x):` blocks, removing 17 `[union-attr]` /
    `[arg-type]` suppressions in `planning/engine.py`.
  * `_GriddedWindField` slab attributes (`_u_data`, `_v_data`,
    `_times`, `_levs`, `_lats`, `_lons`, `_times_raw`) given
    explicit class-level `npt.NDArray[...]` annotations,
    replacing the `self._x = None` initialization pattern that
    forced 26 suppressions across `winds/gridded.py` and
    `winds/providers/gfs.py`.  Stale "datetime64" doc on `_times`
    corrected — actual stored type is float Unix-epoch seconds
    (added a separate `_times_raw` for the original datetime64).
* **`[no-any-return]` reduced 45 → 16**: replaced 29 documented
  suppressions with explicit Python coercion at the boundary:
  * `q.m_as(unit)` → `float(q.m_as(unit))` for scalar Quantity.
  * numpy reductions / arithmetic → `float(...)` (atmosphere,
    sun, geometry, lvis, winds/utils, frame_camera).
  * numpy bool comparisons → `bool(...)` (lvis).
  * pandas indexing → `str(...)` / `list(...)` (airports).
  * `ndarray.shape` → `(int(s[0]), int(s[1]))` tuple.
  * The 16 remaining ignores are all genuine library-boundary
    cases (earthengine / earthaccess / rasterio no stubs;
    `requests.json()` Any returns; `np.median` ndarray returns;
    geomag).

### Lint: enable ruff bugbear

* `[tool.ruff]` extends to include `B` (flake8-bugbear) and
  `SIM115` (open-without-context-manager).  Both ship enabled in
  `pyproject.toml`; CI catches new instances going forward.
* **38 lint findings cleared:**
  * **B904** (raise-without-from-inside-except) — 20 sites across
    11 files.  Added `from err` to preserve traceback chains, or
    `from None` for intentional reframes (ImportError →
    HyPlanRuntimeError "library missing" messages).
  * **B007** (unused-loop-control-variable) — 12 sites renamed
    `for x in ...` → `for _x in ...` to flag unused loop
    bindings.
  * **SIM115** (file-open without context-manager) — 6 sites in
    `tests/test_exports.py` migrated `open(path).read()` →
    standard `Path(path).read_text()`.
* Pyupgrade + RUF auto-fixable items (~30 mechanical fixes for
  unused-iterable-allocations, unsorted dunder-slots, etc.)
  applied via `ruff --fix --select UP --select RUF --ignore RUF100`.
  RUF100 (unused-noqa-directive) explicitly excluded because its
  auto-fix incorrectly removes legitimate `# noqa: F401`
  directives that suppress default-on F401 errors on shim
  re-exports.
* **Stylistic rule sets** (full RUF / RET / SIM / UP) plus
  **B905** (zip-strict) queued in `TODO.md` for a separate
  gradual-opt-in pass; ~170 minor findings, none bug-grade.

### Long-function refactors

Two of the largest functions split into thin dispatchers + named
helpers.  Pure extraction; identical behavior.

* **`fetch_phenology`** (`phenology/sources.py`): 280 lines →
  dispatcher + 6 helpers, longest helper 58 lines.
  * `_fetch_phenology_appeears` — AppEEARS server-side extraction
    (point samples; fast).
  * `_fetch_phenology_granules` — HDF4 download + local raster
    processing (full spatial coverage; slow).  Granule path
    further factored into `_resolve_granule_short_names`,
    `_collect_granule_rows`, `_empty_phenology_frame`,
    `_merge_combined_satellite`.
  * Annotated `_PRODUCT_CONFIG: dict[str, dict[str, Any]]` (was
    untyped) — clears 9 surrounding `# type: ignore[index]` /
    `[arg-type]` suppressions on heterogeneous-config dict access.
* **`compute_flight_plan`** (`planning/engine.py`): 383 lines →
  306 lines + 2 record builders.
  * `_build_flight_line_record` (58 lines) — solves the crab-aware
    track-hold problem at the line midpoint, returns the
    GeoDataFrame record for a FlightLine with crab / groundspeed /
    wind metadata.
  * `_build_loiter_record` (41 lines) — renders a hold-orbit
    ground track for a Waypoint with non-zero delay; falls back
    to a Point geometry when altitude is unavailable.
* **Unused-noqa cleanup**: `ruff check --select RUF100 --fix`
  cleared 69 `# noqa` directives that became redundant after the
  recent `__all__` declarations were added and the TypeGuard fix
  made `[union-attr]` ignores unused.

### Verification

* `mypy hyplan` — Success: no issues found in 89 source files.
* `ruff check hyplan tests` — All checks passed (default + B + SIM115).
* `pytest tests` — 1697 passed, 1 skipped, 1 warning.

## v1.6.0 — 2026-05-09

Internationalization of the aircraft fleet plus aircraft-calibration
completion.  HyPlan grows from 14 → **22 pre-configured research
aircraft**, with calibration coverage rising from 8 → **19
calibrated platforms**: 17 via in-situ ICARTT / IWG1 / NetCDF
archives (NASA / NOAA fleet + six new international classes), plus
2 via ADS-B globe-history archives (King Air A90 fleet aggregate,
University of Wyoming UWKA-2 King Air 350).  Two **breaking
renames** in the legacy fleet — see "Naming convention" below.

### Naming convention (BREAKING — no backward-compat aliases)

`C130` → `NASA_C130` and `TwinOtter` → `NOAA_TwinOtter`.  The fleet
now consistently uses operator-prefixed names whenever the
calibration is single-operator-specific.  Both classes were already
calibrated against one operator's tails (NASA Wallops C-130H
N436NA / NOAA Twin Otter N48RF + N46RF); the rename makes that
explicit and signals that other operators (USAF / NCAR / NRL C-130,
CIRPAS / Kenn Borek Twin Otter) would need their own calibration
class.  `KingAirB200` is left as-is despite a similarly NASA-only
calibration source — the airframe sees genuine multi-operator use
across HyPlan's user base.

Anything importing `from hyplan.aircraft import TwinOtter` or `C130`
will fail loudly until updated.  The migration is a mechanical name
change.

### Six new international aircraft classes

| Class | Airframe | Operator | Sorties | Source |
|---|---|---|---|---|
| `FAAM_BAe146` | BAe-146-301 (G-LUXE) | FAAM (UK) | 125 | CEDA FAAM Core Data Product 1 Hz, 27 ASMM-tagged campaigns 2017-2024 |
| `SAFIRE_ATR42` | ATR-42-320 (F-HMTO) | SAFIRE (FR) | 44 | CEDA EUFAR (28, wind-triangle TAS) + AERIS EUREC4A 2020 (19, native TAS) |
| `BAS_TwinOtter` | DHC-6-300 polar | BAS (UK) | 105 | CEDA MASIN: OFCAP 2010-2011 + ACCACIA 2013 + ORCHESTRA 2017-2018 + IGP 2018 + ArcticCyclones 2022 |
| `NERC_DO228` | Dornier Do228-101 (D-CALM) | NERC ARSF (UK) | 34 | CEDA NERC ARSF: ACTIVE 2005-2006 + Eyjafjallajökull 2010 (wind-triangle TAS) |
| `AWI_BaslerBT67` | Basler BT-67 (Polar 5/6) | AWI (DE) | ~78 | PANGAEA: ACLOUD 2017 + HALO-AC3 2022 |
| `DLR_HALO` | Gulfstream G550 (D-ADLR) | DLR (DE) | 18 | DLR HALO BAHAMAS 1 Hz, HALO-AC3 2022 |

Each ships with a `calibrate.py` script and a companion
`calibration.ipynb`.  `SourceRecord` entries carry CEDA / PANGAEA /
AERIS DOIs where available so users can cite the underlying
archives.

### Refreshed NASA + NOAA fleet calibrations

* **`NCAR_GV`** (HIAPER): inferred → calibrated.  22 NSF-GV ICARTT
  NAV sorties from DC3 2012 (NASA LaRC ASD archive); TAS
  reconstructed via wind triangle since DC3 RAF-NAV product omits
  TASX.  CasMachSchedule (M0.80 cruise) carried over from
  `NASA_GV`; DC3 cruise TAS at FL300 (468 kt observed) confirmed
  the schedule applies.
* **`NOAA_TwinOtter`** (was `TwinOtter`): refresh from 17 → **164
  sorties**.  Adds 6 NOAA CSL chemistry campaigns on N46RF
  (TopDown 2014, UWFPS 2017, CalFiDE 2022, AEROMMA 2023, AMMBEC
  2024, USOS 2024) to the FIREX-AQ N48RF baseline.  Service
  ceiling 15 → 17.5 kft; approach 99 → 105 kt.  Per-file unit
  detection corrects PI mislabeling between m/s and kt across
  campaigns.
* **`NOAA_WP3D`** (new class): Lockheed WP-3D Orion, NOAA AOC tails
  N42RF / N43RF.  Calibrated against **96 sorties** combining 18
  NOAA CSL chemistry sorties (ARCPAC 2008, CalNex 2010, SENEX
  2013, SONGNEX 2015) with 78 NOAA AOML HRD hurricane-program
  sorties.  Climb VS profile within 5 fpm of `NASA_P3` at every
  altitude — confirms airframe equivalence.  Cruise schedule runs
  22-28 kt slower than `NASA_P3` (chemistry mission profile favors
  slow-cruise dwell over plumes).
* **`NOAA_GIV`** (new class): Gulfstream IV-SP, NOAA AOC tail
  N49RF "Gonzo".  Calibrated against 93 hurricane
  synoptic-surveillance sorties (2021-2025).  Service ceiling
  47,500 ft (op-p99; certified 45,000), Vapp 146 kt, cruise 443
  KTAS @ FL400.  First calibrated G-IV variant — distinct from
  `NASA_GIV` which remains brochure-only.

### ADS-B globe-history calibrations (new path)

For airframes without IWG1 / ICARTT public archives, v1.6.0 adds an
ADS-B-based calibration path that pulls historical 24-h globe
traces from `airplanes.live`'s `globe_history/<YYYY-MM-DD>/`
endpoint, segments them into sorties, and runs them through the
existing `hyplan.aircraft.adsb` pipeline (groundspeed-as-TAS
still-air baseline; v1.6.1 will switch to MERRA-2 wind-triangle
reconstruction).

* **`KingAirA90`**: brochure → **calibrated**.  124 active US
  65-A90 / 65-A90-1 (civilian) tails enumerated from the FAA
  Releasable Aircraft database; a 30-day archive pull (2026-04-09
  → 2026-05-08) yielded 643 sorties / 298k fix rows across 25
  active tails.  A skydive-sortie filter (short duration + high
  peak altitude + sustained > 2500 fpm descent) drops 215 of 643
  sorties — the active US A90 fleet is heavily skydive-dominated.
  428 retained sorties produce a calibration whose cruise schedule
  and service ceiling track brochure values closely (cruise 222 kt
  @ FL160 vs brochure 226 @ FL150; ceiling 25 kft op-p99 vs
  26.4 kft POH).
* **`KingAir350`**: brochure → **calibrated**.  University of
  Wyoming UWKA-2 (N2UW / hex A18F28); 22 sorties pulled from 18
  active days across 5 science-campaign windows (2025-01 through
  2026-04, 8.7k trace rows).  Hybrid block: data-fit cruise peak
  (318 kt @ FL330, slightly above stock B300 reflecting UWKA-2's
  Blackhawk XP-67A engine upgrade), climb_schedule, and
  climb_profile; brochure-retained approach_speed (110 kt) and
  descent_profile (single-tail upper-altitude bins are too sparse
  for the descent fitter).  Confidence 0.5-0.65.

Both ship with `calibrate.py` only (no `calibration.ipynb`
companion); see
`notebooks/calibration/KingAirA90/_fetch_airplanes_live.py` for the
trace fetcher.

### Calibration toolchain

* **Per-source archive fetchers** under `notebooks/calibration/`:
  * `_larc_asd_fetch.py` — generic NASA LaRC Airborne Science Data
    fetcher (decodes the ArcView `enzFile` URL token scheme).
  * `_noaa_csl_fetch.py` — cookie-authenticated NOAA CSL
    field-project search.
  * `_hrd_fetch.py` + `_hrd_loader.py` — NOAA AOML HRD hurricane
    archive (G-IV `.01.txt` and P-3 `.1sec.txt` ARWO formats).
  * Per-aircraft fetchers: CEDA FAAM, CEDA EUFAR, CEDA BAS MASIN,
    CEDA NERC ARSF, airplanes.live globe-history (`KingAirA90` and
    `KingAir350`).
* **Notebook builder** `_make_notebook.py` emits a standardized
  13-cell `calibration.ipynb` per aircraft that delegates to the
  per-aircraft `calibrate.py` source of truth.  All 17 in-situ
  calibrated aircraft now ship with a notebook (was 8 of 11 in
  v1.5.x).
* **Calibration directory rename**: every
  `notebooks/calibration/<short>/` directory is renamed to match
  its aircraft class name (`giii/` → `NASA_GIII/` etc.) for
  consistency with `data/<class>/` layout.

### ICARTT loader hardening (`hyplan/aircraft/icartt.py`)

Driven by older / non-spec-compliant campaign files:

* **Latin-1 encoding fallback** for ARCPAC 2008 et al.
* **Whitespace-delimited line 1 + data rows** auto-detection (some
  ARCPAC files use spaces instead of commas).
* **6-digit sentinels** (`-999999`, `-777777`, `-888888`) for LaRC
  PI-merge convention.
* **0-360°E longitude wrap** for DISCOVER-AQ merges.
* **km altitude** unit branch (the previous `"m" in u` test was
  matching `"km"` substring).
* **Underscore-aware regex** for PI-merge instrument prefixes
  (`FMS_TAS`, `IRS_HEAD`, etc.).
* New canonical-name patterns: `GRD_SPD`, `WNS`, `WND`, and `^...^`
  anchor stripping on heading / track / pitch / roll / AOA.

### `SourceRecord` schema extension

`hyplan.aircraft._base.SourceRecord` gains two optional fields:

* `url: str = ""` — landing-page URL for the data archive.
* `doi: str = ""` — DOI of the cited dataset.

Existing `SourceRecord(...)` calls work unchanged.  Eleven
calibrated classes ship with `url` populated; six ship with
verified DOIs (CEDA UUIDs, AERIS EUREC4A
`10.25326/162`, PANGAEA HALO-AC3 `10.1594/PANGAEA.967719`, AWI
Polar `10.1594/PANGAEA.902849`).

### Auto-generated fleet documentation

New `docs/_gen_fleet_tables.py` introspects the live aircraft
classes and rewrites two marker-delimited regions:

* `docs/api/aircraft.md` — fleet overview table.
* `docs/calibration.md` — data sources grouped by archive
  (NASA ASP / NOAA CSL / CEDA FAAM / CEDA EUFAR / AERIS EUREC4A /
  CEDA BAS MASIN / DLR HALO BAHAMAS / CEDA NERC ARSF / AWI PANGAEA)
  with clickable DOIs / URLs.

Re-run `python -m docs._gen_fleet_tables` after any calibration
change.

### Calibration coverage

| Status | Count | Aircraft |
|---|---|---|
| `calibrated` | **19** | NASA: ER-2, GIII, GV, P-3, WB-57, C-130, B-200; NOAA: TwinOtter, WP-3D *(new)*, GIV *(new)*; international *(all new)*: NCAR_GV, FAAM_BAe146, SAFIRE_ATR42, BAS_TwinOtter, NERC_DO228, AWI_BaslerBT67, DLR_HALO; ADS-B *(new)*: KingAirA90, KingAir350 |
| `inferred` | 1 | NASA_C20A (mirrors NASA_GIII) |
| `uncalibrated` | 2 | NASA_GIV (NASA AFRC tail; NOAA G-IV is the calibrated variant), NASA_B777 |

### Other housekeeping

* `docs/api/aircraft.md` autoclass directives updated for renames +
  new `NOAA_WP3D`, `NOAA_GIV`, and the six international classes.
* `docs/stability.md` experimental tier: `NCAR_GV` removed (now
  calibrated).
* `docs/calibration.md` adds a "Performance envelope cross-check"
  section comparing all 19 calibrated classes against brochure /
  POH values, flagging known artifacts (NERC_DO228 cruise inversion
  above FL100, DLR_HALO flat cruise FL200-FL300) for v1.6.1
  follow-up.
* HRD tail-letter mapping in HRD filenames (`H` = N42RF Kermit, `I`
  = N43RF Miss Piggy, `N` = N49RF Gonzo, `U` = USAF WC-130J — the
  USAF data on disk is unused in v1.6.0).

## v1.5.2 — 2026-05-07

Maintenance + performance release.  No public-API changes.

### Faster `compute_isochrone` (~5× on refuel sweeps)

Two memoization layers in the bisection hot path:

* `Aircraft._climb` / `_descend` now cache `(time, distance)`
  results on a per-instance dict keyed by `(start_alt_ft,
  end_alt_ft, true_air_speed_kt, wind_along_track_kt)`.  Iso­chrone
  bisection drivers and refuel-itinerary evaluators were calling
  these thousands of times with identical altitude pairs (e.g.
  cruise climb from runway to FL250 is the same regardless of
  bisection state).  Each call previously ran a 64-step
  trapezoidal integration through expensive pint Quantity
  arithmetic.
* `_track_hold_solution_from_uv` (`hyplan/winds/utils.py`) now
  caches its return dict on a 4096-entry FIFO keyed by
  rounded `(tas_mps, track_deg, u_mps, v_mps)`.  The hot loop
  also lifts `np.sin/cos/arcsin` calls into floats up front,
  sidestepping numpy's `__array_ufunc__` dispatch on each call.

Combined effect on the v1.5.1-baseline refuel-extended fixture
(B-200, 2 candidates, 36 rays, 30 kt const wind, 4-hr sortie /
8-hr day): **12.83 s → 2.60 s.**  Direct still-air and
constant-wind cases were already sub-second; unchanged.
Boundaries are numerically identical to v1.5.1.

A new `tests/test_isochrone_perf.py` under `@pytest.mark.perf`
gates the savings against future regressions.  Local profiling
fixtures in `bench/run_isochrone.py`.

### Drop Python 3.9 support; minimum is now 3.10

mypy 2.x dropped 3.9 entirely (CI was printing `Python 3.9 is
not supported (must be 3.10 or higher)` on every run).  Bumping
the minimum to 3.10 unlocks PEP 604 unions and builtin generics,
and lets us drop the `timezonefinder.*` mypy `follow_imports`
override that existed because 3.9 mypy couldn't parse
match-statement syntax.  `numpy` CI pin loosened from
`>=1.26,<2.3` to `>=2.0,<2.3` (the lower bound was held back
solely to keep 3.9 wheels resolvable).

No code-level API changes; existing 3.9 users will see a pip
resolution error on upgrade rather than a runtime break.

### `var-annotated` mypy suppression removed

The codebase has been clean against this error code since the
v1.5 numpy-stub pins settled.  Drop the suppression and the
explanatory comment block from `pyproject.toml`; CI exercises
mypy with the suppression removed so future PRs can't slip
unannotated `np.empty/zeros` assignments.

### Other housekeeping

* `paper/paper.pdf` rebuilt against the current source.
* `notebooks/isochrone.ipynb` re-executed on v1.5.2

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
* `notebooks/calibration/NASA_ER2/calibration.ipynb` — narrative no
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

New supporting tooling: the `hyplan.aircraft.iwg1` loader (with `trim_ground_taxi`), `IWG1TraceWindField`, a planned-sortie parser (Green Card XLSX/PDF via `pdfplumber` + KML), and three calibration notebooks under `notebooks/calibration/NASA_ER2/` (`iwg1_calibration`, `sortie_replay`, `planned_vs_flown`).

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
