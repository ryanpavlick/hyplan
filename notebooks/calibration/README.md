# Aircraft calibration

This directory contains the per-aircraft calibration scripts and
notebooks that produce the performance values in
[`hyplan/aircraft/_models.py`](../../hyplan/aircraft/_models.py).

For an end-user view (which aircraft is calibrated, sortie counts,
data-archive citations) see [`docs/calibration.md`](../../docs/calibration.md).
This README is the **developer guide** — how to add a new aircraft
or refresh an existing one.

## Layout

```
notebooks/calibration/
├── _common.py             # Shared recipe helpers (filter / phase-label /
│                          # bin / TAS schedule / approach / manifest CSV)
├── _make_notebook.py      # Template-based notebook builder
├── _hrd_loader.py         # NOAA AOML HRD ARWO + 1-sec-text loaders
├── _hrd_fetch.py          # NOAA AOML HRD downloader
├── _asp_fetch.py          # NASA Airborne Science archive helpers
├── _larc_asd_fetch.py     # NASA LaRC ASD ArcView helpers
├── _noaa_csl_fetch.py     # NOAA CSL field-project (cookie-auth) helper
└── <aircraft>/
    ├── calibrate.py       # Standalone calibration script
    ├── calibration.ipynb  # Companion interactive notebook
    └── _fetch_*.py        # Per-archive fetcher (when applicable)
```

Run any script standalone from the repo root::

    python -m notebooks.calibration.<aircraft>.calibrate

## The shared recipe

Every calibration follows the same six-step pipeline, implemented as
helpers in [`_common.py`](_common.py):

| Step | Helper | Purpose |
|------|--------|---------|
| 1. Load | per-aircraft loader | Read native nav data; reconstruct TAS via wind triangle if not shipped |
| 2. Filter | `apply_sortie_filters` | Drop too-short, too-long, low-altitude sorties |
| 3. Label | `label_phases` | Tag each fix as climb / cruise / descent (vertical-rate gates) |
| 4. Bin | `per_bin` / `tas_per_bin` | 5-kft median + IQR; n>=30/bin (VS) or n>=200 (TAS) |
| 5. Pick | `schedule_pts` | Pick TAS schedule breakpoints at aircraft-specific target altitudes |
| 6. Emit | `summary_table(manifest_path=...)` | Print summary + write per-sortie provenance CSV |

Aircraft-specific knobs live in the per-aircraft `calibrate.py`:

* `ACTIVE_VS_THR_FPM` — climb / descent gate (1500 fpm jets / turboprops,
  1000 fpm B-200, 500 fpm Twin Otter, 700 fpm Do-228)
* `MIN_DUR_MIN` / `MAX_DUR_MIN` — sortie duration filter
* `MIN_PEAK_ALT_FT` / `MAX_PEAK_ALT_FT` — peak-altitude filter
* `TARGET_ALTS_FT` — TAS schedule breakpoint altitudes
* `<glob>_GLOB` — per-archive file pattern

Everything else — the IQR statistics, the centered-difference vertical-
rate filter, the wind-triangle TAS reconstruction (`wind_triangle_tas_kt`),
the per-bin median, and the manifest CSV — lives in `_common.py` and is
shared.

## Adding a new aircraft

1. **Pick or write a fetcher.**  Look at the existing `_*_fetch.py`
   helpers; CEDA, AERIS, and NOAA AOML patterns are well-trodden.
   The fetcher should be idempotent and write a per-sortie manifest
   under `data/<aircraft>/calibration_manifest.csv` so reviewers can
   trace which sorties contributed to each calibrated value.

2. **Write a loader** that returns a DataFrame with the canonical
   columns:

   * `timestamp` (pandas Timestamp)
   * `altitude` (feet, MSL pressure altitude is fine)
   * `vertical_rate` (fpm — use `vertical_rate_fpm()` from `_common`)
   * `tas_kt` (knots — use `wind_triangle_tas_kt()` if not shipped)
   * Optional: `roll_deg`, `pitch_deg`, `heading_deg`, `groundspeed`,
     `radar_alt_ft`

3. **Copy an existing `calibrate.py`** as a template (recommended:
   [`FAAM_BAe146/calibrate.py`](FAAM_BAe146/calibrate.py) for NetCDF native-TAS
   archives, [`SAFIRE_ATR42/calibrate.py`](SAFIRE_ATR42/calibrate.py) for archives
   needing wind-triangle TAS reconstruction).  Set the per-aircraft
   knobs above; the rest is mechanical.

4. **Run it.**  The script prints a paste-ready constructor block.
   Paste into [`hyplan/aircraft/_models.py`](../../hyplan/aircraft/_models.py).

5. **Add the SourceRecord** to the new class with `url=` and `doi=`
   (fields on `hyplan.aircraft.SourceRecord`) so users can cite the
   underlying archive.

6. **Wire up the imports** in
   [`hyplan/aircraft/__init__.py`](../../hyplan/aircraft/__init__.py),
   [`hyplan/__init__.py`](../../hyplan/__init__.py), and
   [`docs/api/aircraft.md`](../../docs/api/aircraft.md).

7. **Generate the companion notebook** — add an entry to
   `AIRCRAFT_CONFIGS` in [`_make_notebook.py`](_make_notebook.py) and
   run `python -m notebooks.calibration._make_notebook`.

8. **Refresh the auto-generated fleet tables** —
   `python -m docs._gen_fleet_tables`.

9. **Add a smoke test** to
   [`tests/test_aircraft.py`](../../tests/test_aircraft.py) — verifies
   the class instantiates and reports sane cruise speed at altitude.

## Calibration manifest CSVs

Each calibration script writes a per-sortie provenance file under
`data/<aircraft>/calibration_manifest.csv` with columns:

* `key` — source-file identifier (archive/year/storm/flight)
* `status` — `kept` or `dropped`
* `reason` — why dropped (empty for kept sorties)
* `date`, `duration_min`, `peak_alt_ft`, `n_fixes`

This lets reviewers (and JOSS reviewers in particular) answer
"did sortie X contribute to the final climb profile?" without
re-running the script.  Commit the manifest alongside any
calibration refresh.

## Provenance and reproducibility

* Calibrations are **not** committed as one-shot snapshots — every
  number in `_models.py` traces back to (a) a public archive URL/DOI
  in the `SourceRecord`, (b) the calibration script, and (c) the
  manifest CSV.
* When a new sortie batch arrives (e.g., the next FAAM season), the
  flow is: extend the fetcher → re-run `calibrate.py` → paste the
  refreshed constructor block → commit `_models.py` + the new
  `calibration_manifest.csv`.
* The companion `calibration.ipynb` is a thin interactive wrapper
  that delegates to `calibrate.py`; it's safe to re-run any time
  to inspect bin medians, IQR, and the paste-ready block.

## Aircraft-specific quirks

A few aircraft don't follow the "one archive, one loader" pattern:

* **`NOAA_TwinOtter`** ([`NOAA_TwinOtter/calibrate.py`](NOAA_TwinOtter/calibrate.py))
  — combines FIREX-AQ N48RF with six NOAA CSL N46RF campaigns.
  Per-file unit detection corrects PI mislabeled m/s vs kt.
* **`NOAA_WP3D`** ([`NOAA_WP3D/calibrate.py`](NOAA_WP3D/calibrate.py)) — merges
  NOAA CSL chemistry ICARTT (3-file MET+POS+MIS join on AOCTimewave)
  with NOAA AOML HRD hurricane 1-sec text via the shared HRD loader.
* **`NOAA_GIV`** ([`NOAA_GIV/calibrate.py`](NOAA_GIV/calibrate.py)) — uses the
  same HRD 1-sec-text loader as `NOAA_WP3D`'s P-3 portion (G-IV "N"
  prefix files share the format).
* **`BAS_TwinOtter`** ([`BAS_TwinOtter/calibrate.py`](BAS_TwinOtter/calibrate.py))
  — five archives across two GPS-unit suffixes (`_JAVAD` vs `_OXTS`)
  and two file flavours (`_1hz.nc` native-TAS vs `*-qc.nc` wind-
  triangle reconstruction).
