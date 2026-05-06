# Aircraft Calibration

How HyPlan's aircraft performance models are derived from in-situ
field-campaign data.

## What "calibrated" means

For each calibrated platform, the following parameters in
[hyplan/aircraft/_models.py](../hyplan/aircraft/_models.py) come
from per-altitude-bin medians of real flight data rather than
manufacturer brochures:

* `climb_profile` and `descent_profile` (`VerticalProfile`) —
  active-VS bin medians (5-kft bins, n≥30/bin) from sustained-
  climb and sustained-descent fixes.  "Active" means
  `vertical_rate * sign ≥ ACTIVE_VS_THR_FPM`, where the threshold
  is per-aircraft (1500 fpm for jets and most turboprops, 1000
  fpm for the B-200, 500 fpm for the Twin Otter).
* `climb_schedule` / `cruise_schedule` / `descent_schedule`
  (`TasSchedule`) — per-phase TAS-vs-altitude medians at the same
  5-kft binning, anchored at the typical rotation TAS at SL and
  picked at aircraft-specific target altitudes.
* `turn_model.max_bank_deg` — usually `max(AFM normal-ops 30°,
  data p90)` of `|roll_deg|` over fixes with `|roll| > 5°`.
* `approach_speed` — median TAS in the last ~500 ft AGL with
  `vertical_rate < -200 fpm`.
* `service_ceiling` — operational p99 of per-sortie peak
  altitudes (not the certified service ceiling at MTOW).
* `sources` (`SourceRecord`) — campaign / archive citation +
  number of sorties + confidence.

We don't enforce monotonicity on `climb_profile` /
`descent_profile`.  Jets and turboprops typically peak ROC near
FL050-FL100 (limited below by 250-KCAS ATC procedures); descent
VS typically peaks around FL150-FL200 (descent at VMO in CAS) and
then declines in the upper levels (Mach-limited descent at
constant M).  Clamping these to monotone shapes pushes bins
outside their IQRs and obscures the real envelope, so the
calibration step ships the raw bin medians and lets reviewers see
the shape directly.

## Calibration status

| Aircraft class | Status | Source | Sorties | Notebook |
|----------------|--------|--------|---------|----------|
| `NASA_ER2` | Data-fit | NASA AFRC IWG1 | cache-dependent | [er2/calibration.ipynb](../notebooks/calibration/er2/calibration.ipynb) |
| `NASA_GIII` | Data-fit | NASA ASP archive IWG1 | ~153 | [giii/calibration.ipynb](../notebooks/calibration/giii/calibration.ipynb) |
| `NASA_GV` | Data-fit | NASA ASP archive IWG1 | ~101 | [gv/calibration.ipynb](../notebooks/calibration/gv/calibration.ipynb) |
| `NASA_WB57` | Data-fit | NASA ASP archive IWG1 | ~100 | [wb57/calibration.ipynb](../notebooks/calibration/wb57/calibration.ipynb) |
| `C130` | Data-fit | NASA ASP archive IWG1 (ACT-America) | ~87 | [c130/calibration.ipynb](../notebooks/calibration/c130/calibration.ipynb) |
| `NASA_P3` | Data-fit | NASA ASP archive IWG1 | ~252 | [p3/calibration.ipynb](../notebooks/calibration/p3/calibration.ipynb) |
| `KingAirB200` | Data-fit | NASA ICARTT (ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS) | 250 | [b200/calibration.ipynb](../notebooks/calibration/b200/calibration.ipynb) |
| `TwinOtter` | Data-fit | NASA / NOAA ICARTT (FIREX-AQ N48RF) | ~17 | [twin_otter/calibration.ipynb](../notebooks/calibration/twin_otter/calibration.ipynb) |
| `NCAR_GV` | Inferred | Mirrored from `NASA_GV` (same airframe class) | — | (deferred — see below) |
| `NASA_C20A` | Inferred | Mirrored from `NASA_GIII` (same type certificate) | — | (deferred — see below) |
| `KingAirA90` | Brochure | None public | — | (deferred) |
| Other classes | Brochure | Manufacturer specs | — | — |

Sortie counts may shift as data deliveries refresh.  See each
notebook's §1 summary table for the current count.

## Methodology

Each `calibration.ipynb` notebook follows the same recipe (the
ER-2 notebook predates this builder pattern but the structure is
the same):

1. **Load** every IWG1 / ICARTT file from the campaign
   directory, trim ground taxi (or fall back to altitude-only
   airborne detection when groundspeed is missing).
2. **Filter** sorties on duration (60–600/900 min, per-aircraft)
   and peak altitude (per-aircraft floor and ceiling), drop
   sorties with no valid altitude or vertical rate.
3. **Phase-label** each fix as `climb` / `cruise` / `descent`
   from `vertical_rate` against ±300 fpm gates.
4. **Bin** climb / descent fixes by 5-kft altitude bin and
   compute median, p25, p75 of vertical rate.
5. **Bin** TAS the same way for each phase.
6. **Pick breakpoints** at aircraft-specific target altitudes;
   anchor SL at typical rotation / approach TAS so the SL point
   isn't contaminated by takeoff-roll or pattern fixes.
7. **Validate** by overlaying the proposed VerticalProfile on the
   per-bin IQR + median plot.
8. **Emit a paste-ready cell** with the calibrated constants,
   `SourceRecord`, and labeled `service_ceiling` / `approach_speed`
   so it's clear which numbers are operational vs aircraft-
   intrinsic.

Shared helpers live in [notebooks/calibration/_common.py](../notebooks/calibration/_common.py):
`label_phases`, `apply_sortie_filters`, `per_bin`, `tas_per_bin`,
`schedule_pts`, `evaluate_profile`, `summary_table`.  The
aircraft-specific knobs (active-VS threshold, target altitudes,
rotation TAS, hold bands for the ER-2) stay in the per-aircraft
builder.

## Operational vs aircraft-intrinsic numbers

Several reported metrics describe **operational** behavior across
the sortie set rather than aircraft-intrinsic performance:

* Wall-clock **time-to-cruise** (TOC) includes pre-cruise
  level-offs, ATC routing, and weight-management step climbs.
  Reviewers should not read TOC variance as a performance bound;
  the planner's modeled TOC comes from integrating the calibrated
  `climb_profile`, which excludes those operational delays.
* **`service_ceiling`** as shipped is the operational p99 of
  per-sortie peak altitudes for the mission mix.  This is below
  the airframe service ceiling under MTOW (e.g., B-200 ships
  30000 ft vs 35000 ft brochure; Twin Otter ships ~15000 ft vs
  25000 ft brochure).  Each calibration notebook explicitly
  labels this in its paste-ready cell.
* **`approach_speed`** is the median final-approach TAS for the
  mission mix.  The Twin Otter notebook's 99 kt is on the high
  side of the DHC-6's 80–90 kt typical AFM approach, reflecting
  the NOAA mission profile.

## Deferred calibrations

Three aircraft classes remain on inferred or brochure values
because their public data sources are auth-walled or unavailable:

* **`NCAR_GV`** (HIAPER, N677F) — sources at
  https://data.eol.ucar.edu/ (HIPPO / SOCRATES / ORCAS /
  ATTREX / WE-CAN), accessible via NCAR EOL ORDER request.  The
  generic ICARTT loader recognizes HIAPER LRT variable names
  via [hyplan/aircraft/eol_ncar.py](../hyplan/aircraft/eol_ncar.py).
* **`NASA_C20A`** (NASA 502, AFRC G-III variant) — currently
  inferred from `NASA_GIII` (same airframe + type certificate).
  No public ICARTT/IWG1 nav data; AFRC mission ops contact
  required for NASDAT housekeeping logs.
* **`KingAirA90`** — no public IWG1-grade A-90 data; the only
  available files are ADS-B-grade tracks (`n94s_alltracks.csv`)
  with no TAS / Roll / IAS.  A wind-derivation fallback (GPS
  groundspeed + MERRA-2 wind field) is the long-term path.

Order of effort if access becomes available: HIAPER (loader is
closest to ready, ~50–100 sorties available across HIPPO /
SOCRATES / ORCAS) → C-20A (only meaningful if AFRC ops
shares data) → A-90 (wind-derivation last resort).

## Conventions

* **Active-VS thresholds** (per-aircraft):
  - Jets and most turboprops: 1500 fpm
  - King Air B-200: 1000 fpm (climb rates drop below 1500 fpm
    above FL150; the lower threshold extends bin coverage to
    FL250-FL280)
  - Twin Otter: 500 fpm (slow climber; 1500 fpm loses everything
    above FL050, 800 fpm cuts off above FL100)
* **5-kft bin width** is universal; smaller bins thin the
  per-bin sample below the n=30 floor at high altitude.
* **n≥30/bin** floor for VS bins; **n≥50–200/bin** for TAS bins
  (per-aircraft, set in the builder).
* **n≥30/bin** for active-VS keeps every bin that survives
  contamination filtering above autopilot precision.
* The ER-2 notebook excludes weight-management hold bands
  (FL220–FL260, FL240–FL280, FL336–FL376) when computing
  active-climb medians so sortie-specific level-offs and holds do
  not depress the aircraft-intrinsic climb profile.
* Default `confidence=0.85` for data-fit calibrations,
  `confidence=0.7` for inferred-from-sibling values.
* `max_bank_deg` ships as `max(AFM normal-ops 30°, data p90)` so
  the planner uses a bank that the aircraft is willing to fly,
  not the typical-mix median or a steep-turn / emergency value.
