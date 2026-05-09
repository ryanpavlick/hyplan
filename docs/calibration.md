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

Each aircraft class exposes its provenance via the
`Aircraft.calibration_status` attribute, with values
`"calibrated"`, `"inferred"`, or `"uncalibrated"` corresponding
to the *Status* column below.

| Aircraft class | Status | Source | Sorties | Notebook |
|----------------|--------|--------|---------|----------|
| `NASA_ER2` | calibrated | NASA AFRC IWG1 | cache-dependent | [NASA_ER2/calibration.ipynb](../notebooks/calibration/NASA_ER2/calibration.ipynb) |
| `NASA_GIII` | calibrated | NASA ASP archive IWG1 | ~152 | [NASA_GIII/calibration.ipynb](../notebooks/calibration/NASA_GIII/calibration.ipynb) |
| `NASA_GV` | calibrated | NASA ASP archive IWG1 | ~84 | [NASA_GV/calibration.ipynb](../notebooks/calibration/NASA_GV/calibration.ipynb) |
| `NASA_WB57` | calibrated | NASA ASP archive IWG1 | ~100 | [NASA_WB57/calibration.ipynb](../notebooks/calibration/NASA_WB57/calibration.ipynb) |
| `NASA_C130` | calibrated | NASA ASP archive IWG1 (ACT-America) | ~91 | [NASA_C130/calibration.ipynb](../notebooks/calibration/NASA_C130/calibration.ipynb) |
| `NASA_P3` | calibrated | NASA ASP archive IWG1 | ~252 | [NASA_P3/calibration.ipynb](../notebooks/calibration/NASA_P3/calibration.ipynb) |
| `NOAA_WP3D` | calibrated | NOAA CSL ICARTT (ARCPAC 2008, CalNex 2010, SENEX 2013, SONGNEX 2015) | ~18 | [NOAA_WP3D/calibrate.py](../notebooks/calibration/NOAA_WP3D/calibrate.py) |
| `KingAirB200` | calibrated | NASA ICARTT (ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS) | 250 | [KingAirB200/calibration.ipynb](../notebooks/calibration/KingAirB200/calibration.ipynb) |
| `NOAA_TwinOtter` | calibrated | NOAA FIREX-AQ + NOAA CSL ICARTT (TopDown 2014, UWFPS 2017, CalFiDE 2022, AEROMMA 2023, AMMBEC 2024, USOS 2024) | ~164 | [NOAA_TwinOtter/calibrate.py](../notebooks/calibration/NOAA_TwinOtter/calibrate.py) |
| `NCAR_GV` | calibrated | NSF/NCAR HIAPER ICARTT NAV (DC3 2012, LaRC ASD); TAS reconstructed via wind triangle | ~22 | [NCAR_GV/calibrate.py](../notebooks/calibration/NCAR_GV/calibrate.py) |
| `FAAM_BAe146` | calibrated | FAAM Core Data Product 1 Hz, CEDA archive 2017-2024 (27 ASMM-tagged campaigns: DCMEX, ACSIS/ARNA, MPHASE, CCREST, CLARIFY, ACRUISE, ICE-D, WESCON, …) | 125 | [FAAM_BAe146/calibrate.py](../notebooks/calibration/FAAM_BAe146/calibrate.py) |
| `SAFIRE_ATR42` | calibrated | CEDA EUFAR (TAS reconstructed via wind triangle; 7 transnational-access projects) + AERIS EUREC4A 2020 (native TAS) | 44 | [SAFIRE_ATR42/calibrate.py](../notebooks/calibration/SAFIRE_ATR42/calibrate.py) |
| `DLR_HALO` | calibrated | DLR HALO BAHAMAS (HALO-AC3 2022, Arctic, single campaign — confidence 0.7) | 18 | [DLR_HALO/calibrate.py](../notebooks/calibration/DLR_HALO/calibrate.py) |
| `BAS_TwinOtter` | calibrated | BAS MASIN core nav, CEDA: OFCAP 2010-2011 (23) + ACCACIA 2013 (34) + ORCHESTRA 2017-18 (22) + IGP 2018 (14) + ArcticCyclones 2022 (15) | 105 | [BAS_TwinOtter/calibrate.py](../notebooks/calibration/BAS_TwinOtter/calibrate.py) |
| `NERC_DO228` | calibrated | NERC ARSF D-CALM CEDA: ACTIVE 2005-2006 NetCDFs + Eyjafjallajökull 2010 ARSF CSV (TAS reconstructed via wind triangle) | 34 | [NERC_DO228/calibrate.py](../notebooks/calibration/NERC_DO228/calibrate.py) |
| `NASA_C20A` | inferred | Mirrored from `NASA_GIII` (same type certificate) | — | (deferred — see below) |
| `NASA_GIV` | uncalibrated | Manufacturer brochure | — | — |
| `NASA_B777` | uncalibrated | Manufacturer brochure | — | — |
| `KingAirA90` | calibrated | airplanes.live ADS-B (30-day archive, skydive-filtered) | 428 | [KingAirA90/calibrate.py](../notebooks/calibration/KingAirA90/calibrate.py) |
| `KingAir350` | calibrated | airplanes.live ADS-B (UWKA-2 single-tail, science-campaign sample) | 22 | [KingAir350/calibrate.py](../notebooks/calibration/KingAir350/calibrate.py) |

Sortie counts may shift as data deliveries refresh.  See each
notebook's §1 summary table for the current count.

## Data sources

HyPlan's calibrated performance models draw on a handful of
public archives.  The table below groups aircraft by source archive,
auto-generated from the live class definitions by
`python -m docs._gen_fleet_tables`.

<!-- BEGIN AUTOGEN: data_sources -->

| Archive / source | Aircraft | Citation | Notes |
|---|---|---|---|
| **NASA AFRC IWG1** | `NASA_ER2` | [link](https://www.nasa.gov/centers-and-facilities/armstrong/er-2/) | Armstrong Flight Research Center IWG1 / nav telemetry. Used for ER-2. |
| **NASA Airborne Science Program archive** | `NASA_GIII`, `NASA_GV`, `NASA_P3`, `NASA_WB57`, `NASA_C130` | [link](https://airbornescience.nasa.gov/data) | Public asp-archive.arc.nasa.gov IWG1 archive. Used for G-III, G-V, WB-57, P-3, C-130. |
| **NASA ICARTT campaign archive** | `KingAirB200` | [link](https://www-air.larc.nasa.gov/missions.htm) | Multi-campaign NASA ICARTT data (ACT-America, DISCOVER-AQ, KORUS-AQ, LMOS). Used for KingAirB200. |
| **NOAA CSL ICARTT** | `NOAA_WP3D`, `NOAA_TwinOtter` | [link](https://csl.noaa.gov/groups/csl4/measurements/) | ICARTT archive of NOAA Chemical Sciences Laboratory chemistry missions. Used for NOAA_TwinOtter (6 campaigns) and NOAA_WP3D (4 campaigns). |
| **NASA FIREX-AQ** | `NOAA_TwinOtter` | [link](https://csl.noaa.gov/groups/csl4/measurements/) | Firefighter EXperiment for Air Quality 2019. Used for the original NOAA_TwinOtter calibration. |
| **NSF/NCAR HIAPER** | `NCAR_GV` | [link](https://www-air.larc.nasa.gov/missions/dc3-seac4rs/) | NSF/NCAR HIAPER (N677F) DC3 ICARTT NAV files via LaRC ASD. TAS reconstructed via wind triangle. Used for NCAR_GV. |
| **CEDA FAAM** | `FAAM_BAe146` | [doi:10.5285/86433ad261a64a82a3b7b56ed29e4717](https://doi.org/10.5285/86433ad261a64a82a3b7b56ed29e4717) | FAAM Core Data Product 1 Hz NetCDFs from CEDA, 2017-2024 ASMM-tagged campaigns. Used for FAAM_BAe146. |
| **CEDA EUFAR** | `SAFIRE_ATR42` | [link](https://catalogue.ceda.ac.uk/uuid/3d27ed6d44614ea6a9f4f59ff7e4e1ec) | European Facility for Airborne Research transnational-access archive. Used for SAFIRE_ATR42 (7 EUFAR projects). |
| **AERIS EUREC4A** | `SAFIRE_ATR42` | [doi:10.25326/162](https://doi.org/10.25326/162) | French AERIS portal — EUREC4A 2020 SAFIRE ATR-42 L2 1 Hz NetCDFs. |
| **CEDA BAS MASIN** | `BAS_TwinOtter` | [doi:10.5285/8be3dd7cdf44090d89aeb8f105421506](https://doi.org/10.5285/8be3dd7cdf44090d89aeb8f105421506) | British Antarctic Survey MASIN core nav across 5 campaign archives (OFCAP, ACCACIA, ORCHESTRA, IGP, ArcticCyclones). |
| **DLR HALO BAHAMAS** | `DLR_HALO` | [doi:10.1594/PANGAEA.967719](https://doi.org/10.1594/PANGAEA.967719) | DLR HALO native sensor system. Used for DLR_HALO (HALO-AC3). |
| **CEDA NERC ARSF** | `NERC_DO228` | [doi:10.5285/d2c5c36981824b71a98a2906394d61f3](https://doi.org/10.5285/d2c5c36981824b71a98a2906394d61f3) | NERC Airborne Research and Survey Facility D-CALM Dornier 228 (ACTIVE 2005-2006 + Eyjafjallajökull 2010). TAS reconstructed via wind triangle. |
| **AWI PANGAEA Polar 5/6** | `AWI_BaslerBT67` | [doi:10.1594/PANGAEA.902849](https://doi.org/10.1594/PANGAEA.902849) | Alfred Wegener Institute Basler BT-67 nav/met. Used for AWI_BaslerBT67. |

<!-- END AUTOGEN: data_sources -->

Adding a new aircraft from one of these archives is mostly a matter of
writing a per-source loader; the shared recipe in
[`notebooks/calibration/_common.py`](../notebooks/calibration/_common.py)
handles filtering, phase labeling, and binning.  Adding a new source
archive means writing a fetcher (auth and idempotent download) plus a
loader that returns a DataFrame with the canonical columns
(`timestamp`, `altitude`, `vertical_rate`, `tas_kt`, optionally
`roll_deg`, `pitch_deg`, `heading_deg`, `groundspeed`).

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
  30000 ft vs 35000 ft brochure; Twin Otter ships 17500 ft vs
  25000 ft brochure).  Each calibration notebook explicitly
  labels this in its paste-ready cell.
* **`approach_speed`** is the median final-approach TAS for the
  mission mix.  The Twin Otter notebook's 105 kt is on the high
  side of the DHC-6's 80–90 kt typical AFM approach, reflecting
  the NOAA mission profile.

## Performance envelope cross-check

Each calibrated aircraft was cross-checked against manufacturer
brochures and published specifications.  This section summarizes the
comparison so users can judge confidence in HyPlan's planning output
and so reviewers can see exactly which numbers come from data, which
from brochures, and where the two diverge.

Most discrepancies trace to the operational context discussed
above: HyPlan's `service_ceiling` is op-p99 of the actual mission
mix (typically below the certificated AFM ceiling at MTOW); cruise
schedules reflect the research-mission profile (often slow-cruise
dwell over plumes); the approach-speed measure is final-descent TAS
at light-fuel landing weight, which can sit above the certificated
Vref for some types and below for others.

### Calibrated platforms — consistent with brochure

| Class | Ceiling | Cruise | Approach |
|---|---|---|---|
| `NASA_ER2` | 70k ft ✓ | 401 kt @ FL650 (M0.65) ✓ | 130 kt ✓ |
| `NASA_GIII` | 45k ft ✓ | 472 kt @ FL300 (M0.80) ✓ | 139 kt ✓ |
| `NASA_GV` | 51k ft ✓ | M0.80 cruise ✓ | 126 kt — below 130-140 Vref; light science fuel state |
| `NASA_P3` | 27k ft (op-p99 vs 28k brochure) ✓ | 333 kt @ FL200 ✓ | 132 kt ✓ |
| `NASA_WB57` | 63k ft (op-p99) ✓ | 382 kt @ FL550 (M0.65) ✓ | 117 kt — below 130 Vref; light science fuel state |
| `NASA_C130` | 28k ft (op-p99 vs 30k brochure) ✓ | 305 kt @ FL200 ✓ | 126 kt ✓ |
| `KingAirB200` | 30k ft (op-p99 vs 35k brochure) ✓ | 239 kt @ FL250 ✓ | 112 kt — above 100 brochure; mission profile |
| `NCAR_GV` | 51k ft ✓ | M0.80 ✓ | 141 kt ✓ |
| `NOAA_WP3D` | 27.6k ft (op-p99 vs 28k) ✓ | 343 kt plateau @ FL200+ ✓ | 132 kt ✓ |
| `NOAA_TwinOtter` | 17.5k ft (op-p99 vs 25k brochure) ✓ | 142 kt @ FL100 ✓ | 105 kt — above 75-80 brochure Vref; NOAA mission |
| `BAS_TwinOtter` | 14.6k ft (polar BL ops) ✓ | 141 kt @ FL100 ✓ | 95 kt — above 75-80 Vref |
| `FAAM_BAe146` | 34.5k ft vs 35k brochure ✓ | 369 kt @ FL300 (M0.62) — slow vs brochure M0.70; FAAM science profile | 121 kt ✓ |
| `SAFIRE_ATR42` | 24.7k ft vs 25k brochure ✓ | 227 kt @ FL150 — slow vs 265 brochure max cruise; EUFAR survey | 117 kt ✓ |
| `AWI_BaslerBT67` | 25k ft ✓ | 186 kt @ FL100 ✓ | 90 kt ✓ |
| `KingAirA90` | 25k ft (op-p99 vs 26.4k POH) ✓ | 222 kt @ FL160 ✓ vs 226 kt @ FL150 brochure | 149 kt — TAS in last 500 ft AGL during descent; AFM Vref 95-100 KIAS |
| `KingAir350` | 35k ft ✓ | 318 kt @ FL330 ✓ (vs brochure 312 @ FL280; UWKA-2 has Blackhawk XP-67A upgrade) | 110 kt (brochure retained — single-tail descent sample artifacts above FL200) |

### Calibrated platforms — known artifacts

These aircraft have published values that trace to identifiable
sample-size or selection artifacts.  None block release; flagged
here so reviewers and users planning around these envelopes know
where the calibration is on thinner ice.

* **`NOAA_GIV`** — `service_ceiling=47500 ft` exceeds the
  certificated G-IV-SP brochure ceiling of 45000 ft.  Op-p99 over
  the 93-sortie hurricane-surveillance sample where the airframe
  was light enough to overshoot AFM in practice.  Realistic for the
  NOAA mission mix but worth noting.
* **`NERC_DO228`** — cruise schedule decreases above FL100
  (176 kt → 146 kt by FL200), which is unphysical for a turboprop.
  Small-sample artifact at the upper bins (n=34 sorties across two
  CEDA campaigns; the FL150-FL200 cruise bins are populated by
  fewer than 200 fixes).  Approach speed 110 kt is also above the
  80-90 kt brochure Vref.  Queued for v1.6.1 (truncate cruise
  schedule at FL100 or add caveat).
* **`DLR_HALO`** — cruise schedule is implausibly flat at 457 kt
  through FL200-FL300; the brochure long-range cruise band starts
  above FL400 and a G550 climbing through these levels should see
  ~360 kt @ FL200 and ~430 kt @ FL300.  18-sortie HALO-AC3 sample
  is heavily biased toward high-altitude transit, and sub-FL300
  bins are populated by descent fixes mislabeled as cruise.  The
  class already ships with `confidence=0.7` reflecting the
  single-campaign sample size.  Queued for v1.6.1 (truncate cruise
  schedule below FL300).

### Uncalibrated brochure values — reasonableness

| Class | Source | Notes |
|---|---|---|
| `NASA_GIV` | G-IV-SP brochure | Coherent: 45k ceiling, M0.80 cruise, 5130 nm range |
| `NASA_C20A` | Mirrors `NASA_GIII` | Coherent (same airframe + type certificate) |
| `NASA_B777` | 777 brochure | Generic; suitable for transit-style planning |

## Deferred calibrations

One aircraft class remains on inferred values because its public
data source is auth-walled:

* **`NASA_C20A`** (NASA 502, AFRC G-III variant) — currently
  inferred from `NASA_GIII` (same airframe + type certificate).
  No public ICARTT/IWG1 nav data; AFRC mission ops contact
  required for NASDAT housekeeping logs.

Order of effort if access becomes available: C-20A (only
meaningful if AFRC ops shares data) → A-90 (wind-derivation last
resort).  `NCAR_GV` (HIAPER) was deferred in earlier releases but
is now calibrated against 22 NSF-GV ICARTT NAV sorties from DC3
2012 (LaRC ASD); see the row above.

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
