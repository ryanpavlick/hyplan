# HyPlan Notebooks: Learning Path & Reference Guide

This directory contains Jupyter notebooks that teach you how to plan airborne remote sensing campaigns using HyPlan. The notebooks are organized as a guided curriculum, from introductory concepts through advanced mission types, wind-aware reachability, calibration, and campaign workflows.

---

## How to Use These Notebooks

1. **Start with the tutorial** to understand the end-to-end workflow.
2. **Follow the learning path** below, or jump directly to the topic you need.
3. Each notebook is self-contained and can be run independently.
4. Notebooks that require internet access or external credentials are clearly marked.

---

## Start Here

| Notebook | Description |
|----------|-------------|
| [tutorial.ipynb](tutorial.ipynb) | **Your first stop.** Walks through a complete airborne campaign planning workflow: define an instrument and aircraft, generate flight lines, compute a flight plan, and visualize the results. Start here to understand how all the pieces fit together. |

---

## Core Geometry & Planning

These notebooks cover the fundamental building blocks of flight planning: individual flight lines, flight boxes (groups of parallel lines), and full flight plans with timing and segments.

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [flight_line_operations.ipynb](flight_line_operations.ipynb) | Create, inspect, and manipulate individual flight lines. Covers azimuth, offsets, splitting, merging, and geodetic properties. | When you need to understand or customize individual flight line geometry. |
| [flight_box_generation.ipynb](flight_box_generation.ipynb) | Generate sets of parallel flight lines (flight boxes) from a center line or study area polygon, with configurable overlap and spacing. | When you need to cover a study area with parallel flight lines. |
| [flight_plan_computation.ipynb](flight_plan_computation.ipynb) | Compute a complete flight plan from flight lines: segment expansion, aircraft-performance-aware timing, phase labels, and summary metrics. | When you need to go from flight lines to a realistic, timed mission plan. |
| [flight_patterns.ipynb](flight_patterns.ipynb) | Generate standard survey patterns: racetracks, lawnmowers, expanding squares, spirals, and more. | When you need a pre-built survey pattern rather than custom flight lines. |
| [dubins_path_planning.ipynb](dubins_path_planning.ipynb) | Compute minimum-radius turn paths (Dubins paths) between waypoints, respecting aircraft turning constraints. | When you need smooth, flyable transitions between waypoints or flight lines. |
| [flight_optimizer_demo.ipynb](flight_optimizer_demo.ipynb) | Optimize flight line ordering and sortie grouping subject to endurance, daily flight-time, and refuel-airport constraints. | When you have many flight lines and want to build efficient multi-sortie or multi-day plans. |
| [isochrone.ipynb](isochrone.ipynb) | Compute wind-aware reachability polygons — `one_way`, `round_trip`, `return_safe`, concentric, and refuel-aware examples — with direct, ellipse, and adaptive ray strategies. | When you need to determine which sites are feasible from a base airport, compare recovery/refuel options, or estimate where an airborne aircraft can still reach and recover. |

**Suggested order:** flight_line_operations &rarr; flight_box_generation &rarr; flight_plan_computation &rarr; flight_patterns &rarr; dubins_path_planning &rarr; flight_optimizer_demo &rarr; isochrone

---

## Instruments & Aircraft

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [sensor_comparison.ipynb](sensor_comparison.ipynb) | Compare sensors across the HyPlan registry: GSD, swath width, spectral range, and altitude constraints. | When choosing between sensors or understanding how sensor parameters affect planning. |
| [aircraft_performance.ipynb](aircraft_performance.ipynb) | Compare aircraft performance: ceiling, endurance, payload capacity, calibrated speed schedules, climb/descent profiles, and turn behavior. | When selecting an aircraft or understanding how aircraft limits constrain your mission. |

**Suggested order:** sensor_comparison &rarr; aircraft_performance

---

## Environmental Constraints

These notebooks help you account for real-world environmental factors that affect when and how to fly.

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [solar_planning.ipynb](solar_planning.ipynb) | Compute solar geometry (elevation, azimuth) and identify optimal illumination windows for a study site. | When your science requires specific solar illumination conditions (e.g., avoiding long shadows). |
| [winds.ipynb](winds.ipynb) | Obtain wind data from constant assumptions or reanalysis / forecast providers, and understand wind conventions (from-direction, U/V components). | When you need wind inputs for flight planning or want to assess wind conditions at your site. |
| [wind_effects.ipynb](wind_effects.ipynb) | Analyze how wind affects flight execution: ground speed, crab angle, swath distortion, mission timing, and long-leg wind-sampling assumptions. | When you need to understand how wind changes your flight plan's timing and coverage. |
| [cloud_analysis.ipynb](cloud_analysis.ipynb) | Retrieve and analyze cloud fraction data from Open-Meteo to assess clear-sky probability for your study area. | When scheduling missions around cloud cover using freely available data (no credentials needed). |
| [cloud_analysis_gee.ipynb](cloud_analysis_gee.ipynb) | Retrieve cloud fraction from Google Earth Engine (MODIS/ERA5) for longer historical records and climatological analysis. | When you need multi-year cloud climatology and have GEE credentials. |
| [phenology_analysis.ipynb](phenology_analysis.ipynb) | Analyze vegetation phenology (green-up, peak, senescence) from satellite data to time missions to target phenological stages. | When your science targets specific vegetation states (e.g., peak greenness, leaf-off). |
| [glint_analysis.ipynb](glint_analysis.ipynb) | Compute sun glint angles and identify conditions where specular reflection affects water/ocean observations. | When planning over-water missions and need to avoid or target sun glint. |

**Suggested order:** solar_planning &rarr; winds &rarr; wind_effects &rarr; cloud_analysis &rarr; phenology_analysis &rarr; glint_analysis

---

## Mission Types

Specialized notebooks for planning missions with different instrument types, each with unique geometry and constraints.

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [lidar_lvis_planning.ipynb](lidar_lvis_planning.ipynb) | Plan LVIS lidar missions: pulse rate, swath geometry, altitude constraints, and coverage optimization. | When planning lidar missions where pulse density and footprint size drive the design. |
| [als_lidar_planning.ipynb](als_lidar_planning.ipynb) | Plan scanning-mirror discrete-return topographic ALS missions (RIEGL VQ-480 II as reference): swath, footprint, nominal point density, contiguity regime maps, MTA timing envelope, inverse solvers, terrain-aware coverage over a real DEM, crab-aware swath polygons, and the G-LiHT dual VQ-480i multi-lidar rig. | When planning corridor / topographic / vegetation / infrastructure ALS surveys where point density and along-track contiguity drive the design. |
| [profiling_lidar_planning.ipynb](profiling_lidar_planning.ipynb) | Plan nadir-pointing single-beam profiling lidars (NASA HSRL-2, HALO, CPL): footprint diameter, horizontal resolution, pulses-per-profile. | When planning vertical-column atmospheric profiling missions (aerosol/cloud backscatter, water-vapor or methane DIAL). |
| [awp_planning.ipynb](awp_planning.ipynb) | Plan Aerosol Wind Profiler missions: dual-LOS geometry, profile spacing, stable-leg feasibility, and vector-profile placement along a flight plan. | When planning coherent Doppler wind-lidar missions where long straight legs and profile density matter more than swath width. |
| [radar_sar_missions.ipynb](radar_sar_missions.ipynb) | Plan SAR radar missions: side-looking geometry, incidence angle, swath width, and look-direction constraints. | When planning SAR missions where look angle and offset geometry matter. |
| [frame_camera_planning.ipynb](frame_camera_planning.ipynb) | Plan frame camera missions: GSD, footprint, forward/side overlap, and frame rate requirements. | When planning aerial photography or photogrammetry with frame cameras. |
| [stereo_oblique_planning.ipynb](stereo_oblique_planning.ipynb) | Plan stereo and oblique camera missions: convergence angle, base-to-height ratio, and tilted sensor geometry. | When planning stereo photogrammetry or oblique imaging missions. |
| [glint_arc_planning.ipynb](glint_arc_planning.ipynb) | Plan curved flight arcs that maintain a constant sun glint angle over water targets. | When you need to fly curved paths to maintain optimal glint geometry. |

**Suggested order:** Start with the mission type that matches your instrument.

---

## Terrain & Airspace

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [terrain_aware_planning.ipynb](terrain_aware_planning.ipynb) | Incorporate terrain elevation into flight planning: DEM retrieval, terrain profiles, and altitude adjustments to maintain constant AGL. | When flying over mountainous or variable terrain where constant AGL matters. |
| [airspace_check.ipynb](airspace_check.ipynb) | Check flight lines against airspace boundaries (FAA NASR data) to identify potential conflicts with restricted or controlled airspace. | When you need to verify that your flight plan avoids airspace conflicts. |
| [airport_selection.ipynb](airport_selection.ipynb) | Find and rank nearby airports by distance, runway length, and surface type for mission staging. | When choosing a base of operations for your campaign. |

**Suggested order:** terrain_aware_planning &rarr; airspace_check &rarr; airport_selection

---

## Campaign Management & Coordination

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [campaign_management.ipynb](campaign_management.ipynb) | Organize multi-flight, multi-day campaigns: define study areas, group flight plans, and track campaign-level metadata. | When managing a campaign with multiple flights or study sites. |
| [satellite_coordination.ipynb](satellite_coordination.ipynb) | Coordinate airborne flights with satellite overpasses: compute ground tracks, find coincidence windows, and plan coordinated observations. | When you need to time airborne flights to coincide with satellite overpasses. |

---

## Aircraft Calibration

How HyPlan's aircraft performance models are derived from real-world telemetry.  19 research platforms are data-calibrated from NASA / NOAA ICARTT campaigns, NASA AFRC IWG1 logs, ADS-B globe-history archives, and international sources (CEDA, AERIS, PANGAEA, DLR).  The per-aircraft notebooks document the methodology, source data, validation diagnostics, and the per-aircraft JSON profile each calibration writes to at [`hyplan/data/aircraft/<short_name>.json`](../hyplan/data/aircraft/).  See [docs/calibration.md](../docs/calibration.md) for the calibration concepts overview.

### NASA fleet

| Notebook | Aircraft | Source | When to Use |
|----------|----------|--------|-------------|
| [calibration/NASA_ER2/calibration.ipynb](calibration/NASA_ER2/calibration.ipynb) | NASA ER-2 (N806/N809) | NASA AFRC IWG1 | Reproduce the ER-2 calibration; methodology reference for the other aircraft. |
| [calibration/NASA_GIII/calibration.ipynb](calibration/NASA_GIII/calibration.ipynb) | NASA G-III (N520) | NASA ASP archive IWG1 | Reproduce the G-III calibration (152 sorties). |
| [calibration/NASA_GV/calibration.ipynb](calibration/NASA_GV/calibration.ipynb) | NASA G-V (N95) | NASA ASP archive IWG1 | Reproduce the G-V calibration (84 sorties). |
| [calibration/NASA_WB57/calibration.ipynb](calibration/NASA_WB57/calibration.ipynb) | NASA WB-57 (N926/N927) | NASA ASP archive IWG1 | Reproduce the WB-57 calibration (100 sorties). |
| [calibration/NASA_C130/calibration.ipynb](calibration/NASA_C130/calibration.ipynb) | NASA C-130H (N436/N439) | NASA ASP archive IWG1 (ACT-America) | Reproduce the C-130H calibration (91 sorties). |
| [calibration/NASA_P3/calibration.ipynb](calibration/NASA_P3/calibration.ipynb) | NASA P-3 (N426) | NASA ASP archive IWG1 | Reproduce the P-3 calibration (252 sorties). |
| [calibration/KingAirB200/calibration.ipynb](calibration/KingAirB200/calibration.ipynb) | NASA King Air B-200 / UC-12 | NASA ICARTT (multi-campaign: ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS) | Reproduce the B-200 calibration (250 sorties). |

### NOAA fleet

| Notebook | Aircraft | Source | When to Use |
|----------|----------|--------|-------------|
| [calibration/NOAA_WP3D/calibration.ipynb](calibration/NOAA_WP3D/calibration.ipynb) | NOAA WP-3D Orion (N42RF/N43RF) | NOAA CSL ICARTT + NOAA AOML HRD hurricane 1-sec text | Reproduce the WP-3D calibration (NOAA chemistry + Hurricane Hunter sorties). |
| [calibration/NOAA_GIV/calibration.ipynb](calibration/NOAA_GIV/calibration.ipynb) | NOAA G-IV "Gonzo" (N49RF) | NOAA AOML HRD 1-sec text | Reproduce the G-IV calibration (93 hurricane synoptic-surveillance sorties). |
| [calibration/NOAA_TwinOtter/calibration.ipynb](calibration/NOAA_TwinOtter/calibration.ipynb) | NOAA Twin Otter (N48RF/N46RF) | NASA / NOAA ICARTT (FIREX-AQ + 6 NOAA CSL campaigns) | Reproduce the Twin Otter calibration (164 sorties; per-file unit detection). |

### International fleet (UK / EU / DE / DLR / BAS / AWI)

| Notebook | Aircraft | Source | When to Use |
|----------|----------|--------|-------------|
| [calibration/NCAR_GV/calibration.ipynb](calibration/NCAR_GV/calibration.ipynb) | NCAR HIAPER (N677F) | NSF/NCAR HIAPER ICARTT NAV (DC3 2012, LaRC ASD) | Reproduce the HIAPER calibration (22 sorties; TAS reconstructed via wind triangle). |
| [calibration/FAAM_BAe146/calibration.ipynb](calibration/FAAM_BAe146/calibration.ipynb) | FAAM BAe-146 (G-LUXE, UK) | CEDA FAAM Core Data Product 1 Hz | Reproduce the FAAM calibration (125 sorties across 27 ASMM-tagged campaigns 2017-2024). |
| [calibration/SAFIRE_ATR42/calibration.ipynb](calibration/SAFIRE_ATR42/calibration.ipynb) | SAFIRE ATR-42 (F-HMTO, FR) | CEDA EUFAR + AERIS EUREC4A 2020 | Reproduce the ATR-42 calibration (44 sorties; mixed wind-triangle and native-TAS sources). |
| [calibration/BAS_TwinOtter/calibration.ipynb](calibration/BAS_TwinOtter/calibration.ipynb) | BAS Twin Otter polar (UK) | CEDA BAS MASIN: OFCAP, ACCACIA, ORCHESTRA, IGP, ArcticCyclones | Reproduce the polar Twin Otter calibration (105 sorties across 5 archives). |
| [calibration/NERC_DO228/calibration.ipynb](calibration/NERC_DO228/calibration.ipynb) | NERC ARSF Dornier 228 (D-CALM, UK) | CEDA NERC ARSF: ACTIVE 2005-2006 + Eyjafjallajökull 2010 | Reproduce the Do-228 calibration (34 sorties; wind-triangle TAS). |
| [calibration/AWI_BaslerBT67/calibration.ipynb](calibration/AWI_BaslerBT67/calibration.ipynb) | AWI Polar 5/6 (Basler BT-67, DE) | PANGAEA: ACLOUD 2017 + HALO-AC3 2022 | Reproduce the BT-67 calibration (~78 sorties). |
| [calibration/DLR_HALO/calibration.ipynb](calibration/DLR_HALO/calibration.ipynb) | DLR HALO (G550, D-ADLR, DE) | DLR HALO BAHAMAS (HALO-AC3 2022) | Reproduce the HALO calibration (18 sorties; single campaign, confidence 0.7). |

### Special-purpose ER-2 notebooks

| Notebook | Aircraft | Source | When to Use |
|----------|----------|--------|-------------|
| [calibration/NASA_ER2/sortie_replay.ipynb](calibration/NASA_ER2/sortie_replay.ipynb) | NASA ER-2 | IWG1 + planned trace | Validate the calibrated ER-2 model against historical sorties. |
| [calibration/NASA_ER2/planned_vs_flown.ipynb](calibration/NASA_ER2/planned_vs_flown.ipynb) | NASA ER-2 | Green Card / KML / IWG1 | Compare planned vs flown vs modeled for NM17 B / CO07v4 / CO06. |

The companion `calibration.ipynb` is a thin interactive wrapper around the per-aircraft `calibrate.py` script — both share helpers from [`notebooks/calibration/_common.py`](calibration/_common.py) (`label_phases`, `per_bin`, `tas_per_bin`, `schedule_pts`, `evaluate_profile`, `summary_table`, `apply_sortie_filters`, `apply_calibration_to_profile`).  Notebooks are regenerated by [`python -m notebooks.calibration._make_notebook`](calibration/_make_notebook.py) from the `AIRCRAFT_CONFIGS` table and the per-aircraft `calibrate.py` source of truth.  Each `calibrate.py` writes the refreshed values directly to `hyplan/data/aircraft/<short_name>.json`; reviewers can inspect the diff on that single file to see exactly what shifted.

These notebooks read locally-cached IWG1 / ICARTT / NetCDF files from `data/<aircraft-class>/` (gitignored — `data/NASA_GIII/`, `data/FAAM_BAe146/`, etc.).  Bring your own from the relevant archive — see [`docs/calibration.md`](../docs/calibration.md) for per-archive citation and access notes.  Install `pip install hyplan[planned]` for Green Card XLSX/PDF parsing.

---

## Export & Sharing

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [export_formats.ipynb](export_formats.ipynb) | Export flight plans to KML, KMZ, GPX, XLSX, CSV, IWG1 / ICARTT-style files, TrackAir, ForeFlight, Honeywell, and pilot-facing briefing formats. | When you need to share flight plans with collaborators, GIS tools, notebooks, or flight-team review workflows. |

---

## Validation & Testing

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [validation.ipynb](validation.ipynb) | Validate HyPlan's computational results against independent reference values (geodetic distances, solar angles, etc.). | When you want to verify HyPlan's accuracy or understand its validation methodology. |

---

## Quick Reference: Prerequisites & Requirements

| Notebook | Internet | Credentials | Optional Deps | Example Data |
|----------|----------|-------------|----------------|--------------|
| tutorial | Yes | None | None | Yes |
| flight_line_operations | No | None | None | No |
| flight_box_generation | No | None | None | No |
| flight_plan_computation | Yes | None | None | No |
| flight_patterns | No | None | None | No |
| dubins_path_planning | No | None | None | No |
| flight_optimizer_demo | Yes | None | None | No |
| isochrone | Optional (MERRA-2 section) | Optional (NASA Earthdata for MERRA-2) | `[winds]` for MERRA-2 | No |
| sensor_comparison | No | None | None | No |
| aircraft_performance | No | None | None | No |
| solar_planning | No | None | None | No |
| winds | Yes | None | None | No |
| wind_effects | Yes | None | None | No |
| cloud_analysis | Yes | None | None | Yes (`exampledata/`) |
| cloud_analysis_gee | Yes | Google Earth Engine | `earthengine-api` | Yes (`exampledata/`) |
| phenology_analysis | Yes | NASA Earthdata | None | Yes (`exampledata/`) |
| glint_analysis | No | None | None | No |
| glint_arc_planning | No | None | None | No |
| lidar_lvis_planning | No | None | None | No |
| als_lidar_planning | Yes (terrain section auto-downloads Copernicus GLO-30 DEM) | None | None | No |
| profiling_lidar_planning | No | None | None | No |
| awp_planning | Optional (terrain demo) | None | None | No |
| radar_sar_missions | No | None | None | No |
| frame_camera_planning | No | None | None | No |
| stereo_oblique_planning | No | None | None | No |
| terrain_aware_planning | Yes | None | None | No |
| airspace_check | Yes | None | None | No |
| airport_selection | Yes | None | None | No |
| campaign_management | No | None | None | Yes (`exampledata/`) |
| calibration/NASA_ER2/calibration | No | None | None | Local `data/NASA_ER2/` (gitignored) |
| calibration/NASA_GIII/calibration | No | None | None | Local `data/NASA_GIII/` (gitignored) |
| calibration/NASA_GV/calibration | No | None | None | Local `data/NASA_GV/` (gitignored) |
| calibration/NASA_WB57/calibration | No | None | None | Local `data/NASA_WB57/` (gitignored) |
| calibration/NASA_C130/calibration | No | None | None | Local `data/NASA_C130/` (gitignored) |
| calibration/NASA_P3/calibration | No | None | None | Local `data/NASA_P3/` (gitignored) |
| calibration/KingAirB200/calibration | No | None | None | Local `data/KingAirB200/` (gitignored) |
| calibration/NOAA_WP3D/calibration | No | None | None | Local `data/WP3D/` + `data/HRD/` (gitignored) |
| calibration/NOAA_GIV/calibration | No | None | None | Local `data/HRD/G-IV-SP_N49RF/` (gitignored) |
| calibration/NOAA_TwinOtter/calibration | No | None | None | Local `data/NOAA_TwinOtter/` (gitignored) |
| calibration/NCAR_GV/calibration | No | None | None | Local `data/HIAPER/` (gitignored) |
| calibration/FAAM_BAe146/calibration | No | None | None | Local `data/FAAM/` (gitignored) |
| calibration/SAFIRE_ATR42/calibration | No | None | None | Local `data/ATR42/` (gitignored) |
| calibration/BAS_TwinOtter/calibration | No | None | None | Local `data/BAS_TwinOtter/` (gitignored) |
| calibration/NERC_DO228/calibration | No | None | None | Local `data/DO228/` (gitignored) |
| calibration/AWI_BaslerBT67/calibration | No | None | None | Local `data/BT67/` (gitignored) |
| calibration/DLR_HALO/calibration | No | None | None | Local `data/HALO/` (gitignored) |
| calibration/NASA_ER2/sortie_replay | No | None | None | Local `data/NASA_ER2/` (gitignored) |
| calibration/NASA_ER2/planned_vs_flown | No | None | `[planned]` | Local `data/NASA_ER2/` (gitignored) |
| satellite_coordination | Yes | None | None | No |
| export_formats | Yes | None | None | No |
| validation | No | None | None | No |
