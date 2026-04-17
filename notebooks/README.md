# HyPlan Notebooks: Learning Path & Reference Guide

This directory contains Jupyter notebooks that teach you how to plan airborne remote sensing campaigns using HyPlan. The notebooks are organized as a guided curriculum, from introductory concepts through advanced mission types and operational workflows.

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
| [flight_plan_computation.ipynb](flight_plan_computation.ipynb) | Compute a complete flight plan from flight lines: segment expansion, timing, fuel estimates, and summary metrics. | When you need to go from flight lines to a flyable, timed mission plan. |
| [flight_patterns.ipynb](flight_patterns.ipynb) | Generate standard survey patterns: racetracks, lawnmowers, expanding squares, spirals, and more. | When you need a pre-built survey pattern rather than custom flight lines. |
| [dubins_path_planning.ipynb](dubins_path_planning.ipynb) | Compute minimum-radius turn paths (Dubins paths) between waypoints, respecting aircraft turning constraints. | When you need smooth, flyable transitions between waypoints or flight lines. |
| [flight_optimizer_demo.ipynb](flight_optimizer_demo.ipynb) | Optimize flight line ordering to minimize transit time and total mission duration. | When you have many flight lines and want to find the most efficient ordering. |

**Suggested order:** flight_line_operations &rarr; flight_box_generation &rarr; flight_plan_computation &rarr; flight_patterns &rarr; dubins_path_planning &rarr; flight_optimizer_demo

---

## Instruments & Aircraft

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [sensor_comparison.ipynb](sensor_comparison.ipynb) | Compare sensors across the HyPlan registry: GSD, swath width, spectral range, and altitude constraints. | When choosing between sensors or understanding how sensor parameters affect planning. |
| [aircraft_performance.ipynb](aircraft_performance.ipynb) | Compare aircraft performance: ceiling, endurance, payload capacity, and speed envelopes. | When selecting an aircraft or understanding how aircraft limits constrain your mission. |

**Suggested order:** sensor_comparison &rarr; aircraft_performance

---

## Environmental Constraints

These notebooks help you account for real-world environmental factors that affect when and how to fly.

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [solar_planning.ipynb](solar_planning.ipynb) | Compute solar geometry (elevation, azimuth) and identify optimal illumination windows for a study site. | When your science requires specific solar illumination conditions (e.g., avoiding long shadows). |
| [winds.ipynb](winds.ipynb) | Obtain wind data from constant assumptions or MERRA-2 reanalysis, and understand wind conventions (from-direction, U/V components). | When you need wind inputs for flight planning or want to assess wind conditions at your site. |
| [wind_effects.ipynb](wind_effects.ipynb) | Analyze how wind affects flight execution: ground speed, crab angle, swath distortion, and mission timing. | When you need to understand how wind changes your flight plan's timing and coverage. |
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

## Export & Sharing

| Notebook | Description | When to Use |
|----------|-------------|-------------|
| [export_formats.ipynb](export_formats.ipynb) | Export flight plans to KML, GeoJSON, CSV, IWG1, and other formats for use in Google Earth, GIS tools, and pilot briefing packages. | When you need to share flight plans with pilots, collaborators, or load them into external tools. |

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
| radar_sar_missions | No | None | None | No |
| frame_camera_planning | No | None | None | No |
| stereo_oblique_planning | No | None | None | No |
| terrain_aware_planning | Yes | None | None | No |
| airspace_check | Yes | None | None | No |
| airport_selection | Yes | None | None | No |
| campaign_management | No | None | None | Yes (`exampledata/`) |
| satellite_coordination | Yes | None | None | No |
| export_formats | Yes | None | None | No |
| validation | No | None | None | No |
