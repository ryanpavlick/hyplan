---
title: 'HyPlan: An Open-Source Python Library for Planning Airborne Remote Sensing Campaigns'
tags:
  - Python
  - remote sensing
  - flight planning
  - airborne science
  - hyperspectral
  - Earth observation
authors:
  - name: Ryan Pavlick
    orcid: 0000-0001-8772-3508
    corresponding: true
    affiliation: 1
affiliations:
  - name: National Aeronautics and Space Administration, Washington, DC, USA
    index: 1
date: 16 April 2026
bibliography: paper.bib
---

# Summary

Airborne remote sensing---using instrumented aircraft to collect data over the Earth's surface---is a critical tool for Earth science, enabling observations at spatial and temporal scales that bridge ground-based measurements and satellite data. Airborne platforms carry imaging spectrometers, lidars, radars, and other instruments over study areas at altitudes ranging from a few hundred meters to over 20 kilometers, producing data products across Earth science disciplines. Planning these campaigns requires simultaneously reasoning about aircraft flight dynamics, sensor characteristics and ground sampling, environmental conditions, airport logistics, airspace restrictions, satellite overpass coordination, and more. Scientists typically address these constraints using ad hoc spreadsheets, manual calculations, and institutional knowledge, producing workflows that are error-prone, hard to reproduce, and hard to transfer between campaigns.

HyPlan is an open-source Python library that provides a unified, reproducible, and extensible framework for planning airborne remote sensing missions. It encodes the physics of sensor--platform--environment interactions into composable building blocks covering the full mission-planning lifecycle, from study-area definition and flight line generation through swath and GSD calculations to multi-day mission plans with line-ordering optimization. It ships with 19 pre-configured NASA instrument models and 15 research aircraft, and provides tools for solar geometry, terrain-aware swath modeling, cloud climatology, airport selection, and satellite overpass prediction.

HyPlan's core technical contribution is **terrain-aware swath modeling using ray--terrain intersection**, which captures terrain-induced variations in swath width and position that flat-Earth approximations miss; the effect can be substantial in mountainous regions, where swath width may vary by hundreds of meters along a single flight line. A complementary contribution is **wind-aware trajectory modeling subject to aircraft performance constraints**, in which Dubins paths (trochoidal under non-zero wind) are combined with altitude-dependent climb, cruise, and descent models to produce physically realistic flight paths and segment-by-segment timing.

# Statement of Need

Airborne science campaigns are major investments---a single ER-2 deployment costs tens of thousands of dollars per flight hour, and campaigns typically span weeks to months---yet the planning process has remained largely manual and fragmented across disconnected tools. Scientists planning these campaigns must simultaneously reason about several interacting domains:

**Sensor performance.** Ground sample distance, swath width, and critical ground speed all depend on flight altitude and speed; terrain relief causes swath width to vary along a single flight line, and required swath overlap depends on both science goals and this variability. Planning must account for these relationships to meet spatial sampling requirements.

**Solar geometry.** Passive optical instruments require minimum solar elevation for adequate signal-to-noise, and aquatic applications must also contend with sun glint---specular reflection from the water surface that can either contaminate observations or, when deliberately exploited, provide surface-roughness information [@cox1954glint]. The glint angle varies continuously along a line with solar position, view geometry, and heading.

**Logistics.** Research aircraft have finite endurance and runway requirements, and a large study area may need hundreds of flight lines across many sorties. Flight line ordering, refueling choices, and per-day line allocation all drive total campaign duration and cost, and the resulting schedule must be compatible with the solar and environmental constraints above.

**Satellite coordination.** Airborne campaigns designed for calibration, validation, or science synergy with satellite instruments must predict overpass times and ground-track geometry to identify windows of overlap.

**Multi-instrument coordination.** Many campaigns deploy several instruments simultaneously---imaging spectrometers, lidars, radars, and in situ sensors flown together or on coordinated aircraft. Planning must produce a shared flight geometry that satisfies every instrument's altitude, speed, and overlap requirements, and must communicate those decisions across instrument teams and across science domains in a common, reproducible language.

# State of the Field

Several categories of tools address subsets of the airborne mission planning problem, but none in the integrated, programmatic manner that science campaign design requires.

**Commercial flight planning software** (ForeFlight, Jeppesen FliteStar) targets pilot navigation, fuel planning, and regulatory compliance, not scientific objectives such as GSD, spectral coverage, or sensor-specific swath overlap. **GIS platforms** (QGIS, ArcGIS) can visualize flight lines and study areas but lack domain-specific calculations for sensor modeling, aircraft performance, and mission timing.

**UAS mission planners and agency-internal tools.** Tools such as Mission Planner and QGroundControl [@qgroundcontrol] target small drones at low altitudes and short ranges, and do not model the sensor physics, solar geometry, or logistics constraints relevant to crewed research aircraft. Various NASA centers maintain internal planning tools tailored to specific instruments or aircraft, but these are typically proprietary, undocumented, and tightly coupled to particular missions.

**Moving Lines.** The most directly comparable tool is Moving Lines [@leblanc2018movinglines], a Python-based application developed for NASA airborne science campaigns. Moving Lines excels at real-time, interactive flight plan creation and modification during campaign operations, with a graphical user interface combining an interactive map, spreadsheet-based waypoint editing, and overlays of satellite imagery and weather model output. It has been used operationally across numerous NASA campaigns including ORACLES, IMPACTS, and ARCSIX. Moving Lines is designed as a GUI for manual, interactive planning rather than as a programmatic library, and does not provide sensor-specific swath and GSD calculations, terrain-aware analysis, automated flight line generation, or ordering optimization. HyPlan and Moving Lines are complementary: HyPlan addresses the pre-campaign science planning phase---determining how many flight lines are needed, what altitude and speed satisfy sensor requirements, when solar conditions permit data collection, and how to schedule lines across multiple flight days---while Moving Lines supports the tactical, day-of-flight planning and replanning that occurs during campaign execution.

HyPlan complements these tools by providing an open-source, composable Python framework that integrates sensor modeling, flight planning, environmental analysis, and logistics optimization in a single library.

While several existing tools address components of airborne mission planning, extending them to support HyPlan's use case would require substantial architectural changes. In particular, Moving Lines is designed as an interactive GUI for real-time operations, whereas HyPlan targets programmatic, pre-campaign design workflows that require reproducibility, automation, and integration with scientific analysis pipelines. Similarly, GIS platforms and UAS mission planners lack native representations of sensor physics, aircraft performance, and mission-level scheduling constraints. HyPlan was therefore developed as a composable Python library that unifies these domains within a single framework, enabling workflows that cannot be readily achieved by extending existing tools.

# Typical Workflow

A typical HyPlan workflow proceeds through five stages (\autoref{fig:workflow}), each a few lines of Python: (1) define a study area from polygons or center-and-dimension parameters; (2) generate parallel flight lines for a given sensor, altitude, and cross-track overlap; (3) compute terrain-aware swaths from 30 m Copernicus DEM data; (4) assemble the full sortie---takeoff, climb, transit, data collection, descent, approach---using an aircraft performance model and, optionally, wind fields from MERRA-2, GEOS-FP, or GFS; and (5) export to standard pilot, science, and GIS formats. Because every step is a Python function call, the entire workflow is version-controllable and reproducible.

![HyPlan workflow. The five-stage pipeline integrates spatial planning ("how to fly"---sensor, terrain, aircraft, winds) with environmental timing analysis ("when to fly"). Phenology and cloud climatology inform when to schedule the campaign based on the study area; cloud climatology additionally iterates with the mission plan, since campaign duration depends jointly on clear-sky probability and per-sortie flight time. Solar geometry and cloud forecasts enter at the mission plan for daily illumination, glint, and go/no-go decisions.\label{fig:workflow}](figures/fig1_workflow.png)

# Software Design

HyPlan is organized into three functional groups---Flight Planning, Instruments, and Environment and Logistics---built on NumPy [@harris2020numpy], pandas [@mckinney2010pandas], GeoPandas [@jordahl2020geopandas], and Shapely [@gillies2007shapely], with coordinate transformations and geodesics via pymap3d [@hirsch2018pymap3d] and pyproj [@pyproj]. Two cross-cutting design decisions shape the library: all physical quantities carry explicit units via Pint [@pint], preventing the unit-conversion errors that have historically caused failures in aerospace applications [@mco1999] and letting aviation and scientific conventions mix freely; and all distance and bearing calculations use Vincenty's formulae [@vincenty1975direct] on the WGS84 ellipsoid, avoiding the >0.3% errors of spherical approximations at survey distances. Results export to standard geospatial and aviation formats and render as interactive Folium web maps [@folium]. Source code, installation instructions, and full API documentation are available at the project repository, <https://github.com/ryanpavlick/hyplan>, with rendered documentation at <https://ryanpavlick.github.io/hyplan/>.

HyPlan's design reflects a series of trade-offs between physical realism, computational efficiency, and usability. Aircraft trajectories are modeled using Dubins paths rather than full optimal-control solutions, providing fast, deterministic approximations suitable for interactive planning while capturing first-order maneuver constraints. Aircraft performance is represented using simplified altitude-dependent climb and cruise models derived from published specifications, prioritizing robustness and ease of parameterization over high-fidelity flight dynamics. Flight-line ordering uses a greedy heuristic rather than global optimization to ensure tractable runtimes for large surveys, with the option to incorporate more advanced solvers in future work. Terrain-aware swath modeling is implemented via ray--terrain intersection rather than analytic approximations, trading computational cost for improved accuracy in complex terrain.

## Flight Planning

The core abstraction is the `FlightLine`, a geodesic segment defined by two endpoints and an altitude above mean sea level, supporting clipping to polygon boundaries, splitting, perpendicular and along-track offsetting, rotation, and reversal---all preserving geodesic properties, altitude, and metadata. The `FlightBox` module distributes parallel flight lines across arbitrary study areas given a sensor, altitude, and desired cross-track overlap, oriented along a user-specified azimuth or along the study area's minimum rotated bounding rectangle, with an optional boustrophedon (alternating-direction) pattern that minimizes transit between lines. Beyond parallel surveys, the `flight_patterns` module provides generators for racetrack patterns, rosettes, sawtooth profiles, spirals, and coordinated multi-aircraft lines used for instrument calibration and atmospheric profiling.

\autoref{fig:flightbox} illustrates the spatial planning pipeline for a survey of Rincón de la Vieja National Park in Costa Rica using AVIRIS-3 at 20,000 ft MSL.

![Terrain-aware flight box generation over Rincón de la Vieja National Park, Costa Rica (0--2,017 m elevation) for AVIRIS-3 at 20,000 ft MSL. (a) Study area on shaded relief. (b) Minimum rotated bounding rectangle with sensor-driven line spacing. (c) Parallel flight lines with terrain-aware spacing. (d) Lines clipped to the park boundary with ray--terrain-intersected swath polygons, narrowing over the summit where aircraft--terrain separation decreases.\label{fig:flightbox}](figures/fig2_flight_box.png)

**Aircraft performance** is modeled for 15 research aircraft spanning the range of platforms used in airborne Earth science, from high-altitude jets (NASA ER-2, WB-57) and business jets (Gulfstream III, IV, V) through medium-altitude turboprops (King Air B200, Twin Otter, P-3 Orion, C-130) to smaller platforms. Each aircraft model specifies service ceiling, cruise speed, sea-level and service-ceiling rates of climb, descent rate, approach speed, endurance, range, and maximum bank angle. Cruise speed varies with altitude as a linear interpolation between low-altitude and service-ceiling speeds (or a user-defined piecewise profile); rate of climb follows a linear model that decreases with altitude and admits an analytical climb-time solution, producing a realistic exponential climb profile. The approach phase follows an IFR profile with an intermediate fix and final approach fix at standard distances from the runway.

Complete mission plans are assembled by the `flight_plan` module, which models every phase of a sortie---takeoff, climb, transit, data collection, transit between lines, descent, approach, and landing---using the aircraft's altitude-dependent speed profile. The output is a GeoDataFrame containing the full trajectory with segment classification and segment-by-segment timing, distance, heading, and altitude.

Per-segment wind corrections are provided by the `winds` module, which exposes a `WindField` abstraction returning U/V wind components at arbitrary ``(lat, lon, altitude, time)`` points. Five implementations are available: `StillAirField` (zero-wind baseline), `ConstantWindField` (a single speed and direction), `MERRA2WindField` (MERRA-2 reanalysis via OPeNDAP, for historical planning), `GMAOWindField` (GEOS-FP near-real-time analysis), and `GFSWindField` (NOAA GFS 0.25° forecasts via the NOMADS GRIB filter, with up to a 16-day horizon and server-side subsetting that keeps downloads small). The same abstraction feeds both `compute_flight_plan`---where it determines crab angle and ground speed on every segment---and the trochoidal Dubins solver described below, where it shapes the time-optimal turn geometry.

For campaigns with many flight lines, the `flight_optimizer` module formulates line ordering as a graph problem: nodes in a NetworkX [@hagberg2008networkx] graph represent airport locations and flight line endpoints, and edges are weighted by transit time computed from Dubins path distances and the aircraft performance model. A greedy nearest-neighbor heuristic subject to endurance, daily flight-time, and refueling-airport constraints produces a multi-day schedule with ordered lines, refueling stops, and per-day flight time accounting. The optimizer is experimental: it does not yet incorporate solar windows, cloud forecasts, or airspace conflicts as constraints, and the greedy algorithm does not guarantee global optimality.

Realistic aircraft maneuvering between waypoints is modeled using Dubins curves [@dubins1957curves; @walker2011dubins]: the shortest path between two oriented points subject to a minimum turning radius $R = v^2 / (g \tan\phi)$ derived from true airspeed $v$ and maximum bank angle $\phi$. In still air the path is a combination of circular arcs and straight segments; under wind the arcs become trochoidal ground tracks (circles drifting with the wind), solved in the air-relative frame following Sachdev et al. [@sachdev2023trochoid] and mapped to ground coordinates with the cumulative wind drift. The solution also yields the crab angle---the offset between aircraft heading and ground track---which affects both mission timing and the orientation of the sensor's cross-track field of view relative to the ground. \autoref{fig:dubins} compares still-air and wind-perturbed trajectories for a complete mission from Liberia airport (MRLB) to the Rincón de la Vieja flight lines. To our knowledge, HyPlan is the only open-source flight planning tool that implements wind-aware trochoidal Dubins paths for airborne science applications.

![Complete mission from MRLB (Liberia, Costa Rica) to Rincón de la Vieja. Top: map view comparing still-air circular Dubins arcs (left) with wind-aware trochoidal arcs under a 60 kt northeasterly (right). Bottom: altitude profile of the full sortie; the still-air trace is offset by 500 ft for visibility.\label{fig:dubins}](figures/fig3_wind_dubins.png)

## Instruments

HyPlan provides sensor models for the major instrument types used in airborne remote sensing. **Line-scanning imagers** (pushbroom or whiskbroom) are characterized by their cross-track field of view, number of across-track pixels, and frame rate. Each sensor model exposes the geometric relationships---nadir GSD, swath width, and critical ground speed (the maximum speed at which along-track pixels remain contiguous)---needed to plan altitudes and speeds that satisfy science requirements. Pre-configured models are provided for 14 NASA line-scanning instruments covering visible-to-shortwave-infrared imaging spectrometers (AVIRIS-3 [@thompson2022aviris3], AVIRIS-5, PRISM [@mouroulis2014prism], G-LiHT VNIR), thermal emission spectrometers and scanners (HyTES [@johnson2011hytes], G-LiHT Thermal, MASTER, eMAS), a UV/visible trace-gas spectrometer (GCAS), and a solar-induced fluorescence imager (G-LiHT SIF).

**Frame cameras** are modeled with two-dimensional sensor arrays, computing ground footprint, GSD, and along-track sampling from focal length, sensor dimensions, pixel count, and frame rate. A `MultiCameraRig` class composes multiple cameras into a single sensor with a combined cross-track field of view, forward/aft stereo pairs, and along-track overlap analysis; a factory method reproduces the eight-camera QUAKES-I stereoimaging instrument [@donnellan2025quakes].

**Lidar** is represented by the LVIS (Land, Vegetation, and Ice Sensor) full-waveform scanning lidar [@blair1999lvis]. The model supports three standard lens configurations (narrow, medium, wide) that trade footprint size against the ability to tile the conical swath contiguously at the sensor's pulse rate and aircraft ground speed.

**Synthetic aperture radar** is modeled for the UAVSAR instrument [@hensley2008uavsar] in its L-band, P-band (AirMOSS), and Ka-band (GLISTIN-A) configurations. For this side-looking geometry the model computes slant- and ground-range resolution, swath width and ground offsets defined by near- and far-range incidence angles, and interferometric line spacing for repeat-pass or cross-track interferometry applications.

**Terrain-aware swath polygons** are generated by the `swath` module, which traces rays from the sensor at the cross-track swath edges to the terrain surface, then closes the polygon along-track. The resulting polygons capture the terrain-induced variations in swath width and position visible in \autoref{fig:flightbox}d, which can be substantial in mountainous terrain.

## Environment and Logistics

**Solar position** calculations use the `sunposition` library [@reda2004sunposition]. The `sun` module determines data collection windows by finding when solar elevation crosses specified thresholds (e.g., 20° or 30° minimum) across a range of dates, supporting seasonal planning across latitude.

**Sun glint** is computed in the `glint` module as the angle between the specular reflection direction and the sensor line-of-sight for each pixel in the cross-track field of view, evaluated along the full flight line at configurable spacing to produce a spatially resolved glint map. This identifies flight headings and times that minimize (or, for glint-targeting applications, maximize) specular contamination. For applications that deliberately exploit sun glint---notably methane plume detection over water---the `GlintArc` class generates a banked, arc-shaped flight path that tilts the sensor through the specular reflection geometry, following the observing strategy of Ayasse et al. [@ayasse2022glint].

**Terrain analysis** uses 30-meter Copernicus DEM data, downloaded automatically from AWS and cached locally with R-tree spatial indexing. The terrain module supports bulk elevation queries, Rasterio [@rasterio]-based tile merging, and a vectorized ray--terrain intersection algorithm that returns the first ground intersection for an arbitrary off-nadir view ray at a configurable precision (default 10 m).

**Cloud climatology and forecasts.** The `clouds` module estimates clear-sky probability by day of year from MODIS Terra and Aqua quality-assurance bands via Google Earth Engine [@gorelick2017gee] and from ERA5 reanalysis via the Open-Meteo API [@openmeteo], and runs a campaign simulation that estimates how many campaign days are required to achieve complete coverage under a given clear-sky threshold. Because MODIS provides at most two overpasses per day, this supports seasonal planning but does not resolve diurnal cloud development. For short-range scheduling and day-of-flight go/no-go decisions, the `forecast` submodule fetches up to 16 days of cloud-cover forecast from the Open-Meteo Forecast API [@openmeteo].

**Vegetation phenology** is assessed from MODIS products via NASA EarthData---NDVI/EVI vegetation indices (MOD13A1/MYD13A1), leaf area index (MOD15A2H), and phenological transition dates (MCD12Q2)---producing seasonal profiles and phenology calendars that identify collection windows when vegetation is at the desired phenological stage. The cloud and phenology modules together enable joint temporal optimization (\autoref{fig:timing}), allowing planners to target specific ecosystem states---peak greenness, senescence, snowmelt onset---that can shift by weeks between years.

![Cloud fraction (top) and vegetation index (bottom) climatology for three California study regions. The shaded band marks the joint observation window in which clear-sky probability is high and montane vegetation is near peak greenness.\label{fig:timing}](figures/fig4_cloud_phenology.png)

**Airport selection** leverages the OurAirports global database [@ourairports], filtering by proximity, country, airport type, runway length, and surface type, and exposing runway dimensions, headings, and elevation for operational matching against a given aircraft.

**Airspace conflict detection** checks planned flight lines against controlled airspace, temporary flight restrictions, and special-use airspace using OpenAIP [@openaip] for international boundaries and FAA TFR/NASR for US-specific restrictions, reporting intersections with airspace class, altitude floors and ceilings, and effective schedules so that conflict resolution can begin during pre-campaign planning rather than being deferred to flight operations.

**Satellite overpass prediction** supports coordination of airborne observations with 15 missions including PACE, Landsat-8/9, Sentinel-2A/B, Sentinel-3A/B, JPSS-1/2, Aqua, Terra, ICESat-2, CALIPSO, CloudSat, and EarthCARE. The module fetches TLE sets from CelesTrak [@celestrak], propagates orbits with Skyfield [@rhodes2019skyfield], and returns per-overpass ground tracks and swath footprint polygons filtered by solar zenith angle and spatial overlap with the study area.

## Limitations

HyPlan is designed for pre-campaign planning and does not currently model real-time operational constraints such as dynamic weather avoidance, air traffic control restrictions, or in-flight replanning. These capabilities are typically addressed by operational tools such as Moving Lines during campaign execution.

Aircraft performance parameters are drawn from published specifications and operator-provided values and have not yet been calibrated against real-world telemetry. Infrastructure for fitting performance models to ADS-B tracks is included in the library (`hyplan.aircraft.adsb`), but that calibration work is ongoing.

# Research Impact Statement

Early versions of HyPlan have been evaluated in the context of NASA airborne science campaigns including BioSCape [@cardoso2025bioscape], SHIFT [@chadwick2025shift], and the 2022--2023 ABoVE AVIRIS deployments [@miller2025above], where it was applied in exploratory and pre-campaign planning workflows. These applications informed the design of core functionality including terrain-aware swath modeling, flight-line generation, and mission-level scheduling.

HyPlan is designed as a reproducible, programmatic alternative to ad hoc planning workflows. All planning steps are expressed as Python function calls, enabling version-controlled, re-executable campaign plans that can be shared across instrument teams and revisited as study designs evolve.

The software demonstrates strong community-readiness signals, including over 1,200 automated tests with greater than 80% code coverage, continuous integration on every commit, comprehensive API documentation, and more than 20 Jupyter notebooks that serve as both tutorials and integration tests. Core calculations are validated against independent references including Vincenty geodesic test cases, NOAA solar geometry calculations, and analytical sensor models.

HyPlan is under active development for integration into future NASA airborne campaign planning workflows, where its ability to unify sensor modeling, aircraft performance, and environmental constraints in a single framework is expected to support more reproducible and efficient mission design.

# AI Usage Disclosure

Generative AI tools (Claude and ChatGPT) were used to assist with drafting and editing this manuscript. The software itself was developed with AI coding assistance (Claude and ChatGPT). All AI-generated content was reviewed and verified by the author.

# Acknowledgements

The author thanks Samuel LeBlanc for developing Moving Lines and for contributions to the airborne science planning community that informed HyPlan's design.

This work was supported by the National Aeronautics and Space Administration.

# References
