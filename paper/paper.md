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

HyPlan is an open-source Python library that provides a unified, reproducible, and extensible framework for planning airborne remote sensing missions. It encodes the physics of sensor--platform--environment interactions into composable building blocks covering the full mission-planning lifecycle, from study-area definition and flight line generation through swath and GSD calculations to multi-day mission plans with line-ordering optimization. It includes pre-configured models for a range of NASA airborne instruments (e.g., AVIRIS-3, PRISM, LVIS, UAVSAR, HyTES) and multiple research aircraft, along with tools for solar geometry, terrain-aware swath modeling, cloud climatology, airport selection, and satellite overpass prediction.

HyPlan's core technical contribution is **terrain-aware swath modeling using ray--terrain intersection**, which captures terrain-induced variations in swath width and position that flat-Earth approximations miss; the effect can be substantial in mountainous regions, where swath width may vary by hundreds of meters along a single flight line. A complementary contribution is **wind-aware trajectory modeling subject to aircraft performance constraints**, combining Dubins paths (trochoidal under non-zero wind) with altitude-dependent climb, cruise, and descent models to produce physically realistic flight paths and segment-by-segment timing.

# Statement of Need

Airborne science campaigns are major investments---a single ER-2 deployment costs tens of thousands of dollars per flight hour, and campaigns typically span weeks to months---yet the planning process has remained largely manual and fragmented across disconnected tools. Planning these campaigns is a coupled problem spanning sensor performance, solar geometry, environmental conditions, logistics, and satellite coordination. Sensor requirements such as ground sample distance and swath overlap depend on altitude and terrain; solar elevation and glint constrain when observations are viable; clouds and ecosystem state determine data usability; and aircraft endurance and airport availability constrain how flight lines can be scheduled.

These constraints are interdependent: altitude affects both sensor performance and coverage, wind alters trajectory and timing, and environmental conditions determine when planned flight lines can be executed. In practice, existing workflows treat these elements separately, making it difficult to evaluate trade-offs or ensure reproducibility.

HyPlan addresses this need by treating airborne mission planning as a **coupled physical and spatiotemporal problem**, unifying aircraft dynamics, sensor modeling, environmental constraints, and mission-level scheduling within a single reproducible system.

# State of the Field

Several categories of tools address subsets of the airborne mission planning problem, but none in the integrated, programmatic manner that science campaign design requires.

**Commercial flight planning software** (ForeFlight, Jeppesen FliteStar) targets pilot navigation, fuel planning, and regulatory compliance, not scientific objectives such as GSD, spectral coverage, or sensor-specific swath overlap. **GIS platforms** (QGIS, ArcGIS) can visualize flight lines and study areas but lack domain-specific calculations for sensor modeling, aircraft performance, and mission timing.

**UAS mission planners and agency-internal tools.** Tools such as Mission Planner and QGroundControl [@qgroundcontrol] target small drones at low altitudes and short ranges, and do not model the sensor physics, solar geometry, or logistics constraints relevant to crewed research aircraft. Various NASA centers maintain internal planning tools tailored to specific instruments or aircraft, but these are typically proprietary, undocumented, and tightly coupled to particular missions.

**Moving Lines.** The most directly comparable tool is Moving Lines [@leblanc2018movinglines], a Python-based application developed for NASA airborne science campaigns. Moving Lines excels at real-time, interactive flight plan creation and modification during campaign operations, with a graphical user interface combining an interactive map, spreadsheet-based waypoint editing, and overlays of satellite imagery and weather model output. It has been used operationally across numerous NASA campaigns including ORACLES, IMPACTS, and ARCSIX. Moving Lines is designed as a GUI for manual, interactive planning rather than as a programmatic library, and does not provide sensor-specific swath and GSD calculations, terrain-aware analysis, automated flight line generation, or ordering optimization. HyPlan and Moving Lines are complementary: HyPlan addresses the pre-campaign science planning phase---determining how many flight lines are needed, what altitude and speed satisfy sensor requirements, when solar conditions permit data collection, and how to schedule lines across multiple flight days---while Moving Lines supports the tactical, day-of-flight planning and replanning that occurs during campaign execution.

# Software Design

HyPlan is designed as a **programmatic, physics-based flight planning library** for airborne remote sensing campaigns. The architecture reflects a set of deliberate design choices aimed at supporting reproducible, science-driven mission planning while accommodating the complexity of aircraft performance, sensor geometry, and environmental constraints.

### Programmatic workflows over interactive tools

A central design decision is to expose all functionality through a **Python API** rather than a graphical user interface. While interactive tools such as Moving Lines are well-suited for real-time, tactical flight planning, they make it difficult to reproduce, audit, or systematically explore planning decisions.

HyPlan instead represents mission design as a sequence of function calls operating on explicit data structures such as `FlightLine`, `FlightBox`, and flight plan GeoDataFrames built on NumPy [@harris2020numpy], pandas [@mckinney2010pandas], GeoPandas [@jordahl2020geopandas], and Shapely [@gillies2007shapely]. Modules such as `flight_plan` and `flight_optimizer` combine these abstractions to produce complete multi-segment sortie descriptions, including takeoff, climb, transit, data collection, and landing.

The trade-off is reduced interactivity in exchange for **reproducibility and automation**, enabling version-controlled workflows and systematic exploration of planning parameters.

### Physics-based modeling over geometric approximation

HyPlan explicitly models the physical relationships between aircraft motion, sensor geometry, and the environment. This is reflected across several modules:

- The `swath` module performs **terrain-aware swath modeling** by tracing rays from the sensor to the terrain surface using digital elevation models accessed via rasterio [@rasterio] and GDAL [@gdal2024], rather than assuming flat-Earth geometry (\autoref{fig:terrain}).
- The `flight_plan` module incorporates **wind-aware trajectory modeling**, computing crab angles and ground speeds from airspeed and wind vectors (\autoref{fig:wind}).
- The `dubins3d` module models **aircraft-constrained motion in three dimensions** [@vana2020dubins3d], with pitch constraints derived from each aircraft model's climb and descent performance. Under non-zero wind, the air-relative Dubins paths [@dubins1957curves] produce **trochoidal ground trajectories** [@sachdev2023trochoid] (\autoref{fig:wind}).

![Flat-earth vs terrain-aware flight planning over Rincón de la Vieja National Park, Costa Rica (elevation 234--2,072 m). (a) A flat-earth planner assumes constant swath width and produces uniformly spaced lines. (b) Terrain-aware planning uses ray--terrain intersection to measure actual swath width at each line position, requiring additional lines where terrain narrows the swath. (c) Coverage gaps (red) show areas that would be missed by the flat-earth plan but are covered by the terrain-aware plan.\label{fig:terrain}](figures/fig1_terrain_comparison.png)

The trade-off is increased complexity compared to planners that assume straight-line motion or constant ground speed. The benefit is **physically realistic trajectory, timing, and coverage estimates**---particularly when wind speeds are non-negligible relative to aircraft speed or altitude changes are required---ensuring that planned trajectories are both kinematically feasible and scientifically valid.

![Wind-aware mission planning over Rincón de la Vieja. Top: map views comparing still-air Dubins transit arcs (left) with trochoidal arcs under 60 kt northeasterly wind (right). Bottom: altitude profile showing segment-by-segment timing for both conditions. Wind increases total mission time and distorts transit geometry, effects that must be accounted for in fuel planning and airspace coordination.\label{fig:wind}](figures/fig2_wind_dubins.png)

### Composable abstractions rather than monolithic workflows

HyPlan decomposes the planning problem into **composable abstractions** that can be combined into flexible workflows. Core abstractions include flight lines, survey patterns, sensor models, swath generators, glint analyses, and sortie/mission planners.

Each component is designed to operate independently while maintaining compatibility with others. For example, `FlightLine` objects can be generated independently of sensor choice, sensor models can be applied to arbitrary flight geometries, and swath or glint analyses can be layered onto the same flight lines without modifying the underlying planning objects.

The trade-off is that users must assemble workflows themselves rather than relying on a single “one-click” planner. However, this modular design allows HyPlan to support a wide range of research scenarios, from simple coverage estimation to complex multi-day campaign optimization. It also enables extensibility: new aircraft, sensors, flight patterns, or analysis methods can be introduced without modifying core modules.

### Explicit units and geodesic accuracy

All physical quantities in HyPlan carry explicit units using the Pint library [@pint], and all spatial calculations are performed on the WGS84 ellipsoid using geodesic methods implemented via `pymap3d` [@hirsch2018pymap3d] and pyproj [@pyproj].

This design avoids the class of unit conversion and projection errors that commonly arise in mixed-domain workflows---the kind of error famously responsible for the loss of Mars Climate Orbiter [@mco1999]. The trade-off is additional implementation complexity compared to unitless or planar approaches. The benefit is **numerical correctness and consistency across domains**, allowing seamless integration of aviation conventions (feet, knots, nautical miles) with scientific conventions (meters, m/s, kilometers).

For airborne campaigns spanning hundreds to thousands of kilometers, these geodesic considerations are essential to maintaining spatial accuracy.

### Integration of flight execution and environmental timing

A distinguishing design feature of HyPlan is the integration of **flight execution modeling (“how to fly”)** with **environmental timing constraints (“when to fly”)** within a single framework.

In addition to geometric and aircraft modeling, HyPlan includes modules for:

- **Solar geometry and glint analysis** (`sun`, `glint`): computation of solar elevation [@reda2004sunposition] and specular reflection geometry, including glint arcs that constrain flight heading and timing  
- **Cloud climatology** (`clouds`): estimation of clear-sky probability from reanalysis data via Open-Meteo [@openmeteo] or Google Earth Engine [@gorelick2017gee], with campaign simulation tools  
- **Vegetation phenology** (`phenology`): extraction of NDVI/LAI time series and phenological transition dates from MODIS products  
- **Satellite coordination** (`satellites`): prediction of overpass timing and swath overlap using TLE propagation via Skyfield [@rhodes2019skyfield] and CelesTrak [@celestrak]  

These modules enable users to determine not only where and how to fly, but also **when environmental conditions are suitable for data collection**. The trade-off is increased system scope compared to planners focused solely on geometry or navigation. However, airborne campaigns are inherently constrained by environmental conditions, and optimal data acquisition depends on jointly satisfying spatial and temporal constraints.

### Logistics-aware mission planning

HyPlan incorporates logistics considerations directly into the planning process. The `flight_plan` module models complete sorties, while `flight_optimizer` addresses the problem of ordering flight lines across multiple days subject to constraints such as aircraft endurance, maximum daily flight time, and the availability of refueling airports.

Airport selection is supported through integration with the OurAirports database [@ourairports], enabling filtering by runway length, surface type, and proximity. Airspace conflict detection uses FAA NASR and OpenAIP [@openaip] data to identify intersections with restricted, prohibited, or controlled airspace. These logistics constraints interact with solar and environmental timing, creating a coupled optimization problem that HyPlan addresses through heuristic graph-based methods using NetworkX [@hagberg2008networkx].

The trade-off is that the optimizer uses heuristic approaches rather than guaranteeing global optimality. The benefit is **computational efficiency and practical applicability**, producing usable multi-day schedules for large campaigns without requiring expensive optimization algorithms.

### Design implications for research applications

These design choices collectively enable HyPlan to function as a **reproducible research tool** rather than a static planning utility. Researchers can encode mission design assumptions in code, evaluate sensitivity to parameters such as altitude, overlap, or cloud thresholds, and generate planning products that are directly traceable to scientific requirements.

This architecture supports a range of applications, including:

- design of airborne calibration/validation campaigns for satellite missions  
- planning of ecological and biogeochemical surveys with phenology constraints  
- optimization of multi-day campaigns under cloud, solar, and logistics constraints  
- integration of airborne observations with satellite overpasses  

## Limitations

HyPlan is designed for pre-campaign planning and does not currently model real-time operational constraints such as dynamic weather avoidance, air traffic control restrictions, or in-flight replanning. These capabilities are typically addressed by operational tools such as Moving Lines during campaign execution.

Aircraft performance parameters are drawn from published specifications and operator-provided values and have not yet been calibrated against real-world telemetry. Infrastructure for fitting performance models to ADS-B tracks is included in the library (`hyplan.aircraft.adsb`), but that calibration work is ongoing.

# Research Impact Statement

Early versions of HyPlan have been applied in exploratory and pre-campaign planning workflows for multiple NASA airborne science campaigns, including the SHIFT campaign [@chadwick2025shift], AVIRIS deployments for the NASA Arctic Boreal Vulnerability Experiment (ABoVE) in 2022 and 2023 [@miller2025above], the BioSCaPE campaign [@cardoso2025bioscape], and the AVUELO AVIRIS campaign (2025). 

In these applications, HyPlan supported evaluation of flight line configurations, terrain and solar constraints, and mission feasibility under logistical and environmental limitations, directly informing the design of core capabilities such as terrain-aware swath modeling, flight pattern generation, and mission-level scheduling.

The software demonstrates strong community-readiness signals, including over 1,200 automated tests with greater than 80% code coverage, continuous integration on every commit, comprehensive API documentation, and more than 20 Jupyter notebooks that serve as both tutorials and integration tests. Core calculations are validated against independent references including Vincenty geodesic [@vincenty1975direct] test cases, NOAA solar geometry calculations [@reda2004sunposition], and analytical sensor models.

HyPlan is under active development for integration into future NASA airborne campaign planning workflows, where its ability to unify sensor modeling, aircraft performance, and environmental constraints in a single framework is expected to support more reproducible and efficient mission design.

# AI Usage Disclosure

Generative AI tools (Claude and ChatGPT) were used to assist with drafting and editing this manuscript. The software itself was developed with AI coding assistance (Claude and ChatGPT). All AI-generated content was reviewed and verified by the author.

# Acknowledgements

The author thanks Samuel LeBlanc for developing Moving Lines and for contributions to the airborne science planning community that informed HyPlan's design.

This work was supported by the National Aeronautics and Space Administration.

# References
