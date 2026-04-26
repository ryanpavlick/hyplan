# HyPlan

An open-source Python library for planning airborne remote sensing campaigns.

HyPlan helps scientists and engineers design flight missions for hyperspectral,
multispectral, lidar, and radar remote sensing. It handles flight line generation,
sensor modeling, swath coverage, solar glint prediction, cloud analysis,
terrain-aware calculations, and mission logistics including airport selection
and aircraft performance.

```{tip}
HyPlan modules are classified as **Stable** or **Experimental**.
Stable APIs will not change in backward-incompatible ways within a
major version.  See {doc}`stability` for details.
```

## Getting Started

```{toctree}
:maxdepth: 2

installation
concepts
architecture
stability
tutorial
```

## API Reference

```{toctree}
:maxdepth: 2
:caption: Flight planning

api/flight_line
api/waypoint
api/flight_box
api/flight_patterns
api/flight_plan
api/flight_optimizer
```

```{toctree}
:maxdepth: 2
:caption: Aircraft & navigation

api/aircraft
api/dubins3d
```

```{toctree}
:maxdepth: 2
:caption: Instruments

api/sensors
api/profiling_lidar
api/awp
```

```{toctree}
:maxdepth: 2
:caption: Winds & atmosphere

api/winds
api/atmosphere
```

```{toctree}
:maxdepth: 2
:caption: Terrain & swath

api/terrain
api/swath
```

```{toctree}
:maxdepth: 2
:caption: Environment & timing

api/sun
api/glint
api/clouds
api/phenology
api/satellites
api/airspace
```

```{toctree}
:maxdepth: 2
:caption: Mission management

api/campaign
api/exports
api/plotting
```

```{toctree}
:maxdepth: 2
:caption: Internals

api/geometry
api/units
api/download
api/exceptions
```

## Developer Guide

```{toctree}
:maxdepth: 2

developer
```
