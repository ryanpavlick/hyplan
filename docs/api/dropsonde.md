# Dropsondes

```{warning}
**Experimental module (v1.9).**  The public API of
`hyplan.instruments.dropsondes` is provisional and may change in
subsequent releases as it is validated against operational use.  Pin
the HyPlan version if you depend on the current surface; track release
notes for breaking changes.
```

Event-based sampling — each release is a single (lat, lon, alt, time)
event, not a swath.  The science footprint is the slant column from
release to splash, drifting through the ambient wind.  HyPlan's
dropsondes are first-class objects: `DropsondeRelease` (event),
`DropsondeTrajectory` (simulated descent), and `DropsondePlan`
(collection).  All three are frozen with identity equality;
`DropsondePlan.simulate()` is functional — it returns a new plan,
leaving the original pre-sim plan intact.

The {class}`~hyplan.instruments.DropsondeSystem` class extends
{class}`~hyplan.instruments.Sensor` for naming / registry uniformity
but does **not** implement the `ScanningSensor` protocol — dropsondes
have no swath.  The pre-configured
{data}`~hyplan.instruments.AVAPS_NRD41` reference is the standard
NCAR-EOL Vaisala NRD41 sonde used in hurricane and airborne campaigns.
Reference instances are **shared singletons** — to customise a
parameter, build a new `DropsondeSystem(...)` rather than mutating
the reference.

Worked planning example:
[`notebooks/dropsonde_objects.ipynb`](../../notebooks/dropsonde_objects.ipynb).

## Core objects

```{eval-rst}
.. autoclass:: hyplan.instruments.DropsondeRelease
   :members:
   :show-inheritance:

.. autoclass:: hyplan.instruments.DropsondeTrajectory
   :members:
   :show-inheritance:

.. autoclass:: hyplan.instruments.DropsondePlan
   :members:
   :show-inheritance:
```

## Sensor

```{eval-rst}
.. autoclass:: hyplan.instruments.DropsondeSystem
   :members:
   :show-inheritance:

.. autodata:: hyplan.instruments.AVAPS_NRD41
   :no-value:

.. autodata:: hyplan.instruments.RD94
   :no-value:

.. autodata:: hyplan.instruments.AXCTD
   :no-value:

.. autofunction:: hyplan.instruments.terminal_velocity_nrd41

.. autofunction:: hyplan.instruments.terminal_velocity_sippican_axctd
```

## Simulation

```{eval-rst}
.. autofunction:: hyplan.instruments.simulate_release

.. autofunction:: hyplan.instruments.simulate_descent_trajectory
```

## Planning helpers

```{eval-rst}
.. autofunction:: hyplan.instruments.releases_along_flight_line
```

## Inverse targeting

```{eval-rst}
.. autoclass:: hyplan.instruments.DropsondeReleaseSolution
   :members:

.. autofunction:: hyplan.instruments.solve_release_for_target
```

## Flight-plan adapter

```{eval-rst}
.. autoclass:: hyplan.instruments.FlightPlanTrack
   :members:

.. autoclass:: hyplan.instruments.PlannedSegment
   :members:

.. autoclass:: hyplan.instruments.AircraftTrackSample
   :members:
```

## GeoDataFrame exports

`DropsondePlan` exposes three GeoDataFrame views — manifest (one row
per planned release), trajectories (one row per integration step), and
summary (per-release diagnostics with splash ellipse) — via
`to_manifest_gdf()`, `trajectories_gdf()`, and `summary()`.  See the
worked notebook for the column schemas in context.

```{eval-rst}
.. autofunction:: hyplan.instruments.summarize_trajectories
```
