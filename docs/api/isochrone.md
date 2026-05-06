# Isochrone

Wind-aware reachability boundaries.  Given a starting state, a time budget,
and (optionally) a recovery destination + on-station observation
requirement, compute the polygon around the start where the aircraft can
operate within the time budget.

## Modes

* **`one_way`** — single-leg reach.  "Where can I be after `budget`?"
* **`round_trip`** — out and back from the same place.  Default; the
  recovery destination defaults to ``start``.
* **`return_safe`** — out, observe, recover at a *different* airport
  within `budget`.  Required for missions where the target is far from
  base.

## Compute

```{eval-rst}
.. autofunction:: hyplan.planning.compute_isochrone
.. autofunction:: hyplan.planning.compute_concentric_isochrones
.. autofunction:: hyplan.planning.compute_refuel_isochrone
.. autofunction:: hyplan.planning.evaluate_target_reachability
```

### Concentric reach

`compute_concentric_isochrones` sweeps multiple budgets in one call and
amortizes the bracket search across them — each budget seeds the next
larger budget's lower bound, so total cost is roughly O(M+N) rather than
O(M·N) for M budgets and N rays.  Returns one stacked GeoDataFrame
tagged with `budget_min` / `budget_hr` columns.

### Refuel-aware reach

`compute_refuel_isochrone` extends the standard isochrone with a single
optional refuel stop drawn from a list of pre-cleared candidates.  Two
clocks are tracked: a per-fuel-cycle `sortie_budget` (resets after each
refuel) and a total wall-clock `flight_day_budget` (does not reset and
absorbs `refuel_time`).  `reserve` applies *per fuel cycle* only.

For every azimuth, three itineraries are evaluated and the most-extending
one wins:

* `direct` — `start → target → recovery`
* `outbound_refuel(R)` — `start → R → target → recovery`
* `return_refuel(R)` — `start → target → R → recovery`

v1 limits `max_refuel_stops` to 1 (a single sortie touches at most two
tanks).  Chained refuels are deferred.

### Single-target reachability

`evaluate_target_reachability` is the complement of
`compute_refuel_isochrone`: rather than sweeping azimuths to find the
boundary, it asks "given *this* target, which itineraries reach it?"
and returns a structured dict with the best route plus all feasible
alternatives.

## Helpers

```{eval-rst}
.. autofunction:: hyplan.planning.isochrone_polygon
.. autofunction:: hyplan.planning.plot_isochrone
.. autofunction:: hyplan.plot_isochrone_static
```

`plot_isochrone` is the interactive Folium plotter.
`plot_isochrone_static` (in `hyplan/plotting.py`) is the
publication-quality Cartopy plotter — accepts either a single GDF or
a list of `(gdf, color, label)` tuples for layered comparison, and
auto-handles concentric and refuel results.

## See also

* [`flight_plan`](flight_plan.md) — once you've selected a target site
  inside the isochrone, hand that target to `compute_flight_plan` to
  produce the actual sortie.
* The [isochrone tutorial notebook](../../notebooks/isochrone.ipynb) walks
  through all three modes plus a multi-aircraft fleet comparison.
