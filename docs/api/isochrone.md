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
```

## Helpers

```{eval-rst}
.. autofunction:: hyplan.planning.isochrone_polygon
.. autofunction:: hyplan.planning.plot_isochrone
```

## See also

* [`flight_plan`](flight_plan.md) — once you've selected a target site
  inside the isochrone, hand that target to `compute_flight_plan` to
  produce the actual sortie.
* The [isochrone tutorial notebook](../../notebooks/isochrone.ipynb) walks
  through all three modes plus a multi-aircraft fleet comparison.
