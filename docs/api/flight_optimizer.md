# Flight Optimizer

Graph-based flight line ordering with endurance constraints, refueling
stops, and multi-day scheduling.  Uses a greedy nearest-neighbour
heuristic over a transit-time weighted directed graph.

## Graph construction

```{eval-rst}
.. autofunction:: hyplan.flight_optimizer.build_graph
```

## Optimization

```{eval-rst}
.. autofunction:: hyplan.flight_optimizer.greedy_optimize
```
