# Dubins Path Planning

2D Dubins path planning with optional wind for realistic aircraft maneuvering
between waypoints. Used by `Aircraft._hybrid_path` for the horizontal layout;
the vertical profile is integrated separately from each aircraft's calibrated
`climb_profile` / `descent_profile`. Trochoidal ground tracks under wind
follow Sachdev et al. (2023).

```{eval-rst}
.. autoclass:: hyplan.dubins3d.DubinsPath2D
   :members:
   :show-inheritance:
```
