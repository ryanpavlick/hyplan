# Winds

Wind field models for per-segment wind correction in flight planning.

Provides a {py:class}`~hyplan.winds.WindField` abstraction with
implementations for still air, constant wind, NASA MERRA-2 reanalysis,
NOAA GFS forecast, and GMAO GEOS-FP near-real-time analysis.

## Wind field classes

```{eval-rst}
.. autoclass:: hyplan.winds.WindField
   :members:

.. autoclass:: hyplan.winds.StillAirField
   :members:
   :show-inheritance:

.. autoclass:: hyplan.winds.ConstantWindField
   :members:
   :show-inheritance:

.. autoclass:: hyplan.winds.MERRA2WindField
   :members:
   :show-inheritance:

.. autoclass:: hyplan.winds.GFSWindField
   :members:
   :show-inheritance:

.. autoclass:: hyplan.winds.GMAOWindField
   :members:
   :show-inheritance:
```

## Factory function

```{eval-rst}
.. autofunction:: hyplan.winds.wind_field_from_plan
```
