# Airspace

Fetch airspace data from [OpenAIP](https://www.openaip.net/), the FAA NASR
database, FAA TFR feed, and FlightPlanDB oceanic-track service, and check
flight lines for conflicts. OpenAIP requires a free API key — set the
``OPENAIP_API_KEY`` environment variable or pass it to
{py:class}`~hyplan.airspace.OpenAIPClient`.

## Data model

```{eval-rst}
.. autoclass:: hyplan.airspace.Airspace
.. autoclass:: hyplan.airspace.AirspaceConflict
.. autoclass:: hyplan.airspace.OceanicTrack
```

## Conflict detection

```{eval-rst}
.. autofunction:: hyplan.airspace.check_airspace_conflicts
.. autofunction:: hyplan.airspace.check_airspace_proximity
.. autofunction:: hyplan.airspace.fetch_and_check
```

## Severity and schedule helpers

```{eval-rst}
.. autofunction:: hyplan.airspace.classify_severity
.. autofunction:: hyplan.airspace.filter_by_schedule
.. autofunction:: hyplan.airspace.convert_agl_floors
.. autofunction:: hyplan.airspace.summarize_airspaces
```

## Data sources

```{eval-rst}
.. autoclass:: hyplan.airspace.OpenAIPClient
   :members:

.. autoclass:: hyplan.airspace.FAATFRClient
   :members:

.. autoclass:: hyplan.airspace.NASRAirspaceSource
   :members:

.. autoclass:: hyplan.airspace.FlightPlanDBClient
   :members:
```

## Cache management

```{eval-rst}
.. autofunction:: hyplan.airspace.clear_airspace_cache
```

## Parsing helpers

```{eval-rst}
.. autofunction:: hyplan.airspace.parse_airspace_items
```
