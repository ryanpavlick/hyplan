# Airspace

Fetch airspace data from [OpenAIP](https://www.openaip.net/) and check flight
lines for conflicts.  Requires a free OpenAIP API key — set the
``OPENAIP_API_KEY`` environment variable or pass it to
{py:class}`~hyplan.airspace.OpenAIPClient`.

## Data model

```{eval-rst}
.. autoclass:: hyplan.airspace.Airspace
.. autoclass:: hyplan.airspace.AirspaceConflict
```

## Conflict detection

```{eval-rst}
.. autofunction:: hyplan.airspace.check_airspace_conflicts
.. autofunction:: hyplan.airspace.fetch_and_check
```

## OpenAIP client

```{eval-rst}
.. autoclass:: hyplan.airspace.OpenAIPClient
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
