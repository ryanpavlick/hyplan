# Flight Patterns

Generate common flight patterns for use with
{py:func}`~hyplan.planning.compute_flight_plan`.

All public generators return a
{py:class}`~hyplan.pattern.Pattern`, not a bare list. This lets HyPlan
carry generator parameters, regenerate patterns after edits, persist
them inside a {py:class}`~hyplan.campaign.Campaign`, and round-trip them
through GeoJSON/JSON-friendly structures used by interactive planning
tools.

See also: [Pattern](pattern.md)

## Pattern generators

```{eval-rst}
.. autofunction:: hyplan.flight_patterns.racetrack
.. autofunction:: hyplan.flight_patterns.rosette
.. autofunction:: hyplan.flight_patterns.sawtooth
.. autofunction:: hyplan.flight_patterns.spiral
.. autofunction:: hyplan.flight_patterns.polygon
.. autofunction:: hyplan.flight_patterns.glint_arc
```

## Multi-aircraft coordination

```{eval-rst}
.. autofunction:: hyplan.flight_patterns.coordinated_line
```

## Utilities

```{eval-rst}
.. autofunction:: hyplan.flight_patterns.flight_lines_to_waypoint_path
```
