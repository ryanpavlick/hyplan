# Pattern

{py:class}`~hyplan.pattern.Pattern` is the reusable container that sits
between the flight-pattern generators and campaign/planning workflows.

- Pattern generators such as {py:func}`~hyplan.flight_patterns.racetrack`
  and {py:func}`~hyplan.flight_patterns.spiral` return `Pattern`
  instances.
- Line-based patterns store ordered
  {py:class}`~hyplan.flight_line.FlightLine` objects.
- Waypoint-based patterns store ordered
  {py:class}`~hyplan.waypoint.Waypoint` objects.
- {py:func}`~hyplan.planning.compute_flight_plan` accepts `Pattern`
  objects directly and expands them into their underlying elements.
- {py:class}`~hyplan.campaign.Campaign` assigns stable `pattern_id` and
  `line_id` values when a pattern is added to a campaign.

## Pattern class

```{eval-rst}
.. autoclass:: hyplan.pattern.Pattern
   :members:
   :show-inheritance:
```
