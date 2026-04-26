# Campaign

A {py:class}`~hyplan.campaign.Campaign` organises a study area, airspace data,
flight lines, reusable patterns, and line groups into a single object that can
be saved to and loaded from a plain folder.

## Folder structure

```text
my_campaign/
  campaign.json        # name, version, campaign_id, revision metadata
  domain.geojson       # study area polygon
  airspaces.json       # raw OpenAIP items (re-parsed on load)
  flight_lines/
    all_lines.geojson  # free-standing flight lines only
    groups.json        # logical groupings with generation params
  patterns/
    all_patterns.json  # serialized Pattern objects
```

## What changed in the pattern-aware campaign model

- Campaigns now manage both free-standing lines and first-class
  {py:class}`~hyplan.pattern.Pattern` objects.
- Line-based patterns receive stable campaign-global `line_id` values
  when added to a campaign.
- Campaign mutation bumps `revision` and `updated_at`, which makes the
  object easier to synchronize with interactive clients.
- `patterns_to_geojson()` exposes waypoint-based pattern geometry for map
  display, while `flight_lines_to_geojson()` continues to expose all
  line geometry.

## Common workflow

1. Build a pattern with a generator such as
   {py:func}`~hyplan.flight_patterns.racetrack`.
2. Add it to a campaign with
   {py:meth}`~hyplan.campaign.Campaign.add_pattern`.
3. Edit the pattern or individual legs with
   {py:meth}`~hyplan.campaign.Campaign.replace_pattern`,
   {py:meth}`~hyplan.campaign.Campaign.replace_line_anywhere`, or
   {py:meth}`~hyplan.pattern.Pattern.replace_line`.
4. Save the campaign and use the stored IDs and revision metadata to
   track changes across planning sessions.

## Campaign class

```{eval-rst}
.. autoclass:: hyplan.campaign.Campaign
   :members:
   :show-inheritance:
```
