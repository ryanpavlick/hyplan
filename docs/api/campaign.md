# Campaign

A {py:class}`~hyplan.campaign.Campaign` organises a study area, airspace data,
flight lines, and line groups into a single object that can be saved to and
loaded from a plain folder.

## Folder structure

```text
my_campaign/
  campaign.json        # name, bounds, country, version
  domain.geojson       # study area polygon
  airspaces.json       # raw OpenAIP items (re-parsed on load)
  flight_lines/
    all_lines.geojson  # master library of flight lines
    groups.json        # logical groupings with generation params
```

## Campaign class

```{eval-rst}
.. autoclass:: hyplan.campaign.Campaign
   :members:
   :show-inheritance:
```
