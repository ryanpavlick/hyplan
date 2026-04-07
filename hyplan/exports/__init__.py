"""Export flight plans to MovingLines-compatible file formats.

All formats produce output that is drop-in compatible with MovingLines
(github.com/samuelleblanc/fp) so existing flight crew workflows are
preserved.

Typical usage::

    from hyplan import compute_flight_plan, DynamicAviation_B200
    from hyplan.exports import to_pilot_excel, to_foreflight_csv

    plan = compute_flight_plan(aircraft, waypoints, ...)
    to_pilot_excel(plan, "flight_plan_for_pilots.xlsx", aircraft=aircraft)
    to_foreflight_csv(plan, "flight_plan_FOREFLIGHT.csv")

This module is a thin facade — each export format lives in its own
submodule (``excel``, ``csv``, ``icartt``, ``kml``, ``gpx``, ``text``)
and the shared waypoint extraction / formatting helpers live in
``_common``. All public names are re-exported here so existing imports
like ``from hyplan.exports import to_excel`` continue to work.

References
----------
LeBlanc, S.E. (2018). Moving Lines: NASA airborne research flight
planning tool. Zenodo. doi:10.5281/zenodo.1478126
"""

from ._common import extract_waypoints, generate_wp_names
from .excel import to_excel, to_pilot_excel
from .csv import to_foreflight_csv, to_honeywell_fms, to_er2_csv
from .icartt import to_icartt
from .kml import to_kml
from .gpx import to_gpx
from .text import to_txt, to_trackair

__all__ = [
    "extract_waypoints",
    "generate_wp_names",
    "to_excel",
    "to_pilot_excel",
    "to_foreflight_csv",
    "to_honeywell_fms",
    "to_er2_csv",
    "to_icartt",
    "to_kml",
    "to_gpx",
    "to_txt",
    "to_trackair",
]
