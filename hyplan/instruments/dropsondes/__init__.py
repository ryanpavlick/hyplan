"""Dropsonde planning — first-class HyPlan objects.

.. warning::

   **Experimental module (v1.9).**  The public API of
   ``hyplan.instruments.dropsondes`` is provisional and may change in
   subsequent releases as it is validated against operational use.
   Pin the HyPlan version if you depend on the current surface; track
   release notes for breaking changes.

This sub-package contains:

- :class:`DropsondeSystem` + reference instances :data:`AVAPS_NRD41` /
  :data:`RD94` (shared singletons; build a new ``DropsondeSystem``
  rather than mutating the references).
- :class:`DropsondeRelease` — a frozen, identity-equal record of one
  planned release event with provenance and tri-state QC flags.
- :class:`DropsondeTrajectory` — one simulated descent.
- :class:`DropsondePlan` — collection of releases (and their
  simulations) with manifest / trajectory / summary exports.
- :class:`FlightPlanTrack` + :class:`AircraftTrackSample` — typed
  adapter over the ``compute_flight_plan`` GeoDataFrame, with the
  canonical aircraft-trajectory sampler used by both release planning
  and inverse targeting.
- :func:`simulate_release` — object-aware descent wrapper.
- :func:`simulate_descent_trajectory` — the low-level RK4 kernel.
- :func:`releases_along_flight_line` — emit releases along a single
  :class:`hyplan.flight_line.FlightLine` (no computed plan required).
- :func:`solve_release_for_target` — inverse Lagrangian targeting.

The dropsonde abstraction is intentionally distinct from HyPlan's
swath sensors (line scanners, LVIS, lidar, AWP): a release is an
**event**, not a continuous strip.  See :doc:`/api/dropsonde` for the
API tour.
"""

from .flight_plan_track import AircraftTrackSample, FlightPlanTrack, PlannedSegment
from .models import DropsondeRelease, DropsondeTrajectory
from .plan import DropsondePlan, summarize_trajectories
from .planning import releases_along_flight_line, releases_along_segment
from .sensor import (
    AVAPS_NRD41,
    AXCTD,
    RD94,
    DropsondeSystem,
    terminal_velocity_nrd41,
    terminal_velocity_sippican_axctd,
)
from .simulate import simulate_descent_trajectory, simulate_release
from .targeting import (
    DropsondeReleaseSolution,
    golden_section_search,
    solve_release_for_target,
)

__all__ = [
    "AVAPS_NRD41",
    "AXCTD",
    "RD94",
    "AircraftTrackSample",
    "DropsondePlan",
    "DropsondeRelease",
    "DropsondeReleaseSolution",
    "DropsondeSystem",
    "DropsondeTrajectory",
    "FlightPlanTrack",
    "PlannedSegment",
    "golden_section_search",
    "releases_along_flight_line",
    "releases_along_segment",
    "simulate_descent_trajectory",
    "simulate_release",
    "solve_release_for_target",
    "summarize_trajectories",
    "terminal_velocity_nrd41",
    "terminal_velocity_sippican_axctd",
]
