"""Inverse Lagrangian targeting: given a desired splash, find a release.

The constraint is the **aircraft trajectory** — we do not search
"release anywhere", only "at what elapsed time along the planned
trajectory should we release so the simulated splash lands closest to
the target."

The solver does a coarse scan via
:meth:`FlightPlanTrack.iter_samples` to bracket the minimum, then a
small in-tree golden-section search refines it.  SciPy is **not** a
HyPlan dependency, so we ship our own ~30-LOC golden-section helper
rather than pulling in ``scipy.optimize.minimize_scalar``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from ...exceptions import HyPlanValueError
from ...units import ureg
from ...waypoint import Waypoint
from ...winds.base import WindField
from .flight_plan_track import AircraftTrackSample, FlightPlanTrack
from .models import DropsondeRelease, DropsondeTrajectory
from .plan import DropsondePlan
from .sensor import AVAPS_NRD41, DropsondeSystem
from .simulate import simulate_release

if TYPE_CHECKING:
    from ...aircraft._base import Aircraft

__all__ = [
    "DropsondeReleaseSolution",
    "golden_section_search",
    "solve_release_for_target",
]


_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """Minimise ``f`` on ``[a, b]`` via golden-section search.

    Returns the argmin.  ``tol`` is the absolute tolerance on the
    bracket width.  Pure-Python, no SciPy dependency.
    """
    if b < a:
        a, b = b, a
    width = b - a
    x1 = b - _GOLDEN * width
    x2 = a + _GOLDEN * width
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(max_iter):
        if (b - a) <= tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - _GOLDEN * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _GOLDEN * (b - a)
            f2 = f(x2)
    return 0.5 * (a + b)


@dataclass(frozen=True, eq=False)
class DropsondeReleaseSolution:
    """Result of an inverse targeting solve."""

    target: Waypoint
    release: DropsondeRelease
    trajectory: DropsondeTrajectory
    miss_distance: Quantity
    feasible: bool
    reason: str | None = None


def _resolve_track(
    flight_plan: FlightPlanTrack | DropsondePlan,
) -> FlightPlanTrack:
    if isinstance(flight_plan, FlightPlanTrack):
        return flight_plan
    if isinstance(flight_plan, DropsondePlan):
        if flight_plan.flight_track is None:
            raise HyPlanValueError(
                "solve_release_for_target requires a DropsondePlan built "
                "from a FlightPlanTrack (got flight_track=None — likely "
                "built from a Pattern or bare releases)."
            )
        return flight_plan.flight_track
    raise HyPlanValueError(
        "flight_plan must be FlightPlanTrack or DropsondePlan, "
        f"got {type(flight_plan).__name__}"
    )


def _sample_to_release(
    sample: AircraftTrackSample,
    *,
    sensor: DropsondeSystem,
    aircraft: "Aircraft | None",
    takeoff_time: _dt.datetime,
    release_id: int = 0,
) -> DropsondeRelease:
    wp = Waypoint(
        latitude=sample.latitude,
        longitude=sample.longitude,
        heading=sample.heading_deg if sample.heading_deg is not None else 0.0,
        altitude_msl=sample.altitude_m * ureg.meter,
        name=f"target_candidate_{sample.elapsed_s:.0f}s",
    )
    ac_vel: tuple[float, float] | None = None
    if sample.heading_deg is not None and sample.groundspeed_mps and sample.groundspeed_mps > 0:
        az = np.radians(float(sample.heading_deg))
        ac_vel = (
            sample.groundspeed_mps * float(np.sin(az)),
            sample.groundspeed_mps * float(np.cos(az)),
        )
    return DropsondeRelease(
        waypoint=wp,
        sensor=sensor,
        aircraft=aircraft,
        release_time=takeoff_time + _dt.timedelta(seconds=sample.elapsed_s),
        aircraft_velocity_mps=ac_vel,
        source=sample.segment_index,
        source_id=sample.segment_index,
        source_segment_type=sample.segment_type,
        release_id=release_id,
    )


def solve_release_for_target(
    target: Waypoint,
    flight_plan: FlightPlanTrack | DropsondePlan,
    *,
    takeoff_time: _dt.datetime,
    sensor: DropsondeSystem = AVAPS_NRD41,
    wind_field: WindField,
    aircraft: "Aircraft | None" = None,
    search_window: tuple[_dt.datetime, _dt.datetime] | None = None,
    segment_types: tuple[str, ...] = ("flight_line", "transit"),
    coarse_step: Quantity = 10 * ureg.second,
    tolerance: Quantity = 100 * ureg.meter,
    max_iter: int = 30,
) -> DropsondeReleaseSolution:
    """Find the release time along ``flight_plan`` whose splash hits ``target``.

    Coarse scan + golden-section refine.  Returns a solution whose
    ``feasible`` flag is ``True`` only when the miss distance is within
    ``tolerance``.
    """
    track = _resolve_track(flight_plan)

    # Map search_window to elapsed-seconds bounds.
    total = track.total_duration_s()
    t_lo, t_hi = 0.0, total
    if search_window is not None:
        sw_lo = (search_window[0] - takeoff_time).total_seconds()
        sw_hi = (search_window[1] - takeoff_time).total_seconds()
        t_lo = max(t_lo, sw_lo)
        t_hi = min(t_hi, sw_hi)
    if t_hi <= t_lo:
        raise HyPlanValueError("search_window has zero or negative duration after clamping")

    tol_m = float(tolerance.m_as("meter"))

    # Helper: simulate a release at elapsed time t and return (miss_m, release, traj).
    def _evaluate(t_elapsed: float) -> tuple[float, DropsondeRelease, DropsondeTrajectory]:
        sample = track.sample_at_elapsed(t_elapsed * ureg.second)
        if sample.segment_type not in segment_types:
            # Penalise out-of-window samples heavily.
            release = _sample_to_release(
                sample, sensor=sensor, aircraft=aircraft, takeoff_time=takeoff_time,
            )
            return (np.inf, release, None)  # type: ignore[return-value]
        release = _sample_to_release(
            sample, sensor=sensor, aircraft=aircraft, takeoff_time=takeoff_time,
        )
        traj = simulate_release(release, wind_field=wind_field)
        miss_m, _ = pymap3d.vincenty.vdist(
            float(target.latitude),
            float(target.longitude),
            float(traj.splash_waypoint.latitude),
            float(traj.splash_waypoint.longitude),
        )
        return (float(miss_m), release, traj)

    # Coarse scan.
    step_s = float(coarse_step.m_as("second"))
    if step_s <= 0:
        raise HyPlanValueError("coarse_step must be positive")
    samples_t: list[float] = []
    t = t_lo
    while t <= t_hi + 1e-6:
        samples_t.append(t)
        t += step_s
    if not samples_t:
        raise HyPlanValueError("coarse scan produced no candidates")

    best_idx = 0
    best_miss = float("inf")
    best_release: DropsondeRelease | None = None
    best_traj: DropsondeTrajectory | None = None
    cache: dict[int, tuple[float, DropsondeRelease, DropsondeTrajectory]] = {}
    for i, t_i in enumerate(samples_t):
        miss, rel, tr = _evaluate(t_i)
        cache[i] = (miss, rel, tr)
        if miss < best_miss:
            best_miss = miss
            best_idx = i
            best_release = rel
            best_traj = tr

    if best_release is None or best_traj is None or not np.isfinite(best_miss):
        return DropsondeReleaseSolution(
            target=target,
            release=DropsondeRelease(waypoint=target, sensor=sensor),
            trajectory=None,  # type: ignore[arg-type]
            miss_distance=float("inf") * ureg.meter,
            feasible=False,
            reason="no feasible release within search_window/segment_types",
        )

    # Refine via golden-section over the bracket around the coarse minimum.
    lo_idx = max(0, best_idx - 1)
    hi_idx = min(len(samples_t) - 1, best_idx + 1)
    lo_t = samples_t[lo_idx]
    hi_t = samples_t[hi_idx]

    def _miss_only(t_elapsed: float) -> float:
        return _evaluate(t_elapsed)[0]

    t_refined = golden_section_search(
        _miss_only, lo_t, hi_t, tol=max(0.1, step_s / 100.0), max_iter=max_iter,
    )
    miss, rel, traj = _evaluate(t_refined)
    if miss > best_miss:
        # Refined point was worse; stick with the coarse best.
        miss = best_miss
        rel = best_release
        traj = best_traj

    return DropsondeReleaseSolution(
        target=target,
        release=rel,
        trajectory=traj,
        miss_distance=miss * ureg.meter,
        feasible=bool(miss <= tol_m),
        reason=(
            None if miss <= tol_m
            else "aircraft trajectory cannot place splash within tolerance"
        ),
    )
