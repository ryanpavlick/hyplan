"""Wind-aware isochrones for research mission planning.

Given a starting state, a time budget, an optional return destination, and a
wind field, return the boundary of all reachable target points.  Three modes:

* ``"one_way"`` — "Where can I be after ``budget``?"  Single-leg reach.
* ``"return_safe"`` — "Where can I go *and* still get to ``return_destination``
  within ``budget``?"  The operational science envelope.
* ``"round_trip"`` — degenerate case where ``return_destination == start``;
  out-and-back from the same place.

v1 ships **direct great-circle reachability**: each leg is an arc with
bearing-aligned endpoint headings.  No Dubins turn overhead is added.
Heading-constrained Dubins reachability is reserved for a follow-up.

The core algorithm sweeps 360° in azimuth steps (default 5°), and for each
ray uses an expanding bracket + binary search to find the maximum reachable
distance along that ray such that the round-trip leg-time constraint holds.
Each leg is timed by the private :func:`_leg_time` helper, which integrates
:meth:`Aircraft._climb`, the cruise speed schedule, and
:meth:`Aircraft._descend` directly with along-track wind from the supplied
``WindField``.

Ray sampling
------------

The default ``ray_strategy="uniform"`` preserves the original fixed-azimuth
sweep, which is convenient when consumers index rows by cardinal bearings.
For distinct recovery fields, ``ray_strategy="auto"`` uses a two-focus
ellipse approximation (start and recovery are the foci) to choose a better
first-pass azimuth distribution before the real leg-time solver runs.
``"adaptive"`` and ``"ellipse_adaptive"`` additionally insert midpoint rays
where solved boundary chords are longer than ``adaptive_spacing_nmi``.

Wind sampling
-------------

The public functions accept three kwargs that select how wind is
sampled per leg:

* ``wind_sampling="cruise_midpoint"`` (default) — one wind sample at
  the cruise-segment geographic midpoint, climb and descent still-air.
  Strict v1.5 behavior; cheapest.
* ``wind_sampling="phase_midpoint"`` — adds one wind sample each at
  the climb-segment-mid and descent-segment-mid altitudes (projected
  onto the great-circle bearing and passed into
  ``Aircraft._climb`` / ``_descend`` as ``wind_along_track``).  Wind
  changes the *ground distance* allocated to climb and descent, not
  the climb/descent times themselves (climb rate is air-mass-relative
  and independent of along-track wind).  Cruise wind sampled once at
  the cruise midpoint; the net effect on total leg time arrives
  through the cruise term, which sees a different
  ``cruise_distance_nmi``.
* ``wind_sampling="segmented_cruise"`` — same climb/descent treatment
  as ``phase_midpoint`` plus splits cruise into N subsegments
  (``min(max_wind_samples_per_leg, ceil(cruise_distance / wind_sample_spacing))``),
  each with its own midpoint wind sample at a cumulative time anchor.
  Recommended for transit legs > ~500 nmi or strong wind gradients
  (jet streams, fronts).  Two-pass fixed-point on per-subsegment
  times.

Limitations
-----------

* Climb / descent times remain wind-independent in all modes.  Wind
  only redistributes ground distance between phases.  True vertical-
  wind integration during climb (where stronger winds aloft would
  reduce climb-rate effectiveness) requires modifying
  ``Aircraft._climb_with_wind``'s integrator and is out of scope.
* Cruise sampling within ``segmented_cruise`` uses uniform spacing.
  Adaptive segmentation (denser samples where the wind gradient is
  high) is a follow-up.
* The reported per-ray ``outbound_headwind_kt`` /
  ``return_headwind_kt`` is the distance-weighted average across
  cruise subsegments; per-segment headwinds are not surfaced in the
  GeoDataFrame schema.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..aircraft._base import Aircraft
from ..aircraft.wind_path import (
    climb_with_wind_field,
    descend_with_wind_field,
)
from ..airports import Airport
from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from ..geometry import wrap_to_180
from ..units import ureg
from ..waypoint import Waypoint
from ..winds.base import WindField
from ..winds.simple import ConstantWindField, StillAirField
from ..winds.utils import _track_hold_solution_from_uv

__all__ = [
    "compute_concentric_isochrones",
    "compute_isochrone",
    "compute_multi_base_isochrone",
    "compute_multi_refuel_isochrone",
    "compute_refuel_isochrone",
    "evaluate_target_reachability",
    "isochrone_polygon",
    "plot_isochrone",
]


_VALID_MODES = ("one_way", "round_trip", "return_safe")
_VALID_REFUEL_MODES = ("round_trip", "return_safe")
_VALID_WIND_SAMPLING = ("cruise_midpoint", "phase_midpoint", "segmented_cruise")
_VALID_RAY_STRATEGIES = (
    "uniform",
    "auto",
    "ellipse",
    "adaptive",
    "ellipse_adaptive",
)
_DEFAULT_WIND_SAMPLE_SPACING_NMI = 100.0
_DEFAULT_MAX_WIND_SAMPLES = 20
_DEFAULT_ADAPTIVE_SPACING_NMI = 100.0
_DEFAULT_MAX_ADAPTIVE_RAYS = 180


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _validate_common_kwargs(
    *,
    start: Airport | Waypoint,
    cruise_altitude: Quantity | None,
    on_station_altitude: Quantity | None,
    on_station_time: Quantity,
    reserve: Quantity,
    mode: str,
    valid_modes: tuple[str, ...],
    azimuth_resolution_deg: float,
    distance_tolerance_nmi: float,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
    ray_strategy: str = "uniform",
    adaptive_spacing_nmi: float | None = None,
    max_adaptive_rays: int = _DEFAULT_MAX_ADAPTIVE_RAYS,
) -> tuple[Waypoint, Quantity, float, float]:
    """Validate kwargs common to ``compute_isochrone`` and
    ``compute_refuel_isochrone``.

    Returns ``(start_wp, cruise_altitude, reserve_min, on_station_min)`` —
    the coerced start Waypoint, the resolved cruise altitude (defaulting
    to ``start.altitude_msl`` when not provided), and the reserve /
    on-station times in minutes.

    Budget-specific validation (e.g., ``budget > reserve``) stays in the
    individual public functions because the budget surfaces differ
    (``budget`` vs ``sortie_budget`` + ``flight_day_budget``).
    """
    if mode not in valid_modes:
        raise HyPlanValueError(
            f"Invalid mode {mode!r}; must be one of {valid_modes}."
        )

    # Coerce Airport → Waypoint.
    start_wp = (
        _airport_or_wp_to_waypoint(start)
        if isinstance(start, Airport)
        else start
    )
    if start_wp.altitude_msl is None:
        raise HyPlanValueError(
            "`start.altitude_msl` is required (start altitude drives "
            "climb/descent dispatch and wind sampling)."
        )

    if azimuth_resolution_deg <= 0 or azimuth_resolution_deg > 120:
        raise HyPlanValueError(
            f"azimuth_resolution_deg must be in (0, 120] (need at "
            f"least 3 rays for a polygon boundary), got "
            f"{azimuth_resolution_deg}."
        )
    if distance_tolerance_nmi <= 0:
        raise HyPlanValueError(
            f"distance_tolerance_nmi must be positive, got "
            f"{distance_tolerance_nmi}."
        )

    reserve_min = reserve.m_as(ureg.minute)
    on_station_min = on_station_time.m_as(ureg.minute)
    if reserve_min < 0:
        raise HyPlanValueError(
            f"`reserve` must be non-negative, got {reserve_min:.1f} min."
        )
    if on_station_min < 0:
        raise HyPlanValueError(
            f"`on_station_time` must be non-negative, got "
            f"{on_station_min:.1f} min."
        )

    if cruise_altitude is None:
        cruise_altitude = start_wp.altitude_msl

    if cruise_altitude.m_as(ureg.meter) < start_wp.altitude_msl.m_as(ureg.meter):
        raise HyPlanValueError(
            f"v1 requires `cruise_altitude` "
            f"({cruise_altitude.m_as(ureg.feet):.0f} ft) to be at or "
            f"above `start.altitude_msl` "
            f"({start_wp.altitude_msl.m_as(ureg.feet):.0f} ft).  Initial "
            f"descent to cruise is deferred — for now, set "
            f"`cruise_altitude` to the current altitude or higher."
        )

    if on_station_altitude is not None and not np.isclose(
        on_station_altitude.m_as(ureg.meter),
        cruise_altitude.m_as(ureg.meter),
    ):
        raise HyPlanValueError(
            "v1 requires `on_station_altitude` to equal "
            "`cruise_altitude` (or be None).  Multi-altitude "
            "isochrones are deferred."
        )

    if wind_sampling not in _VALID_WIND_SAMPLING:
        raise HyPlanValueError(
            f"Invalid wind_sampling {wind_sampling!r}; must be one of "
            f"{_VALID_WIND_SAMPLING}."
        )
    if wind_sample_spacing.m_as(ureg.nautical_mile) <= 0:
        raise HyPlanValueError(
            f"wind_sample_spacing must be positive, got "
            f"{wind_sample_spacing}."
        )
    if max_wind_samples_per_leg < 1:
        raise HyPlanValueError(
            f"max_wind_samples_per_leg must be >= 1, got "
            f"{max_wind_samples_per_leg}."
        )

    if ray_strategy not in _VALID_RAY_STRATEGIES:
        raise HyPlanValueError(
            f"Invalid ray_strategy {ray_strategy!r}; must be one of "
            f"{_VALID_RAY_STRATEGIES}."
        )
    if adaptive_spacing_nmi is not None and adaptive_spacing_nmi <= 0:
        raise HyPlanValueError(
            f"adaptive_spacing_nmi must be positive when provided, got "
            f"{adaptive_spacing_nmi}."
        )
    if max_adaptive_rays < 3:
        raise HyPlanValueError(
            f"max_adaptive_rays must be >= 3, got {max_adaptive_rays}."
        )

    return start_wp, cruise_altitude, reserve_min, on_station_min


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_isochrone(
    aircraft: Aircraft,
    start: Airport | Waypoint,
    budget: Quantity,
    *,
    cruise_altitude: Quantity | None = None,
    on_station_altitude: Quantity | None = None,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    return_destination: Airport | Waypoint | None = None,
    mode: str = "round_trip",
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
    ray_strategy: str = "uniform",
    adaptive_spacing_nmi: float | None = None,
    max_adaptive_rays: int = _DEFAULT_MAX_ADAPTIVE_RAYS,
) -> gpd.GeoDataFrame:
    """Compute a wind-aware isochrone around ``start``.

    Args:
        aircraft: The aircraft model.  Climb / cruise / descent profiles
            and speed schedules are read from this object.
        start: Starting state.  Accepts either an :class:`Airport`
            (resolved to a Waypoint at runway elevation, heading 0°)
            or a :class:`Waypoint` directly.  When a Waypoint is
            provided, ``altitude_msl`` is required.  Pass an Airport
            for pre-flight planning; pass a Waypoint with a non-runway
            altitude for in-flight re-tasking.
        budget: Total time available (e.g. endurance remaining).
        cruise_altitude: Transit / observation altitude.  Defaults to
            ``start.altitude_msl``.  v1 treats this as the single
            altitude for both the outbound transit and on-station phase.
        on_station_altitude: Reserved for future multi-altitude support.
            v1 requires this to be ``None`` or equal to ``cruise_altitude``;
            a distinct value raises :class:`HyPlanValueError`.
        start_time: UTC datetime when the sortie begins.  Used to query
            time-varying wind providers.  Defaults to ``datetime.now(UTC)``.
        wind_source: A :class:`WindField` provider.  Defaults to
            :class:`StillAirField`.
        return_destination: Where the aircraft must reach by
            ``budget − reserve``.  Required for ``"return_safe"``;
            defaults to ``start`` for ``"round_trip"``; ignored
            (with a warning) for ``"one_way"``.  Accepts an
            :class:`Airport` (resolved to its runway-elevation Waypoint)
            or a :class:`Waypoint`.
        mode: ``"one_way"`` | ``"round_trip"`` | ``"return_safe"``.
        on_station_time: Required dwell at the target.
        reserve: Mandatory unflown time buffer (fuel/reserve fuel).
        azimuth_resolution_deg: Step in degrees between sweep rays.
            Default 5.0 → 72 rays.
        distance_tolerance_nmi: Binary-search stop criterion (boundary
            distance precision in nmi).
        wind_sampling: How wind is sampled per leg —
            ``"cruise_midpoint"`` (default; one sample at cruise
            midpoint, climb/descent still-air),
            ``"phase_midpoint"`` (adds one wind sample each at the
            climb-segment-mid and descent-segment-mid altitudes,
            redistributing ground distance between phases), or
            ``"segmented_cruise"`` (splits cruise into multiple
            subsegments, ~``wind_sample_spacing`` apart).  See the
            module docstring's "Wind sampling" section.
        wind_sample_spacing: Cruise-segment spacing for
            ``wind_sampling="segmented_cruise"`` (ignored otherwise).
            Default 100 nmi.
        max_wind_samples_per_leg: Hard cap on cruise subsegments to
            keep ``_leg_time`` cost bounded for very long legs.
            Default 20.
        ray_strategy: How sweep azimuths are chosen. ``"uniform"``
            preserves the original fixed-angle sweep. ``"auto"`` uses
            ellipse-aware seed rays when start and recovery differ,
            otherwise uniform rays. ``"adaptive"`` adds midpoint rays
            where solved boundary chords exceed ``adaptive_spacing_nmi``;
            ``"ellipse"`` and ``"ellipse_adaptive"`` force the
            two-focus seed distribution when possible.
        adaptive_spacing_nmi: Target maximum boundary chord length for
            adaptive refinement. Defaults to 100 nmi when an adaptive
            strategy is selected.
        max_adaptive_rays: Hard cap on adaptive ray count.

    Returns:
        A :class:`geopandas.GeoDataFrame` in ``EPSG:4326``, one row per
        ray, with ``Point`` geometries on the isochrone boundary.  See
        the module docstring for the column schema.  Invocation context
        is stashed in ``gdf.attrs``.

    Raises:
        HyPlanValueError: For invalid arguments (see input validation in
            the plan / tests).
        HyPlanRuntimeError: If the expanding bracket exceeds 20000 nmi
            (pathological wind field or aircraft).
    """
    # --- input validation ---------------------------------------------------
    start, cruise_altitude, reserve_min, on_station_min = _validate_common_kwargs(
        start=start,
        cruise_altitude=cruise_altitude,
        on_station_altitude=on_station_altitude,
        on_station_time=on_station_time,
        reserve=reserve,
        mode=mode,
        valid_modes=_VALID_MODES,
        azimuth_resolution_deg=azimuth_resolution_deg,
        distance_tolerance_nmi=distance_tolerance_nmi,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
        ray_strategy=ray_strategy,
        adaptive_spacing_nmi=adaptive_spacing_nmi,
        max_adaptive_rays=max_adaptive_rays,
    )

    budget_min = budget.m_as(ureg.minute)
    if budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`budget` ({budget_min:.1f} min) must exceed `reserve` "
            f"({reserve_min:.1f} min); otherwise no time is available "
            f"to fly."
        )

    # --- return destination resolution + mode-specific rules ---------------
    return_wp: Waypoint | None
    return_label: str
    if mode == "one_way":
        if return_destination is not None:
            import warnings
            warnings.warn(
                "`return_destination` is ignored when mode='one_way'.",
                stacklevel=2,
            )
        return_wp = None
        return_label = "—"
    elif mode == "round_trip":
        if return_destination is None:
            return_wp = start
            return_label = start.name or "start"
        else:
            return_wp = _resolve_destination(return_destination)
            return_label = _destination_label(return_destination)
    else:  # return_safe
        if return_destination is None:
            raise HyPlanValueError(
                "`return_destination` is required when mode='return_safe'."
            )
        return_wp = _resolve_destination(return_destination)
        return_label = _destination_label(return_destination)

    if start_time is None:
        start_time = datetime.datetime.now(datetime.timezone.utc)
    if wind_source is None:
        wind_source = StillAirField()

    # --- sweep --------------------------------------------------------------
    azimuths, effective_ray_strategy = _initial_ray_azimuths(
        aircraft=aircraft,
        start=start,
        cruise_altitude=cruise_altitude,
        return_wp=return_wp,
        mode=mode,
        on_station_min=on_station_min,
        budget_min=budget_min,
        reserve_min=reserve_min,
        azimuth_resolution_deg=azimuth_resolution_deg,
        ray_strategy=ray_strategy,
    )
    rows = _solve_rays_with_strategy(
        aircraft=aircraft,
        start=start,
        cruise_altitude=cruise_altitude,
        start_time=start_time,
        wind_source=wind_source,
        return_wp=return_wp,
        mode=mode,
        on_station_min=on_station_min,
        budget_min=budget_min,
        reserve_min=reserve_min,
        azimuths_deg=azimuths,
        distance_tolerance_nmi=distance_tolerance_nmi,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
        effective_ray_strategy=effective_ray_strategy,
        adaptive_spacing_nmi=adaptive_spacing_nmi,
        max_adaptive_rays=max_adaptive_rays,
    )

    df = pd.DataFrame(rows)
    geometry = [Point(lon, lat) for lat, lon in zip(df["target_lat"], df["target_lon"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Stash invocation context for the plotter / consumers.
    gdf.attrs.update({
        "mode": mode,
        "start_lat": start.latitude,
        "start_lon": start.longitude,
        "start_altitude_ft": (
            start.altitude_msl.m_as(ureg.feet)
            if start.altitude_msl is not None else None
        ),
        "cruise_altitude_ft": cruise_altitude.m_as(ureg.feet),
        "budget_min": budget_min,
        "budget_hr": budget_min / 60.0,
        "reserve_min": reserve_min,
        "on_station_min": on_station_min,
        "return_destination_label": return_label,
        "return_destination_lat": (
            return_wp.latitude if return_wp is not None else None
        ),
        "return_destination_lon": (
            return_wp.longitude if return_wp is not None else None
        ),
        "aircraft_type": aircraft.aircraft_type,
        "wind_source_kind": type(wind_source).__name__,
        "start_time": start_time.isoformat(),
        "ray_strategy": ray_strategy,
        "effective_ray_strategy": effective_ray_strategy,
        "n_rays": len(gdf),
    })
    return gdf


def compute_concentric_isochrones(
    aircraft: Aircraft,
    start: Airport | Waypoint,
    budgets: Sequence[Quantity],
    *,
    cruise_altitude: Quantity | None = None,
    on_station_altitude: Quantity | None = None,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    return_destination: Airport | Waypoint | None = None,
    mode: str = "round_trip",
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
    ray_strategy: str = "uniform",
    adaptive_spacing_nmi: float | None = None,
    max_adaptive_rays: int = _DEFAULT_MAX_ADAPTIVE_RAYS,
) -> gpd.GeoDataFrame:
    """Compute multiple isochrone contours in one call (e.g., 1/2/3 hr).

    Sweeps ``budgets`` in ascending order and seeds each budget's
    bracket search from the previous (smaller) budget's converged
    distances per ray, exploiting the fact that feasibility is
    monotone in budget.  Total cost is roughly ``O(M + N)`` rather
    than ``O(M·N)`` for ``M`` budgets and ``N`` rays.

    Args:
        aircraft, start, cruise_altitude, on_station_altitude,
        start_time, wind_source, return_destination, mode,
        on_station_time, reserve, azimuth_resolution_deg,
        distance_tolerance_nmi, wind_sampling, wind_sample_spacing,
        max_wind_samples_per_leg, ray_strategy,
        adaptive_spacing_nmi, max_adaptive_rays: same semantics as
            :func:`compute_isochrone`.
        budgets: iterable of ``Quantity`` time values to sweep.
            Must be non-empty; sorted ascending internally.

    Returns:
        A :class:`geopandas.GeoDataFrame` in EPSG:4326, one row per
        ``(budget, azimuth)`` combination.  Columns match
        :func:`compute_isochrone` plus ``budget_min`` and
        ``budget_hr`` tagging which contour each row belongs to.
        ``gdf.attrs["budgets_hr"]`` lists budgets in computed order
        (ascending).

    Refuel-aware concentric reach is intentionally out of scope —
    sweeping ``(sortie, day)`` tuples is a different UI problem.
    """
    if not budgets:
        raise HyPlanValueError(
            "`budgets` must contain at least one Quantity."
        )

    # Reuse compute_isochrone's validation by validating each budget
    # via _validate_common_kwargs once; this also coerces start to a
    # Waypoint and resolves cruise_altitude defaults.
    start, cruise_altitude, reserve_min, on_station_min = _validate_common_kwargs(
        start=start,
        cruise_altitude=cruise_altitude,
        on_station_altitude=on_station_altitude,
        on_station_time=on_station_time,
        reserve=reserve,
        mode=mode,
        valid_modes=_VALID_MODES,
        azimuth_resolution_deg=azimuth_resolution_deg,
        distance_tolerance_nmi=distance_tolerance_nmi,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
        ray_strategy=ray_strategy,
        adaptive_spacing_nmi=adaptive_spacing_nmi,
        max_adaptive_rays=max_adaptive_rays,
    )

    budgets_min = sorted(b.m_as(ureg.minute) for b in budgets)
    smallest = budgets_min[0]
    if smallest <= reserve_min:
        raise HyPlanValueError(
            f"smallest budget ({smallest:.1f} min) must exceed reserve "
            f"({reserve_min:.1f} min)."
        )

    # Resolve recovery destination once (shared across budgets).
    if mode == "one_way":
        if return_destination is not None:
            import warnings
            warnings.warn(
                "`return_destination` is ignored when mode='one_way'.",
                stacklevel=2,
            )
        return_wp: Waypoint | None = None
        return_label = "—"
    elif mode == "round_trip":
        if return_destination is None:
            return_wp = start
            return_label = start.name or "start"
        else:
            return_wp = _resolve_destination(return_destination)
            return_label = _destination_label(return_destination)
    else:  # return_safe
        if return_destination is None:
            raise HyPlanValueError(
                "`return_destination` is required when mode='return_safe'."
            )
        return_wp = _resolve_destination(return_destination)
        return_label = _destination_label(return_destination)

    if start_time is None:
        start_time = datetime.datetime.now(datetime.timezone.utc)
    if wind_source is None:
        wind_source = StillAirField()

    azimuths, effective_ray_strategy = _initial_ray_azimuths(
        aircraft=aircraft,
        start=start,
        cruise_altitude=cruise_altitude,
        return_wp=return_wp,
        mode=mode,
        on_station_min=on_station_min,
        budget_min=budgets_min[-1],
        reserve_min=reserve_min,
        azimuth_resolution_deg=azimuth_resolution_deg,
        ray_strategy=ray_strategy,
    )
    n_rays = len(azimuths)

    all_rows: list[dict[str, Any]] = []
    seed_d_lo = np.zeros(n_rays, dtype=float)

    for budget_min in budgets_min:
        rows = _solve_rays_with_strategy(
            aircraft=aircraft,
            start=start,
            cruise_altitude=cruise_altitude,
            start_time=start_time,
            wind_source=wind_source,
            return_wp=return_wp,
            mode=mode,
            on_station_min=on_station_min,
            budget_min=budget_min,
            reserve_min=reserve_min,
            azimuths_deg=azimuths,
            distance_tolerance_nmi=distance_tolerance_nmi,
            seed_d_lo=seed_d_lo,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
            effective_ray_strategy=effective_ray_strategy,
            adaptive_spacing_nmi=adaptive_spacing_nmi,
            max_adaptive_rays=max_adaptive_rays,
        )
        # Tag rows with budget; capture per-ray distances for next seed.
        # Adaptive strategies may add rays, so carry the refined azimuth set
        # forward to larger budgets.
        rows = sorted(rows, key=lambda r: r["azimuth_deg"])
        azimuths = np.array([row["azimuth_deg"] for row in rows], dtype=float)
        n_rays = len(azimuths)
        new_seed = np.zeros(n_rays, dtype=float)
        for i, row in enumerate(rows):
            row["budget_min"] = budget_min
            row["budget_hr"] = budget_min / 60.0
            new_seed[i] = float(row["distance_nmi"])
            all_rows.append(row)
        seed_d_lo = new_seed

    df = pd.DataFrame(all_rows)
    geometry = [Point(lon, lat) for lat, lon in zip(df["target_lat"], df["target_lon"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf.attrs.update({
        "mode": mode,
        "start_lat": start.latitude,
        "start_lon": start.longitude,
        "start_altitude_ft": (
            start.altitude_msl.m_as(ureg.feet)
            if start.altitude_msl is not None else None
        ),
        "cruise_altitude_ft": cruise_altitude.m_as(ureg.feet),
        "budgets_min": budgets_min,
        "budgets_hr": [b / 60.0 for b in budgets_min],
        "reserve_min": reserve_min,
        "on_station_min": on_station_min,
        "return_destination_label": return_label,
        "return_destination_lat": (
            return_wp.latitude if return_wp is not None else None
        ),
        "return_destination_lon": (
            return_wp.longitude if return_wp is not None else None
        ),
        "aircraft_type": aircraft.aircraft_type,
        "wind_source_kind": type(wind_source).__name__,
        "start_time": start_time.isoformat(),
        "ray_strategy": ray_strategy,
        "effective_ray_strategy": effective_ray_strategy,
        "n_rays": n_rays,
    })
    return gdf


def compute_refuel_isochrone(
    aircraft: Aircraft,
    start: Airport | Waypoint,
    sortie_budget: Quantity,
    *,
    flight_day_budget: Quantity,
    cruise_altitude: Quantity | None = None,
    refuel_airports: Sequence[Airport | Waypoint],
    refuel_time: Quantity = 60 * ureg.minute,
    return_destination: Airport | Waypoint | None = None,
    mode: str = "return_safe",
    on_station_altitude: Quantity | None = None,
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    max_refuel_stops: int = 1,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> gpd.GeoDataFrame:
    """Wind-aware isochrone with a single optional refuel stop.

    Two clocks are tracked: ``sortie_budget`` per fuel cycle (resets after
    each refuel) and ``flight_day_budget`` total wall-clock (does not
    reset, includes refuel time).  ``reserve`` applies *per fuel cycle*,
    not to the day budget.

    For every azimuth, three itinerary templates are evaluated and the
    one reaching the largest distance wins:

    * ``direct`` — ``start → target → recovery``
    * ``outbound_refuel(R)`` — ``start → R → target → recovery``
    * ``return_refuel(R)`` — ``start → target → R → recovery``

    Each refuel airport in ``refuel_airports`` is tested in both
    placements (where eligible).  v1 limits ``max_refuel_stops`` to 1 —
    a single sortie touches at most two tanks.

    Args:
        aircraft: Aircraft model.
        start: Departure point (Airport or Waypoint).
        sortie_budget: Per-fuel-cycle endurance.  Positional to mirror
            ``compute_isochrone(..., budget=...)``.
        flight_day_budget: Total wall-clock budget (kw-only — the new
            required clock).
        refuel_airports: Pre-cleared refuel candidates.  Empty raises;
            use ``compute_isochrone`` if no refuel option applies.
        refuel_time: Per-refuel wall-clock cost.  Default 60 min.
        return_destination: Recovery field.  Required for
            ``"return_safe"``; defaults to ``start`` for
            ``"round_trip"``.
        mode: ``"return_safe"`` (default) or ``"round_trip"``.
            ``"one_way"`` is rejected — multi-hop one-way reach is
            deferred.
        max_refuel_stops: v1 must be ``1``.  Reserved for future chained
            refuels.
        Other kwargs match ``compute_isochrone``.

    Returns:
        A GeoDataFrame with one row per ray.  Per-leg time columns
        (``start_to_target_time_min``, ``start_to_refuel_time_min``,
        ``refuel_to_target_time_min``, ``target_to_refuel_time_min``,
        ``refuel_to_return_time_min``, ``target_to_return_time_min``) are
        NaN unless the itinerary uses that leg.  ``itinerary``,
        ``refuel_airport``, ``refuel_count``, ``day_total_time_min``,
        ``sortie_cycle_1_min``, ``sortie_cycle_2_min``,
        ``sortie_margin_min``, ``day_margin_min``,
        ``limiting_leg`` describe the chosen path.  ``limiting_leg``
        values for refuel results are ``"sortie"`` | ``"flight_day"`` |
        ``"both"`` | ``"slack"`` | ``"unflyable"`` (a different value
        vocabulary than ``compute_isochrone``'s, but the column name is
        shared so consumers holding both gdfs use the same accessor).

        ``gdf.attrs`` includes ``refuel_airports_evaluated``,
        ``refuel_airports_unreachable``, and ``refuel_airports_used``.
    """
    import warnings

    # --- input validation --------------------------------------------------
    start, cruise_altitude, reserve_min, on_station_min = _validate_common_kwargs(
        start=start,
        cruise_altitude=cruise_altitude,
        on_station_altitude=on_station_altitude,
        on_station_time=on_station_time,
        reserve=reserve,
        mode=mode,
        valid_modes=_VALID_REFUEL_MODES,
        azimuth_resolution_deg=azimuth_resolution_deg,
        distance_tolerance_nmi=distance_tolerance_nmi,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    if max_refuel_stops != 1:
        raise HyPlanValueError(
            f"v1 of compute_refuel_isochrone requires max_refuel_stops=1, "
            f"got {max_refuel_stops}.  Chained refuels are deferred."
        )

    if not refuel_airports:
        raise HyPlanValueError(
            "`refuel_airports` is empty; use `compute_isochrone` when no "
            "refuel option applies."
        )

    sortie_budget_min = sortie_budget.m_as(ureg.minute)
    flight_day_budget_min = flight_day_budget.m_as(ureg.minute)
    refuel_time_min = refuel_time.m_as(ureg.minute)

    if refuel_time_min < 0:
        raise HyPlanValueError(
            f"`refuel_time` must be non-negative, got "
            f"{refuel_time_min:.1f} min."
        )
    if sortie_budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`sortie_budget` ({sortie_budget_min:.1f} min) must exceed "
            f"`reserve` ({reserve_min:.1f} min)."
        )
    if flight_day_budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`flight_day_budget` ({flight_day_budget_min:.1f} min) must "
            f"exceed `reserve` ({reserve_min:.1f} min)."
        )

    if flight_day_budget_min < sortie_budget_min:
        warnings.warn(
            f"flight_day_budget ({flight_day_budget_min:.1f} min) is "
            f"less than sortie_budget ({sortie_budget_min:.1f} min); the "
            f"day clock will bind the boundary instead of the sortie clock.",
            stacklevel=2,
        )
    if flight_day_budget_min < sortie_budget_min + refuel_time_min:
        warnings.warn(
            f"flight_day_budget ({flight_day_budget_min:.1f} min) is less "
            f"than sortie_budget + refuel_time "
            f"({sortie_budget_min + refuel_time_min:.1f} min); a full fuel "
            f"cycle plus refuel will not fit, so the day clock may strongly "
            f"limit refuel routes.",
            stacklevel=2,
        )

    # --- recovery destination -----------------------------------------------
    if mode == "round_trip":
        if return_destination is None:
            recovery_wp = start
            return_label = start.name or "start"
        else:
            recovery_wp = _resolve_destination(return_destination)
            return_label = _destination_label(return_destination)
    else:  # return_safe
        if return_destination is None:
            raise HyPlanValueError(
                "`return_destination` is required when mode='return_safe'."
            )
        recovery_wp = _resolve_destination(return_destination)
        return_label = _destination_label(return_destination)

    if start_time is None:
        start_time = datetime.datetime.now(datetime.timezone.utc)
    if wind_source is None:
        wind_source = StillAirField()

    # --- prefilter refuel airports -----------------------------------------
    eligibility, evaluated, unreachable = _prefilter_refuel_airports(
        aircraft=aircraft,
        start=start,
        recovery_wp=recovery_wp,
        cruise_altitude=cruise_altitude,
        start_time=start_time,
        wind_source=wind_source,
        refuel_airports=refuel_airports,
        sortie_budget_min=sortie_budget_min,
        flight_day_budget_min=flight_day_budget_min,
        reserve_min=reserve_min,
        refuel_time_min=refuel_time_min,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    # --- sweep --------------------------------------------------------------
    azimuths = np.arange(0.0, 360.0, azimuth_resolution_deg)
    rows = _solve_rays_refuel(
        aircraft=aircraft,
        start=start,
        cruise_altitude=cruise_altitude,
        start_time=start_time,
        wind_source=wind_source,
        recovery_wp=recovery_wp,
        on_station_min=on_station_min,
        sortie_budget_min=sortie_budget_min,
        flight_day_budget_min=flight_day_budget_min,
        reserve_min=reserve_min,
        refuel_time_min=refuel_time_min,
        refuel_eligibility=eligibility,
        azimuths_deg=azimuths,
        distance_tolerance_nmi=distance_tolerance_nmi,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    df = pd.DataFrame(rows)
    geometry = [Point(lon, lat) for lat, lon in zip(df["target_lat"], df["target_lon"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    used = {
        r["refuel_airport"]
        for r in rows
        if r.get("refuel_airport") is not None
    }

    gdf.attrs.update({
        "mode": mode,
        "start_lat": start.latitude,
        "start_lon": start.longitude,
        "start_altitude_ft": (
            start.altitude_msl.m_as(ureg.feet)
            if start.altitude_msl is not None else None
        ),
        "cruise_altitude_ft": cruise_altitude.m_as(ureg.feet),
        "sortie_budget_min": sortie_budget_min,
        "sortie_budget_hr": sortie_budget_min / 60.0,
        "flight_day_budget_min": flight_day_budget_min,
        "flight_day_budget_hr": flight_day_budget_min / 60.0,
        "refuel_time_min": refuel_time_min,
        "max_refuel_stops": max_refuel_stops,
        "reserve_min": reserve_min,
        "on_station_min": on_station_min,
        "return_destination_label": return_label,
        "return_destination_lat": recovery_wp.latitude,
        "return_destination_lon": recovery_wp.longitude,
        "aircraft_type": aircraft.aircraft_type,
        "wind_source_kind": type(wind_source).__name__,
        "start_time": start_time.isoformat(),
        "refuel_airports_evaluated": evaluated,
        "refuel_airports_unreachable": unreachable,
        "refuel_airports_used": sorted(used),
    })
    return gdf


def evaluate_target_reachability(
    aircraft: Aircraft,
    start: Airport | Waypoint,
    target: Airport | Waypoint,
    *,
    sortie_budget: Quantity,
    flight_day_budget: Quantity | None = None,
    cruise_altitude: Quantity | None = None,
    refuel_airports: Sequence[Airport | Waypoint] = (),
    refuel_time: Quantity = 60 * ureg.minute,
    return_destination: Airport | Waypoint | None = None,
    mode: str = "return_safe",
    on_station_altitude: Quantity | None = None,
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> dict[str, Any]:
    """Evaluate reachability of a single target via direct + refuel paths.

    The complement of :func:`compute_refuel_isochrone`: rather than
    sweeping azimuths to find the boundary, this asks "given *this*
    target, which itineraries reach it within budget?" and reports
    every feasible option.

    Args:
        aircraft, start, sortie_budget, flight_day_budget, cruise_altitude,
        refuel_airports, refuel_time, return_destination, mode,
        on_station_altitude, on_station_time, reserve, start_time,
        wind_source: same semantics as
            :func:`compute_refuel_isochrone`.  ``flight_day_budget``
            defaults to ``sortie_budget`` (the day clock then never
            binds).  ``refuel_airports`` may be empty — only the direct
            itinerary is then evaluated.
        target: the point to evaluate.  Accepts an :class:`Airport` or
            a :class:`Waypoint`.

    Returns:
        Dict with keys:
            ``reachable``: ``bool``.
            ``best``: itinerary diagnostic dict for the chosen route,
                or ``None`` when ``reachable`` is False.
            ``alternatives``: list of dicts for the other feasible
                itineraries, sorted by ascending ``day_total_time_min``.
            ``unreachable_reason``: short human-readable string when
                ``reachable`` is False, else ``None``.
            ``target_lat``, ``target_lon``: coordinates of the target.
    """
    import warnings

    start, cruise_altitude, reserve_min, on_station_min = _validate_common_kwargs(
        start=start,
        cruise_altitude=cruise_altitude,
        on_station_altitude=on_station_altitude,
        on_station_time=on_station_time,
        reserve=reserve,
        mode=mode,
        valid_modes=_VALID_REFUEL_MODES,
        azimuth_resolution_deg=5.0,  # not used for single-point eval
        distance_tolerance_nmi=0.5,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    sortie_budget_min = sortie_budget.m_as(ureg.minute)
    if flight_day_budget is None:
        flight_day_budget_min = sortie_budget_min
    else:
        flight_day_budget_min = flight_day_budget.m_as(ureg.minute)
    refuel_time_min = refuel_time.m_as(ureg.minute)

    if refuel_time_min < 0:
        raise HyPlanValueError(
            f"`refuel_time` must be non-negative, got "
            f"{refuel_time_min:.1f} min."
        )
    if sortie_budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`sortie_budget` ({sortie_budget_min:.1f} min) must exceed "
            f"`reserve` ({reserve_min:.1f} min)."
        )
    if flight_day_budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`flight_day_budget` ({flight_day_budget_min:.1f} min) must "
            f"exceed `reserve` ({reserve_min:.1f} min)."
        )
    if flight_day_budget_min < sortie_budget_min:
        warnings.warn(
            f"flight_day_budget ({flight_day_budget_min:.1f} min) is "
            f"less than sortie_budget ({sortie_budget_min:.1f} min); "
            f"the day clock will bind.",
            stacklevel=2,
        )

    # Recovery destination.
    if mode == "round_trip":
        recovery_wp = (
            _resolve_destination(return_destination)
            if return_destination is not None
            else start
        )
    else:  # return_safe
        if return_destination is None:
            raise HyPlanValueError(
                "`return_destination` is required when mode='return_safe'."
            )
        recovery_wp = _resolve_destination(return_destination)

    if start_time is None:
        start_time = datetime.datetime.now(datetime.timezone.utc)
    if wind_source is None:
        wind_source = StillAirField()

    # Coerce target.
    target_wp = _airport_or_wp_to_waypoint(target, require_altitude=False)
    if target_wp.altitude_msl is None:
        target_wp = Waypoint(
            latitude=target_wp.latitude,
            longitude=target_wp.longitude,
            heading=target_wp.heading,
            altitude_msl=cruise_altitude,
            name=target_wp.name,
        )

    # Refuel prefilter (or empty list when no refuel airports given).
    if refuel_airports:
        eligibility, _, _ = _prefilter_refuel_airports(
            aircraft=aircraft,
            start=start,
            recovery_wp=recovery_wp,
            cruise_altitude=cruise_altitude,
            start_time=start_time,
            wind_source=wind_source,
            refuel_airports=refuel_airports,
            sortie_budget_min=sortie_budget_min,
            flight_day_budget_min=flight_day_budget_min,
            reserve_min=reserve_min,
            refuel_time_min=refuel_time_min,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )
    else:
        eligibility = []

    candidates = _evaluate_refuel_at_d(
        aircraft=aircraft,
        start=start,
        target=target_wp,
        recovery_wp=recovery_wp,
        cruise_altitude=cruise_altitude,
        start_time=start_time,
        wind_source=wind_source,
        on_station_min=on_station_min,
        sortie_budget_min=sortie_budget_min,
        flight_day_budget_min=flight_day_budget_min,
        reserve_min=reserve_min,
        refuel_time_min=refuel_time_min,
        refuel_eligibility=eligibility,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    if not candidates:
        return {
            "reachable": False,
            "best": None,
            "alternatives": [],
            "unreachable_reason": (
                "no feasible itinerary within sortie_budget + "
                "flight_day_budget"
            ),
            "target_lat": target_wp.latitude,
            "target_lon": target_wp.longitude,
        }

    # candidates is sorted by extension headroom (descending) — that's
    # the right order for the solver, but for "spot-check a target" the
    # natural ordering is by elapsed wall-clock time.  Re-sort.
    candidates.sort(key=lambda c: c["day_total_time_min"])
    return {
        "reachable": True,
        "best": candidates[0],
        "alternatives": candidates[1:],
        "unreachable_reason": None,
        "target_lat": target_wp.latitude,
        "target_lon": target_wp.longitude,
    }


def compute_multi_base_isochrone(
    aircraft: Aircraft,
    bases: Sequence[Airport | Waypoint],
    budget: Quantity,
    *,
    return_mode: str = "union",
    cruise_altitude: Quantity | None = None,
    on_station_altitude: Quantity | None = None,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    return_destinations: Sequence[Airport | Waypoint] | None = None,
    mode: str = "round_trip",
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
    ray_strategy: str = "uniform",
    adaptive_spacing_nmi: float | None = None,
    max_adaptive_rays: int = _DEFAULT_MAX_ADAPTIVE_RAYS,
) -> gpd.GeoDataFrame:
    """Compute reach polygons for multiple candidate bases.

    Calls :func:`compute_isochrone` once per base and aggregates the
    per-base reach polygons according to ``return_mode``.

    Args:
        aircraft: Aircraft model used for every base.
        bases: One or more candidate base airports / waypoints.  Each is
            passed through ``compute_isochrone(start=base, ...)``.
        budget: Total time available; same for every base.
        return_mode: How to aggregate per-base reach polygons.  v1
            supports ``"union"`` (returns a single-row GeoDataFrame whose
            geometry is the ``shapely.ops.unary_union`` of the per-base
            polygons).  ``"per_base"`` and ``"best_base"`` are reserved
            for future versions and raise :class:`HyPlanValueError`.
        return_destinations: Optional per-base recovery airfields,
            parallel to ``bases``.  Pass ``None`` (default) to recover
            at the same base each aircraft launches from.  Length must
            match ``bases`` exactly when provided.
        wind_source: A :class:`WindField` shared across all bases.
            For gridded providers, ensure the slab covers the bounding
            box of every base; reuse the same instance to amortize
            slab fetches across the per-base solves.
        cruise_altitude, on_station_altitude, start_time, mode,
        on_station_time, reserve, azimuth_resolution_deg,
        distance_tolerance_nmi, wind_sampling, wind_sample_spacing,
        max_wind_samples_per_leg, ray_strategy, adaptive_spacing_nmi,
        max_adaptive_rays:
            Forwarded to :func:`compute_isochrone` unchanged.

    Returns:
        A single-row :class:`geopandas.GeoDataFrame` (``EPSG:4326``)
        with columns:

        * ``geometry`` — union polygon (Polygon or MultiPolygon)
        * ``n_bases`` — total bases passed in
        * ``n_contributing_bases`` — bases that produced ≥ 3 reachable
          rays (and therefore a polygon)
        * ``base_labels`` — list of base labels in input order
        * ``return_mode`` — echoed for downstream code
        * ``budget_minutes`` — budget echoed in minutes
        * ``mode`` — single-base mode echoed (``"round_trip"`` etc.)

        ``gdf.attrs["per_base_gdfs"]`` carries the per-base
        :func:`compute_isochrone` GeoDataFrames in input order; useful
        for inspection or reuse in plotting.

    Raises:
        HyPlanValueError: If ``bases`` is empty, ``return_mode`` is
            unsupported, or ``return_destinations`` length does not
            match ``bases``.
        HyPlanRuntimeError: If no base produced ≥ 3 reachable rays.

    Notes:
        Cost is proportional to ``len(bases)``.  For ``N`` bases at
        ``M`` rays this is ``N × M`` ray solves.  Performance for large
        sweeps (gridded winds + refuel + many bases) is documented in
        ``docs/performance.md``.
    """
    if return_mode != "union":
        raise HyPlanValueError(
            f"return_mode={return_mode!r} not yet supported; v1 only "
            f"implements 'union'.  'per_base' and 'best_base' are "
            f"planned follow-ups."
        )
    if not bases:
        raise HyPlanValueError("`bases` must be non-empty.")
    if return_destinations is not None and len(return_destinations) != len(bases):
        raise HyPlanValueError(
            f"`return_destinations` must be None or have the same "
            f"length as `bases` ({len(bases)}); got "
            f"{len(return_destinations)}."
        )

    per_base_gdfs: list[gpd.GeoDataFrame] = []
    per_base_polygons: list[Polygon] = []
    for i, base in enumerate(bases):
        return_dest = (
            return_destinations[i] if return_destinations is not None else None
        )
        gdf = compute_isochrone(
            aircraft, base, budget,
            cruise_altitude=cruise_altitude,
            on_station_altitude=on_station_altitude,
            start_time=start_time,
            wind_source=wind_source,
            return_destination=return_dest,
            mode=mode,
            on_station_time=on_station_time,
            reserve=reserve,
            azimuth_resolution_deg=azimuth_resolution_deg,
            distance_tolerance_nmi=distance_tolerance_nmi,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
            ray_strategy=ray_strategy,
            adaptive_spacing_nmi=adaptive_spacing_nmi,
            max_adaptive_rays=max_adaptive_rays,
        )
        per_base_gdfs.append(gdf)
        try:
            per_base_polygons.append(isochrone_polygon(gdf))
        except HyPlanValueError:
            # Fewer than 3 reachable rays — base contributes nothing.
            continue

    if not per_base_polygons:
        raise HyPlanRuntimeError(
            f"No base produced a valid reach polygon (all {len(bases)} "
            f"bases had fewer than 3 reachable rays).  Check budget, "
            f"wind, and aircraft envelope."
        )

    union = unary_union(per_base_polygons)
    base_labels = [_destination_label(b) for b in bases]

    out = gpd.GeoDataFrame(
        {
            "geometry": [union],
            "n_bases": [len(bases)],
            "n_contributing_bases": [len(per_base_polygons)],
            "base_labels": [base_labels],
            "return_mode": [return_mode],
            "budget_minutes": [budget.m_as(ureg.minute)],
            "mode": [mode],
        },
        crs="EPSG:4326",
    )
    out.attrs["per_base_gdfs"] = per_base_gdfs
    return out


def compute_multi_refuel_isochrone(
    aircraft: Aircraft,
    start: Airport | Waypoint,
    sortie_budget: Quantity,
    *,
    flight_day_budget: Quantity,
    refuel_airports: Sequence[Airport | Waypoint],
    return_mode: str = "union",
    cruise_altitude: Quantity | None = None,
    refuel_time: Quantity = 60 * ureg.minute,
    return_destination: Airport | Waypoint | None = None,
    mode: str = "return_safe",
    on_station_altitude: Quantity | None = None,
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    max_refuel_stops: int = 1,
    start_time: datetime.datetime | None = None,
    wind_source: WindField | None = None,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> gpd.GeoDataFrame:
    """Per-refuel reach decomposition + union polygon.

    Calls :func:`compute_refuel_isochrone` once per refuel candidate
    (each treated as the sole refuel option) and aggregates the
    per-refuel reach polygons according to ``return_mode``.

    The on-station dwell (``on_station_time``) is forwarded to every
    underlying solve so the per-refuel polygons are directly comparable
    — each one represents reach with the *same* science requirement,
    differing only in which refuel airfield is available.  This is
    distinct from the all-refuels-at-once solve done by
    ``compute_refuel_isochrone(refuel_airports=[...])``: the union
    *region* here matches that single-call result (per-azimuth max is
    identical by construction), but the polygon *discretizations*
    differ slightly because :func:`shapely.ops.unary_union`
    triangulates two star-polygons differently than per-azimuth
    boundary connection.  The added value is the per-refuel
    decomposition, which enables sensitivity analysis ("what would we
    lose if airfield X dropped out?") and contribution-by-refuel
    visualization.

    Args:
        aircraft: Aircraft model.
        start: Departure point (single base; mirrors
            :func:`compute_refuel_isochrone`).
        sortie_budget: Per-fuel-cycle endurance.
        flight_day_budget: Total wall-clock budget across the sortie.
        refuel_airports: One or more refuel candidates.  Each is run as
            the only refuel for an independent solve.  Empty raises;
            use :func:`compute_isochrone` if no refuel option applies.
        return_mode: How to aggregate per-refuel reach polygons.  v1
            supports ``"union"``; ``"per_refuel"`` and ``"best_refuel"``
            are reserved for future versions and raise
            :class:`HyPlanValueError`.
        on_station_time: Required dwell at the target.  Charged against
            the sortie cycle that visits the target, just as in
            ``compute_refuel_isochrone``.  Forwarded unchanged.
        cruise_altitude, refuel_time, return_destination, mode,
        on_station_altitude, reserve, max_refuel_stops, start_time,
        wind_source, azimuth_resolution_deg, distance_tolerance_nmi,
        wind_sampling, wind_sample_spacing, max_wind_samples_per_leg:
            Forwarded to :func:`compute_refuel_isochrone` unchanged.

    Returns:
        A single-row :class:`geopandas.GeoDataFrame` (``EPSG:4326``)
        with columns:

        * ``geometry`` — union polygon (Polygon or MultiPolygon)
        * ``n_refuels`` — total refuel candidates passed in
        * ``n_contributing_refuels`` — refuels that produced ≥ 3 reachable
          rays (and therefore a polygon)
        * ``refuel_labels`` — list of refuel labels in input order
        * ``return_mode`` — echoed
        * ``sortie_budget_minutes`` — echoed in minutes
        * ``flight_day_budget_minutes`` — echoed in minutes
        * ``on_station_minutes`` — echoed in minutes
        * ``mode`` — single-base mode echoed
          (``"return_safe"`` or ``"round_trip"``)

        ``gdf.attrs["per_refuel_gdfs"]`` carries the per-refuel
        :func:`compute_refuel_isochrone` GeoDataFrames in input order;
        useful for inspection or reuse in plotting.

    Raises:
        HyPlanValueError: If ``refuel_airports`` is empty or
            ``return_mode`` is unsupported.
        HyPlanRuntimeError: If no refuel produced a valid reach polygon.

    Notes:
        Cost is proportional to ``len(refuel_airports)``.  For ``N``
        candidates at ``M`` rays this is ``N`` refuel-solver calls,
        each evaluating ~3 itinerary templates per ray.  Reuse a
        single :class:`WindField` instance across the call so any
        gridded slab fetch amortizes.
    """
    if return_mode != "union":
        raise HyPlanValueError(
            f"return_mode={return_mode!r} not yet supported; v1 only "
            f"implements 'union'.  'per_refuel' and 'best_refuel' are "
            f"planned follow-ups."
        )
    if not refuel_airports:
        raise HyPlanValueError("`refuel_airports` must be non-empty.")

    per_refuel_gdfs: list[gpd.GeoDataFrame] = []
    per_refuel_polygons: list[Polygon] = []
    for refuel in refuel_airports:
        gdf = compute_refuel_isochrone(
            aircraft, start, sortie_budget,
            flight_day_budget=flight_day_budget,
            cruise_altitude=cruise_altitude,
            refuel_airports=[refuel],
            refuel_time=refuel_time,
            return_destination=return_destination,
            mode=mode,
            on_station_altitude=on_station_altitude,
            on_station_time=on_station_time,
            reserve=reserve,
            max_refuel_stops=max_refuel_stops,
            start_time=start_time,
            wind_source=wind_source,
            azimuth_resolution_deg=azimuth_resolution_deg,
            distance_tolerance_nmi=distance_tolerance_nmi,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )
        per_refuel_gdfs.append(gdf)
        try:
            per_refuel_polygons.append(isochrone_polygon(gdf))
        except HyPlanValueError:
            # Fewer than 3 reachable rays — refuel contributes nothing.
            continue

    if not per_refuel_polygons:
        raise HyPlanRuntimeError(
            f"No refuel candidate produced a valid reach polygon (all "
            f"{len(refuel_airports)} candidates had fewer than 3 "
            f"reachable rays).  Check budgets, wind, and aircraft "
            f"envelope."
        )

    union = unary_union(per_refuel_polygons)
    refuel_labels = [_destination_label(r) for r in refuel_airports]

    out = gpd.GeoDataFrame(
        {
            "geometry": [union],
            "n_refuels": [len(refuel_airports)],
            "n_contributing_refuels": [len(per_refuel_polygons)],
            "refuel_labels": [refuel_labels],
            "return_mode": [return_mode],
            "sortie_budget_minutes": [sortie_budget.m_as(ureg.minute)],
            "flight_day_budget_minutes": [flight_day_budget.m_as(ureg.minute)],
            "on_station_minutes": [on_station_time.m_as(ureg.minute)],
            "mode": [mode],
        },
        crs="EPSG:4326",
    )
    out.attrs["per_refuel_gdfs"] = per_refuel_gdfs
    return out


def isochrone_polygon(gdf: gpd.GeoDataFrame) -> Polygon:
    """Connect the isochrone boundary points into a closed polygon.

    Boundary points are taken in order of ``azimuth_deg``.  Result is
    returned in EPSG:4326 (matches the input GeoDataFrame's CRS).
    """
    sorted_gdf = gdf.sort_values("azimuth_deg")
    coords = [(p.x, p.y) for p in sorted_gdf.geometry]
    if len(coords) < 3:
        raise HyPlanValueError(
            "Need at least 3 boundary points to form a polygon."
        )
    return Polygon(coords)


def plot_isochrone(
    gdf: gpd.GeoDataFrame,
    base_map: folium.Map | None = None,
    *,
    color: str = "steelblue",
    fill_opacity: float = 0.2,
    tiles: str = "OpenStreetMap",
    zoom_start: int = 6,
) -> folium.Map:
    """Render the isochrone on a Folium map.

    Args:
        gdf: A GeoDataFrame returned by :func:`compute_isochrone` or
            :func:`compute_refuel_isochrone`.  Refuel results are
            recognized via ``gdf.attrs["refuel_airports_evaluated"]``
            and trigger refuel-airport markers + per-itinerary dot
            coloring (see ``color`` below).
        base_map: Optional existing Folium map to add to.  If ``None``,
            a new one is created centered on ``start`` with the
            ``tiles`` basemap loaded.
        color: Polygon stroke + fill color.  For refuel-isochrone
            results, this color is also used for ``"direct"`` ray dots,
            but ``"outbound_refuel"`` and ``"return_refuel"`` dots use
            a fixed palette (steel blue / red) so the itinerary
            structure is legible regardless of the polygon color.
        fill_opacity: Polygon fill opacity (0–1).
        tiles: Folium basemap identifier when constructing a new map.
            Built-in choices include ``"OpenStreetMap"`` (default),
            ``"CartoDB positron"`` (clean light gray, good for science
            figures), ``"CartoDB dark_matter"``, and ``"Esri.WorldImagery"``
            (satellite).  Ignored when ``base_map`` is supplied.
        zoom_start: Initial zoom level (Folium scale 1–18).  Ignored
            when ``base_map`` is supplied.

    Returns:
        The Folium map.
    """
    attrs = gdf.attrs
    start_lat = attrs.get("start_lat", float(gdf.geometry.y.mean()))
    start_lon = attrs.get("start_lon", float(gdf.geometry.x.mean()))

    if base_map is None:
        base_map = folium.Map(
            location=[start_lat, start_lon],
            zoom_start=zoom_start,
            tiles=tiles,
        )

    # Polygon boundary.
    poly = isochrone_polygon(gdf)
    if "sortie_budget_hr" in attrs:
        budget_label = (
            f"sortie={attrs['sortie_budget_hr']:.1f} hr, "
            f"day={attrs.get('flight_day_budget_hr', float('nan')):.1f} hr"
        )
    else:
        budget_label = f"budget={attrs.get('budget_hr', float('nan')):.1f} hr"
    folium.Polygon(
        locations=[(y, x) for x, y in poly.exterior.coords],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        weight=2,
        popup=f"Isochrone — mode={attrs.get('mode', '?')}, {budget_label}",
    ).add_to(base_map)

    # Start marker.
    folium.Marker(
        location=[start_lat, start_lon],
        popup=f"start: {attrs.get('aircraft_type', '?')} "
              f"@ {attrs.get('start_altitude_ft', float('nan')):.0f} ft",
        icon=folium.Icon(color="green", icon="plane", prefix="fa"),
    ).add_to(base_map)

    # Return-destination marker (when distinct from start and applicable).
    ret_lat = attrs.get("return_destination_lat")
    ret_lon = attrs.get("return_destination_lon")
    if (
        ret_lat is not None
        and ret_lon is not None
        and not (
            abs(ret_lat - start_lat) < 1e-6
            and abs(ret_lon - start_lon) < 1e-6
        )
    ):
        folium.Marker(
            location=[ret_lat, ret_lon],
            popup=f"recovery: {attrs.get('return_destination_label', '?')}",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(base_map)

    # Per-ray points with diagnostics popups.
    itinerary_colors = {
        "direct": color,
        "outbound_refuel": "#1f77b4",
        "return_refuel": "#d62728",
        "unflyable": "0.6",
    }
    for _, row in gdf.iterrows():
        popup_html = (
            f"<b>az {row['azimuth_deg']:.0f}°</b><br>"
            f"distance: {row['distance_nmi']:.1f} nmi<br>"
        )
        itinerary = row.get("itinerary")
        if itinerary is not None:
            popup_html += f"itinerary: <b>{itinerary}</b><br>"
            if row.get("refuel_airport"):
                popup_html += f"refuel: {row['refuel_airport']}<br>"
        popup_html += f"outbound: {row['outbound_time_min']:.1f} min<br>"
        if not pd.isna(row.get("return_time_min")):
            popup_html += (
                f"return: {row['return_time_min']:.1f} min<br>"
                f"net headwind: {row['net_headwind_kt']:.1f} kt<br>"
                f"asymmetry: {row['headwind_asymmetry_kt']:.1f} kt<br>"
            )
        else:
            popup_html += (
                f"headwind: {row['outbound_headwind_kt']:.1f} kt<br>"
            )
        dot_color = (
            itinerary_colors.get(itinerary, color)
            if itinerary is not None
            else color
        )
        folium.CircleMarker(
            location=[row["target_lat"], row["target_lon"]],
            radius=2,
            color=dot_color,
            fill=True,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(base_map)

    # Refuel airport markers (when this is a refuel-isochrone result).
    evaluated = attrs.get("refuel_airports_evaluated")
    if evaluated:
        used = set(attrs.get("refuel_airports_used", []))
        for rec in evaluated:
            label = rec["label"]
            is_used = label in used
            t_to = rec["time_to_reach_min"]
            t_to_str = (
                f"{t_to:.1f} min" if t_to is not None else "unflyable"
            )
            popup = (
                f"<b>{label}</b><br>"
                f"{'used' if is_used else 'evaluated, not selected'}<br>"
                f"templates: "
                f"{', '.join(rec['templates_eligible']) or '—'}<br>"
                f"start→R: {t_to_str}"
            )
            folium.Marker(
                location=[rec["lat"], rec["lon"]],
                popup=folium.Popup(popup, max_width=260),
                icon=folium.Icon(
                    color="darkblue" if is_used else "gray",
                    icon="gas-pump", prefix="fa",
                ),
            ).add_to(base_map)
        for u in attrs.get("refuel_airports_unreachable", []):
            if "lat" not in u or "lon" not in u:
                continue
            folium.Marker(
                location=[u["lat"], u["lon"]],
                popup=folium.Popup(
                    f"<b>{u['label']}</b><br>unreachable<br>{u['reason']}",
                    max_width=260,
                ),
                icon=folium.Icon(color="lightgray", icon="ban", prefix="fa"),
            ).add_to(base_map)

    return base_map


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _airport_or_wp_to_waypoint(
    obj: Airport | Waypoint,
    *,
    require_altitude: bool = True,
) -> Waypoint:
    """Coerce an Airport-or-Waypoint argument to a Waypoint.

    Airports become a Waypoint at runway elevation, heading 0°, named
    by ICAO code.  Waypoints are returned as-is, with optional
    altitude validation.
    """
    if isinstance(obj, Waypoint):
        if require_altitude and obj.altitude_msl is None:
            raise HyPlanValueError(
                "Waypoint argument must have altitude_msl set."
            )
        return obj
    # Airport
    elev = obj.elevation if obj.elevation is not None else 0 * ureg.meter
    return Waypoint(
        latitude=obj.latitude,
        longitude=obj.longitude,
        heading=0.0,
        altitude_msl=elev,
        name=obj.icao_code,
    )


def _resolve_destination(
    dest: Airport | Waypoint,
) -> Waypoint:
    """Coerce a return-destination argument to a Waypoint."""
    return _airport_or_wp_to_waypoint(dest, require_altitude=True)


def _destination_label(dest: Airport | Waypoint) -> str:
    if isinstance(dest, Airport):
        return str(dest.icao_code)
    return dest.name or f"({dest.latitude:.2f}, {dest.longitude:.2f})"


def _unflyable_ray(
    *,
    azimuth_deg: float,
    start: Waypoint,
    mode: str,
    on_station_min: float,
) -> dict[str, Any]:
    """Return a zero-distance row marking this ray as infeasible."""
    return {
        "azimuth_deg": azimuth_deg,
        "distance_nmi": 0.0,
        "target_lat": start.latitude,
        "target_lon": start.longitude,
        "outbound_time_min": 0.0,
        "on_station_min": on_station_min,
        "return_time_min": float("nan") if mode == "one_way" else 0.0,
        "total_time_min": 0.0,
        "outbound_headwind_kt": float("nan"),
        "return_headwind_kt": float("nan"),
        "net_headwind_kt": float("nan"),
        "headwind_asymmetry_kt": float("nan"),
        "limiting_leg": "unflyable",
        "time_slack_min": 0.0,
    }


def _initial_bracket_distance_nmi(
    *,
    aircraft: Aircraft,
    cruise_altitude: Quantity,
    mode: str,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
) -> float:
    """Heuristic first upper-bound guess for the expanding bracket.

    Correctness still comes from the expanding-bracket loop; this only
    starts that loop near the expected answer instead of repeatedly
    testing 25, 50, 100, ... nmi for every ray.
    """
    feasible_min = max(0.0, budget_min - reserve_min)
    if mode == "one_way":
        leg_budget_min = feasible_min
    else:
        leg_budget_min = max(0.0, feasible_min - on_station_min) / 2.0

    tas_kt = aircraft.cruise_speed_at(cruise_altitude).m_as(ureg.knot)
    estimate_nmi = tas_kt * (leg_budget_min / 60.0)
    # Pad above the still-air/no-overhead guess.  Strong tailwinds or
    # different recovery geometry may still exceed it, so the normal
    # doubling loop remains in force.
    return float(max(25.0, 1.15 * estimate_nmi))


def _target_coordinates(
    start: Waypoint,
    distances_nmi: npt.NDArray[np.floating[Any]],
    azimuths_deg: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized great-circle target coordinates for radial candidates."""
    lats, lons = pymap3d.vincenty.vreckon(
        start.latitude,
        start.longitude,
        distances_nmi * 1852.0,
        azimuths_deg,
    )
    lats_arr = np.atleast_1d(np.asarray(lats, dtype=float))
    lons_arr = (
        (np.atleast_1d(np.asarray(lons, dtype=float)) + 180.0) % 360.0
    ) - 180.0
    return lats_arr, lons_arr


def _same_horizontal_position(a: Waypoint, b: Waypoint | None) -> bool:
    """Return True when two waypoints are horizontally indistinguishable."""
    if b is None:
        return False
    dist_m, _ = pymap3d.vincenty.vdist(
        a.latitude, a.longitude, b.latitude, b.longitude
    )
    return float(np.asarray(dist_m).ravel()[0]) < 100.0


def _unique_sorted_azimuths(
    azimuths_deg: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Normalize, de-duplicate, and sort azimuths in [0, 360)."""
    wrapped = np.mod(np.asarray(azimuths_deg, dtype=float), 360.0)
    rounded = np.round(wrapped, 8)
    out: npt.NDArray[np.float64] = np.array(
        sorted(set(float(v) for v in rounded)), dtype=float,
    )
    return out


def _uniform_azimuths(azimuth_resolution_deg: float) -> npt.NDArray[np.float64]:
    return _unique_sorted_azimuths(np.arange(0.0, 360.0, azimuth_resolution_deg))


def _ellipse_seed_azimuths(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    return_wp: Waypoint | None,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
    azimuth_resolution_deg: float,
) -> npt.NDArray[np.float64]:
    """Approximate two-focus return-safe boundary and sample its perimeter.

    The solved isochrone still comes from the real leg-time binary search.
    This helper only chooses better first-pass ray bearings for cases where
    the start and recovery point are distinct.
    """
    uniform = _uniform_azimuths(azimuth_resolution_deg)
    if return_wp is None or _same_horizontal_position(start, return_wp):
        return uniform

    dist_m, bearing_deg = pymap3d.vincenty.vdist(
        start.latitude, start.longitude, return_wp.latitude, return_wp.longitude
    )
    focus_distance_nmi = float(np.asarray(dist_m).ravel()[0]) / 1852.0
    if focus_distance_nmi <= 1e-6:
        return uniform

    tas_kt = aircraft.cruise_speed_at(cruise_altitude).m_as(ureg.knot)
    available_min = max(0.0, budget_min - reserve_min - on_station_min)
    path_distance_nmi = tas_kt * (available_min / 60.0)
    if path_distance_nmi <= focus_distance_nmi:
        return uniform

    n_rays = max(3, len(uniform))
    a = 0.5 * path_distance_nmi
    c = 0.5 * focus_distance_nmi
    b_sq = max(0.0, a * a - c * c)
    if b_sq <= 1e-6:
        return uniform
    b = float(np.sqrt(b_sq))

    # Sample equally by approximate perimeter distance, not by ellipse
    # parameter. That is the part that fixes vertex crowding most directly.
    n_dense = max(720, 24 * n_rays)
    t = np.linspace(0.0, 2.0 * np.pi, n_dense, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    dx = np.diff(np.r_[x, x[0]])
    dy = np.diff(np.r_[y, y[0]])
    seg = np.hypot(dx, dy)
    cumulative = np.r_[0.0, np.cumsum(seg)]
    perimeter = cumulative[-1]
    if perimeter <= 0:
        return uniform

    sample_s = np.linspace(0.0, perimeter, n_rays, endpoint=False)
    x_s = np.interp(sample_s, cumulative, np.r_[x, x[0]])
    y_s = np.interp(sample_s, cumulative, np.r_[y, y[0]])

    along_from_start = x_s + c
    right_from_start = y_s
    relative_deg = np.degrees(np.arctan2(right_from_start, along_from_start))
    azimuths = float(np.asarray(bearing_deg).ravel()[0]) + relative_deg
    return _unique_sorted_azimuths(azimuths)


def _initial_ray_azimuths(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    return_wp: Waypoint | None,
    mode: str,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
    azimuth_resolution_deg: float,
    ray_strategy: str,
) -> tuple[npt.NDArray[np.float64], str]:
    """Choose first-pass ray azimuths and report the effective strategy."""
    two_focus = (
        mode in {"round_trip", "return_safe"}
        and not _same_horizontal_position(start, return_wp)
    )

    if ray_strategy == "auto":
        effective = "ellipse" if two_focus else "uniform"
    elif ray_strategy in {"ellipse", "ellipse_adaptive"} and not two_focus:
        effective = "adaptive" if ray_strategy == "ellipse_adaptive" else "uniform"
    else:
        effective = ray_strategy

    if effective in {"ellipse", "ellipse_adaptive"}:
        azimuths = _ellipse_seed_azimuths(
            aircraft=aircraft,
            start=start,
            cruise_altitude=cruise_altitude,
            return_wp=return_wp,
            on_station_min=on_station_min,
            budget_min=budget_min,
            reserve_min=reserve_min,
            azimuth_resolution_deg=azimuth_resolution_deg,
        )
        if len(azimuths) < 3:
            azimuths = _uniform_azimuths(azimuth_resolution_deg)
            effective = "uniform"
    else:
        azimuths = _uniform_azimuths(azimuth_resolution_deg)

    return azimuths, effective


def _boundary_chord_lengths_nmi(
    rows: list[dict[str, Any]],
) -> npt.NDArray[np.float64]:
    """Geodesic chord lengths between adjacent solved boundary vertices."""
    if len(rows) < 2:
        empty: npt.NDArray[np.float64] = np.array([], dtype=float)
        return empty
    ordered = sorted(rows, key=lambda r: r["azimuth_deg"])
    lats = np.array([r["target_lat"] for r in ordered], dtype=float)
    lons = np.array([r["target_lon"] for r in ordered], dtype=float)
    dist_m, _ = pymap3d.vincenty.vdist(
        lats, lons, np.roll(lats, -1), np.roll(lons, -1)
    )
    chords: npt.NDArray[np.float64] = np.asarray(dist_m, dtype=float) / 1852.0
    return chords


def _mid_azimuth_deg(a: float, b: float) -> float:
    """Midpoint bearing from a to b around the 0/360 wrap."""
    delta = (b - a) % 360.0
    return (a + 0.5 * delta) % 360.0


def _refined_azimuths_from_rows(
    rows: list[dict[str, Any]],
    *,
    spacing_nmi: float,
    max_rays: int,
) -> npt.NDArray[np.float64]:
    """Add midpoint rays across long boundary chords."""
    ordered = sorted(rows, key=lambda r: r["azimuth_deg"])
    az: npt.NDArray[np.float64] = np.array(
        [r["azimuth_deg"] for r in ordered], dtype=float,
    )
    if len(az) >= max_rays:
        return az

    chords = _boundary_chord_lengths_nmi(ordered)
    if not len(chords) or np.nanmax(chords) <= spacing_nmi:
        return az

    candidates: list[tuple[float, float]] = []
    for i, chord in enumerate(chords):
        if chord <= spacing_nmi:
            continue
        a = float(az[i])
        b = float(az[(i + 1) % len(az)])
        candidates.append((float(chord), _mid_azimuth_deg(a, b)))

    candidates.sort(reverse=True, key=lambda item: item[0])
    remaining = max(0, max_rays - len(az))
    additions = [mid for _, mid in candidates[:remaining]]
    if not additions:
        return az
    return _unique_sorted_azimuths(np.r_[az, additions])


def _solve_rays(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    return_wp: Waypoint | None,
    mode: str,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
    azimuths_deg: npt.NDArray[np.floating[Any]],
    distance_tolerance_nmi: float,
    seed_d_lo: npt.NDArray[np.floating[Any]] | None = None,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> list[dict[str, Any]]:
    """Solve all radial rays, vectorizing candidate geometry per iteration.

    ``seed_d_lo`` (when provided) sets the per-ray lower bound for the
    bracket search.  Used by ``compute_concentric_isochrones`` to
    amortize work across budgets: the previous (smaller) budget's
    converged ``d_lo`` is feasible at the current (larger) budget by
    monotonicity, so we skip the doubling phase below that distance.
    Rays seeded with ``0`` are treated as fresh (matches the
    no-seed default behavior).
    """
    feasible_budget_min = budget_min - reserve_min
    n_rays = len(azimuths_deg)
    initial_hi = _initial_bracket_distance_nmi(
        aircraft=aircraft,
        cruise_altitude=cruise_altitude,
        mode=mode,
        on_station_min=on_station_min,
        budget_min=budget_min,
        reserve_min=reserve_min,
    )

    if seed_d_lo is None:
        seed_d_lo = np.zeros(n_rays, dtype=float)
    seeded_mask = seed_d_lo > 0
    d_lo = seed_d_lo.astype(float).copy()
    # Ensure d_hi[i] > d_lo[i] for every ray; the doubling loop expands
    # from there.
    d_hi = np.maximum(initial_hi, d_lo + max(initial_hi, 1.0))
    active = np.ones(n_rays, dtype=bool)
    zero_unflyable = np.zeros(n_rays, dtype=bool)

    def _evaluate_many(
        indices: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.floating[Any]],
    ) -> tuple[npt.NDArray[np.float64], list[dict[str, Any]]]:
        """Evaluate feasibility for selected ray indices."""
        az = azimuths_deg[indices]
        target_lats, target_lons = _target_coordinates(start, distances, az)
        totals = np.empty(len(indices), dtype=float)
        diags: list[dict[str, Any]] = []

        for j, _idx in enumerate(indices):
            target = Waypoint(
                latitude=float(target_lats[j]),
                longitude=float(target_lons[j]),
                heading=float(az[j]),
                altitude_msl=cruise_altitude,
            )
            t_out_min, hw_out_kt = _leg_time(
                aircraft=aircraft,
                start_wp=start,
                end_wp=target,
                cruise_altitude=cruise_altitude,
                t_anchor=start_time,
                wind_source=wind_source,
                wind_sampling=wind_sampling,
                wind_sample_spacing=wind_sample_spacing,
                max_wind_samples_per_leg=max_wind_samples_per_leg,
            )

            if mode == "one_way":
                t_back_min = float("nan")
                hw_back_kt = float("nan")
                total_min = t_out_min
            else:
                assert return_wp is not None
                return_anchor = start_time + datetime.timedelta(
                    minutes=t_out_min + on_station_min
                )
                t_back_min, hw_back_kt = _leg_time(
                    aircraft=aircraft,
                    start_wp=target,
                    end_wp=return_wp,
                    cruise_altitude=cruise_altitude,
                    t_anchor=return_anchor,
                    wind_source=wind_source,
                    wind_sampling=wind_sampling,
                    wind_sample_spacing=wind_sample_spacing,
                    max_wind_samples_per_leg=max_wind_samples_per_leg,
                )
                total_min = t_out_min + on_station_min + t_back_min

            totals[j] = total_min
            diags.append({
                "azimuth_deg": float(az[j]),
                "distance_nmi": float(distances[j]),
                "target_lat": float(target_lats[j]),
                "target_lon": float(target_lons[j]),
                "outbound_time_min": t_out_min,
                "on_station_min": on_station_min,
                "return_time_min": t_back_min,
                "total_time_min": total_min,
                "outbound_headwind_kt": hw_out_kt,
                "return_headwind_kt": hw_back_kt,
            })

        return totals, diags

    # Expanding bracket.  Work on all still-feasible rays at once.  First
    # verify that the zero-distance target is itself feasible; otherwise
    # binary search would converge to d=0 and report an over-budget "boundary".
    #
    # Note: this d=0 probe is intentionally *liberal* — _leg_time short-
    # circuits at distance < 1e-6 nmi and returns (0, 0), so zero_totals
    # captures only on_station_min and ignores climb/descent overhead.
    # That's fine here: rays whose climb+descent alone exceed the budget
    # cannot expand the bracket (d_hi is infeasible → expanding stays
    # False → binary search keeps d_lo = 0), and the post-convergence
    # probe at distance_tolerance_nmi below (`if final_diag["distance_nmi"]
    # == 0.0: ...`) catches them and marks them unflyable.  Do not
    # remove that probe without also tightening this one.
    active_indices = np.flatnonzero(active)
    totals, _ = _evaluate_many(active_indices, d_hi[active_indices])
    zero_totals, _ = _evaluate_many(
        active_indices, np.zeros_like(active_indices, dtype=float)
    )
    zero_infeasible = (
        ~np.isfinite(zero_totals) | (zero_totals > feasible_budget_min)
    )
    # Seeded rays were known feasible at d=seed_d_lo>0 from a smaller
    # budget; do not mark them unflyable just because d=0 is over the
    # new budget (return-leg-binding rays where the boundary is
    # strictly outside the start can have zero infeasibility but a
    # valid annular feasible region).
    zero_infeasible_for_unflyable = zero_infeasible & ~seeded_mask[active_indices]
    if np.any(zero_infeasible_for_unflyable):
        zero_indices = active_indices[zero_infeasible_for_unflyable]
        zero_unflyable[zero_indices] = True
        active[zero_indices] = False

    feasible = np.isfinite(totals) & (totals <= feasible_budget_min)
    expanding = np.zeros(n_rays, dtype=bool)
    expanding[active_indices[feasible & ~zero_infeasible_for_unflyable]] = True

    while np.any(expanding):
        d_lo[expanding] = d_hi[expanding]
        d_hi[expanding] *= 2.0
        if np.any(d_hi[expanding] > 20000.0):
            first_bad = int(np.flatnonzero(expanding & (d_hi > 20000.0))[0])
            raise HyPlanRuntimeError(
                f"Isochrone bracket exceeded 20000 nmi at azimuth "
                f"{azimuths_deg[first_bad]:.1f}°.  Pathological wind "
                f"field or aircraft performance — refusing to loop further."
            )

        idx = np.flatnonzero(expanding)
        totals, _ = _evaluate_many(idx, d_hi[idx])
        still_feasible = np.isfinite(totals) & (totals <= feasible_budget_min)
        expanding[:] = False
        expanding[idx[still_feasible]] = True

    # Binary search.  Active excludes rays that were unflyable at the
    # first positive seed distance; all other rays have d_lo / d_hi brackets.
    while True:
        widths = d_hi - d_lo
        search = active & (widths > distance_tolerance_nmi)
        if not np.any(search):
            break
        idx = np.flatnonzero(search)
        d_mid = 0.5 * (d_lo[idx] + d_hi[idx])
        totals, _ = _evaluate_many(idx, d_mid)
        feasible_mid = np.isfinite(totals) & (totals <= feasible_budget_min)
        d_lo[idx[feasible_mid]] = d_mid[feasible_mid]
        d_hi[idx[~feasible_mid]] = d_mid[~feasible_mid]

    # Final diagnostics in original azimuth order.
    rows: list[dict[str, Any]] = []
    final_indices = np.flatnonzero(active)
    final_by_index: dict[int, dict[str, Any]] = {}
    if len(final_indices):
        _, final_diags = _evaluate_many(final_indices, d_lo[final_indices])
        final_by_index = dict(zip(final_indices.tolist(), final_diags))

    for i, azimuth_deg in enumerate(azimuths_deg):
        if zero_unflyable[i]:
            rows.append(
                _unflyable_ray(
                    azimuth_deg=float(azimuth_deg), start=start, mode=mode,
                    on_station_min=on_station_min,
                )
            )
            continue

        final_diag = final_by_index[i]
        final_total = final_diag["total_time_min"]
        if final_diag["distance_nmi"] == 0.0:
            # Safety net for the liberal d=0 probe above: if the search
            # converged to zero distance, re-evaluate at a distance just
            # large enough to incur climb/descent overhead.  This is what
            # actually catches rays whose phase overhead alone exceeds the
            # budget (the d=0 probe wouldn't flag them since _leg_time
            # short-circuits at distance < 1e-6 nmi).
            probe_totals, _ = _evaluate_many(
                np.array([i]), np.array([distance_tolerance_nmi])
            )
            if (
                not np.isfinite(probe_totals[0])
                or probe_totals[0] > feasible_budget_min
            ):
                rows.append(
                    _unflyable_ray(
                        azimuth_deg=float(azimuth_deg), start=start, mode=mode,
                        on_station_min=on_station_min,
                    )
                )
                continue

        if mode == "one_way":
            final_diag["net_headwind_kt"] = float("nan")
            final_diag["headwind_asymmetry_kt"] = float("nan")
            final_diag["limiting_leg"] = "outbound"
        else:
            hw_o = final_diag["outbound_headwind_kt"]
            hw_r = final_diag["return_headwind_kt"]
            final_diag["net_headwind_kt"] = 0.5 * (hw_o + hw_r)
            final_diag["headwind_asymmetry_kt"] = 0.5 * (hw_o - hw_r)
            legs = {
                "outbound": final_diag["outbound_time_min"],
                "on_station": final_diag["on_station_min"],
                "return": final_diag["return_time_min"],
            }
            final_diag["limiting_leg"] = max(legs, key=lambda k: legs[k])

        final_diag["time_slack_min"] = max(
            0.0, feasible_budget_min - final_total
        )
        rows.append(final_diag)

    return rows


def _solve_rays_with_strategy(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    return_wp: Waypoint | None,
    mode: str,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
    azimuths_deg: npt.NDArray[np.floating[Any]],
    distance_tolerance_nmi: float,
    seed_d_lo: npt.NDArray[np.floating[Any]] | None = None,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
    effective_ray_strategy: str = "uniform",
    adaptive_spacing_nmi: float | None = None,
    max_adaptive_rays: int = _DEFAULT_MAX_ADAPTIVE_RAYS,
) -> list[dict[str, Any]]:
    """Solve rays, optionally refining boundary gaps adaptively."""
    azimuths = _unique_sorted_azimuths(azimuths_deg)
    rows = _solve_rays(
        aircraft=aircraft,
        start=start,
        cruise_altitude=cruise_altitude,
        start_time=start_time,
        wind_source=wind_source,
        return_wp=return_wp,
        mode=mode,
        on_station_min=on_station_min,
        budget_min=budget_min,
        reserve_min=reserve_min,
        azimuths_deg=azimuths,
        distance_tolerance_nmi=distance_tolerance_nmi,
        seed_d_lo=seed_d_lo if seed_d_lo is not None and len(seed_d_lo) == len(azimuths) else None,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )

    if effective_ray_strategy not in {"adaptive", "ellipse_adaptive"}:
        return rows

    spacing = adaptive_spacing_nmi or _DEFAULT_ADAPTIVE_SPACING_NMI
    while len(azimuths) < max_adaptive_rays:
        refined = _refined_azimuths_from_rows(
            rows,
            spacing_nmi=spacing,
            max_rays=max_adaptive_rays,
        )
        if len(refined) == len(azimuths):
            break
        azimuths = refined
        rows = _solve_rays(
            aircraft=aircraft,
            start=start,
            cruise_altitude=cruise_altitude,
            start_time=start_time,
            wind_source=wind_source,
            return_wp=return_wp,
            mode=mode,
            on_station_min=on_station_min,
            budget_min=budget_min,
            reserve_min=reserve_min,
            azimuths_deg=azimuths,
            distance_tolerance_nmi=distance_tolerance_nmi,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )

    return rows


def _leg_time(
    *,
    aircraft: Aircraft,
    start_wp: Waypoint,
    end_wp: Waypoint,
    cruise_altitude: Quantity,
    t_anchor: datetime.datetime,
    wind_source: WindField,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> tuple[float, float]:
    """Compute leg time (minutes) and along-track headwind (knots).

    Three wind-sampling modes (``wind_sampling``):

    * ``"cruise_midpoint"`` (default) — single cruise wind sample at
      the cruise-segment geographic midpoint, with up to 3 fixed-point
      iterations on cruise time.  Climb and descent are still-air
      (v1.5 behavior, strictly backward-compatible).
    * ``"phase_midpoint"`` — adds a single climb-segment-mid and
      descent-segment-mid wind sample (each at the phase-mid
      altitude), projected onto the great-circle bearing and passed
      to :meth:`Aircraft._climb` / :meth:`Aircraft._descend` as
      ``wind_along_track``.  This redistributes ground distance
      between the climb/descent and cruise segments; cruise time is
      then computed via the same one-sample logic as
      ``"cruise_midpoint"``.  Climb/descent **times** are unchanged
      (climb rate is air-mass-relative, not wind-dependent).
    * ``"segmented_cruise"`` — same climb/descent treatment as
      ``"phase_midpoint"`` plus splits cruise into N subsegments
      (``min(max_wind_samples_per_leg, ceil(cruise_distance / wind_sample_spacing))``).
      Each subsegment gets its own midpoint wind sample with
      cumulative time anchor; per-subsegment groundspeed via
      :func:`_track_hold_solution_from_uv`; total cruise time is the
      sum.  Two-pass fixed-point on per-subsegment times.

    Returns ``(time_minutes, headwind_kt)``.  ``headwind_kt`` is the
    signed cruise-leg along-track headwind (positive = headwind,
    negative = tailwind), distance-weighted across subsegments when
    ``segmented_cruise`` is in use.  Returns ``(inf, inf)`` when the
    leg is unflyable (crosswind > TAS or groundspeed ≤ 0 at any
    sample).
    """
    # Geodesy.
    distance_m, bearing_deg = pymap3d.vincenty.vdist(
        start_wp.latitude, start_wp.longitude,
        end_wp.latitude, end_wp.longitude,
    )
    distance_nmi = float(distance_m) / 1852.0
    if distance_nmi < 1e-6:
        return 0.0, 0.0

    track_deg = float(bearing_deg)
    start_alt = start_wp.altitude_msl
    end_alt = end_wp.altitude_msl
    assert start_alt is not None and end_alt is not None

    # --- Climb / descent: still-air for cruise_midpoint, wind-corrected
    # ground distance for phase_midpoint / segmented_cruise. -----------
    # Wind affects the *ground distance* covered during climb/descent
    # (a tailwind extends the ground arc), not the times themselves
    # (climb rate is air-mass-relative).  Net effect on total leg
    # time arrives through the cruise term, which sees a different
    # cruise_distance_nmi.
    phase_aware = wind_sampling != "cruise_midpoint"

    if start_alt < cruise_altitude:
        if phase_aware:
            t_climb_q, d_climb_q, _ = climb_with_wind_field(
                aircraft,
                start_lat=start_wp.latitude,
                start_lon=start_wp.longitude,
                start_alt=start_alt,
                cruise_alt=cruise_altitude,
                track_deg=track_deg,
                t_anchor=t_anchor,
                wind_source=wind_source,
            )
        else:
            t_climb_q, d_climb_q = aircraft._climb(start_alt, cruise_altitude)
        t_climb_min = t_climb_q.m_as(ureg.minute)
        d_climb_nmi = d_climb_q.m_as(ureg.nautical_mile)
    else:
        t_climb_min = 0.0
        d_climb_nmi = 0.0

    if end_alt < cruise_altitude:
        if phase_aware:
            t_desc_q, d_desc_q, _ = descend_with_wind_field(
                aircraft,
                start_lat=start_wp.latitude,
                start_lon=start_wp.longitude,
                total_distance_nmi=distance_nmi,
                cruise_alt=cruise_altitude,
                end_alt=end_alt,
                track_deg=track_deg,
                t_anchor=t_anchor,
                t_climb_min=t_climb_min,
                d_climb_nmi=d_climb_nmi,
                wind_source=wind_source,
            )
        else:
            t_desc_q, d_desc_q = aircraft._descend(cruise_altitude, end_alt)
        t_desc_min = t_desc_q.m_as(ureg.minute)
        d_desc_nmi = d_desc_q.m_as(ureg.nautical_mile)
    else:
        t_desc_min = 0.0
        d_desc_nmi = 0.0

    cruise_distance_nmi = distance_nmi - d_climb_nmi - d_desc_nmi
    if cruise_distance_nmi < 0:
        # Climb + descent overhead exceed the geodesic distance.  Cruise
        # term collapses to zero; accept still-air climb+descent time
        # as the leg time.  No wind correction applies.
        return t_climb_min + t_desc_min, 0.0

    cruise_tas = aircraft.cruise_speed_at(cruise_altitude)
    cruise_tas_kt = cruise_tas.m_as(ureg.knot)

    # --- Cruise: dispatch on wind_sampling. ----------------------------
    if wind_sampling == "cruise_midpoint":
        # Legacy code path — kept verbatim for backward compatibility.
        return _cruise_midpoint_legacy(
            wind_source=wind_source,
            start_wp=start_wp,
            track_deg=track_deg,
            d_climb_nmi=d_climb_nmi,
            cruise_distance_nmi=cruise_distance_nmi,
            cruise_altitude=cruise_altitude,
            cruise_tas=cruise_tas,
            cruise_tas_kt=cruise_tas_kt,
            t_anchor=t_anchor,
            t_climb_min=t_climb_min,
            t_desc_min=t_desc_min,
        )

    # phase_midpoint and segmented_cruise share the segmented engine,
    # differing only in n_segments.
    if wind_sampling == "phase_midpoint":
        n_segments = 1
    else:  # segmented_cruise
        spacing_nmi = float(wind_sample_spacing.m_as(ureg.nautical_mile))
        n_segments = min(
            int(max_wind_samples_per_leg),
            max(1, int(np.ceil(cruise_distance_nmi / spacing_nmi))),
        )

    cruise_result = _cruise_time_segmented(
        wind_source=wind_source,
        start_wp=start_wp,
        track_deg=track_deg,
        d_climb_nmi=d_climb_nmi,
        cruise_distance_nmi=cruise_distance_nmi,
        cruise_altitude=cruise_altitude,
        cruise_tas=cruise_tas,
        cruise_tas_kt=cruise_tas_kt,
        t_anchor=t_anchor,
        t_climb_min=t_climb_min,
        n_segments=n_segments,
    )
    if cruise_result["status"] != "ok":
        return float("inf"), float("inf")
    return (
        t_climb_min + cruise_result["time_min"] + t_desc_min,
        cruise_result["headwind_kt"],
    )


def _cruise_midpoint_legacy(
    *,
    wind_source: WindField,
    start_wp: Waypoint,
    track_deg: float,
    d_climb_nmi: float,
    cruise_distance_nmi: float,
    cruise_altitude: Quantity,
    cruise_tas: Quantity,
    cruise_tas_kt: float,
    t_anchor: datetime.datetime,
    t_climb_min: float,
    t_desc_min: float,
) -> tuple[float, float]:
    """v1.5 cruise-midpoint code path, factored out unchanged.  Returns
    ``(total_leg_time_min, headwind_kt)``."""
    cruise_mid_dist_nmi = max(
        1e-3, d_climb_nmi + 0.5 * cruise_distance_nmi
    )
    mid_lat, mid_lon = pymap3d.vincenty.vreckon(
        start_wp.latitude, start_wp.longitude,
        cruise_mid_dist_nmi * 1852.0, track_deg,
    )
    mid_lat_f = float(mid_lat)
    mid_lon_f = float(wrap_to_180(float(mid_lon)))

    if isinstance(wind_source, StillAirField):
        t_cruise_min = cruise_distance_nmi / cruise_tas_kt * 60.0
        return t_climb_min + t_cruise_min + t_desc_min, 0.0

    if isinstance(wind_source, ConstantWindField):
        u_q, v_q = wind_source.wind_at(
            mid_lat_f, mid_lon_f, cruise_altitude, t_anchor,
        )
        try:
            sol = _track_hold_solution_from_uv(
                cruise_tas, track_deg, u_q, v_q,
            )
        except HyPlanValueError:
            return float("inf"), float("inf")
        gs_kt = sol["groundspeed"].m_as(ureg.knot)
        headwind_kt = -sol["alongtrack_wind"].m_as(ureg.knot)
        t_cruise_min = cruise_distance_nmi / gs_kt * 60.0
        return t_climb_min + t_cruise_min + t_desc_min, headwind_kt

    # Bounded fixed-point loop on cruise leg time.
    headwind_kt = 0.0
    t_cruise_min = cruise_distance_nmi / cruise_tas_kt * 60.0  # still-air seed
    t_cruise_min_prev: float | None = None

    for _ in range(3):
        sample_time = t_anchor + datetime.timedelta(
            minutes=t_climb_min + 0.5 * t_cruise_min
        )
        u_q, v_q = wind_source.wind_at(
            mid_lat_f, mid_lon_f, cruise_altitude, sample_time,
        )
        try:
            sol = _track_hold_solution_from_uv(
                cruise_tas, track_deg, u_q, v_q,
            )
        except HyPlanValueError:
            return float("inf"), float("inf")

        gs_kt = sol["groundspeed"].m_as(ureg.knot)
        headwind_kt = -sol["alongtrack_wind"].m_as(ureg.knot)
        t_cruise_min_new = cruise_distance_nmi / gs_kt * 60.0

        if t_cruise_min_prev is not None and abs(
            t_cruise_min_new - t_cruise_min
        ) * 60.0 < 5.0:  # |Δt_cruise| < 5 sec
            t_cruise_min = t_cruise_min_new
            break

        t_cruise_min_prev = t_cruise_min
        t_cruise_min = t_cruise_min_new

    return t_climb_min + t_cruise_min + t_desc_min, headwind_kt


def _cruise_time_segmented(
    *,
    wind_source: WindField,
    start_wp: Waypoint,
    track_deg: float,
    d_climb_nmi: float,
    cruise_distance_nmi: float,
    cruise_altitude: Quantity,
    cruise_tas: Quantity,
    cruise_tas_kt: float,
    t_anchor: datetime.datetime,
    t_climb_min: float,
    n_segments: int,
) -> dict[str, Any]:
    """Cruise-segment time + distance-weighted headwind under
    segmented or phase-midpoint sampling.

    Returns a dict ``{time_min, headwind_kt, n_samples, status}``
    where ``status`` is ``"ok"`` or ``"unflyable"`` (in which case
    ``time_min`` is ``inf`` and ``headwind_kt`` is ``inf``).

    Two-pass loop:

    1. Seed each subsegment time with the still-air estimate.
    2. For each subsegment in order, sample wind at its midpoint
       lat/lon at the cumulative anchor; solve track-hold; update
       per-subsegment time.  Repeat once.
    """
    n_samples = max(1, int(n_segments))
    subseg_nmi = cruise_distance_nmi / n_samples

    # StillAirField: short-circuit (no wind queries).
    if isinstance(wind_source, StillAirField):
        return {
            "time_min": cruise_distance_nmi / cruise_tas_kt * 60.0,
            "headwind_kt": 0.0,
            "n_samples": n_samples,
            "status": "ok",
        }

    # ConstantWindField: a single (u, v) is enough — cache it and
    # still run the segmented math (each subsegment's groundspeed
    # solve is identical, so output is degenerate but correct).
    constant_uv: tuple[Any, Any] | None = None
    if isinstance(wind_source, ConstantWindField):
        u_q, v_q = wind_source.wind_at(
            start_wp.latitude, start_wp.longitude,
            cruise_altitude, t_anchor,
        )
        constant_uv = (u_q, v_q)

    # Pre-compute subsegment midpoint coordinates (lat/lon).  These
    # don't change between passes; only the time anchor does (and
    # only for time-varying winds).
    mid_lats = np.empty(n_samples, dtype=float)
    mid_lons = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        mid_dist_nmi = max(
            1e-3, d_climb_nmi + (i + 0.5) * subseg_nmi,
        )
        mid_lat, mid_lon = pymap3d.vincenty.vreckon(
            start_wp.latitude, start_wp.longitude,
            mid_dist_nmi * 1852.0, track_deg,
        )
        mid_lats[i] = float(mid_lat)
        mid_lons[i] = float(wrap_to_180(float(mid_lon)))

    # Seed: still-air per-subsegment time.
    t_subseg_min = np.full(n_samples, subseg_nmi / cruise_tas_kt * 60.0)
    headwind_kt_per = np.zeros(n_samples, dtype=float)

    for _pass in range(2):
        cumulative_min = t_climb_min
        for i in range(n_samples):
            sample_offset_min = cumulative_min + 0.5 * float(t_subseg_min[i])
            if constant_uv is not None:
                u_q, v_q = constant_uv
            else:
                sample_time = t_anchor + datetime.timedelta(
                    minutes=sample_offset_min,
                )
                u_q, v_q = wind_source.wind_at(
                    float(mid_lats[i]), float(mid_lons[i]),
                    cruise_altitude, sample_time,
                )
            try:
                sol = _track_hold_solution_from_uv(
                    cruise_tas, track_deg, u_q, v_q,
                )
            except HyPlanValueError:
                return {
                    "time_min": float("inf"),
                    "headwind_kt": float("inf"),
                    "n_samples": n_samples,
                    "status": "unflyable",
                }
            gs_kt = sol["groundspeed"].m_as(ureg.knot)
            headwind_kt_per[i] = -sol["alongtrack_wind"].m_as(ureg.knot)
            t_subseg_min[i] = subseg_nmi / gs_kt * 60.0
            cumulative_min += float(t_subseg_min[i])

    total_time_min = float(np.sum(t_subseg_min))
    if cruise_distance_nmi > 0:
        # Distance-weighted average headwind (each subsegment has
        # equal length, so this is just the arithmetic mean).
        headwind_avg = float(np.mean(headwind_kt_per))
    else:
        headwind_avg = 0.0
    return {
        "time_min": total_time_min,
        "headwind_kt": headwind_avg,
        "n_samples": n_samples,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Refuel-aware solver
# ---------------------------------------------------------------------------

def _anchor_invariant(wind_source: WindField) -> bool:
    """True when ``wind_at`` returns the same (u, v) regardless of the
    sample time, so leg-time results can be cached across anchors.

    ``StillAirField`` and ``ConstantWindField`` are time-invariant by
    construction; everything else (MERRA-2, GFS, GMAO) varies with time
    and must be re-sampled at the correct anchor.
    """
    return isinstance(wind_source, (StillAirField, ConstantWindField))


def _prefilter_refuel_airports(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    recovery_wp: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    refuel_airports: Sequence[Airport | Waypoint],
    sortie_budget_min: float,
    flight_day_budget_min: float,
    reserve_min: float,
    refuel_time_min: float,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Coerce refuel airports and prefilter per template.

    Returns ``(eligibility, evaluated, unreachable)`` where ``eligibility``
    is a list of dicts the solver consumes (one per usable airport), and
    ``evaluated`` / ``unreachable`` go to ``gdf.attrs``.

    Per-template eligibility is computed independently — an airport that
    fails one placement is still kept for the other.  The leg times here
    are sampled at fixed anchors and serve as a *lenient* prefilter; the
    per-ray Phase 2 always recomputes leg times with the correct
    in-itinerary anchors so time-varying winds are honored at the
    boundary.
    """
    cycle_cap_min = sortie_budget_min - reserve_min

    eligibility: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []

    later_anchor = start_time + datetime.timedelta(minutes=sortie_budget_min)

    for raw in refuel_airports:
        r_wp = _airport_or_wp_to_waypoint(raw, require_altitude=True)
        label = (
            raw.icao_code if isinstance(raw, Airport)
            else (r_wp.name or f"({r_wp.latitude:.2f},{r_wp.longitude:.2f})")
        )

        t_sR, hw_sR = _leg_time(
            aircraft=aircraft, start_wp=start, end_wp=r_wp,
            cruise_altitude=cruise_altitude, t_anchor=start_time,
            wind_source=wind_source,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )
        t_Rrec, hw_Rrec = _leg_time(
            aircraft=aircraft, start_wp=r_wp, end_wp=recovery_wp,
            cruise_altitude=cruise_altitude, t_anchor=later_anchor,
            wind_source=wind_source,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )

        outbound_ok = (
            np.isfinite(t_sR)
            and t_sR <= cycle_cap_min
            and t_sR + refuel_time_min <= flight_day_budget_min
        )
        return_ok = (
            np.isfinite(t_Rrec) and t_Rrec <= cycle_cap_min
        )

        templates: list[str] = []
        if outbound_ok:
            templates.append("outbound_refuel")
        if return_ok:
            templates.append("return_refuel")

        rec = {
            "label": label,
            "lat": r_wp.latitude,
            "lon": r_wp.longitude,
            "time_to_reach_min": (
                float(t_sR) if np.isfinite(t_sR) else None
            ),
            "time_from_to_recovery_min": (
                float(t_Rrec) if np.isfinite(t_Rrec) else None
            ),
            "templates_eligible": templates,
        }

        if not templates:
            reasons = []
            if not np.isfinite(t_sR):
                reasons.append("start→R unflyable")
            elif t_sR > cycle_cap_min:
                reasons.append(
                    f"start→R = {t_sR:.0f} min > cycle cap "
                    f"{cycle_cap_min:.0f} min"
                )
            elif t_sR + refuel_time_min > flight_day_budget_min:
                reasons.append("no day budget left after cycle 1 + refuel")
            if not np.isfinite(t_Rrec):
                reasons.append("R→recovery unflyable")
            elif t_Rrec > cycle_cap_min:
                reasons.append(
                    f"R→recovery = {t_Rrec:.0f} min > cycle cap "
                    f"{cycle_cap_min:.0f} min"
                )
            unreachable.append({
                "label": label,
                "lat": r_wp.latitude,
                "lon": r_wp.longitude,
                "reason": "; ".join(reasons) or "no eligible template",
            })
            continue

        evaluated.append(rec)
        eligibility.append({
            "label": label,
            "wp": r_wp,
            "outbound_ok": outbound_ok,
            "return_ok": return_ok,
            "t_sR_min": float(t_sR) if np.isfinite(t_sR) else float("inf"),
            "hw_sR_kt": float(hw_sR) if np.isfinite(hw_sR) else float("nan"),
            "t_Rrec_min": float(t_Rrec) if np.isfinite(t_Rrec) else float("inf"),
            "hw_Rrec_kt": float(hw_Rrec) if np.isfinite(hw_Rrec) else float("nan"),
        })

    return eligibility, evaluated, unreachable


def _evaluate_refuel_at_d(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    target: Waypoint,
    recovery_wp: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    on_station_min: float,
    sortie_budget_min: float,
    flight_day_budget_min: float,
    reserve_min: float,
    refuel_time_min: float,
    refuel_eligibility: list[dict[str, Any]],
    template: str | None = None,
    refuel_label: str | None = None,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> list[dict[str, Any]]:
    """Evaluate one or more refuel itinerary templates at the given target.

    Returns the list of feasible itineraries sorted by extension
    headroom (``min(sortie_margin, day_margin)``, descending).  Empty
    list when no itinerary is feasible.

    When ``template`` is provided, only that template is evaluated.  For
    refuel templates, ``refuel_label`` narrows the evaluation to one airport.
    The refuel solver uses this filtered mode so each template gets its own
    monotonic bracket/binary-search boundary before the best route is selected.
    """
    cycle_cap_min = sortie_budget_min - reserve_min
    candidates: list[dict[str, Any]] = []

    # --- direct: start → target → recovery -------------------------------
    t_st, hw_st = _leg_time(
        aircraft=aircraft, start_wp=start, end_wp=target,
        cruise_altitude=cruise_altitude, t_anchor=start_time,
        wind_source=wind_source,
        wind_sampling=wind_sampling,
        wind_sample_spacing=wind_sample_spacing,
        max_wind_samples_per_leg=max_wind_samples_per_leg,
    )
    if (template is None or template == "direct") and np.isfinite(t_st):
        anchor_tr = start_time + datetime.timedelta(
            minutes=t_st + on_station_min
        )
        t_tr, hw_tr = _leg_time(
            aircraft=aircraft, start_wp=target, end_wp=recovery_wp,
            cruise_altitude=cruise_altitude, t_anchor=anchor_tr,
            wind_source=wind_source,
            wind_sampling=wind_sampling,
            wind_sample_spacing=wind_sample_spacing,
            max_wind_samples_per_leg=max_wind_samples_per_leg,
        )
        if np.isfinite(t_tr):
            cycle_1 = t_st + on_station_min + t_tr
            day_total = cycle_1
            sortie_margin = cycle_cap_min - cycle_1
            day_margin = flight_day_budget_min - day_total
            if sortie_margin >= 0 and day_margin >= 0:
                candidates.append({
                    "itinerary": "direct",
                    "refuel_airport": None,
                    "refuel_count": 0,
                    "refuel_time_min": float("nan"),
                    "start_to_target_time_min": t_st,
                    "start_to_refuel_time_min": float("nan"),
                    "refuel_to_target_time_min": float("nan"),
                    "target_to_refuel_time_min": float("nan"),
                    "refuel_to_return_time_min": float("nan"),
                    "target_to_return_time_min": t_tr,
                    "outbound_time_min": t_st,
                    "return_time_min": t_tr,
                    "total_time_min": day_total,
                    "outbound_headwind_kt": hw_st,
                    "return_headwind_kt": hw_tr,
                    "day_total_time_min": day_total,
                    "sortie_cycle_1_min": cycle_1,
                    "sortie_cycle_2_min": float("nan"),
                    "sortie_margin_min": sortie_margin,
                    "day_margin_min": day_margin,
                })

    # --- outbound_refuel(R): start → R → target → recovery -----------------
    if template is None or template == "outbound_refuel":
        for elig in refuel_eligibility:
            if not elig["outbound_ok"]:
                continue
            if refuel_label is not None and elig["label"] != refuel_label:
                continue
            r_wp = elig["wp"]
            # start→R is sampled at start_time both in the prefilter and
            # here, so the cached value is always the correct anchor.
            t_sR = elig["t_sR_min"]
            if not np.isfinite(t_sR) or t_sR > cycle_cap_min:
                continue
            anchor_2 = start_time + datetime.timedelta(
                minutes=t_sR + refuel_time_min
            )
            t_Rt, hw_Rt = _leg_time(
                aircraft=aircraft, start_wp=r_wp, end_wp=target,
                cruise_altitude=cruise_altitude, t_anchor=anchor_2,
                wind_source=wind_source,
                wind_sampling=wind_sampling,
                wind_sample_spacing=wind_sample_spacing,
                max_wind_samples_per_leg=max_wind_samples_per_leg,
            )
            if not np.isfinite(t_Rt):
                continue
            anchor_tr = anchor_2 + datetime.timedelta(
                minutes=t_Rt + on_station_min
            )
            t_tr, hw_tr = _leg_time(
                aircraft=aircraft, start_wp=target, end_wp=recovery_wp,
                cruise_altitude=cruise_altitude, t_anchor=anchor_tr,
                wind_source=wind_source,
                wind_sampling=wind_sampling,
                wind_sample_spacing=wind_sample_spacing,
                max_wind_samples_per_leg=max_wind_samples_per_leg,
            )
            if not np.isfinite(t_tr):
                continue
            cycle_1 = t_sR
            cycle_2 = t_Rt + on_station_min + t_tr
            day_total = cycle_1 + refuel_time_min + cycle_2
            sortie_margin = cycle_cap_min - max(cycle_1, cycle_2)
            day_margin = flight_day_budget_min - day_total
            if sortie_margin < 0 or day_margin < 0:
                continue
            candidates.append({
                "itinerary": "outbound_refuel",
                "refuel_airport": elig["label"],
                "refuel_count": 1,
                "refuel_time_min": refuel_time_min,
                "start_to_target_time_min": float("nan"),
                "start_to_refuel_time_min": t_sR,
                "refuel_to_target_time_min": t_Rt,
                "target_to_refuel_time_min": float("nan"),
                "refuel_to_return_time_min": float("nan"),
                "target_to_return_time_min": t_tr,
                "outbound_time_min": t_sR + refuel_time_min + t_Rt,
                "return_time_min": t_tr,
                "total_time_min": day_total,
                "outbound_headwind_kt": hw_Rt,  # R → target
                "return_headwind_kt": hw_tr,
                "day_total_time_min": day_total,
                "sortie_cycle_1_min": cycle_1,
                "sortie_cycle_2_min": cycle_2,
                "sortie_margin_min": sortie_margin,
                "day_margin_min": day_margin,
            })

    # --- return_refuel(R): start → target → R → recovery ------------------
    if template is None or template == "return_refuel":
        for elig in refuel_eligibility:
            if not elig["return_ok"]:
                continue
            if refuel_label is not None and elig["label"] != refuel_label:
                continue
            r_wp = elig["wp"]
            # Cycle 1's start→target leg was computed unconditionally
            # at the top of this function; bail this refuel placement
            # if that leg is unflyable.
            if not np.isfinite(t_st):
                continue
            anchor_tR = start_time + datetime.timedelta(
                minutes=t_st + on_station_min
            )
            t_tR, _ = _leg_time(
                aircraft=aircraft, start_wp=target, end_wp=r_wp,
                cruise_altitude=cruise_altitude, t_anchor=anchor_tR,
                wind_source=wind_source,
                wind_sampling=wind_sampling,
                wind_sample_spacing=wind_sample_spacing,
                max_wind_samples_per_leg=max_wind_samples_per_leg,
            )
            if not np.isfinite(t_tR):
                continue
            cycle_1 = t_st + on_station_min + t_tR
            if cycle_1 > cycle_cap_min:
                continue
            if _anchor_invariant(wind_source):
                # Wind sampled in the prefilter is anchor-invariant for
                # StillAirField/ConstantWindField — reuse it without
                # calling _leg_time again.
                t_Rrec = elig["t_Rrec_min"]
                hw_Rrec = elig["hw_Rrec_kt"]
            else:
                # Time-varying wind: R→recovery anchor depends on
                # cycle_1 (which depends on d), so the prefilter sample
                # is at the wrong time and we must recompute.
                anchor_Rrec = start_time + datetime.timedelta(
                    minutes=cycle_1 + refuel_time_min
                )
                t_Rrec, hw_Rrec = _leg_time(
                    aircraft=aircraft, start_wp=r_wp, end_wp=recovery_wp,
                    cruise_altitude=cruise_altitude, t_anchor=anchor_Rrec,
                    wind_source=wind_source,
                    wind_sampling=wind_sampling,
                    wind_sample_spacing=wind_sample_spacing,
                    max_wind_samples_per_leg=max_wind_samples_per_leg,
                )
            if not np.isfinite(t_Rrec):
                continue
            cycle_2 = t_Rrec
            day_total = cycle_1 + refuel_time_min + cycle_2
            sortie_margin = cycle_cap_min - max(cycle_1, cycle_2)
            day_margin = flight_day_budget_min - day_total
            if sortie_margin < 0 or day_margin < 0:
                continue
            candidates.append({
                "itinerary": "return_refuel",
                "refuel_airport": elig["label"],
                "refuel_count": 1,
                "refuel_time_min": refuel_time_min,
                "start_to_target_time_min": t_st,
                "start_to_refuel_time_min": float("nan"),
                "refuel_to_target_time_min": float("nan"),
                "target_to_refuel_time_min": t_tR,
                "refuel_to_return_time_min": t_Rrec,
                "target_to_return_time_min": float("nan"),
                "outbound_time_min": t_st,
                "return_time_min": t_tR + refuel_time_min + t_Rrec,
                "total_time_min": day_total,
                "outbound_headwind_kt": hw_st,  # start → target
                "return_headwind_kt": hw_Rrec,  # R → recovery
                "day_total_time_min": day_total,
                "sortie_cycle_1_min": cycle_1,
                "sortie_cycle_2_min": cycle_2,
                "sortie_margin_min": sortie_margin,
                "day_margin_min": day_margin,
            })

    # Sort feasible candidates by extension headroom — the itinerary
    # whose own per-itinerary boundary lies furthest beyond this d
    # comes first.  Callers consume ``[0]`` for "best" or scan the
    # whole list for "all alternatives."
    candidates.sort(
        key=lambda c: min(c["sortie_margin_min"], c["day_margin_min"]),
        reverse=True,
    )
    return candidates


def _solve_rays_refuel(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    recovery_wp: Waypoint,
    on_station_min: float,
    sortie_budget_min: float,
    flight_day_budget_min: float,
    reserve_min: float,
    refuel_time_min: float,
    refuel_eligibility: list[dict[str, Any]],
    azimuths_deg: npt.NDArray[np.floating[Any]],
    distance_tolerance_nmi: float,
    wind_sampling: str = "cruise_midpoint",
    wind_sample_spacing: Quantity = 100 * ureg.nautical_mile,
    max_wind_samples_per_leg: int = _DEFAULT_MAX_WIND_SAMPLES,
) -> list[dict[str, Any]]:
    """Per-ray expanding-bracket + binary-search across three itinerary
    templates.  Scalar per ray (no cross-ray vectorization) — simpler than
    the plain solver, and acceptable since each probe already calls
    ``_leg_time`` 2–3 times.
    """
    cruise_tas_kt = aircraft.cruise_speed_at(cruise_altitude).m_as(ureg.knot)
    tol_min = distance_tolerance_nmi / max(cruise_tas_kt, 1.0) * 60.0

    # Initial bracket: budget the day clock since refueling can use it
    # all.  Borrow the existing heuristic with the larger of the two
    # budgets.
    initial_hi = _initial_bracket_distance_nmi(
        aircraft=aircraft,
        cruise_altitude=cruise_altitude,
        mode="round_trip",
        on_station_min=on_station_min,
        budget_min=flight_day_budget_min,
        reserve_min=0.0,  # day clock has no reserve
    )

    def _make_target(az: float, d_nmi: float) -> Waypoint:
        lats, lons = _target_coordinates(
            start, np.array([d_nmi]), np.array([az]),
        )
        return Waypoint(
            latitude=float(lats[0]),
            longitude=float(lons[0]),
            heading=float(az),
            altitude_msl=cruise_altitude,
        )

    specs: list[tuple[str, str | None]] = [("direct", None)]
    for elig in refuel_eligibility:
        if elig["outbound_ok"]:
            specs.append(("outbound_refuel", elig["label"]))
        if elig["return_ok"]:
            specs.append(("return_refuel", elig["label"]))

    rows: list[dict[str, Any]] = []
    for az in azimuths_deg:
        az_f = float(az)
        solved: list[tuple[float, dict[str, Any]]] = []

        for template, refuel_label in specs:
            def _evaluate_spec(
                d_nmi: float,
                *,
                template: str = template,
                refuel_label: str | None = refuel_label,
            ) -> dict[str, Any] | None:
                target = _make_target(az_f, d_nmi)
                cands = _evaluate_refuel_at_d(
                    aircraft=aircraft,
                    start=start,
                    target=target,
                    recovery_wp=recovery_wp,
                    cruise_altitude=cruise_altitude,
                    start_time=start_time,
                    wind_source=wind_source,
                    on_station_min=on_station_min,
                    sortie_budget_min=sortie_budget_min,
                    flight_day_budget_min=flight_day_budget_min,
                    reserve_min=reserve_min,
                    refuel_time_min=refuel_time_min,
                    refuel_eligibility=refuel_eligibility,
                    template=template,
                    refuel_label=refuel_label,
                    wind_sampling=wind_sampling,
                    wind_sample_spacing=wind_sample_spacing,
                    max_wind_samples_per_leg=max_wind_samples_per_leg,
                )
                return cands[0] if cands else None

            if _evaluate_spec(0.0) is None:
                continue

            d_lo = 0.0
            d_hi = initial_hi
            if _evaluate_spec(d_hi) is not None:
                while True:
                    d_lo = d_hi
                    d_hi *= 2.0
                    if d_hi > 20000.0:
                        raise HyPlanRuntimeError(
                            f"Refuel-isochrone bracket exceeded 20000 nmi "
                            f"at azimuth {az_f:.1f}°."
                        )
                    if _evaluate_spec(d_hi) is None:
                        break

            while d_hi - d_lo > distance_tolerance_nmi:
                d_mid = 0.5 * (d_lo + d_hi)
                if _evaluate_spec(d_mid) is not None:
                    d_lo = d_mid
                else:
                    d_hi = d_mid

            final_for_spec = _evaluate_spec(d_lo)
            if final_for_spec is not None:
                solved.append((d_lo, final_for_spec))

        if not solved:
            rows.append(_unflyable_refuel_row(
                azimuth_deg=az_f, start=start,
                on_station_min=on_station_min,
            ))
            continue

        d_lo, final = max(
            solved,
            key=lambda item: (
                item[0],
                min(item[1]["sortie_margin_min"], item[1]["day_margin_min"]),
            ),
        )

        target = _make_target(az_f, d_lo)
        hw_o = final["outbound_headwind_kt"]
        hw_r = final["return_headwind_kt"]

        # Determine binding constraint.
        sm = final["sortie_margin_min"]
        dm = final["day_margin_min"]
        if sm <= tol_min and dm <= tol_min:
            limiting = "both"
        elif sm <= tol_min:
            limiting = "sortie"
        elif dm <= tol_min:
            limiting = "flight_day"
        else:
            limiting = "slack"

        row = {
            "azimuth_deg": az_f,
            "distance_nmi": float(d_lo),
            "target_lat": target.latitude,
            "target_lon": target.longitude,
            "on_station_min": on_station_min,
            "net_headwind_kt": 0.5 * (hw_o + hw_r),
            "headwind_asymmetry_kt": 0.5 * (hw_o - hw_r),
            "limiting_leg": limiting,
        }
        row.update(final)
        rows.append(row)

    return rows


def _unflyable_refuel_row(
    *,
    azimuth_deg: float,
    start: Waypoint,
    on_station_min: float,
) -> dict[str, Any]:
    nan = float("nan")
    return {
        "azimuth_deg": azimuth_deg,
        "distance_nmi": 0.0,
        "target_lat": start.latitude,
        "target_lon": start.longitude,
        "on_station_min": on_station_min,
        "itinerary": "unflyable",
        "refuel_airport": None,
        "refuel_count": 0,
        "refuel_time_min": nan,
        "start_to_target_time_min": nan,
        "start_to_refuel_time_min": nan,
        "refuel_to_target_time_min": nan,
        "target_to_refuel_time_min": nan,
        "refuel_to_return_time_min": nan,
        "target_to_return_time_min": nan,
        "outbound_time_min": 0.0,
        "return_time_min": 0.0,
        "total_time_min": 0.0,
        "outbound_headwind_kt": nan,
        "return_headwind_kt": nan,
        "net_headwind_kt": nan,
        "headwind_asymmetry_kt": nan,
        "day_total_time_min": 0.0,
        "sortie_cycle_1_min": 0.0,
        "sortie_cycle_2_min": nan,
        "sortie_margin_min": 0.0,
        "day_margin_min": 0.0,
        "limiting_leg": "unflyable",
    }
