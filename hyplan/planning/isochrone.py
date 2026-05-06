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

Limitations
-----------

v1 applies the wind field only to the cruise segment of each leg.  Climb
and descent are computed in still air.  This matches the simplest
interpretation of the underlying ``Aircraft._climb`` / ``_descend`` calls
when no ``wind_along_track`` is provided.  Vertically-varying wind (a
jet stream that's stronger in the climb-out band than at cruise altitude
and ditto for descent) is therefore not captured.  Per-phase wind
sampling — sampling once at climb-mid altitude, once at cruise altitude,
once at descent-mid altitude — is a planner-wide concern rather than an
isochrone-specific one (it would also tighten ``compute_flight_plan``
and the ER-2 ``planned_vs_flown`` validation), and is deferred to a
follow-up improvement to ``Aircraft._hybrid_path`` / its callers.
"""

from __future__ import annotations

import datetime
from typing import Optional, Tuple, Union

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import Point, Polygon

from ..aircraft._base import Aircraft
from ..airports import Airport
from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from ..geometry import wrap_to_180
from ..units import ureg
from ..waypoint import Waypoint
from ..winds.base import WindField
from ..winds.simple import ConstantWindField, StillAirField
from ..winds.utils import _track_hold_solution_from_uv

__all__ = [
    "compute_isochrone",
    "isochrone_polygon",
    "plot_isochrone",
]


_VALID_MODES = ("one_way", "round_trip", "return_safe")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_isochrone(
    aircraft: Aircraft,
    start: Union[Airport, Waypoint],
    budget: Quantity,
    *,
    cruise_altitude: Optional[Quantity] = None,
    on_station_altitude: Optional[Quantity] = None,
    start_time: Optional[datetime.datetime] = None,
    wind_source: Optional[WindField] = None,
    return_destination: Union[Airport, Waypoint, None] = None,
    mode: str = "round_trip",
    on_station_time: Quantity = 0 * ureg.minute,
    reserve: Quantity = 0 * ureg.minute,
    azimuth_resolution_deg: float = 5.0,
    distance_tolerance_nmi: float = 0.5,
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
    if mode not in _VALID_MODES:
        raise HyPlanValueError(
            f"Invalid mode {mode!r}; must be one of {_VALID_MODES}."
        )
    # Coerce Airport → Waypoint (uniform downstream handling).
    if isinstance(start, Airport):
        start = _airport_or_wp_to_waypoint(start)
    if start.altitude_msl is None:
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

    budget_min = budget.m_as(ureg.minute)
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
    if budget_min <= reserve_min:
        raise HyPlanValueError(
            f"`budget` ({budget_min:.1f} min) must exceed `reserve` "
            f"({reserve_min:.1f} min); otherwise no time is available "
            f"to fly."
        )

    if cruise_altitude is None:
        cruise_altitude = start.altitude_msl

    if cruise_altitude.m_as(ureg.meter) < start.altitude_msl.m_as(ureg.meter):
        raise HyPlanValueError(
            f"v1 requires `cruise_altitude` "
            f"({cruise_altitude.m_as(ureg.feet):.0f} ft) to be at or "
            f"above `start.altitude_msl` "
            f"({start.altitude_msl.m_as(ureg.feet):.0f} ft).  Initial "
            f"descent to cruise is deferred — for now, set "
            f"`cruise_altitude` to the current altitude or higher."
        )

    if on_station_altitude is not None:
        if not np.isclose(
            on_station_altitude.m_as(ureg.meter),
            cruise_altitude.m_as(ureg.meter),
        ):
            raise HyPlanValueError(
                "v1 requires `on_station_altitude` to equal "
                "`cruise_altitude` (or be None).  Multi-altitude "
                "isochrones are deferred."
            )

    # --- return destination resolution + mode-specific rules ---------------
    return_wp: Optional[Waypoint]
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
    azimuths = np.arange(0.0, 360.0, azimuth_resolution_deg)
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
    )

    df = pd.DataFrame(rows)
    geometry = [Point(lon, lat) for lat, lon in zip(df["target_lat"], df["target_lon"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Stash invocation context for the plotter / consumers.
    gdf.attrs.update({
        "mode": mode,
        "start_lat": start.latitude,
        "start_lon": start.longitude,
        "start_altitude_ft": start.altitude_msl.m_as(ureg.feet),
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
    })
    return gdf


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
    base_map: Optional[folium.Map] = None,
    *,
    color: str = "steelblue",
    fill_opacity: float = 0.2,
    tiles: str = "OpenStreetMap",
    zoom_start: int = 6,
) -> folium.Map:
    """Render the isochrone on a Folium map.

    Args:
        gdf: A GeoDataFrame returned by :func:`compute_isochrone`.
        base_map: Optional existing Folium map to add to.  If ``None``,
            a new one is created centered on ``start`` with the
            ``tiles`` basemap loaded.
        color: Polygon stroke + fill color.
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
    folium.Polygon(
        locations=[(y, x) for x, y in poly.exterior.coords],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        weight=2,
        popup=f"Isochrone — mode={attrs.get('mode', '?')}, "
              f"budget={attrs.get('budget_hr', float('nan')):.1f} hr",
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
    for _, row in gdf.iterrows():
        popup_html = (
            f"<b>az {row['azimuth_deg']:.0f}°</b><br>"
            f"distance: {row['distance_nmi']:.1f} nmi<br>"
            f"outbound: {row['outbound_time_min']:.1f} min<br>"
        )
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
        folium.CircleMarker(
            location=[row["target_lat"], row["target_lon"]],
            radius=2,
            color=color,
            fill=True,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(base_map)

    return base_map


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _airport_or_wp_to_waypoint(
    obj: Union[Airport, Waypoint],
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
    dest: Union[Airport, Waypoint],
) -> Waypoint:
    """Coerce a return-destination argument to a Waypoint."""
    return _airport_or_wp_to_waypoint(dest, require_altitude=True)


def _destination_label(dest: Union[Airport, Waypoint]) -> str:
    if isinstance(dest, Airport):
        return str(dest.icao_code)
    return dest.name or f"({dest.latitude:.2f}, {dest.longitude:.2f})"


def _unflyable_ray(
    *,
    azimuth_deg: float,
    start: Waypoint,
    mode: str,
    on_station_min: float,
) -> dict:
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
    distances_nmi: np.ndarray,
    azimuths_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def _solve_rays(
    *,
    aircraft: Aircraft,
    start: Waypoint,
    cruise_altitude: Quantity,
    start_time: datetime.datetime,
    wind_source: WindField,
    return_wp: Optional[Waypoint],
    mode: str,
    on_station_min: float,
    budget_min: float,
    reserve_min: float,
    azimuths_deg: np.ndarray,
    distance_tolerance_nmi: float,
) -> list[dict]:
    """Solve all radial rays, vectorizing candidate geometry per iteration."""
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

    d_lo = np.zeros(n_rays, dtype=float)
    d_hi = np.full(n_rays, initial_hi, dtype=float)
    active = np.ones(n_rays, dtype=bool)
    zero_unflyable = np.zeros(n_rays, dtype=bool)

    def _evaluate_many(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Evaluate feasibility for selected ray indices."""
        az = azimuths_deg[indices]
        target_lats, target_lons = _target_coordinates(start, distances, az)
        totals = np.empty(len(indices), dtype=float)
        diags: list[dict] = []

        for j, idx in enumerate(indices):
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

    # Expanding bracket.  Work on all still-feasible rays at once.
    active_indices = np.flatnonzero(active)
    totals, _ = _evaluate_many(active_indices, d_hi[active_indices])
    infeasible_seed = ~np.isfinite(totals)
    if np.any(infeasible_seed):
        seed_indices = active_indices[infeasible_seed]
        zero_unflyable[seed_indices] = True
        active[seed_indices] = False

    feasible = np.isfinite(totals) & (totals <= feasible_budget_min)
    expanding = np.zeros(n_rays, dtype=bool)
    expanding[active_indices[feasible]] = True

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
    rows: list[dict] = []
    final_indices = np.flatnonzero(active)
    final_by_index: dict[int, dict] = {}
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


def _leg_time(
    *,
    aircraft: Aircraft,
    start_wp: Waypoint,
    end_wp: Waypoint,
    cruise_altitude: Quantity,
    t_anchor: datetime.datetime,
    wind_source: WindField,
) -> Tuple[float, float]:
    """Compute leg time (minutes) and along-track headwind (knots).

    Implements the v1 leg-timing recipe:

    1. Geodetic distance + bearing from ``start_wp`` to ``end_wp``.
    2. If ``start_wp.altitude_msl < cruise_altitude``, climb to
       ``cruise_altitude`` in still air.  Wind during climb is a known
       v1 simplification — see "Limitations" in the module docstring.
    3. Cruise from end-of-climb to top-of-descent.  Crab + groundspeed
       are solved via :func:`_track_hold_solution_from_uv` so a pure
       crosswind correctly slows the aircraft (it must crab into the
       wind to hold the great-circle track) and an unflyable wind
       (crosswind > TAS or groundspeed ≤ 0) returns infinity, marking
       the ray as infeasible.
    4. If ``end_wp.altitude_msl < cruise_altitude``, descend in still
       air.  Same v1 simplification as climb.
    5. Cruise wind sampled at the cruise-leg midpoint and at time
       ``t_anchor + t_climb + t_cruise/2``.  Up to three fixed-point
       iterations on cruise time, early exit when ``|Δt_cruise| < 5 sec``.

    Returns ``(time_minutes, headwind_kt)``.  ``headwind_kt`` is the
    signed *cruise-leg* along-track headwind (positive = headwind,
    negative = tailwind); ``float("inf")`` when the leg is unflyable.
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

    # Climb segment (still-air; v1 simplification — see module docstring).
    if start_alt < cruise_altitude:
        t_climb_q, d_climb_q = aircraft._climb(start_alt, cruise_altitude)
        t_climb_min = t_climb_q.m_as(ureg.minute)
        d_climb_nmi = d_climb_q.m_as(ureg.nautical_mile)
    else:
        t_climb_min = 0.0
        d_climb_nmi = 0.0

    # Descent segment (still-air; v1 simplification).
    if end_alt < cruise_altitude:
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

    # Cruise midpoint (geographic): the geographic midpoint of the
    # cruise segment, which lies between (d_climb_nmi) and (distance −
    # d_desc_nmi) along the bearing.
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
    t_cruise_min_prev: Optional[float] = None

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
            # Crosswind > TAS or unflyable headwind.  Mark this leg
            # infeasible — the binary search above will pull the
            # boundary back.
            return float("inf"), float("inf")

        gs_kt = sol["groundspeed"].m_as(ureg.knot)
        # Along-track headwind = −(along-track tailwind) in knots.
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
