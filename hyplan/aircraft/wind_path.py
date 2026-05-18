"""Wind-aware climb and descent helpers.

These functions wrap :meth:`Aircraft._climb` / :meth:`Aircraft._descend`
with wind-field sampling so callers don't have to reproduce the
geometry-and-time-anchor dance.

What the helpers actually do
----------------------------

Wind affects the *ground distance* covered during climb and descent
(a tailwind extends the ground arc; a headwind shortens it), not the
times themselves — climb rate is air-mass-relative and independent of
along-track wind.  So the recipe is:

1. Run :meth:`Aircraft._climb` (or ``_descend``) once with no wind to
   seed the still-air phase time and ground distance.
2. Sample the wind at the geometric midpoint of the seeded arc, at the
   phase-mid altitude, at the correct cumulative time anchor.
3. Project the sampled (u, v) onto the great-circle bearing to get a
   signed along-track wind (positive = tailwind).
4. Re-run ``_climb`` / ``_descend`` with ``wind_along_track`` set; the
   returned ground distance now reflects the wind, but the returned
   time is unchanged.

This is the "phase_midpoint" recipe used by the isochrone solver under
``wind_sampling="phase_midpoint"`` and ``"segmented_cruise"``, and is
suitable for any planner-physics code that needs to allocate ground
distance between climb / descent / cruise under a time- or
space-varying wind.
"""

from __future__ import annotations

import datetime

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from ..geometry import wrap_to_180
from ..units import ureg
from ..winds.base import WindField
from ..winds.simple import StillAirField
from ._base import Aircraft

__all__ = [
    "climb_with_wind_field",
    "descend_with_wind_field",
]


def _project_wind_along_track(
    *,
    wind_source: WindField,
    start_lat: float,
    start_lon: float,
    track_deg: float,
    mid_dist_nmi: float,
    altitude: Quantity,
    sample_time: datetime.datetime,
) -> Quantity:
    """Sample wind at one point along a great-circle bearing and project
    it onto the track.

    Returns a :class:`pint.Quantity` in knots; positive = tailwind.
    For :class:`StillAirField` the function short-circuits to ``0 kt``
    without querying the provider.
    """
    if isinstance(wind_source, StillAirField):
        return 0.0 * ureg.knot
    mid_lat, mid_lon = pymap3d.vincenty.vreckon(
        start_lat, start_lon, mid_dist_nmi * 1852.0, track_deg,
    )
    u_q, v_q = wind_source.wind_at(
        float(mid_lat), float(wrap_to_180(float(mid_lon))),
        altitude, sample_time,
    )
    # Track unit vector: (sin(track), cos(track)) in (u, v) convention.
    track_rad = float(np.radians(track_deg))
    sin_t = float(np.sin(track_rad))
    cos_t = float(np.cos(track_rad))
    return (u_q * sin_t + v_q * cos_t).to(ureg.knot)


def climb_with_wind_field(
    aircraft: Aircraft,
    *,
    start_lat: float,
    start_lon: float,
    start_alt: Quantity,
    cruise_alt: Quantity,
    track_deg: float,
    t_anchor: datetime.datetime,
    wind_source: WindField,
) -> tuple[Quantity, Quantity, Quantity]:
    """Wind-aware climb time + ground distance.

    Args:
        aircraft: The aircraft model.
        start_lat, start_lon: Departure point (decimal degrees).
        start_alt: Starting altitude (Quantity).
        cruise_alt: Top-of-climb altitude (Quantity).
        track_deg: Great-circle bearing along which the aircraft is
            climbing (degrees true).
        t_anchor: Wall-clock time at the start of the climb.
        wind_source: A :class:`WindField` provider.  Wind is sampled at
            the geometric midpoint of the still-air-seeded climb arc,
            at altitude ``(start_alt + cruise_alt) / 2``, at time
            ``t_anchor + 0.5 * t_climb_seed``.

    Returns:
        ``(t_climb, d_climb, along_track_wind_kt)`` where ``t_climb``
        is in minutes (Quantity), ``d_climb`` is the ground distance
        in nautical miles (Quantity), and ``along_track_wind_kt`` is
        the signed wind component (positive = tailwind, knots) used
        in the wind-corrected climb call.
    """
    t_seed_q, d_seed_q = aircraft._climb(start_alt, cruise_alt)
    t_seed_min = t_seed_q.m_as(ureg.minute)
    d_seed_nmi = d_seed_q.m_as(ureg.nautical_mile)
    along_track_wind = _project_wind_along_track(
        wind_source=wind_source,
        start_lat=start_lat,
        start_lon=start_lon,
        track_deg=track_deg,
        mid_dist_nmi=max(1e-3, 0.5 * d_seed_nmi),
        altitude=(start_alt + cruise_alt) / 2,
        sample_time=(
            t_anchor + datetime.timedelta(minutes=0.5 * t_seed_min)
        ),
    )
    t_climb_q, d_climb_q = aircraft._climb(
        start_alt, cruise_alt, wind_along_track=along_track_wind,
    )
    return t_climb_q, d_climb_q, along_track_wind


def descend_with_wind_field(
    aircraft: Aircraft,
    *,
    start_lat: float,
    start_lon: float,
    total_distance_nmi: float,
    cruise_alt: Quantity,
    end_alt: Quantity,
    track_deg: float,
    t_anchor: datetime.datetime,
    t_climb_min: float,
    d_climb_nmi: float,
    wind_source: WindField,
) -> tuple[Quantity, Quantity, Quantity]:
    """Wind-aware descent time + ground distance.

    The descent midpoint sits at ``total_distance_nmi − d_descent / 2``
    along the bearing from ``(start_lat, start_lon)``; the sample time
    is anchored after the climb + still-air cruise + half the descent.

    Args:
        aircraft: The aircraft model.
        start_lat, start_lon: Departure point (decimal degrees) — the
            origin of the bearing along which the descent's midpoint
            sits.
        total_distance_nmi: Geodesic length of the leg.
        cruise_alt: Top-of-descent altitude (Quantity).
        end_alt: Recovery altitude (Quantity).
        track_deg: Great-circle bearing (degrees true).
        t_anchor: Wall-clock time at the start of the leg (matches
            :func:`climb_with_wind_field`'s ``t_anchor``).
        t_climb_min: Already-computed climb time in minutes.  Use the
            return from :func:`climb_with_wind_field` for consistency.
        d_climb_nmi: Already-computed climb ground distance in nautical
            miles.
        wind_source: A :class:`WindField` provider.

    Returns:
        ``(t_descent, d_descent, along_track_wind_kt)``.  Same units
        and conventions as :func:`climb_with_wind_field`.
    """
    t_seed_q, d_seed_q = aircraft._descend(cruise_alt, end_alt)
    t_seed_min = t_seed_q.m_as(ureg.minute)
    d_seed_nmi = d_seed_q.m_as(ureg.nautical_mile)
    cruise_tas_kt = aircraft.cruise_speed_at(cruise_alt).m_as(ureg.knot)
    cruise_seed_nmi = max(0.0, total_distance_nmi - d_climb_nmi - d_seed_nmi)
    cruise_seed_min = cruise_seed_nmi / cruise_tas_kt * 60.0
    along_track_wind = _project_wind_along_track(
        wind_source=wind_source,
        start_lat=start_lat,
        start_lon=start_lon,
        track_deg=track_deg,
        mid_dist_nmi=max(1e-3, total_distance_nmi - 0.5 * d_seed_nmi),
        altitude=(cruise_alt + end_alt) / 2,
        sample_time=(
            t_anchor + datetime.timedelta(
                minutes=t_climb_min + cruise_seed_min + 0.5 * t_seed_min,
            )
        ),
    )
    t_desc_q, d_desc_q = aircraft._descend(
        cruise_alt, end_alt, wind_along_track=along_track_wind,
    )
    return t_desc_q, d_desc_q, along_track_wind
