"""Shared utilities for wind field modules.

Includes lazy-import helpers, authentication, wind vector conversions,
and wind-correction functions for flight planning.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from pint import Quantity

from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from ..units import ureg

if TYPE_CHECKING:
    from .base import WindField


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _require_xarray():
    """Import and return xarray, raising a clear error if not installed."""
    try:
        import xarray as xr
        return xr
    except ImportError:
        raise HyPlanRuntimeError(
            "xarray and netcdf4 are required for gridded wind fields. "
            "Install them with: pip install hyplan[winds]"
        )


def _earthdata_login():
    """Authenticate with NASA Earthdata using ``earthaccess``.

    Tries strategies in order: ``EARTHDATA_TOKEN`` env var, ``~/.netrc``,
    then interactive prompt.  Returns an authenticated ``requests.Session``
    with a bearer token suitable for OPeNDAP access.

    Raises :class:`~hyplan.exceptions.HyPlanRuntimeError` if ``earthaccess``
    is not installed or login fails.

    .. deprecated::
        This is a thin wrapper around :func:`hyplan._auth._earthdata_login`.
    """
    from .._auth import _earthdata_login as _login

    return _login()


# ---------------------------------------------------------------------------
# Wind vector conversions
# ---------------------------------------------------------------------------

def wind_uv_from_speed_dir(speed_mps: float, from_deg: float) -> Tuple[float, float]:
    """Convert meteorological wind (speed, direction-from) to (u, v) components.

    Args:
        speed_mps: Wind speed in m/s.
        from_deg: Direction the wind is blowing *from* in degrees true
            (meteorological convention: 0 = from north, 90 = from east).

    Returns:
        Tuple of (u, v) in m/s where u is eastward and v is northward.
    """
    from_rad = np.radians(from_deg)
    u = -speed_mps * np.sin(from_rad)
    v = -speed_mps * np.cos(from_rad)
    return float(u), float(v)


def wind_speed_dir_from_uv(u_mps: float, v_mps: float) -> Tuple[float, float]:
    """Convert (u, v) wind components to meteorological (speed, direction-from).

    Args:
        u_mps: Eastward wind component in m/s.
        v_mps: Northward wind component in m/s.

    Returns:
        Tuple of (speed_mps, from_deg) where from_deg is in [0, 360).
        For zero wind, returns (0.0, 0.0).
    """
    speed = float(np.hypot(u_mps, v_mps))
    if speed < 1e-12:
        return 0.0, 0.0
    from_deg = float(np.degrees(np.arctan2(-u_mps, -v_mps))) % 360.0
    return speed, from_deg


# ---------------------------------------------------------------------------
# Wind-correction functions for flight planning
# ---------------------------------------------------------------------------

def _wind_factor(
    tas: Quantity,
    heading_deg: float,
    wind_speed: Optional[Quantity],
    wind_from_deg: Optional[float],
) -> float:
    """Multiplicative wind-correction factor for a segment's no-wind time.

    Given the true airspeed and a constant wind vector (speed + direction
    *from which* the wind is blowing, per meteorological convention), returns
    ``TAS / ground_speed`` where

        ground_speed = TAS - wind_speed * cos(wind_from_deg - heading_deg)

    Multiply a no-wind ``distance / TAS`` time by this factor to obtain the
    wind-corrected time. Returns 1.0 when no wind is supplied. Crosswind
    effects on ground speed are ignored (valid small-angle approximation
    that matches standard pre-flight planning practice).

    Raises:
        HyPlanValueError: If the headwind exceeds TAS, yielding a
            non-positive ground speed (unflyable).
    """
    if wind_speed is None or wind_speed.magnitude == 0 or wind_from_deg is None:
        return 1.0
    if heading_deg is None:
        return 1.0
    rel_angle_rad = np.radians(wind_from_deg - heading_deg)
    headwind = wind_speed * float(np.cos(rel_angle_rad))
    ground_speed = tas - headwind
    if ground_speed.m_as(ureg.knot) <= 0:
        raise HyPlanValueError(
            f"Headwind {headwind.to(ureg.knot):.1f} exceeds TAS "
            f"{tas.to(ureg.knot):.1f} on heading {heading_deg:.0f}°; unflyable."
        )
    return float((tas / ground_speed).to_base_units().magnitude)


def _wind_factor_from_uv(
    tas: Quantity,
    heading_deg: float,
    u: Quantity,
    v: Quantity,
) -> float:
    """Multiplicative wind-correction factor from U/V components.

    Args:
        tas: True airspeed.
        heading_deg: Segment heading in degrees true.
        u: Eastward wind component (positive = from west).
        v: Northward wind component (positive = from south).

    Returns:
        ``TAS / ground_speed`` as a float.
    """
    if heading_deg is None:
        return 1.0
    heading_rad = np.radians(heading_deg)
    u_mps = u.m_as(ureg.meter / ureg.second)
    v_mps = v.m_as(ureg.meter / ureg.second)
    # Headwind is the component of wind opposing the direction of flight.
    # Flight direction unit vector: (sin(hdg), cos(hdg)).
    # Wind vector: (u, v).
    # Tailwind component (wind along flight direction) = u*sin(hdg) + v*cos(hdg)
    # Headwind = -tailwind
    headwind_mps = -(u_mps * np.sin(heading_rad) + v_mps * np.cos(heading_rad))

    tas_mps = tas.m_as(ureg.meter / ureg.second)
    ground_speed_mps = tas_mps - headwind_mps  # TAS - headwind = groundspeed

    if ground_speed_mps <= 0:
        raise HyPlanValueError(
            f"Headwind {headwind_mps:.1f} m/s exceeds TAS "
            f"{tas_mps:.1f} m/s on heading {heading_deg:.0f}°; unflyable."
        )
    return tas_mps / ground_speed_mps  # type: ignore[no-any-return]


def _resolve_wind_factor(
    tas: Quantity,
    heading_deg: float,
    lat: float,
    lon: float,
    altitude: Quantity,
    segment_time: Optional[datetime.datetime],
    wind_source: Optional["WindField"],
    wind_speed: Optional[Quantity],
    wind_direction: Optional[float],
) -> float:
    """Compute wind factor using wind_source (preferred) or legacy scalars."""
    if wind_source is not None:
        u, v = wind_source.wind_at(lat, lon, altitude, segment_time)  # type: ignore[arg-type]
        return _wind_factor_from_uv(tas, heading_deg, u, v)
    return _wind_factor(tas, heading_deg, wind_speed, wind_direction)


def _track_hold_solution_from_uv(
    tas: Quantity,
    track_deg: float,
    u: Quantity,
    v: Quantity,
) -> dict:
    """Solve for crab angle and groundspeed when holding a desired ground track.

    Given TAS, desired track, and wind (u, v), computes the heading the
    aircraft must fly to maintain the track, the resulting crab angle,
    and the along-track groundspeed.

    Args:
        tas: True airspeed.
        track_deg: Desired ground-track azimuth (degrees true).
        u: Eastward wind component (positive = from west).
        v: Northward wind component (positive = from south).

    Returns:
        Dict with keys: ``track_deg``, ``heading_deg``, ``crab_angle_deg``,
        ``groundspeed``, ``alongtrack_wind``, ``crosstrack_wind``.

    Raises:
        HyPlanValueError: If crosswind exceeds TAS (track cannot be held)
            or if resulting groundspeed is non-positive (unflyable).
    """
    tas_mps = tas.m_as(ureg.meter / ureg.second)
    u_mps = u.m_as(ureg.meter / ureg.second)
    v_mps = v.m_as(ureg.meter / ureg.second)

    track_rad = np.radians(track_deg)

    # Decompose wind into along-track and cross-track components
    # Track unit vector: (sin(track), cos(track))
    tailwind = u_mps * np.sin(track_rad) + v_mps * np.cos(track_rad)
    crosswind = u_mps * np.cos(track_rad) - v_mps * np.sin(track_rad)

    # Crab angle: aircraft must point into the crosswind
    sin_crab = -crosswind / tas_mps
    sin_crab = float(np.clip(sin_crab, -1.0, 1.0))

    if abs(crosswind) > tas_mps:
        raise HyPlanValueError(
            f"Crosswind {abs(crosswind):.1f} m/s exceeds TAS "
            f"{tas_mps:.1f} m/s on track {track_deg:.0f}°; "
            f"cannot hold desired ground track."
        )

    crab_rad = np.arcsin(sin_crab)
    crab_deg = float(np.degrees(crab_rad))
    heading_deg = (track_deg + crab_deg) % 360.0

    # Along-track groundspeed
    groundspeed_mps = tas_mps * np.cos(crab_rad) + tailwind

    if groundspeed_mps <= 0:
        raise HyPlanValueError(
            f"Groundspeed {groundspeed_mps:.1f} m/s is non-positive on "
            f"track {track_deg:.0f}° (headwind {-tailwind:.1f} m/s, "
            f"TAS {tas_mps:.1f} m/s); unflyable."
        )

    return {
        "track_deg": track_deg,
        "heading_deg": heading_deg,
        "crab_angle_deg": crab_deg,
        "groundspeed": groundspeed_mps * ureg.meter / ureg.second,
        "alongtrack_wind": tailwind * ureg.meter / ureg.second,
        "crosstrack_wind": crosswind * ureg.meter / ureg.second,
    }


def _resolve_track_hold_solution(
    tas: Quantity,
    track_deg: float,
    lat: float,
    lon: float,
    altitude: Quantity,
    segment_time: Optional[datetime.datetime],
    wind_source: Optional["WindField"],
    wind_speed: Optional[Quantity],
    wind_direction: Optional[float],
) -> dict:
    """Compute track-hold solution using wind_source or legacy scalars.

    Returns a no-wind identity solution when no wind is provided.
    """
    u = None
    v = None

    if wind_source is not None:
        u, v = wind_source.wind_at(lat, lon, altitude, segment_time)  # type: ignore[arg-type]
    elif wind_speed is not None and wind_speed.magnitude != 0 and wind_direction is not None:
        ws = wind_speed.m_as(ureg.meter / ureg.second)
        wind_from_rad = np.radians(wind_direction)
        u = float(-ws * np.sin(wind_from_rad)) * ureg.meter / ureg.second
        v = float(-ws * np.cos(wind_from_rad)) * ureg.meter / ureg.second

    if u is None or v is None:
        # No wind — identity solution
        return {
            "track_deg": track_deg,
            "heading_deg": track_deg,
            "crab_angle_deg": 0.0,
            "groundspeed": tas,
            "alongtrack_wind": 0.0 * ureg.meter / ureg.second,
            "crosstrack_wind": 0.0 * ureg.meter / ureg.second,
        }

    return _track_hold_solution_from_uv(tas, track_deg, u, v)


def _resolve_wind_uv(
    lat: float,
    lon: float,
    altitude: Quantity,
    segment_time: Optional[datetime.datetime],
    wind_source: Optional["WindField"],
    wind_speed: Optional[Quantity],
    wind_direction: Optional[float],
) -> Optional[Tuple[float, float]]:
    """Extract wind as ``(u_east, v_north)`` in m/s for :class:`DubinsPath3D`.

    Returns ``None`` when no wind is available (still-air path).
    """
    if wind_source is not None:
        u, v = wind_source.wind_at(lat, lon, altitude, segment_time)  # type: ignore[arg-type]
        u_mps = u.m_as(ureg.meter / ureg.second)
        v_mps = v.m_as(ureg.meter / ureg.second)
        if abs(u_mps) < 1e-10 and abs(v_mps) < 1e-10:
            return None
        return (u_mps, v_mps)
    if wind_speed is not None and wind_speed.magnitude != 0 and wind_direction is not None:
        ws = wind_speed.m_as(ureg.meter / ureg.second)
        wind_from_rad = np.radians(wind_direction)
        u_mps = float(-ws * np.sin(wind_from_rad))
        v_mps = float(-ws * np.cos(wind_from_rad))
        return (u_mps, v_mps)
    return None
