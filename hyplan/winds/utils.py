"""Shared utilities for wind field modules.

Includes lazy-import helpers, authentication, and wind vector conversions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..exceptions import HyPlanRuntimeError


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
    """
    try:
        import earthaccess
    except ImportError:
        raise HyPlanRuntimeError(
            "earthaccess is required for NASA Earthdata authentication. "
            "Install with: pip install hyplan[winds]"
        )

    # Try non-interactive strategies first
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth.authenticated:
                return earthaccess.get_requests_https_session()
        except Exception:
            continue

    raise HyPlanRuntimeError(
        "NASA Earthdata login failed. Authenticate via one of:\n"
        "  1. Set EARTHDATA_TOKEN environment variable\n"
        "  2. Add to ~/.netrc:\n"
        "     machine urs.earthdata.nasa.gov login <user> password <pass>\n"
        "Register at https://urs.earthdata.nasa.gov if needed."
    )


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
