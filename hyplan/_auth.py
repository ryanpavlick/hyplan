"""Shared NASA Earthdata authentication utilities."""

from __future__ import annotations

from .exceptions import HyPlanRuntimeError


def _require_earthaccess():
    """Import and return earthaccess, raising a clear error if not installed."""
    try:
        import earthaccess

        return earthaccess
    except ImportError:
        raise HyPlanRuntimeError(
            "earthaccess is required for NASA Earthdata authentication. "
            "Install with: pip install earthaccess"
        )


def _earthdata_login():
    """Authenticate with NASA Earthdata using ``earthaccess``.

    Tries strategies in order: ``EARTHDATA_TOKEN`` env var, ``~/.netrc``,
    then interactive prompt.  Returns an authenticated ``requests.Session``
    with a bearer token suitable for OPeNDAP access.

    Raises :class:`~hyplan.exceptions.HyPlanRuntimeError` if ``earthaccess``
    is not installed or login fails.
    """
    earthaccess = _require_earthaccess()

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
