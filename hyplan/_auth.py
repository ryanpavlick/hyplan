"""Shared NASA Earthdata authentication utilities."""

from __future__ import annotations

from typing import Any

from .exceptions import HyPlanRuntimeError


def _require_earthaccess() -> Any:
    """Import and return earthaccess, raising a clear error if not installed."""
    try:
        import earthaccess

        return earthaccess
    except ImportError:
        raise HyPlanRuntimeError(
            "earthaccess is required for NASA Earthdata authentication. "
            "Install with: pip install earthaccess"
        ) from None


def _earthdata_login() -> Any:
    """Authenticate with NASA Earthdata using ``earthaccess``.

    Tries non-interactive strategies in order: environment variables
    (``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD``, or
    ``EARTHDATA_TOKEN``), then ``~/.netrc``.  No interactive prompt is
    attempted.  Returns an authenticated ``requests.Session`` with a
    bearer token suitable for OPeNDAP access.

    Raises :class:`~hyplan.exceptions.HyPlanRuntimeError` if ``earthaccess``
    is not installed or every strategy fails; per-strategy failure causes
    are included in the error message.
    """
    earthaccess = _require_earthaccess()

    failures: list[str] = []
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth.authenticated:
                return earthaccess.get_requests_https_session()
            failures.append(f"{strategy}: not authenticated")
        except Exception as exc:
            failures.append(f"{strategy}: {exc}")

    raise HyPlanRuntimeError(
        f"NASA Earthdata login failed ({'; '.join(failures)}). "
        "Authenticate via one of:\n"
        "  1. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD (or "
        "EARTHDATA_TOKEN) environment variables\n"
        "  2. Add to ~/.netrc:\n"
        "     machine urs.earthdata.nasa.gov login <user> password <pass>\n"
        "Register at https://urs.earthdata.nasa.gov if needed."
    )
