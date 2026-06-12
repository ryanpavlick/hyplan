"""MERRA-2 reanalysis wind field provider."""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any

import numpy as np

from ..gridded import _GriddedWindField
from ..utils import _earthdata_login

logger = logging.getLogger(__name__)

# MERRA-2 standard pressure levels (hPa), descending (surface -> top of atm)
_MERRA2_LEVELS_HPA = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 650, 600, 550, 500, 450, 400, 350,
    300, 250, 200, 150, 100, 70, 50, 40, 30, 20,
    10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1,
], dtype=float)

# Months GES DISC republished under stream 401 instead of 400.
_MERRA2_REPROCESSED_MONTHS: dict[tuple[int, int], int] = {
    (2020, 9): 401,
    (2021, 6): 401,
    (2021, 7): 401,
    (2021, 8): 401,
    (2021, 9): 401,
}


def _merra2_stream(year: int, month: int | None = None) -> int:
    """Return the MERRA-2 stream number for a given year (and month)."""
    if month is not None and (year, month) in _MERRA2_REPROCESSED_MONTHS:
        return _MERRA2_REPROCESSED_MONTHS[(year, month)]
    if year <= 1991:
        return 100
    if year <= 2000:
        return 200
    if year <= 2010:
        return 300
    return 400


def _merra2_url(dt: datetime.date) -> str:
    """Build the OPeNDAP URL for a single MERRA-2 daily file."""
    stream = _merra2_stream(dt.year, dt.month)
    return (
        f"dap2://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/"
        f"MERRA2/M2I3NPASM.5.12.4/{dt.year:04d}/{dt.month:02d}/"
        f"MERRA2_{stream}.inst3_3d_asm_Np.{dt.year:04d}{dt.month:02d}{dt.day:02d}.nc4"
    )


def _alternate_stream_url(url: str) -> str | None:
    """Swap a MERRA-2 URL between the base and reprocessed stream (400<->401)."""
    match = re.search(r"MERRA2_(\d{3})\.", url)
    if match is None:
        return None
    stream = int(match.group(1))
    alt = stream + 1 if stream % 10 == 0 else stream - 1
    return url.replace(f"MERRA2_{stream}.", f"MERRA2_{alt}.", 1)


class MERRA2WindField(_GriddedWindField):
    """MERRA-2 reanalysis wind field for historical planning.

    Fetches 3-hourly instantaneous U/V winds on pressure levels from
    NASA GES DISC via OPeNDAP.

    **Prerequisites:**

    1. Install: ``pip install hyplan[winds]``
    2. Register at https://urs.earthdata.nasa.gov
    3. Authenticate via one of:
       - Set ``EARTHDATA_TOKEN`` environment variable (recommended)
       - Add to ``~/.netrc``::

             machine urs.earthdata.nasa.gov login <user> password <pass>

    Authentication is handled by ``earthaccess``, which tries the
    ``EARTHDATA_TOKEN`` env var first, then ``~/.netrc``, then an
    interactive prompt.

    Args:
        lat_min: Southern latitude bound (degrees).
        lat_max: Northern latitude bound (degrees).
        lon_min: Western longitude bound (degrees).
        lon_max: Eastern longitude bound (degrees).
        time_start: Start of time window (UTC).
        time_end: End of time window (UTC).
        pressure_min_hpa: Top pressure level to fetch (hPa). Default 50.
        pressure_max_hpa: Bottom pressure level to fetch (hPa). Default 1000.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._session = _earthdata_login()
        super().__init__(*args, **kwargs)

    def _open_dataset(self, url: str) -> Any:
        """Open OPeNDAP dataset with Earthdata-authenticated session.

        Falls back to the alternate stream number (400<->401) when the
        primary URL fails to open — GES DISC republished some months
        under a different stream, so the canonical URL can 404.
        """
        try:
            return self._open_url(url)
        except Exception:
            alt_url = _alternate_stream_url(url)
            if alt_url is None:
                raise
            logger.info(
                "Failed to open %s; retrying alternate MERRA-2 stream %s",
                url, alt_url,
            )
            ds = self._open_url(alt_url)
            logger.info("MERRA-2 alternate stream succeeded: %s", alt_url)
            return ds

    def _open_url(self, url: str) -> Any:
        store = self._xr.backends.PydapDataStore.open(url, session=self._session)
        return self._xr.open_dataset(store)

    def _build_urls(self) -> list[str]:
        """One URL per day in the time range."""
        urls = []
        dt = self._time_start.date()
        end_date = self._time_end.date()
        while dt <= end_date:
            urls.append(_merra2_url(dt))
            dt += datetime.timedelta(days=1)
        return urls
