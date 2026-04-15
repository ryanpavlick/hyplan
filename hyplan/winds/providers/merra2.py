"""MERRA-2 reanalysis wind field provider."""

from __future__ import annotations

import datetime
from typing import List

import numpy as np

from ..gridded import _GriddedWindField
from ..utils import _earthdata_login

# MERRA-2 standard pressure levels (hPa), descending (surface -> top of atm)
_MERRA2_LEVELS_HPA = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 650, 600, 550, 500, 450, 400, 350,
    300, 250, 200, 150, 100, 70, 50, 40, 30, 20,
    10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1,
], dtype=float)


def _merra2_stream(year: int) -> int:
    """Return the MERRA-2 stream number for a given year."""
    if year <= 1991:
        return 100
    if year <= 2000:
        return 200
    if year <= 2010:
        return 300
    return 400


def _merra2_url(dt: datetime.date) -> str:
    """Build the OPeNDAP URL for a single MERRA-2 daily file."""
    stream = _merra2_stream(dt.year)
    return (
        f"dap2://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/"
        f"MERRA2/M2I3NPASM.5.12.4/{dt.year:04d}/{dt.month:02d}/"
        f"MERRA2_{stream}.inst3_3d_asm_Np.{dt.year:04d}{dt.month:02d}{dt.day:02d}.nc4"
    )


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

    def __init__(self, *args, **kwargs):
        self._session = _earthdata_login()
        super().__init__(*args, **kwargs)

    def _open_dataset(self, url: str):
        """Open OPeNDAP dataset with Earthdata-authenticated session."""
        store = self._xr.backends.PydapDataStore.open(url, session=self._session)
        return self._xr.open_dataset(store)

    def _build_urls(self) -> List[str]:
        """One URL per day in the time range."""
        urls = []
        dt = self._time_start.date()
        end_date = self._time_end.date()
        while dt <= end_date:
            urls.append(_merra2_url(dt))
            dt += datetime.timedelta(days=1)
        return urls
