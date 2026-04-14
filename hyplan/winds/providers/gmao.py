"""GMAO GEOS-FP near-real-time wind field provider."""

from __future__ import annotations

import datetime
from typing import List, Optional, Tuple

import numpy as np

from ..gridded import _GriddedWindField

_GMAO_FP_URL = (
    "dap2://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Np"
)

# Epoch for GMAO time encoding: "days since 1-1-1 00:00:0.0"
_GMAO_EPOCH = np.datetime64("0001-01-01T00:00:00", "ns")


class GMAOWindField(_GriddedWindField):
    """GMAO GEOS-FP near-real-time wind field for operational planning.

    Fetches 3-hourly instantaneous U/V winds on pressure levels from
    NCCS OPeNDAP server via the pydap DAP2 protocol.  Typically covers
    the last ~30 days of analysis plus short-range forecasts.  No
    credentials required.

    Args:
        lat_min: Southern latitude bound (degrees).
        lat_max: Northern latitude bound (degrees).
        lon_min: Western longitude bound (degrees).
        lon_max: Eastern longitude bound (degrees).
        time_start: Start of time window (UTC).
        time_end: End of time window (UTC).
        pressure_min_hpa: Top pressure level to fetch (hPa). Default 50.
        pressure_max_hpa: Bottom pressure level to fetch (hPa). Default 1000.
        url: Override the default GEOS-FP OPeNDAP URL.
    """

    def __init__(self, *args, url: Optional[str] = None, **kwargs):
        self._base_url = url or _GMAO_FP_URL
        super().__init__(*args, **kwargs)

    def _build_urls(self) -> List[str]:
        """Single URL — GEOS-FP is served as a single aggregated dataset."""
        return [self._base_url]

    def _open_dataset(self, url: str):
        """Open via pydap engine with decode_times=False.

        The NCCS OPeNDAP server's time variable uses a non-standard
        epoch ("days since 1-1-1 00:00:0.0") that netCDF4 cannot decode.
        Using pydap with DAP2 protocol avoids this issue.
        """
        return self._xr.open_dataset(url, engine="pydap", decode_times=False)

    def _var_names(self) -> Tuple[str, str]:
        """GEOS-FP uses lowercase u/v."""
        return ("u", "v")

    @staticmethod
    def _datetime_to_gmao_days(dt: datetime.datetime) -> float:
        """Convert a datetime to GMAO 'days since 1-1-1' float."""
        # proleptic Gregorian ordinal: Jan 1, year 1 = ordinal 1
        naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
        ordinal = naive.toordinal()  # 1-based (Jan 1, 0001 = 1)
        frac = (naive.hour * 3600 + naive.minute * 60 + naive.second) / 86400.0
        return float(ordinal) + frac

    def _time_slice(self, time_coords: np.ndarray) -> slice:
        """Subset the aggregated GMAO time dimension to the request window."""
        lo = self._datetime_to_gmao_days(self._time_start)
        hi = self._datetime_to_gmao_days(self._time_end)
        return self._index_range(time_coords, lo, hi)

    def _decode_time(self, raw_time: np.ndarray) -> np.ndarray:
        """Convert GMAO 'days since 1-1-1' floats to datetime64[ns]."""
        # raw_time values are fractional days since 0001-01-01.
        # Subtract 1 because the epoch day itself is day 1, not day 0.
        days = (raw_time - 1).astype("timedelta64[D]")
        frac_ns = ((raw_time - 1 - np.floor(raw_time - 1)) * 86400e9).astype(
            "timedelta64[ns]"
        )
        return _GMAO_EPOCH + days + frac_ns
