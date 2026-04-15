"""Gridded wind field base class with OPeNDAP fetch and 4-D interpolation."""

from __future__ import annotations

import datetime
import logging
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from pint import Quantity

from ..atmosphere import pressure_at
from ..exceptions import HyPlanRuntimeError
from ..units import ureg
from .base import WindField
from .utils import _require_xarray

logger = logging.getLogger(__name__)


class _GriddedWindField(WindField):
    """Base class for OPeNDAP-backed gridded wind fields.

    Fetches a lat/lon/time/level slab on construction and caches the data
    as numpy arrays for fast 4-D linear interpolation in :meth:`wind_at`.

    Subclasses must implement :meth:`_build_urls` and :meth:`_open_dataset`.
    """

    def __init__(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time_start: datetime.datetime,
        time_end: datetime.datetime,
        pressure_min_hpa: float = 50.0,
        pressure_max_hpa: float = 1000.0,
    ):
        self._xr = _require_xarray()

        self._lat_min = lat_min
        self._lat_max = lat_max
        self._lon_min = lon_min
        self._lon_max = lon_max
        self._time_start = time_start
        self._time_end = time_end
        self._pressure_min_hpa = pressure_min_hpa
        self._pressure_max_hpa = pressure_max_hpa

        # Fetch the slab
        self._u_data = None  # will be numpy array (time, lev, lat, lon)
        self._v_data = None
        self._times = None   # numpy array of datetime64
        self._levs = None    # numpy array of pressure levels in hPa
        self._lats = None    # numpy array
        self._lons = None    # numpy array

        self._fetch_slab()

    @abstractmethod
    def _build_urls(self) -> List[str]:
        """Return one or more OPeNDAP dataset URLs covering the time range."""

    def _open_dataset(self, url: str):
        """Open a single OPeNDAP dataset. Override for auth customization."""
        return self._xr.open_dataset(url, engine="netcdf4")

    def _dim_names(self) -> dict:
        """Return dimension name mapping. Override if names differ."""
        return {"time": "time", "lev": "lev", "lat": "lat", "lon": "lon"}

    def _var_names(self) -> Tuple[str, str]:
        """Return (u_name, v_name) variable names. Override if names differ."""
        return ("U", "V")

    def _decode_time(self, raw_time: np.ndarray) -> np.ndarray:
        """Convert raw time coordinate to datetime64[ns].

        Default implementation assumes the dataset already decoded times.
        Override for datasets opened with ``decode_times=False``.
        """
        return raw_time

    def _time_slice(self, time_coords: np.ndarray) -> slice:
        """Compute an integer index slice for the time dimension.

        Default returns all timesteps (for daily files that are already
        pre-selected).  Override for aggregated datasets that need
        server-side time subsetting.
        """
        return slice(None)

    @staticmethod
    def _index_range(coords: np.ndarray, lo: float, hi: float) -> slice:
        """Compute integer index slice covering [lo, hi] with 1-cell margin."""
        ascending = len(coords) < 2 or coords[0] < coords[-1]
        if not ascending:
            coords = coords[::-1]

        i0 = max(0, int(np.searchsorted(coords, lo, side="left")) - 1)
        i1 = min(len(coords) - 1, int(np.searchsorted(coords, hi, side="right")))

        if not ascending:
            n = len(coords)
            i0, i1 = n - 1 - i1, n - 1 - i0

        return slice(i0, i1 + 1)

    def _fetch_slab(self) -> None:
        """Fetch U/V data slab from OPeNDAP and cache as numpy arrays.

        Uses integer index selection (``isel``) so that pydap translates
        the selection into OPeNDAP server-side constraints, avoiding a
        full-globe download.
        """
        urls = self._build_urls()
        dims = self._dim_names()

        lat_name = dims["lat"]
        lon_name = dims["lon"]
        lev_name = dims["lev"]
        time_name = dims["time"]

        u_name, v_name = self._var_names()

        slabs = []
        for url in urls:
            try:
                logger.info("Fetching wind data from %s", url)
                ds = self._open_dataset(url)
            except Exception as exc:
                raise HyPlanRuntimeError(
                    f"Failed to open wind dataset at {url}: {exc}\n"
                    "Check your network connection and credentials."
                ) from exc

            try:
                # Read coordinate arrays (small metadata)
                lats = ds[lat_name].values
                lons = ds[lon_name].values
                levs = ds[lev_name].values

                times = ds[time_name].values

                # Compute integer index ranges for server-side subsetting
                lat_sl = self._index_range(lats, self._lat_min, self._lat_max)
                lon_sl = self._index_range(lons, self._lon_min, self._lon_max)
                lev_sl = self._index_range(levs, self._pressure_min_hpa, self._pressure_max_hpa)
                time_sl = self._time_slice(times)

                isel_kwargs = {
                    time_name: time_sl,
                    lev_name: lev_sl,
                    lat_name: lat_sl,
                    lon_name: lon_sl,
                }

                slab = ds[[u_name, v_name]].isel(**isel_kwargs).load()
                slabs.append(slab)
            finally:
                ds.close()

        if len(slabs) == 1:
            slab = slabs[0]
        else:
            slab = self._xr.concat(slabs, dim=time_name)

        # Decode raw time values to datetime64 (hook for decode_times=False)
        slab[time_name] = self._decode_time(slab[time_name].values)

        # Filter time to requested window (strip tzinfo for np.datetime64)
        time_vals = slab[time_name].values
        ts = self._time_start.replace(tzinfo=None) if self._time_start.tzinfo else self._time_start
        te = self._time_end.replace(tzinfo=None) if self._time_end.tzinfo else self._time_end
        time_start_np = np.datetime64(ts)
        time_end_np = np.datetime64(te)
        time_mask = (time_vals >= time_start_np) & (time_vals <= time_end_np)
        if np.any(time_mask) and not np.all(time_mask):
            slab = slab.isel({time_name: time_mask})

        self._lats = slab[lat_name].values.astype(float)
        self._lons = slab[lon_name].values.astype(float)
        self._levs = slab[lev_name].values.astype(float)
        time_vals = slab[time_name].values
        self._times_raw = time_vals
        self._times = np.array(
            [(t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
             for t in time_vals],
            dtype=float,
        )
        self._u_data = slab[u_name].values.astype(float)  # (time, lev, lat, lon)
        self._v_data = slab[v_name].values.astype(float)

        # Ensure lat and lev are ascending for np.searchsorted
        if len(self._lats) > 1 and self._lats[0] > self._lats[-1]:  # type: ignore[arg-type, index]
            self._lats = self._lats[::-1]  # type: ignore[index]
            self._u_data = self._u_data[:, :, ::-1, :]  # type: ignore[index]
            self._v_data = self._v_data[:, :, ::-1, :]  # type: ignore[index]
        if len(self._levs) > 1 and self._levs[0] > self._levs[-1]:  # type: ignore[arg-type, index]
            self._levs = self._levs[::-1]  # type: ignore[index]
            self._u_data = self._u_data[:, ::-1, :, :]  # type: ignore[index]
            self._v_data = self._v_data[:, ::-1, :, :]  # type: ignore[index]

        logger.info(
            "Wind slab loaded: %d times, %d levels, %d lats, %d lons",
            len(self._times), len(self._levs), len(self._lats), len(self._lons),  # type: ignore[arg-type]
        )

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        """Interpolate wind at a point from the cached slab."""
        # Convert altitude to ISA pressure
        p_hpa = pressure_at(altitude).m_as(ureg.hectopascal)

        # Convert time to epoch seconds (strip tzinfo for np.datetime64)
        t_naive = time.replace(tzinfo=None) if time.tzinfo else time
        t_epoch = (
            np.datetime64(t_naive) - np.datetime64("1970-01-01T00:00:00")
        ) / np.timedelta64(1, "s")

        u = self._interp4d(self._u_data, t_epoch, p_hpa, lat, lon)  # type: ignore[arg-type]
        v = self._interp4d(self._v_data, t_epoch, p_hpa, lat, lon)  # type: ignore[arg-type]

        return (
            float(u) * (ureg.meter / ureg.second),
            float(v) * (ureg.meter / ureg.second),
        )

    def _interp4d(
        self,
        data: np.ndarray,
        t: float,
        p: float,
        lat: float,
        lon: float,
    ) -> float:
        """4-D linear interpolation on (time, level, lat, lon)."""
        # Clamp and find bounding indices for each dimension
        ti = self._interp_weights(self._times, t)  # type: ignore[arg-type]
        pi = self._interp_weights(self._levs, p)  # type: ignore[arg-type]
        lai = self._interp_weights(self._lats, lat)  # type: ignore[arg-type]
        loi = self._interp_weights(self._lons, lon)  # type: ignore[arg-type]

        # Trilinear over the 16 corners of the 4D hypercube
        result = 0.0
        for it, wt in ti:
            for ip, wp in pi:
                for ila, wla in lai:
                    for ilo, wlo in loi:
                        result += wt * wp * wla * wlo * data[it, ip, ila, ilo]
        return result

    @staticmethod
    def _interp_weights(
        coords: np.ndarray, value: float
    ) -> list:
        """Find bounding indices and weights for linear interpolation.

        Returns a list of (index, weight) tuples (1 or 2 entries).
        Clamps at boundaries.
        """
        if len(coords) == 1:
            return [(0, 1.0)]

        # Clamp to range
        if value <= coords[0]:
            return [(0, 1.0)]
        if value >= coords[-1]:
            return [(len(coords) - 1, 1.0)]

        idx = int(np.searchsorted(coords, value)) - 1
        idx = max(0, min(idx, len(coords) - 2))

        lo = coords[idx]
        hi = coords[idx + 1]
        if hi == lo:
            return [(idx, 1.0)]

        frac = (value - lo) / (hi - lo)
        return [(idx, 1.0 - frac), (idx + 1, frac)]
