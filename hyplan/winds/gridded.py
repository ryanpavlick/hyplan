"""Gridded wind field base class with OPeNDAP fetch and 4-D interpolation."""

from __future__ import annotations

import datetime
import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
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

    # Slab data, populated by :meth:`_fetch_slab` from __init__.
    # _times is Unix epoch seconds (float), converted from datetime64
    # in _fetch_slab; _times_raw keeps the original datetime64 array.
    _u_data: npt.NDArray[np.float64]
    _v_data: npt.NDArray[np.float64]
    _times: npt.NDArray[np.float64]
    _times_raw: npt.NDArray[np.datetime64]
    _levs: npt.NDArray[np.float64]
    _lats: npt.NDArray[np.float64]
    _lons: npt.NDArray[np.float64]

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

        # Slab attrs (_u_data, _v_data, _times, _levs, _lats, _lons)
        # are populated by _fetch_slab; class-level annotations above
        # tell mypy they're always-set NDArrays after construction.
        self._fetch_slab()

    @classmethod
    def for_plan(
        cls,
        plan: Any,
        *,
        time_start: datetime.datetime,
        time_end: datetime.datetime | None = None,
        bbox_buffer_deg: float = 1.0,
        descent_altitude_buffer_ft: float = 0.0,
        **provider_kwargs: Any,
    ) -> _GriddedWindField:
        """Build a gridded wind field sized for a computed flight plan.

        Derives the slab geometry from the plan's columns so callers
        don't have to remember to set ``pressure_min_hpa`` (or the bbox)
        manually:

        * **bbox** — union of every segment's ``geometry`` extents,
          padded by ``bbox_buffer_deg`` (default 1°) so the descent
          drift envelope and any minor mis-tracking is comfortably
          covered.
        * **time window** — ``[time_start, time_end]``; if
          ``time_end`` is omitted it defaults to
          ``time_start + sum(time_to_segment) + 1 hr`` buffer.
        * **pressure range** — derived from the plan's max
          ``start_altitude`` / ``end_altitude`` (in feet); maps that
          altitude to ISA pressure and rounds DOWN to the nearest
          standard MERRA-2 level (taking the lower pressure / higher
          altitude end for safety).  ``descent_altitude_buffer_ft``
          (default 0) adds extra room above the plan's max altitude —
          rarely needed since dropsondes only DESCEND from the release,
          but useful for isochrone planning where an aircraft may climb
          above the plan's nominal ceiling.

        Extra keyword args (e.g. provider-specific options) are passed
        through to the subclass constructor.

        Args:
            plan: Segment-level GeoDataFrame from
                :func:`hyplan.compute_flight_plan` (or equivalent).
                Must carry ``geometry``, ``start_altitude``,
                ``end_altitude``, and ``time_to_segment`` columns.
            time_start: UTC start of the wind window (typically the
                flight's takeoff time).
            time_end: UTC end of the wind window; defaults to
                ``time_start + plan.time_to_segment.sum() + 1 hr``.
            bbox_buffer_deg: Lat/lon padding around the plan's geometric
                extent.  Default 1° handles a ~100 km drift envelope at
                mid-latitudes.
            descent_altitude_buffer_ft: Vertical buffer above the
                plan's max altitude (feet).  Default 0 — appropriate
                for dropsondes (which only descend from release).
        """
        try:
            import geopandas  # noqa: F401  — feature-detection import; used downstream via duck-typing

            from hyplan.atmosphere import pressure_at
            from hyplan.units import ureg
        except ImportError as exc:  # pragma: no cover - missing dep
            raise RuntimeError(
                "for_plan requires geopandas and hyplan.atmosphere"
            ) from exc

        if not hasattr(plan, "geometry") or not hasattr(plan, "columns"):
            raise TypeError("plan must be a GeoDataFrame-like object")

        # Bbox from geometry extents.
        bounds = plan.geometry.total_bounds  # (minx, miny, maxx, maxy)
        if not (len(bounds) == 4 and all(np.isfinite(bounds))):
            raise ValueError("plan geometry has no usable extent")
        lon_min = float(bounds[0]) - bbox_buffer_deg
        lon_max = float(bounds[2]) + bbox_buffer_deg
        lat_min = float(bounds[1]) - bbox_buffer_deg
        lat_max = float(bounds[3]) + bbox_buffer_deg

        # Time window.
        if time_end is None:
            total_min = float(plan["time_to_segment"].sum()) if "time_to_segment" in plan.columns else 0.0
            time_end = time_start + datetime.timedelta(minutes=total_min + 60.0)

        # Pressure range — derive from max altitude across all rows.
        alt_cols = [c for c in ("start_altitude", "end_altitude") if c in plan.columns]
        if not alt_cols:
            raise ValueError(
                "plan needs at least one of 'start_altitude' or "
                "'end_altitude' to derive pressure_min_hpa"
            )
        max_alt_ft = max(float(plan[c].max()) for c in alt_cols)
        max_alt_ft += float(descent_altitude_buffer_ft)
        max_alt_m = max_alt_ft * 0.3048
        p_at_max = float(pressure_at(max_alt_m * ureg.meter).m_as("hPa"))
        # Round DOWN to the nearest convenient standard level so the
        # fetched slab actually covers the altitude.
        from hyplan.winds.providers.merra2 import _MERRA2_LEVELS_HPA
        std_levels_sorted = np.sort(_MERRA2_LEVELS_HPA)  # ascending
        idx = int(np.searchsorted(std_levels_sorted, p_at_max, side="right")) - 1
        idx = max(0, idx)
        pressure_min_hpa = float(std_levels_sorted[idx])

        # Respect provider override if explicitly supplied.
        provider_kwargs.setdefault("pressure_min_hpa", pressure_min_hpa)
        provider_kwargs.setdefault("pressure_max_hpa", 1000.0)

        return cls(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            time_start=time_start,
            time_end=time_end,
            **provider_kwargs,
        )

    # ------------------------------------------------------------------
    # Persistence: round-trip the in-memory slab to a NetCDF file
    # ------------------------------------------------------------------

    def to_netcdf(self, path: str) -> None:
        """Persist the fetched slab to a NetCDF file.

        Useful for caching a one-time MERRA-2 / GFS fetch so example
        notebooks can run reproducibly without network or auth.  The
        file carries the U/V data plus the time / pressure / lat / lon
        coordinate arrays — everything ``wind_at`` needs.  Load it back
        with :meth:`from_netcdf` on any ``_GriddedWindField`` subclass:

        >>> wind = MERRA2WindField.from_netcdf("merra2_cache.nc")
        >>> u, v = wind.wind_at(lat, lon, altitude, time)
        """
        xr = _require_xarray()
        ds = xr.Dataset(
            data_vars={
                "u": (("time", "level", "lat", "lon"), self._u_data),
                "v": (("time", "level", "lat", "lon"), self._v_data),
                "times_epoch": (("time",), self._times),
            },
            coords={
                "time": self._times_raw,
                "level": self._levs,
                "lat": self._lats,
                "lon": self._lons,
            },
            attrs={
                "description": (
                    "HyPlan _GriddedWindField slab cache: u/v winds on "
                    "pressure levels.  Loadable via "
                    "_GriddedWindField.from_netcdf(path)."
                ),
                "u_units": "m s-1",
                "v_units": "m s-1",
                "level_units": "hPa",
            },
        )
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str) -> _GriddedWindField:
        """Load a previously-saved slab cache.

        Bypasses ``__init__`` (no live OPeNDAP fetch, no auth) — simply
        restores the in-memory arrays from the NetCDF written by
        :meth:`to_netcdf`.  The returned instance supports the full
        :meth:`wind_at` API but does not carry the original bbox /
        time-window / pressure-range metadata (those live in the file's
        coordinates).
        """
        xr = _require_xarray()
        ds = xr.open_dataset(path)
        instance = cls.__new__(cls)
        instance._xr = xr
        instance._u_data = ds["u"].values.astype(float)
        instance._v_data = ds["v"].values.astype(float)
        instance._levs = ds["level"].values.astype(float)
        instance._lats = ds["lat"].values.astype(float)
        instance._lons = ds["lon"].values.astype(float)
        instance._times_raw = ds["time"].values
        instance._times = ds["times_epoch"].values.astype(float)
        ds.close()
        return instance

    @abstractmethod
    def _build_urls(self) -> list[str]:
        """Return one or more OPeNDAP dataset URLs covering the time range."""

    def _open_dataset(self, url: str) -> Any:
        """Open a single OPeNDAP dataset. Override for auth customization."""
        return self._xr.open_dataset(url, engine="netcdf4")

    def _dim_names(self) -> dict[str, str]:
        """Return dimension name mapping. Override if names differ."""
        return {"time": "time", "lev": "lev", "lat": "lat", "lon": "lon"}

    def _var_names(self) -> tuple[str, str]:
        """Return (u_name, v_name) variable names. Override if names differ."""
        return ("U", "V")

    def _decode_time(self, raw_time: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Convert raw time coordinate to datetime64[ns].

        Default implementation assumes the dataset already decoded times.
        Override for datasets opened with ``decode_times=False``.
        """
        return raw_time

    def _time_slice(self, time_coords: npt.NDArray[Any]) -> slice:
        """Compute an integer index slice for the time dimension.

        Default returns all timesteps (for daily files that are already
        pre-selected).  Override for aggregated datasets that need
        server-side time subsetting.
        """
        return slice(None)

    @staticmethod
    def _index_range(
        coords: npt.NDArray[np.floating[Any]], lo: float, hi: float,
    ) -> slice:
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
        if len(self._lats) > 1 and self._lats[0] > self._lats[-1]:
            self._lats = self._lats[::-1]
            self._u_data = self._u_data[:, :, ::-1, :]
            self._v_data = self._v_data[:, :, ::-1, :]
        if len(self._levs) > 1 and self._levs[0] > self._levs[-1]:
            self._levs = self._levs[::-1]
            self._u_data = self._u_data[:, ::-1, :, :]
            self._v_data = self._v_data[:, ::-1, :, :]

        logger.info(
            "Wind slab loaded: %d times, %d levels, %d lats, %d lons",
            len(self._times), len(self._levs), len(self._lats), len(self._lons),
        )

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> tuple[Quantity, Quantity]:
        """Interpolate wind at a point from the cached slab."""
        # Convert altitude to ISA pressure
        p_hpa = pressure_at(altitude).m_as(ureg.hectopascal)

        # Convert time to epoch seconds (strip tzinfo for np.datetime64)
        t_naive = time.replace(tzinfo=None) if time.tzinfo else time
        t_epoch = (
            np.datetime64(t_naive) - np.datetime64("1970-01-01T00:00:00")
        ) / np.timedelta64(1, "s")

        u = self._interp4d(self._u_data, t_epoch, p_hpa, lat, lon)
        v = self._interp4d(self._v_data, t_epoch, p_hpa, lat, lon)

        return (
            float(u) * (ureg.meter / ureg.second),
            float(v) * (ureg.meter / ureg.second),
        )

    def _interp4d(
        self,
        data: npt.NDArray[np.floating[Any]],
        t: float,
        p: float,
        lat: float,
        lon: float,
    ) -> float:
        """4-D linear interpolation on (time, level, lat, lon)."""
        # Clamp and find bounding indices for each dimension
        ti = self._interp_weights(self._times, t)
        pi = self._interp_weights(self._levs, p)
        lai = self._interp_weights(self._lats, lat)
        loi = self._interp_weights(self._lons, lon)

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
        coords: npt.NDArray[np.floating[Any]], value: float,
    ) -> list[tuple[int, float]]:
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
