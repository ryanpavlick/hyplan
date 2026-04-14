"""NOAA GFS forecast wind field provider (via NOMADS GRIB filter)."""

from __future__ import annotations

import datetime
import logging
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from ...exceptions import HyPlanRuntimeError
from ..gridded import _GriddedWindField
from ..utils import _require_xarray

logger = logging.getLogger(__name__)

_GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Standard pressure levels available in GFS pgrb2 0.25deg files (hPa)
_GFS_LEVELS_HPA = np.array([
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
    600, 550, 500, 450, 400, 350, 300, 250, 200, 150,
    100, 70, 50, 40, 30, 20, 15, 10, 7, 5, 4, 3, 2, 1,
], dtype=float)


def _gfs_best_cycle(target: datetime.datetime) -> Tuple[datetime.date, int]:
    """Pick the most recent GFS cycle available before *target*.

    GFS cycles run at 00, 06, 12, 18 UTC.  Data is typically available
    ~4-5 hours after the cycle time.  We pick the latest cycle whose
    output would be available by now.
    """
    utc_now = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
    cycles = [0, 6, 12, 18]
    ref = utc_now - datetime.timedelta(hours=5)
    cycle = max(c for c in cycles if c <= ref.hour)
    return ref.date(), cycle


def _gfs_filter_url(
    dt: datetime.date,
    cycle: int,
    fhr: int,
    variables: Tuple[str, ...],
    levels_hpa: list,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """Build a NOMADS GRIB-filter URL for server-side subsetting.

    The GRIB filter returns a small GRIB2 file containing only the
    requested variables, pressure levels, and geographic subregion.
    """
    params = [
        f"dir=%2Fgfs.{dt:%Y%m%d}%2F{cycle:02d}%2Fatmos",
        f"file=gfs.t{cycle:02d}z.pgrb2.0p25.f{fhr:03d}",
    ]
    for var in variables:
        params.append(f"var_{var}=on")
    for lev in sorted(levels_hpa):
        lev_str = f"{int(lev)}" if lev == int(lev) else f"{lev}"
        params.append(f"lev_{lev_str}_mb=on")
    # NOMADS uses 0-360 longitude for the filter
    left = lon_min % 360
    right = lon_max % 360
    params.append(
        f"subregion=&toplat={lat_max}&leftlon={left}"
        f"&rightlon={right}&bottomlat={lat_min}"
    )
    return f"{_GFS_FILTER_URL}?{'&'.join(params)}"


class GFSWindField(_GriddedWindField):
    """NOAA GFS 0.25deg forecast wind field for operational planning.

    Fetches U/V winds on pressure levels from the NOMADS GRIB filter,
    which performs server-side subsetting by variable, pressure level,
    and geographic region — typically downloading only ~10 KB instead
    of the full ~500 MB GRIB2 file.

    No credentials required.  Forecast data available up to 16 days
    ahead, updated 4 times daily (00Z, 06Z, 12Z, 18Z).

    Requires ``cfgrib`` (with eccodes) for GRIB2 decoding.

    Args:
        lat_min: Southern latitude bound (degrees).
        lat_max: Northern latitude bound (degrees).
        lon_min: Western longitude bound (degrees).
        lon_max: Eastern longitude bound (degrees).
        time_start: Start of time window (UTC).
        time_end: End of time window (UTC).
        pressure_min_hpa: Top pressure level to fetch (hPa). Default 50.
        pressure_max_hpa: Bottom pressure level to fetch (hPa). Default 1000.
        cycle_date: Override GFS cycle date (default: auto-select).
        cycle_hour: Override GFS cycle hour (default: auto-select).
        forecast_hour: Specific forecast hour to fetch.  If ``None``
            (default), the hour closest to the midpoint of the time
            window is chosen automatically.
    """

    def __init__(
        self,
        *args,
        cycle_date: Optional[datetime.date] = None,
        cycle_hour: Optional[int] = None,
        forecast_hour: Optional[int] = None,
        **kwargs,
    ):
        self._cycle_date = cycle_date
        self._cycle_hour = cycle_hour
        self._forecast_hour = forecast_hour
        super().__init__(*args, **kwargs)

    def _build_urls(self) -> List[str]:
        # Not used — _fetch_slab is fully overridden.
        return []

    def _fetch_slab(self) -> None:
        """Fetch U/V wind data via the NOMADS GRIB filter.

        Server-side subsetting returns only UGRD/VGRD at the requested
        pressure levels and geographic region, typically ~10 KB.
        """
        try:
            import cfgrib  # noqa: F401 — needed by xarray engine
        except ImportError:
            raise HyPlanRuntimeError(
                "cfgrib (with eccodes) is required for GFS GRIB2 data. "
                "Install with: pip install cfgrib"
            )
        import requests

        xr = _require_xarray()

        # Determine cycle
        if self._cycle_date is not None and self._cycle_hour is not None:
            cycle_date, cycle_hour = self._cycle_date, self._cycle_hour
        else:
            cycle_date, cycle_hour = _gfs_best_cycle(self._time_start)

        cycle_dt = datetime.datetime(
            cycle_date.year, cycle_date.month, cycle_date.day,
            cycle_hour,
        )

        # Pick a single forecast hour
        if self._forecast_hour is not None:
            fhr = self._forecast_hour
        else:
            ts = self._time_start.replace(tzinfo=None) if self._time_start.tzinfo else self._time_start
            te = self._time_end.replace(tzinfo=None) if self._time_end.tzinfo else self._time_end
            mid = ts + (te - ts) / 2
            fhr = max(0, round((mid - cycle_dt).total_seconds() / 3600))

        valid_dt = cycle_dt + datetime.timedelta(hours=fhr)

        # Select GFS levels within the requested pressure range
        levels = [
            float(lev) for lev in _GFS_LEVELS_HPA
            if self._pressure_min_hpa <= lev <= self._pressure_max_hpa
        ]
        if not levels:
            levels = [float(_GFS_LEVELS_HPA[
                np.argmin(np.abs(
                    _GFS_LEVELS_HPA
                    - (self._pressure_min_hpa + self._pressure_max_hpa) / 2
                ))
            ])]

        url = _gfs_filter_url(
            cycle_date, cycle_hour, fhr,
            ("UGRD", "VGRD"), levels,
            self._lat_min, self._lat_max,
            self._lon_min, self._lon_max,
        )

        logger.info(
            "GFS cycle %s/%02dZ f%03d (valid %s), %d levels",
            cycle_date, cycle_hour, fhr, valid_dt, len(levels),
        )

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            raise HyPlanRuntimeError(
                f"Failed to fetch GFS data from NOMADS GRIB filter: {exc}"
            ) from exc

        if len(resp.content) < 100 or resp.content[:4] != b"GRIB":
            raise HyPlanRuntimeError(
                "NOMADS GRIB filter returned invalid data. "
                "The requested cycle/forecast may not be available yet."
            )

        logger.info("Downloaded %d KB from NOMADS GRIB filter", len(resp.content) // 1024)

        # Write to temp file and decode with cfgrib
        tmpfile = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                f.write(resp.content)
                tmpfile = f.name

            ds = xr.open_dataset(
                tmpfile, engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
            )

            # GFS uses 0-360 longitude; convert to -180..180
            lons = ds["longitude"].values
            if np.any(lons > 180):
                lons = np.where(lons > 180, lons - 360, lons)
                sort_idx = np.argsort(lons)
                lons = lons[sort_idx]
                u_data = ds["u"].values[:, :, sort_idx]
                v_data = ds["v"].values[:, :, sort_idx]
            else:
                u_data = ds["u"].values
                v_data = ds["v"].values

            lats = ds["latitude"].values
            levs = ds["isobaricInhPa"].values

            # Store with a single-element time dimension
            self._u_data = u_data[np.newaxis, ...].astype(float)
            self._v_data = v_data[np.newaxis, ...].astype(float)
            self._lats = lats.astype(float)
            self._lons = lons.astype(float)
            self._levs = levs.astype(float)

            t_np = np.datetime64(valid_dt)
            self._times_raw = np.array([t_np], dtype="datetime64[ns]")
            self._times = np.array([
                (t_np - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
            ], dtype=float)

            ds.close()
        finally:
            if tmpfile and os.path.exists(tmpfile):
                os.unlink(tmpfile)

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
            "GFS wind slab loaded: %d levels, %d lats, %d lons",
            len(self._levs), len(self._lats), len(self._lons),
        )
