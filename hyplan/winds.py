"""Wind field models for per-segment wind correction in flight planning.

Provides a :class:`WindField` abstraction that returns wind U/V components
at any (lat, lon, altitude, time) point.  Implementations:

``StillAirField``
    Zero wind everywhere.  Explicit baseline for comparison.

``ConstantWindField``
    Wraps a single speed + direction into constant U/V.  No dependencies
    beyond the hyplan core.

``MERRA2WindField``
    MERRA-2 reanalysis winds (inst3_3d_asm_Np) via OPeNDAP for historical
    planning.  Requires ``pip install hyplan[winds]`` plus NASA Earthdata
    credentials (``EARTHDATA_TOKEN`` env var or ``~/.netrc``).

``GMAOWindField``
    GEOS-FP near-real-time analysis winds via OPeNDAP.  Same dependencies
    as MERRA-2 but typically no credentials required.

``GFSWindField``
    NOAA GFS 0.25° forecast winds via the NOMADS GRIB filter.  No
    credentials required.  Up to 16-day forecast horizon, updated
    4× daily.  Server-side subsetting keeps downloads small (~10 KB).

Usage::

    from hyplan.winds import ConstantWindField, wind_field_from_plan

    # Still air (no wind baseline)
    wf = wind_field_from_plan("still_air", flight_sequence, takeoff_time)

    # Constant wind (backward-compatible with scalar parameters)
    wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=270.0)

    # MERRA-2 historical wind for a planned flight
    wf = wind_field_from_plan("merra2", flight_sequence, takeoff_time)

    # GFS operational forecast
    wf = wind_field_from_plan("gfs", flight_sequence, takeoff_time)

    plan = compute_flight_plan(..., wind_source=wf, takeoff_time=takeoff_time)
"""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from pint import Quantity

from .atmosphere import pressure_at
from .exceptions import HyPlanRuntimeError, HyPlanValueError
from .units import ureg

logger = logging.getLogger(__name__)

__all__ = [
    "WindField",
    "StillAirField",
    "ConstantWindField",
    "MERRA2WindField",
    "GMAOWindField",
    "GFSWindField",
    "wind_field_from_plan",
]

# MERRA-2 standard pressure levels (hPa), descending (surface → top of atm)
_MERRA2_LEVELS_HPA = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 650, 600, 550, 500, 450, 400, 350,
    300, 250, 200, 150, 100, 70, 50, 40, 30, 20,
    10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1,
], dtype=float)


# ---------------------------------------------------------------------------
# Lazy import helper
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
# Abstract base
# ---------------------------------------------------------------------------

class WindField(ABC):
    """Abstract base for wind data providers.

    All subclasses must implement :meth:`wind_at`, which returns eastward
    (U) and northward (V) wind components as ``pint.Quantity`` in m/s.
    """

    @abstractmethod
    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        """Return (u, v) wind components at the given point.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            altitude: Geometric altitude as a :class:`pint.Quantity`.
            time: UTC datetime.

        Returns:
            Tuple of (u, v) as :class:`pint.Quantity` in m/s.
            u is eastward (positive = from west),
            v is northward (positive = from south).
        """


# ---------------------------------------------------------------------------
# Still air (no wind)
# ---------------------------------------------------------------------------

_ZERO_MPS = 0.0 * (ureg.meter / ureg.second)


class StillAirField(WindField):
    """Zero-wind field — always returns U=0, V=0.

    Use this as an explicit "no wind" baseline for comparison or when
    wind data is unavailable.
    """

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        return _ZERO_MPS, _ZERO_MPS


# ---------------------------------------------------------------------------
# Constant wind (backward compatibility)
# ---------------------------------------------------------------------------

class ConstantWindField(WindField):
    """Constant wind field — same U/V everywhere.

    Useful for backward compatibility with the scalar ``wind_speed`` /
    ``wind_direction`` parameters.

    Args:
        wind_speed: Wind speed magnitude.
        wind_from_deg: Direction the wind is blowing *from* in degrees
            true (meteorological convention: 0 = from north, 90 = from east).
    """

    def __init__(self, wind_speed: Quantity, wind_from_deg: float):
        ws = wind_speed.m_as(ureg.meter / ureg.second)
        # Meteorological convention: wind_from_deg is the direction the wind
        # comes FROM.  Convert to U/V components (the direction it blows TO).
        wind_from_rad = np.radians(wind_from_deg)
        self._u = float(-ws * np.sin(wind_from_rad)) * (ureg.meter / ureg.second)
        self._v = float(-ws * np.cos(wind_from_rad)) * (ureg.meter / ureg.second)

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        return self._u, self._v


# ---------------------------------------------------------------------------
# Gridded wind base (shared OPeNDAP + interpolation logic)
# ---------------------------------------------------------------------------

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
    ) -> Tuple[Quantity, Quantity]:
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
        data: np.ndarray,
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


# ---------------------------------------------------------------------------
# MERRA-2
# ---------------------------------------------------------------------------

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
        f"https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/"
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


# ---------------------------------------------------------------------------
# GMAO GEOS-FP
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NOAA GFS (via NOMADS GRIB filter)
# ---------------------------------------------------------------------------

_GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Standard pressure levels available in GFS pgrb2 0.25° files (hPa)
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
    """NOAA GFS 0.25° forecast wind field for operational planning.

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
        import tempfile
        import os

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

            # GFS uses 0–360 longitude; convert to -180..180
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wind_field_from_plan(
    source: str,
    flight_sequence: list,
    takeoff_time: datetime.datetime,
    takeoff_airport=None,
    return_airport=None,
    flight_altitude: Optional[Quantity] = None,
    margin_deg: float = 2.0,
    margin_hours: float = 2.0,
) -> WindField:
    """Create a wind field pre-fetched for a planned flight.

    Computes the geographic and temporal bounding box from the flight
    sequence, adds margins, and constructs the appropriate
    :class:`WindField` with its data slab pre-loaded.

    Args:
        source: ``"merra2"`` for MERRA-2 reanalysis, ``"gmao"`` for
            GEOS-FP near-real-time analysis, ``"gfs"`` for NOAA GFS
            forecast, or ``"still_air"`` for zero wind.
        flight_sequence: Ordered list of flight lines and/or waypoints
            (same format as :func:`~hyplan.flight_plan.compute_flight_plan`).
        takeoff_time: Mission start time (UTC).
        takeoff_airport: Optional departure airport (extends bounding box).
        return_airport: Optional arrival airport (extends bounding box).
        flight_altitude: Representative flight altitude for pressure-level
            selection.  If ``None``, derived from the highest altitude in
            the flight sequence.
        margin_deg: Spatial margin in degrees added to all sides.
        margin_hours: Temporal margin in hours added before/after.

    Returns:
        A :class:`WindField` ready for use with ``compute_flight_plan()``.
    """
    from .flight_line import FlightLine
    from .waypoint import is_waypoint

    # Collect lat/lon/alt from flight sequence
    lats, lons, alts_m = [], [], []

    for item in flight_sequence:
        if isinstance(item, FlightLine):
            for wp in (item.waypoint1, item.waypoint2):
                lats.append(wp.latitude)
                lons.append(wp.longitude)
                if wp.altitude_msl is not None:
                    alts_m.append(wp.altitude_msl.m_as(ureg.meter))
        elif is_waypoint(item):
            lats.append(item.latitude)
            lons.append(item.longitude)
            if item.altitude_msl is not None:
                alts_m.append(item.altitude_msl.m_as(ureg.meter))

    if takeoff_airport is not None:
        lats.append(takeoff_airport.latitude)
        lons.append(takeoff_airport.longitude)
    if return_airport is not None:
        lats.append(return_airport.latitude)
        lons.append(return_airport.longitude)

    if not lats:
        raise HyPlanValueError("Flight sequence is empty — cannot determine wind extent.")

    lat_min = min(lats) - margin_deg
    lat_max = max(lats) + margin_deg
    lon_min = min(lons) - margin_deg
    lon_max = max(lons) + margin_deg

    # Estimate flight duration (~8 hours if we can't compute it)
    estimated_duration_hours = 8.0
    time_start = takeoff_time - datetime.timedelta(hours=margin_hours)
    time_end = takeoff_time + datetime.timedelta(
        hours=estimated_duration_hours + margin_hours
    )

    # Determine pressure range from altitudes
    if flight_altitude is not None:
        max_alt = flight_altitude
    elif alts_m:
        max_alt = max(alts_m) * ureg.meter
    else:
        max_alt = 15000 * ureg.meter  # ~FL500 default

    # Convert to pressure; add margin (go 20% higher in altitude)
    pressure_at_alt = pressure_at(max_alt).m_as(ureg.hectopascal)
    pressure_min_hpa = max(1.0, pressure_at_alt * 0.5)  # higher alt = lower pressure
    pressure_max_hpa = 1000.0  # surface

    source_lower = source.lower().strip()

    if source_lower == "still_air":
        return StillAirField()

    bbox = dict(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        time_start=time_start,
        time_end=time_end,
        pressure_min_hpa=pressure_min_hpa,
        pressure_max_hpa=pressure_max_hpa,
    )

    if source_lower == "merra2":
        return MERRA2WindField(**bbox)
    elif source_lower == "gmao":
        return GMAOWindField(**bbox)
    elif source_lower == "gfs":
        return GFSWindField(**bbox)
    else:
        raise HyPlanValueError(
            f"Unknown wind source '{source}'. "
            "Use 'merra2', 'gmao', 'gfs', or 'still_air'."
        )
