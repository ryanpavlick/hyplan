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
    NOAA GFS 0.25 deg forecast winds via the NOMADS GRIB filter.  No
    credentials required.  Up to 16-day forecast horizon, updated
    4x daily.  Server-side subsetting keeps downloads small (~10 KB).

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

from .base import WindField  # noqa: F401
from .factory import wind_field_from_plan  # noqa: F401
from .gridded import _GriddedWindField  # noqa: F401
from .providers import GFSWindField, GMAOWindField, MERRA2WindField  # noqa: F401
from .providers.gfs import _gfs_best_cycle, _gfs_filter_url  # noqa: F401
from .providers.merra2 import _merra2_stream, _merra2_url  # noqa: F401
from .simple import ConstantWindField, StillAirField  # noqa: F401

__all__ = [
    "WindField",
    "StillAirField",
    "ConstantWindField",
    "MERRA2WindField",
    "GMAOWindField",
    "GFSWindField",
    "wind_field_from_plan",
]
