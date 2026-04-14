"""Factory function for creating wind fields from flight plan parameters."""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from pint import Quantity

from ..atmosphere import pressure_at
from ..exceptions import HyPlanValueError
from ..units import ureg
from .base import WindField
from .simple import StillAirField

logger = logging.getLogger(__name__)


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
    from ..flight_line import FlightLine
    from ..waypoint import is_waypoint

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
        from .providers.merra2 import MERRA2WindField
        return MERRA2WindField(**bbox)
    elif source_lower == "gmao":
        from .providers.gmao import GMAOWindField
        return GMAOWindField(**bbox)
    elif source_lower == "gfs":
        from .providers.gfs import GFSWindField
        return GFSWindField(**bbox)
    else:
        raise HyPlanValueError(
            f"Unknown wind source '{source}'. "
            "Use 'merra2', 'gmao', 'gfs', or 'still_air'."
        )
