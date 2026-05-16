"""Release-candidate generation from `FlightLine` and segment objects.

Two entry points:

- :func:`releases_along_flight_line` — emits :class:`DropsondeRelease`
  objects spaced along a single :class:`FlightLine`.  Works before
  ``compute_flight_plan`` has been run.
- :func:`releases_along_segment` — internal helper that emits releases
  spaced along a :class:`PlannedSegment` (used by
  :meth:`DropsondePlan.from_flight_plan` to walk every segment of a
  computed plan).

Both apply the locked "stricter wins" spacing rule when both
``spacing`` (distance) and ``spacing_time`` (time) are supplied.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

import numpy as np
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import LineString

from ...exceptions import HyPlanValueError
from ...geometry import wrap_to_180
from ...units import ureg
from ...waypoint import Waypoint
from .flight_plan_track import PlannedSegment
from .models import DropsondeRelease
from .sensor import AVAPS_NRD41, DropsondeSystem, _as_quantity

if TYPE_CHECKING:
    from ...aircraft._base import Aircraft
    from ...flight_line import FlightLine

__all__ = ["releases_along_flight_line", "releases_along_segment"]


_TURN_TYPES = frozenset({"turn", "approach", "takeoff", "landing"})


def _resolve_groundspeed_mps(
    explicit: Quantity | None,
    aircraft: "Aircraft | None",
    altitude: Quantity,
) -> float | None:
    if explicit is not None:
        return float(_as_quantity(explicit, "meter / second", "groundspeed").magnitude)
    if aircraft is not None:
        # cruise_speed_at returns a TAS Quantity; we treat it as a
        # GS approximation when wind is unknown.
        tas = aircraft.cruise_speed_at(altitude)
        return float(tas.m_as("meter / second"))
    return None


def _resolve_interval_s(
    spacing_m: float | None,
    spacing_s: float | None,
    gs_mps: float | None,
) -> float:
    """Apply the stricter-wins spacing rule.

    Returns the interval (seconds) at which releases should be emitted.
    """
    if spacing_m is None and spacing_s is None:
        # Default cadence: 10 nm OR 5 min, whichever is longer.
        spacing_m = 10 * 1852.0
        spacing_s = 5 * 60.0

    if spacing_m is not None and spacing_s is not None:
        if gs_mps is None or gs_mps <= 0:
            raise HyPlanValueError(
                "distance spacing requires a positive groundspeed "
                "(explicit groundspeed= or aircraft= must be supplied)"
            )
        return max(spacing_s, spacing_m / gs_mps)
    if spacing_m is not None:
        if gs_mps is None or gs_mps <= 0:
            raise HyPlanValueError(
                "distance spacing requires a positive groundspeed "
                "(explicit groundspeed= or aircraft= must be supplied)"
            )
        return spacing_m / gs_mps
    assert spacing_s is not None
    return spacing_s


def _evaluate_qc(
    *,
    altitude_msl_m: float,
    min_release_altitude_m: float,
    surface_elevation_m: float | None,
    aircraft: "Aircraft | None",
    segment_type: str,
) -> tuple[bool | None, bool | None, bool | None]:
    """Return (qc_min_alt_ok, qc_aircraft_envelope_ok, qc_segment_allowed)."""
    if surface_elevation_m is None:
        qc_min_alt = None
    else:
        agl_m = altitude_msl_m - surface_elevation_m
        qc_min_alt = bool(agl_m >= min_release_altitude_m)

    if aircraft is None:
        qc_env = None
        qc_seg = None
    else:
        ceiling_m = float(aircraft.service_ceiling.m_as("meter"))
        qc_env = bool(altitude_msl_m <= ceiling_m)
        qc_seg = bool(segment_type not in _TURN_TYPES)

    return qc_min_alt, qc_env, qc_seg


def releases_along_flight_line(
    flight_line: "FlightLine",
    *,
    sensor: DropsondeSystem = AVAPS_NRD41,
    aircraft: "Aircraft | None" = None,
    takeoff_time: _dt.datetime | None = None,
    start_elapsed: Quantity = 0 * ureg.second,
    groundspeed: Quantity | None = None,
    spacing: Quantity | None = None,
    spacing_time: Quantity | None = None,
    min_release_altitude: Quantity | None = None,
    surface_elevation_msl: Quantity | None = None,
    first_release_id: int = 0,
) -> list[DropsondeRelease]:
    """Spaced :class:`DropsondeRelease` events along a :class:`FlightLine`.

    Heading comes from ``flight_line.az12``; altitude is the line's
    single altitude.  Spacing follows the same stricter-wins rule as
    :meth:`DropsondePlan.from_flight_plan`.
    """
    if flight_line.altitude_msl is None:
        raise HyPlanValueError("flight_line.altitude_msl must be set")
    alt_msl = flight_line.altitude_msl
    alt_m = float(alt_msl.m_as("meter"))

    gs_mps = _resolve_groundspeed_mps(groundspeed, aircraft, alt_msl)

    spacing_m = (
        float(_as_quantity(spacing, "meter", "spacing").magnitude)
        if spacing is not None else None
    )
    spacing_s = (
        float(_as_quantity(spacing_time, "second", "spacing_time").magnitude)
        if spacing_time is not None else None
    )

    length_m = float(flight_line.length.m_as("meter"))
    az_deg = float(flight_line.az12.m_as("degree"))
    interval_s = _resolve_interval_s(spacing_m, spacing_s, gs_mps)
    if gs_mps is None or gs_mps <= 0:
        # Only spacing_time path can reach here; we have no way to step
        # along the line in metres without a groundspeed.  Emit the line
        # start and end only.
        positions_d_m = [0.0]
    else:
        n = int(np.floor((length_m / gs_mps) / interval_s + 1e-9)) + 1
        positions_d_m = [min(k * interval_s * gs_mps, length_m) for k in range(n)]

    min_alt_m = float(
        _as_quantity(
            min_release_altitude if min_release_altitude is not None else sensor.min_release_altitude,
            "meter", "min_release_altitude",
        ).magnitude
    )
    surface_m = (
        float(_as_quantity(surface_elevation_msl, "meter", "surface_elevation_msl").magnitude)
        if surface_elevation_msl is not None else None
    )
    start_elapsed_s = float(_as_quantity(start_elapsed, "second", "start_elapsed").magnitude)

    # Aircraft velocity (u, v) at release, used for the deployment transient.
    ac_velocity: tuple[float, float] | None = None
    if gs_mps is not None and gs_mps > 0:
        az_rad = np.radians(az_deg)
        ac_velocity = (gs_mps * float(np.sin(az_rad)), gs_mps * float(np.cos(az_rad)))

    qc_min, qc_env, qc_seg = _evaluate_qc(
        altitude_msl_m=alt_m,
        min_release_altitude_m=min_alt_m,
        surface_elevation_m=surface_m,
        aircraft=aircraft,
        segment_type="flight_line",
    )

    releases: list[DropsondeRelease] = []
    for k, d_m in enumerate(positions_d_m):
        lat, lon = pymap3d.vincenty.vreckon(
            flight_line.lat1, flight_line.lon1, d_m, az_deg,
        )
        lon = float(wrap_to_180(lon))
        if takeoff_time is not None and gs_mps is not None and gs_mps > 0:
            t_release = takeoff_time + _dt.timedelta(
                seconds=start_elapsed_s + d_m / gs_mps,
            )
        else:
            t_release = None
        wp = Waypoint(
            latitude=float(lat),
            longitude=float(lon),
            heading=az_deg,
            altitude_msl=alt_msl,
            name=f"{flight_line.site_name or 'line'}_drop_{k}",
        )
        releases.append(
            DropsondeRelease(
                waypoint=wp,
                sensor=sensor,
                aircraft=aircraft,
                release_time=t_release,
                aircraft_velocity_mps=ac_velocity,
                source=flight_line,
                source_id=flight_line.site_name,
                source_segment_type="flight_line",
                qc_min_alt_ok=qc_min,
                qc_aircraft_envelope_ok=qc_env,
                qc_segment_allowed=qc_seg,
                release_id=first_release_id + k,
            )
        )
    return releases


def releases_along_segment(
    segment: PlannedSegment,
    *,
    sensor: DropsondeSystem = AVAPS_NRD41,
    aircraft: "Aircraft | None" = None,
    takeoff_time: _dt.datetime,
    spacing: Quantity | None = None,
    spacing_time: Quantity | None = None,
    min_release_altitude: Quantity | None = None,
    surface_elevation_msl: Quantity | None = None,
    first_release_id: int = 0,
) -> list[DropsondeRelease]:
    """Spaced releases along one :class:`PlannedSegment` of a computed plan."""
    if not isinstance(segment.geometry, LineString) or segment._length_m <= 0:
        return []
    if segment.duration_s <= 0:
        return []

    spacing_m = (
        float(_as_quantity(spacing, "meter", "spacing").magnitude)
        if spacing is not None else None
    )
    spacing_s = (
        float(_as_quantity(spacing_time, "second", "spacing_time").magnitude)
        if spacing_time is not None else None
    )

    gs_mps = segment.groundspeed_mps
    if gs_mps is None or gs_mps <= 0:
        gs_mps = segment._length_m / segment.duration_s
    interval_s = _resolve_interval_s(spacing_m, spacing_s, gs_mps)

    n = int(np.floor((segment.duration_s) / interval_s + 1e-9)) + 1

    min_alt_m = float(
        _as_quantity(
            min_release_altitude if min_release_altitude is not None else sensor.min_release_altitude,
            "meter", "min_release_altitude",
        ).magnitude
    )
    surface_m = (
        float(_as_quantity(surface_elevation_msl, "meter", "surface_elevation_msl").magnitude)
        if surface_elevation_msl is not None else None
    )

    az_deg = segment.planned_track_deg
    ac_velocity: tuple[float, float] | None = None
    if az_deg is not None and gs_mps and gs_mps > 0:
        az_rad = np.radians(float(az_deg))
        ac_velocity = (gs_mps * float(np.sin(az_rad)), gs_mps * float(np.cos(az_rad)))

    from .flight_plan_track import FlightPlanTrack
    track = FlightPlanTrack([segment])

    releases: list[DropsondeRelease] = []
    for k in range(n):
        t_in_seg_s = min(k * interval_s, segment.duration_s)
        elapsed = segment.start_elapsed_s + t_in_seg_s
        sample = track.sample_at_elapsed(elapsed * ureg.second)
        alt_m = sample.altitude_m

        qc_min, qc_env, qc_seg = _evaluate_qc(
            altitude_msl_m=alt_m,
            min_release_altitude_m=min_alt_m,
            surface_elevation_m=surface_m,
            aircraft=aircraft,
            segment_type=segment.segment_type,
        )

        wp = Waypoint(
            latitude=sample.latitude,
            longitude=sample.longitude,
            heading=float(az_deg) if az_deg is not None else 0.0,
            altitude_msl=alt_m * ureg.meter,
            name=(
                f"{segment.segment_name}_drop_{k}"
                if segment.segment_name else f"seg_{segment.index}_drop_{k}"
            ),
        )
        t_release = takeoff_time + _dt.timedelta(seconds=elapsed)
        releases.append(
            DropsondeRelease(
                waypoint=wp,
                sensor=sensor,
                aircraft=aircraft,
                release_time=t_release,
                aircraft_velocity_mps=ac_velocity,
                source=segment.index,
                source_id=segment.index,
                source_segment_type=segment.segment_type,
                qc_min_alt_ok=qc_min,
                qc_aircraft_envelope_ok=qc_env,
                qc_segment_allowed=qc_seg,
                release_id=first_release_id + k,
            )
        )
    return releases
