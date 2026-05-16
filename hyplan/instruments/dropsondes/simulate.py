"""Dropsonde descent simulation.

Two entry points:

- :func:`simulate_descent_trajectory` — the low-level RK4 numerical
  kernel.  Takes raw lat/lon/altitude/time floats and a wind field;
  returns the per-step trajectory GeoDataFrame.
- :func:`simulate_release` — object-aware wrapper that takes a
  :class:`DropsondeRelease`, calls the kernel, and packages the
  result as a :class:`DropsondeTrajectory`.

Most callers should use :func:`simulate_release` (or
:meth:`DropsondePlan.simulate`).  The kernel is exposed for advanced
numerical use.

A release without a ``release_time`` (e.g. one built via
``releases_along_flight_line`` without a ``takeoff_time``) is only
simulable against a time-independent wind field — see
``WindField.is_time_dependent``.  :func:`simulate_release` raises
``HyPlanValueError`` early when the combination is invalid; for
time-independent fields it substitutes a sentinel datetime and
proceeds.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import Point

from ...exceptions import HyPlanTypeError, HyPlanValueError
from ...geometry import wrap_to_180
from ...terrain import get_elevations
from ...units import ureg
from ...waypoint import Waypoint
from ...winds.base import WindField
from .models import DropsondeRelease, DropsondeTrajectory
from .sensor import DropsondeSystem, _as_quantity

__all__ = ["simulate_descent_trajectory", "simulate_release"]


_R_EARTH_M = 6_371_000.0
_DEFAULT_MAX_STEPS = 3600
_SENTINEL_TIME = _dt.datetime(2000, 1, 1, tzinfo=_dt.timezone.utc)


def _wind_uv_mps(
    wind_field: WindField,
    lat: float,
    lon: float,
    altitude_m: float,
    time: _dt.datetime,
) -> tuple[float, float]:
    u, v = wind_field.wind_at(
        lat=lat, lon=lon, altitude=altitude_m * ureg.meter, time=time,
    )
    return float(u.m_as("meter / second")), float(v.m_as("meter / second"))


def _surface_at(
    lat: float,
    lon: float,
    *,
    terrain_aware: bool,
    dem_file: str | None,
    surface_elevation_m: float | None,
) -> float:
    if terrain_aware and dem_file is not None:
        elev = get_elevations(
            np.asarray([lat], dtype=float),
            np.asarray([lon], dtype=float),
            dem_file,
        )
        v = float(elev[0])
        if np.isnan(v):
            return -np.inf  # DEM gap; treat as "above ground", caller tracks
        return v
    if surface_elevation_m is not None:
        return surface_elevation_m
    return 0.0  # sea level


def _derivatives(
    u_mps: float, v_mps: float, w_mps: float, lat_deg: float,
) -> tuple[float, float, float]:
    """Geographic-frame derivatives: (dlat/dt, dlon/dt, dz/dt)."""
    cos_lat = float(np.cos(np.radians(lat_deg)))
    cos_lat = max(cos_lat, 1e-9)
    dlat = float(np.degrees(v_mps / _R_EARTH_M))
    dlon = float(np.degrees(u_mps / (_R_EARTH_M * cos_lat)))
    dz = -float(w_mps)
    return dlat, dlon, dz


# ---------------------------------------------------------------------------
# Low-level kernel
# ---------------------------------------------------------------------------


def simulate_descent_trajectory(
    release_lat: float,
    release_lon: float,
    release_altitude_msl: Quantity,
    release_time_utc: _dt.datetime,
    *,
    sensor: DropsondeSystem,
    wind_field: WindField,
    dem_file: str | None = None,
    terrain_aware: bool = False,
    surface_elevation_msl: Quantity | None = None,
    dt: Quantity = 1 * ureg.second,
    method: Literal["rk4", "euler"] = "rk4",
    max_steps: int = _DEFAULT_MAX_STEPS,
    release_id: int = 0,
    ensemble_member: int = 0,
    u_bias_mps: float = 0.0,
    v_bias_mps: float = 0.0,
    fall_rate_scale: float = 1.0,
    aircraft_velocity_mps: tuple[float, float] | None = None,
) -> gpd.GeoDataFrame:
    """RK4 integrator for a single dropsonde descent.

    Integrates::

        dlat/dt = v_n / R_earth
        dlon/dt = v_e / (R_earth * cos(lat))
        dz/dt   = -w_f(z)

    with per-step wind sampling at the current trajectory point, a
    deployment-transient that decays linearly from the aircraft
    velocity to the local wind over ``sensor.deployment_time``, and
    per-ensemble (u_bias, v_bias) additive offsets.

    Termination, in order: terrain (``terrain_aware=True``),
    ``surface_elevation_msl``, sea level, or ``max_steps``.

    Returns the per-step GeoDataFrame with ``terminated_at_ground`` /
    ``max_steps_exceeded`` / ``dem_gap_count`` attached as ``attrs``.
    """
    if not isinstance(wind_field, WindField):
        raise HyPlanTypeError("wind_field must be a WindField subclass")
    if not isinstance(release_time_utc, _dt.datetime):
        raise HyPlanTypeError("release_time_utc must be a datetime")
    if method not in ("rk4", "euler"):
        raise HyPlanValueError("method must be 'rk4' or 'euler'")
    if max_steps <= 0:
        raise HyPlanValueError("max_steps must be positive")

    dt_s = float(_as_quantity(dt, "second", "dt").magnitude)
    if dt_s <= 0:
        raise HyPlanValueError("dt must be positive")
    z0_m = float(
        _as_quantity(release_altitude_msl, "meter", "release_altitude_msl").magnitude
    )

    surface_m: float | None = None
    if surface_elevation_msl is not None:
        surface_m = float(
            _as_quantity(surface_elevation_msl, "meter", "surface_elevation_msl").magnitude
        )

    deploy_s = float(sensor.deployment_time.m_as("second"))
    if aircraft_velocity_mps is None or deploy_s <= 0.0:
        ac_u_mps = 0.0
        ac_v_mps = 0.0
        deploy_active = False
    else:
        ac_u_mps = float(aircraft_velocity_mps[0])
        ac_v_mps = float(aircraft_velocity_mps[1])
        deploy_active = True

    def _effective_uv(
        elapsed_s: float,
        lat: float,
        lon: float,
        z_m: float,
        t: _dt.datetime,
    ) -> tuple[float, float]:
        u_w, v_w = _wind_uv_mps(wind_field, lat, lon, z_m, t)
        u_eff = u_w + u_bias_mps
        v_eff = v_w + v_bias_mps
        if deploy_active and elapsed_s < deploy_s:
            frac = 1.0 - elapsed_s / deploy_s
            u_eff += (ac_u_mps - u_eff) * frac
            v_eff += (ac_v_mps - v_eff) * frac
        return u_eff, v_eff

    lat = float(release_lat)
    lon = float(release_lon)
    z_m = z0_m
    t = release_time_utc
    u_mps, v_mps = _effective_uv(0.0, lat, lon, z_m, t)
    w_mps = (
        float(sensor.descent_rate_model(z_m * ureg.meter).m_as("meter / second"))
        * fall_rate_scale
    )
    surface_m_now = _surface_at(
        lat, lon,
        terrain_aware=terrain_aware,
        dem_file=dem_file,
        surface_elevation_m=surface_m,
    )
    agl_m = z_m - surface_m_now if np.isfinite(surface_m_now) else np.nan

    rows: list[dict[str, Any]] = [
        {
            "release_id": release_id,
            "ensemble_member": ensemble_member,
            "step": 0,
            "time_utc": t,
            "latitude": lat,
            "longitude": lon,
            "altitude_msl_m": z_m,
            "altitude_agl_m": agl_m,
            "u_wind_mps": u_mps,
            "v_wind_mps": v_mps,
            "w_fall_mps": w_mps,
            "geometry": Point(lon, lat),
        }
    ]

    consec_nan_wind = 0
    last_known_wind = (u_mps, v_mps)
    terminated_at_ground = False
    max_steps_exceeded = False
    dem_gap_count = 0

    for step_idx in range(1, max_steps + 1):
        elapsed_s_start = (step_idx - 1) * dt_s
        u_mps, v_mps = _effective_uv(elapsed_s_start, lat, lon, z_m, t)
        if not (np.isfinite(u_mps) and np.isfinite(v_mps)):
            consec_nan_wind += 1
            if consec_nan_wind > 3:
                warnings.warn(
                    "Wind field returned NaN for >3 consecutive steps; "
                    "terminating trajectory early.",
                    stacklevel=2,
                )
                break
            u_mps, v_mps = last_known_wind
        else:
            consec_nan_wind = 0
            last_known_wind = (u_mps, v_mps)

        w_mps = (
            float(sensor.descent_rate_model(z_m * ureg.meter).m_as("meter / second"))
            * fall_rate_scale
        )

        if method == "rk4":
            k1_lat, k1_lon, k1_z = _derivatives(u_mps, v_mps, w_mps, lat)
            lat_h = lat + 0.5 * dt_s * k1_lat
            lon_h = lon + 0.5 * dt_s * k1_lon
            z_h = z_m + 0.5 * dt_s * k1_z
            t_h = t + _dt.timedelta(seconds=0.5 * dt_s)
            u_h, v_h = _effective_uv(elapsed_s_start + 0.5 * dt_s, lat_h, lon_h, z_h, t_h)
            w_h = (
                float(sensor.descent_rate_model(z_h * ureg.meter).m_as("meter / second"))
                * fall_rate_scale
            )
            k2_lat, k2_lon, k2_z = _derivatives(u_h, v_h, w_h, lat_h)
            lat_h2 = lat + 0.5 * dt_s * k2_lat
            lon_h2 = lon + 0.5 * dt_s * k2_lon
            z_h2 = z_m + 0.5 * dt_s * k2_z
            u_h2, v_h2 = _effective_uv(elapsed_s_start + 0.5 * dt_s, lat_h2, lon_h2, z_h2, t_h)
            w_h2 = (
                float(sensor.descent_rate_model(z_h2 * ureg.meter).m_as("meter / second"))
                * fall_rate_scale
            )
            k3_lat, k3_lon, k3_z = _derivatives(u_h2, v_h2, w_h2, lat_h2)
            lat_f = lat + dt_s * k3_lat
            lon_f = lon + dt_s * k3_lon
            z_f = z_m + dt_s * k3_z
            t_f = t + _dt.timedelta(seconds=dt_s)
            u_f, v_f = _effective_uv(elapsed_s_start + dt_s, lat_f, lon_f, z_f, t_f)
            w_f = (
                float(sensor.descent_rate_model(z_f * ureg.meter).m_as("meter / second"))
                * fall_rate_scale
            )
            k4_lat, k4_lon, k4_z = _derivatives(u_f, v_f, w_f, lat_f)
            lat += dt_s * (k1_lat + 2 * k2_lat + 2 * k3_lat + k4_lat) / 6.0
            lon += dt_s * (k1_lon + 2 * k2_lon + 2 * k3_lon + k4_lon) / 6.0
            z_m += dt_s * (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6.0
        else:  # euler
            dlat, dlon, dz = _derivatives(u_mps, v_mps, w_mps, lat)
            lat += dt_s * dlat
            lon += dt_s * dlon
            z_m += dt_s * dz

        lon = float(wrap_to_180(lon))
        t = t + _dt.timedelta(seconds=dt_s)

        surface_m_now = _surface_at(
            lat, lon,
            terrain_aware=terrain_aware,
            dem_file=dem_file,
            surface_elevation_m=surface_m,
        )
        if terrain_aware and dem_file is not None and not np.isfinite(surface_m_now):
            dem_gap_count += 1
        if np.isfinite(surface_m_now) and z_m <= surface_m_now:
            prev = rows[-1]
            dz_total = z_m - prev["altitude_msl_m"]
            if dz_total != 0.0:
                frac = (surface_m_now - prev["altitude_msl_m"]) / dz_total
                frac = max(0.0, min(1.0, frac))
                lat = prev["latitude"] + frac * (lat - prev["latitude"])
                lon = prev["longitude"] + frac * (lon - prev["longitude"])
                z_m = surface_m_now
                dt_back = (1.0 - frac) * dt_s
                t = t - _dt.timedelta(seconds=dt_back)
            agl_m = 0.0
            rows.append(
                {
                    "release_id": release_id,
                    "ensemble_member": ensemble_member,
                    "step": step_idx,
                    "time_utc": t,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_msl_m": z_m,
                    "altitude_agl_m": agl_m,
                    "u_wind_mps": u_mps,
                    "v_wind_mps": v_mps,
                    "w_fall_mps": w_mps,
                    "geometry": Point(lon, lat),
                }
            )
            terminated_at_ground = True
            break

        agl_m = z_m - surface_m_now if np.isfinite(surface_m_now) else np.nan
        rows.append(
            {
                "release_id": release_id,
                "ensemble_member": ensemble_member,
                "step": step_idx,
                "time_utc": t,
                "latitude": lat,
                "longitude": lon,
                "altitude_msl_m": z_m,
                "altitude_agl_m": agl_m,
                "u_wind_mps": u_mps,
                "v_wind_mps": v_mps,
                "w_fall_mps": w_mps,
                "geometry": Point(lon, lat),
            }
        )
    else:
        max_steps_exceeded = True
        warnings.warn(
            f"max_steps={max_steps} reached before trajectory terminated.",
            stacklevel=2,
        )

    gdf = gpd.GeoDataFrame(
        pd.DataFrame.from_records(rows), geometry="geometry", crs="EPSG:4326",
    )
    gdf.attrs["terminated_at_ground"] = bool(terminated_at_ground)
    gdf.attrs["max_steps_exceeded"] = bool(max_steps_exceeded)
    gdf.attrs["dem_gap_count"] = int(dem_gap_count)
    return gdf


# ---------------------------------------------------------------------------
# Object-aware wrapper
# ---------------------------------------------------------------------------


def _check_release_time(
    release: DropsondeRelease, wind_field: WindField,
) -> _dt.datetime:
    """Resolve the simulation timestamp; raise if missing for a time-dependent field."""
    if release.release_time is not None:
        return release.release_time
    if getattr(wind_field, "is_time_dependent", True):
        raise HyPlanValueError(
            "simulate_release requires release.release_time for "
            "time-dependent wind fields (got release_time=None and "
            f"wind_field={type(wind_field).__name__})"
        )
    return _SENTINEL_TIME


def simulate_release(
    release: DropsondeRelease,
    *,
    wind_field: WindField,
    dem_file: str | None = None,
    terrain_aware: bool = False,
    surface_elevation_msl: Quantity | None = None,
    dt: Quantity = 1 * ureg.second,
    u_bias_mps: float = 0.0,
    v_bias_mps: float = 0.0,
    fall_rate_scale: float = 1.0,
    ensemble_member: int = 0,
) -> DropsondeTrajectory:
    """Forward-simulate a :class:`DropsondeRelease` through ``wind_field``.

    Wraps :func:`simulate_descent_trajectory`.  All inputs except the
    integration kwargs come from ``release``.  Returns a
    :class:`DropsondeTrajectory` whose ``release`` field is the same
    object passed in.
    """
    if release.waypoint.altitude_msl is None:
        raise HyPlanValueError(
            "DropsondeRelease.waypoint.altitude_msl is required for simulation"
        )

    t0 = _check_release_time(release, wind_field)
    track = simulate_descent_trajectory(
        release_lat=float(release.waypoint.latitude),
        release_lon=float(release.waypoint.longitude),
        release_altitude_msl=release.waypoint.altitude_msl,
        release_time_utc=t0,
        sensor=release.sensor,
        wind_field=wind_field,
        dem_file=dem_file,
        terrain_aware=terrain_aware,
        surface_elevation_msl=surface_elevation_msl,
        dt=dt,
        release_id=release.release_id,
        ensemble_member=ensemble_member,
        u_bias_mps=u_bias_mps,
        v_bias_mps=v_bias_mps,
        fall_rate_scale=fall_rate_scale,
        aircraft_velocity_mps=release.aircraft_velocity_mps,
    )

    last = track.iloc[-1]
    splash_wp = Waypoint(
        latitude=float(last["latitude"]),
        longitude=float(last["longitude"]),
        heading=0.0,
        altitude_msl=float(last["altitude_msl_m"]) * ureg.meter,
        name=f"Splash {release.release_id}",
    )
    drift_m, az_fwd = pymap3d.vincenty.vdist(
        float(release.waypoint.latitude),
        float(release.waypoint.longitude),
        float(last["latitude"]),
        float(last["longitude"]),
    )
    t_end = last["time_utc"]
    t_start = track.iloc[0]["time_utc"]
    if isinstance(t_end, _dt.datetime) and isinstance(t_start, _dt.datetime):
        dt_total = (t_end - t_start).total_seconds()
    else:
        dt_total = float("nan")

    return DropsondeTrajectory(
        release=release,
        track=track,
        splash_waypoint=splash_wp,
        time_to_surface=dt_total * ureg.second,
        drift_distance=float(drift_m) * ureg.meter,
        drift_bearing_deg=float(az_fwd) if np.isfinite(az_fwd) else 0.0,
        ensemble_member=ensemble_member,
        qc_terminated_at_ground=bool(track.attrs.get("terminated_at_ground", False)),
        qc_max_steps_exceeded=bool(track.attrs.get("max_steps_exceeded", False)),
        qc_dem_gap_count=int(track.attrs.get("dem_gap_count", 0)),
    )
