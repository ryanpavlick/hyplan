"""Representative compute_isochrone fixtures for profiling.

Three cases ordered by expected runtime, so a single run exercises the
spectrum of cost we care about:

1. Still air, modest azimuth resolution — smallest case; should already
   be fast.
2. Constant wind, fine azimuth resolution — exercises the trochoidal
   path inside `_leg_time` without provider overhead.
3. Refuel isochrone with a couple of refuel candidates — exercises the
   4-6x `_leg_time` calls per ray and is the slowest direct fixture
   that doesn't need network access.

A fourth case (gridded `MERRA2WindField`) is the realistic worst-case
target but requires NASA Earthdata credentials and a network round
trip; gated behind the `BENCH_GRIDDED` env var.

Run as:

    python -m bench.run_isochrone           # cases 1-3
    BENCH_GRIDDED=1 python -m bench.run_isochrone   # also case 4
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from hyplan import Waypoint, ureg
from hyplan.aircraft import NASA_GIII, KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.winds import StillAirField, ConstantWindField
from hyplan.planning.isochrone import (
    compute_isochrone,
    compute_refuel_isochrone,
)


def _kedw_waypoint(altitude: "ureg.Quantity") -> Waypoint:
    """Edwards AFB / Edwards-area starting waypoint at cruise altitude."""
    return Waypoint(
        latitude=34.91, longitude=-117.88,
        altitude_msl=altitude, heading=0.0,
    )


def _kefd_waypoint(altitude: "ureg.Quantity") -> Waypoint:
    return Waypoint(
        latitude=29.61, longitude=-95.16,
        altitude_msl=altitude, heading=0.0,
    )


def _time_call(label: str, fn: Callable[[], object]) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    n = len(result) if hasattr(result, "__len__") else "?"
    print(f"  {label:<46s}  {dt:7.2f}s   n_rays={n}")
    return label, dt


def case_still_air_giii() -> tuple[str, float]:
    aircraft = NASA_GIII()
    cruise_alt = ureg.Quantity(40000, "feet")
    start = _kedw_waypoint(cruise_alt)
    return _time_call(
        "1. still air, GIII, 36 rays",
        lambda: compute_isochrone(
            aircraft=aircraft, start=start,
            budget=4 * ureg.hour,
            cruise_altitude=cruise_alt,
            mode="round_trip",
            wind_source=StillAirField(),
            azimuth_resolution_deg=10.0,
        ),
    )


def case_constant_wind_giii() -> tuple[str, float]:
    aircraft = NASA_GIII()
    cruise_alt = ureg.Quantity(40000, "feet")
    start = _kedw_waypoint(cruise_alt)
    wind = ConstantWindField(
        wind_speed=60 * ureg.knot, wind_from_deg=270.0,
    )
    return _time_call(
        "2. const wind 60 kt, GIII, 72 rays",
        lambda: compute_isochrone(
            aircraft=aircraft, start=start,
            budget=4 * ureg.hour,
            cruise_altitude=cruise_alt,
            mode="round_trip",
            wind_source=wind,
            azimuth_resolution_deg=5.0,
        ),
    )


def case_refuel_b200() -> tuple[str, float]:
    initialize_data(countries=["US"])
    aircraft = KingAirB200()
    cruise_alt = ureg.Quantity(25000, "feet")
    start = _kefd_waypoint(cruise_alt)
    # Two refuel options spread around the gulf coast.
    refuel = [Airport("KLBB"), Airport("KMSY")]
    wind = ConstantWindField(
        wind_speed=30 * ureg.knot, wind_from_deg=270.0,
    )
    return_to = Airport("KEFD")
    return _time_call(
        "3. refuel-extended, B200, 2 candidates, 36 rays",
        lambda: compute_refuel_isochrone(
            aircraft=aircraft, start=start,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            refuel_time=30 * ureg.minute,
            refuel_airports=refuel,
            return_destination=return_to,
            cruise_altitude=cruise_alt,
            wind_source=wind,
            azimuth_resolution_deg=10.0,
        ),
    )


def case_gridded_wind_giii() -> tuple[str, float]:
    """Network + credentials gated; opt in via BENCH_GRIDDED=1."""
    if not os.environ.get("BENCH_GRIDDED"):
        print("  (skipped) 4. gridded MERRA-2 — set BENCH_GRIDDED=1 to enable")
        return ("4. gridded MERRA-2 (skipped)", float("nan"))
    from hyplan.winds.providers import MERRA2WindField
    import datetime as dt
    aircraft = NASA_GIII()
    cruise_alt = ureg.Quantity(40000, "feet")
    start = _kedw_waypoint(cruise_alt)
    wind = MERRA2WindField()
    t0 = dt.datetime(2024, 6, 15, 12, tzinfo=dt.timezone.utc)
    return _time_call(
        "4. MERRA-2 gridded, GIII, 36 rays",
        lambda: compute_isochrone(
            aircraft=aircraft, start=start,
            budget=4 * ureg.hour,
            cruise_altitude=cruise_alt,
            mode="round_trip",
            wind_source=wind,
            takeoff_time=t0,
            azimuth_resolution_deg=10.0,
        ),
    )


def main():
    print("compute_isochrone profiling fixtures")
    print("=" * 70)
    rng = np.random.default_rng(0)  # for any stochastic fallback
    cases = [
        case_still_air_giii,
        case_constant_wind_giii,
        case_refuel_b200,
        case_gridded_wind_giii,
    ]
    results = []
    for c in cases:
        results.append(c())
    print("=" * 70)
    total = sum(dt for _, dt in results if not np.isnan(dt))
    print(f"  TOTAL                                                 {total:7.2f}s")


if __name__ == "__main__":
    main()
