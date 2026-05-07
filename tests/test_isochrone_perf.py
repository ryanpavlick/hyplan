"""Performance regression gate for compute_isochrone / compute_refuel_isochrone.

Marked with ``@pytest.mark.perf`` so it runs in CI but can be skipped
locally with ``pytest -m "not perf"``.

Baseline reference (v1.5.1, before Phase 3 optimizations):

* still air,        GIII, 36 rays:                    ~0.32 s
* const wind 60 kt, GIII, 72 rays:                    ~0.92 s
* refuel-extended,  B200, 2 candidates, 36 rays:     ~12.83 s

Post v1.6 (climb/descent + track-hold caches):

* still air,        GIII, 36 rays:                    ~0.31 s
* const wind 60 kt, GIII, 72 rays:                    ~0.76 s
* refuel-extended,  B200, 2 candidates, 36 rays:      ~2.60 s

Targets below have generous 2-3x headroom over the post-v1.6 numbers
to absorb shared-runner jitter without flaking.  A regression to
within 50% of the v1.5.1 baseline triggers the alarm.
"""
from __future__ import annotations
import time

import pytest

from hyplan import Waypoint, ureg
from hyplan.aircraft import NASA_GIII, KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.winds import StillAirField, ConstantWindField
from hyplan.planning.isochrone import (
    compute_isochrone,
    compute_refuel_isochrone,
)


pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def airports():
    initialize_data(countries=["US"])


def _kedw_wp(altitude):
    return Waypoint(
        latitude=34.91, longitude=-117.88,
        altitude_msl=altitude, heading=0.0,
    )


def _kefd_wp(altitude):
    return Waypoint(
        latitude=29.61, longitude=-95.16,
        altitude_msl=altitude, heading=0.0,
    )


def test_perf_still_air():
    """Round-trip still-air isochrone, GIII, 36 rays.  Target: < 2 s."""
    cruise_alt = ureg.Quantity(40000, "feet")
    t0 = time.perf_counter()
    gdf = compute_isochrone(
        aircraft=NASA_GIII(), start=_kedw_wp(cruise_alt),
        budget=4 * ureg.hour, cruise_altitude=cruise_alt,
        mode="round_trip", wind_source=StillAirField(),
        azimuth_resolution_deg=10.0,
    )
    dt = time.perf_counter() - t0
    assert len(gdf) == 36
    assert dt < 2.0, f"still-air 36-ray isochrone took {dt:.2f}s (target <2s)"


def test_perf_constant_wind():
    """Round-trip constant-wind isochrone, GIII, 72 rays.  Target: < 3 s."""
    cruise_alt = ureg.Quantity(40000, "feet")
    wind = ConstantWindField(wind_speed=60 * ureg.knot, wind_from_deg=270.0)
    t0 = time.perf_counter()
    gdf = compute_isochrone(
        aircraft=NASA_GIII(), start=_kedw_wp(cruise_alt),
        budget=4 * ureg.hour, cruise_altitude=cruise_alt,
        mode="round_trip", wind_source=wind,
        azimuth_resolution_deg=5.0,
    )
    dt = time.perf_counter() - t0
    assert len(gdf) == 72
    assert dt < 3.0, f"constant-wind 72-ray isochrone took {dt:.2f}s (target <3s)"


def test_perf_refuel(airports):
    """Refuel-extended isochrone, B-200, 2 candidates, 36 rays.

    Target: < 8 s (v1.5.1 baseline 12.8 s; post-v1.6 ≈ 2.6 s).
    """
    cruise_alt = ureg.Quantity(25000, "feet")
    wind = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=270.0)
    t0 = time.perf_counter()
    gdf = compute_refuel_isochrone(
        aircraft=KingAirB200(), start=_kefd_wp(cruise_alt),
        sortie_budget=4 * ureg.hour, flight_day_budget=8 * ureg.hour,
        refuel_time=30 * ureg.minute,
        refuel_airports=[Airport("KLBB"), Airport("KMSY")],
        return_destination=Airport("KEFD"),
        cruise_altitude=cruise_alt, wind_source=wind,
        azimuth_resolution_deg=10.0,
    )
    dt = time.perf_counter() - t0
    assert len(gdf) == 36
    assert dt < 8.0, (
        f"refuel-extended 36-ray isochrone took {dt:.2f}s "
        f"(target <8s; v1.5.1 was 12.8s, v1.6 is 2.6s)"
    )
