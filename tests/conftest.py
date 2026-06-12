"""Shared fixtures for HyPlan tests."""

import os
import tempfile

import pytest

from hyplan.flight_line import FlightLine
from hyplan.units import ureg


@pytest.fixture(scope="session", autouse=True)
def _hyplan_cache_root():
    """Keep all HyPlan caches out of the real home directory.

    ``get_cache_root()`` defaults to ``~/.cache/hyplan``; tests must
    never write there.  A stable per-user temp location (rather than a
    fresh tmp_path) is used so large downloads (airports.csv, DEM
    tiles) are still reused across test sessions.  An externally set
    HYPLAN_CACHE_ROOT is respected.
    """
    if os.environ.get("HYPLAN_CACHE_ROOT"):
        yield
        return
    root = os.path.join(tempfile.gettempdir(), "hyplan-test-cache")
    os.makedirs(root, exist_ok=True)
    os.environ["HYPLAN_CACHE_ROOT"] = root
    try:
        yield
    finally:
        os.environ.pop("HYPLAN_CACHE_ROOT", None)


@pytest.fixture
def sample_flight_line():
    """A simple 50 km flight line over Los Angeles."""
    return FlightLine.start_length_azimuth(
        lat1=34.05,
        lon1=-118.25,
        length=ureg.Quantity(50000, "meter"),
        az=45.0,
        altitude_msl=ureg.Quantity(6000, "meter"),
        site_name="Test Line",
    )


@pytest.fixture
def short_flight_line():
    """A short 10 km flight line."""
    return FlightLine.start_length_azimuth(
        lat1=34.0,
        lon1=-118.0,
        length=ureg.Quantity(10000, "meter"),
        az=90.0,
        altitude_msl=ureg.Quantity(3000, "meter"),
        site_name="Short Line",
    )
