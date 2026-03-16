"""Shared fixtures for HyPlan tests."""

import pytest
from hyplan.units import ureg
from hyplan.flight_line import FlightLine


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
