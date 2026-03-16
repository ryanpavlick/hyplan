"""Tests for hyplan.airports."""

import pytest
from hyplan.airports import (
    Airport,
    initialize_data,
    find_nearest_airport,
    find_nearest_airports,
    airports_within_radius,
)


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


class TestAirport:
    def test_create_by_icao(self):
        apt = Airport("KLAX")
        assert apt.icao_code == "KLAX"
        assert apt.latitude is not None
        assert apt.longitude is not None

    def test_properties(self):
        apt = Airport("KLAX")
        assert apt.name is not None
        assert apt.country == "US"
        assert apt.elevation is not None

    def test_geometry(self):
        apt = Airport("KLAX")
        geom = apt.geometry
        assert geom is not None
        assert geom.x == pytest.approx(apt.longitude)
        assert geom.y == pytest.approx(apt.latitude)

    def test_invalid_icao(self):
        with pytest.raises((ValueError, KeyError)):
            Airport("ZZZZ")


class TestAirportSearch:
    def test_find_nearest(self):
        # Near LAX
        icao = find_nearest_airport(33.94, -118.40)
        assert isinstance(icao, str)
        assert len(icao) == 4

    def test_find_nearest_airports(self):
        results = find_nearest_airports(33.94, -118.40, n=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_airports_within_radius(self):
        results = airports_within_radius(33.94, -118.40, radius=50, unit="kilometers")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_airports_within_radius_details(self):
        gdf = airports_within_radius(
            33.94, -118.40, radius=50, unit="kilometers", return_details=True
        )
        assert len(gdf) > 0
        assert "geometry" in gdf.columns
