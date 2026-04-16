"""Tests for hyplan.airports."""

import os
import pytest
import pandas as pd
from hyplan.airports import (
    Airport,
    initialize_data,
    find_nearest_airport,
    find_nearest_airports,
    airports_within_radius,
    get_runway_details,
    get_longest_runway,
    generate_geojson,
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


class TestGetRunwayDetails:
    def test_returns_dataframe(self):
        df = get_runway_details("KSBA")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_contains_expected_columns(self):
        df = get_runway_details("KSBA")
        for col in ["airport_ident", "length_ft", "width_ft", "surface"]:
            assert col in df.columns

    def test_all_rows_match_icao(self):
        df = get_runway_details("KSBA")
        assert (df["airport_ident"] == "KSBA").all()

    def test_accepts_list_of_icao_codes(self):
        df = get_runway_details(["KSBA", "KLAX"])
        assert set(df["airport_ident"].unique()) <= {"KSBA", "KLAX"}
        assert len(df) > 0


class TestGetLongestRunway:
    def test_returns_float(self):
        result = get_longest_runway("KSBA")
        assert isinstance(result, float)
        assert result > 0

    def test_large_airport_has_long_runway(self):
        result = get_longest_runway("KLAX")
        # LAX has runways over 10000 ft
        assert result > 10000

    def test_unknown_airport_returns_none(self):
        result = get_longest_runway("ZZZZ")
        assert result is None


class TestGenerateGeojson:
    def test_creates_file(self, tmp_path):
        filepath = str(tmp_path / "test_airports.geojson")
        generate_geojson(filepath=filepath, icao_codes=["KSBA"])
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_subset_icao_single_string(self, tmp_path):
        filepath = str(tmp_path / "single.geojson")
        generate_geojson(filepath=filepath, icao_codes="KSBA")
        assert os.path.exists(filepath)

    def test_subset_icao_list(self, tmp_path):
        filepath = str(tmp_path / "multi.geojson")
        generate_geojson(filepath=filepath, icao_codes=["KSBA", "KLAX"])
        assert os.path.exists(filepath)


class TestAirportRunwaysProperty:
    def test_runways_property_returns_dataframe(self):
        apt = Airport("KSBA")
        df = apt.runways
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert (df["airport_ident"] == "KSBA").all()
