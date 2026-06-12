"""Tests for hyplan.airports."""

import os

import geopandas as gpd
import pandas as pd
import pytest

from hyplan import airports as airports_module
from hyplan.airports import (
    Airport,
    airports_within_radius,
    find_nearest_airport,
    find_nearest_airports,
    generate_geojson,
    get_longest_runway,
    get_runway_details,
    initialize_data,
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


@pytest.fixture()
def synthetic_airports(monkeypatch):
    """Synthetic airports at high latitude and near the antimeridian.

    Distances from the 70°N query point (70.0, -150.0):
    EAST80 is 80 km east (2.10° of longitude), NORTH95 is 95 km north
    (0.85° of latitude), EAST100 is 100 km east, FAR is 1112 km south.
    """
    df = pd.DataFrame({
        "icao_code": ["EAST80", "NORTH95", "EAST100", "FAR", "WEST01", "EAST02"],
        "name": [
            "East 80 km", "North 95 km", "East 100 km", "Far South",
            "Antimeridian West", "Antimeridian East",
        ],
        "latitude": [70.0, 70.854355, 70.0, 60.0, 52.0, 52.0],
        "longitude": [-147.89881, -150.0, -147.373506, -150.0, -179.9, 178.9],
    })
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
    ).set_index("icao_code", drop=False)
    monkeypatch.setattr(airports_module._db, "gdf_airports", gdf)
    return gdf


class TestHighLatitudeAirportSearch:
    def test_within_radius_returns_airport_due_east(self, synthetic_airports):
        results = airports_within_radius(70.0, -150.0, radius=110, unit="kilometers")
        assert "EAST100" in results

    def test_within_radius_exact_membership(self, synthetic_airports):
        results = airports_within_radius(70.0, -150.0, radius=110, unit="kilometers")
        assert set(results) == {"EAST80", "NORTH95", "EAST100"}

    def test_nearest_airports_rank_by_great_circle(self, synthetic_airports):
        # EAST80 (80 km, 2.10°) must rank above NORTH95 (95 km, 0.85°)
        # even though NORTH95 is closer in raw degrees
        results = find_nearest_airports(70.0, -150.0, n=3)
        assert results == ["EAST80", "NORTH95", "EAST100"]

    def test_find_nearest_airport_high_latitude(self, synthetic_airports):
        assert find_nearest_airport(70.0, -150.0) == "EAST80"


class TestAntimeridianAirportSearch:
    def test_within_radius_across_antimeridian(self, synthetic_airports):
        results = airports_within_radius(52.0, 179.9, radius=30, unit="kilometers")
        assert results == ["WEST01"]

    def test_nearest_across_antimeridian(self, synthetic_airports):
        # WEST01 is 13.7 km away across the antimeridian; EAST02 is 68.5 km
        assert find_nearest_airport(52.0, 179.9) == "WEST01"


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


@pytest.fixture()
def synthetic_airport_db(monkeypatch):
    """Two synthetic airports: one without IATA/municipality, one with."""
    import numpy as np

    df = pd.DataFrame({
        "icao_code": ["TEST1", "TEST2"],
        "iata_code": [np.nan, "TST"],
        "name": ["No Optional Fields", "Full Fields"],
        "iso_country": ["US", "US"],
        "municipality": [np.nan, "Testville"],
        "elevation_ft": [123.0, 456.0],
        "latitude": [40.0, 41.0],
        "longitude": [-100.0, -101.0],
    })
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
    ).set_index("icao_code", drop=False)
    monkeypatch.setattr(airports_module._db, "gdf_airports", gdf)
    return gdf


class TestOptionalStringProperties:
    def test_missing_iata_and_municipality_return_none(
        self, synthetic_airport_db
    ):
        apt = Airport("TEST1")
        assert apt.iata_code is None
        assert apt.municipality is None

    def test_present_iata_and_municipality_return_str(
        self, synthetic_airport_db
    ):
        apt = Airport("TEST2")
        assert apt.iata_code == "TST"
        assert apt.municipality == "Testville"

    def test_real_airport_with_iata(self):
        apt = Airport("KLAX")
        assert apt.iata_code == "LAX"
        assert apt.municipality is None or isinstance(apt.municipality, str)


class TestStaleCacheHint:
    def test_logs_for_old_file(self, tmp_path, caplog):
        import logging
        import time as _time

        f = tmp_path / "airports.csv"
        f.write_text("data")
        old = _time.time() - 40 * 86400
        os.utime(f, (old, old))
        with caplog.at_level(logging.INFO, logger="hyplan.airports"):
            airports_module._warn_if_stale(str(f))
        assert any("days old" in r.getMessage() for r in caplog.records)

    def test_quiet_for_fresh_file(self, tmp_path, caplog):
        import logging

        f = tmp_path / "airports.csv"
        f.write_text("data")
        with caplog.at_level(logging.INFO, logger="hyplan.airports"):
            airports_module._warn_if_stale(str(f))
        assert not caplog.records

    def test_noop_for_missing_file(self, tmp_path):
        airports_module._warn_if_stale(str(tmp_path / "missing.csv"))
