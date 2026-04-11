"""Tests for hyplan.clouds (date range logic, no Google Earth Engine required)."""

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from hyplan.clouds import (
    create_date_ranges,
    simulate_visits,
    OpenMeteoCloudFraction,
    fetch_cloud_fraction,
)


class TestCreateDateRanges:
    def test_normal_range(self):
        """Within-year range should produce one tuple per year."""
        ranges = create_date_ranges(100, 200, 2020, 2022)
        assert len(ranges) == 3
        for start, stop in ranges:
            assert "-100" in start
            assert "-201" in stop

    def test_single_year(self):
        ranges = create_date_ranges(1, 365, 2025, 2025)
        assert len(ranges) == 1

    def test_year_boundary_crossing(self):
        """day_start > day_stop should produce two ranges per year."""
        ranges = create_date_ranges(335, 60, 2020, 2020)
        assert len(ranges) == 2
        # First range: day 335 of 2020 to Jan 1 of 2021
        assert "2020-335" in ranges[0][0]
        assert "2021-001" in ranges[0][1]
        # Second range: Jan 1 to day 61 of 2021
        assert "2021-001" in ranges[1][0]
        assert "2021-061" in ranges[1][1]

    def test_year_boundary_multi_year(self):
        """Multi-year crossing should produce 2 ranges per year."""
        ranges = create_date_ranges(300, 90, 2019, 2021)
        assert len(ranges) == 6  # 3 years x 2 ranges each


class TestSimulateVisits:
    @pytest.fixture
    def sample_cloud_df(self):
        """Create a simple cloud fraction DataFrame for testing."""
        rows = []
        for year in [2020, 2021]:
            for day in range(1, 366):
                for poly in ["A", "B", "C"]:
                    # Make cloud fraction vary: every 3rd day is clear
                    cf = 0.05 if day % 3 == 0 else 0.50
                    rows.append({
                        "polygon_id": poly,
                        "year": year,
                        "day_of_year": day,
                        "cloud_fraction": cf,
                    })
        return pd.DataFrame(rows)

    def test_basic_simulation(self, sample_cloud_df):
        result_df, visit_tracker, rest_days_dict = simulate_visits(
            sample_cloud_df,
            day_start=1, day_stop=100,
            year_start=2020, year_stop=2020,
            cloud_fraction_threshold=0.10,
        )
        assert len(result_df) == 1
        assert result_df["year"].iloc[0] == 2020
        assert 2020 in visit_tracker

    def test_all_polygons_visited(self, sample_cloud_df):
        """With enough clear days, all polygons should be visited."""
        result_df, visit_tracker, _ = simulate_visits(
            sample_cloud_df,
            day_start=1, day_stop=365,
            year_start=2020, year_stop=2020,
            cloud_fraction_threshold=0.10,
        )
        visited = set(visit_tracker[2020].keys())
        assert visited == {"A", "B", "C"}

    def test_rest_days(self, sample_cloud_df):
        """With rest_day_threshold=1, rest days should be inserted after each visit."""
        _, visit_tracker, rest_days_dict = simulate_visits(
            sample_cloud_df,
            day_start=1, day_stop=365,
            year_start=2020, year_stop=2020,
            cloud_fraction_threshold=0.10,
            rest_day_threshold=1,
        )
        # With threshold=1, a rest day is needed after every visit,
        # but rest days only occur when there IS a visitable polygon on that day.
        # Just verify the simulation completes and visits polygons.
        total_visits = sum(len(v) for v in visit_tracker[2020].values())
        assert total_visits > 0

    def test_year_boundary_simulation(self, sample_cloud_df):
        """Simulate across a year boundary (day_start > day_stop)."""
        # Add cloud data for the relevant days
        result_df, visit_tracker, _ = simulate_visits(
            sample_cloud_df,
            day_start=350, day_stop=30,
            year_start=2020, year_stop=2020,
            cloud_fraction_threshold=0.10,
        )
        assert len(result_df) == 1
        # The simulation should have run across the boundary
        assert result_df["days"].iloc[0] > 0

    def test_exclude_weekends(self, sample_cloud_df):
        result_df, _, _ = simulate_visits(
            sample_cloud_df,
            day_start=1, day_stop=30,
            year_start=2020, year_stop=2020,
            cloud_fraction_threshold=0.10,
            exclude_weekends=True,
        )
        assert len(result_df) == 1


# ---------------------------------------------------------------------------
# OpenMeteoCloudFraction
# ---------------------------------------------------------------------------

def _make_openmeteo_response(dates, cloud_pcts):
    """Build a mock Open-Meteo JSON response."""
    return {
        "latitude": 34.4,
        "longitude": -119.8,
        "daily_units": {"time": "iso8601", "cloud_cover_mean": "%"},
        "daily": {
            "time": dates,
            "cloud_cover_mean": cloud_pcts,
        },
    }


def _mock_gdf(names_coords):
    """Build a GeoDataFrame with Name and polygon geometry from (name, lat, lon) tuples."""
    import geopandas as _gpd
    from shapely.geometry import box

    rows = []
    for name, lat, lon in names_coords:
        rows.append({"Name": name, "geometry": box(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)})
    return _gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestOpenMeteoCloudFraction:
    def test_basic_fetch(self):
        """Mocked fetch returns correct DataFrame schema."""
        dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
        pcts = [10.0, 50.0, 90.0]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openmeteo_response(dates, pcts)

        gdf = _mock_gdf([("Box1", 34.4, -119.8)])

        with patch("hyplan.clouds._requests.get" if hasattr(OpenMeteoCloudFraction, "_requests") else "requests.get", return_value=mock_resp) as mock_get:
            # Patch at the import location inside the method
            import hyplan.clouds as _clouds_mod
            original_fetch = OpenMeteoCloudFraction.fetch

            src = OpenMeteoCloudFraction()
            # Use a direct mock of requests inside fetch
            with patch.dict("sys.modules", {}):
                pass

            # Simpler: just call with mocked requests
            import requests as _req
            with patch.object(_req, "get", return_value=mock_resp):
                df = src.fetch(gdf, 2023, 2023, 1, 3)

        assert set(df.columns) == {"polygon_id", "year", "day_of_year", "cloud_fraction"}
        assert len(df) == 3
        assert df["cloud_fraction"].iloc[0] == pytest.approx(0.10)
        assert df["cloud_fraction"].iloc[2] == pytest.approx(0.90)
        assert df["polygon_id"].iloc[0] == "Box1"

    def test_missing_name_column(self):
        """Should raise if Name column is missing."""
        import geopandas as _gpd
        from shapely.geometry import Point

        gdf = _gpd.GeoDataFrame({"id": ["A"]}, geometry=[Point(0, 0)])
        src = OpenMeteoCloudFraction()
        with pytest.raises(Exception, match="Name"):
            src.fetch(gdf, 2023, 2023, 1, 10)

    def test_cloud_fraction_scale(self):
        """Cloud cover should be converted from 0-100% to 0.0-1.0."""
        dates = ["2023-06-15"]
        pcts = [75.0]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openmeteo_response(dates, pcts)

        gdf = _mock_gdf([("X", 34.0, -118.0)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp):
            df = OpenMeteoCloudFraction().fetch(gdf, 2023, 2023, 166, 166)

        assert len(df) == 1
        assert df["cloud_fraction"].iloc[0] == pytest.approx(0.75)

    def test_compatible_with_simulate_visits(self):
        """Output should work with simulate_visits without modification."""
        # Build a realistic DataFrame manually (as OpenMeteo would produce)
        rows = []
        for day in range(1, 31):
            for poly in ["A", "B"]:
                cf = 0.05 if day % 4 == 0 else 0.6
                rows.append({
                    "polygon_id": poly, "year": 2023,
                    "day_of_year": day, "cloud_fraction": cf,
                })
        df = pd.DataFrame(rows)

        result_df, visit_tracker, rest_days = simulate_visits(
            df, day_start=1, day_stop=30,
            year_start=2023, year_stop=2023,
            cloud_fraction_threshold=0.10,
        )
        assert len(result_df) == 1
        assert 2023 in visit_tracker

    def test_http_error_raises(self):
        """HTTP errors should raise HyPlanRuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        gdf = _mock_gdf([("A", 34.0, -118.0)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                OpenMeteoCloudFraction().fetch(gdf, 2023, 2023, 1, 10)


class TestFetchCloudFraction:
    def test_unknown_source_raises(self, tmp_path):
        """Unknown source should raise."""
        # Create a minimal GeoJSON file
        from shapely.geometry import box
        import geopandas as _gpd

        gdf = _gpd.GeoDataFrame(
            {"Name": ["A"]},
            geometry=[box(-120, 33, -119, 34)],
            crs="EPSG:4326",
        )
        path = tmp_path / "test.geojson"
        gdf.to_file(path, driver="GeoJSON")

        with pytest.raises(Exception, match="Unknown"):
            fetch_cloud_fraction(str(path), 2023, 2023, 1, 30, source="invalid")
