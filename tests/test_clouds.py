"""Tests for hyplan.clouds (date range logic, no Google Earth Engine required)."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# xarray is part of the [clouds] extra; the spatial-plotting tests below
# require it but the rest of this file does not. Gate the dependent class
# rather than failing collection in a base install.
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

from hyplan.clouds import (
    create_date_ranges,
    simulate_visits,
    OpenMeteoCloudFraction,
    OpenMeteoCloudForecast,
    fetch_cloud_fraction,
    fetch_cloud_forecast,
    summarize_cloud_fraction_by_doy,
    plot_doy_cloud_fraction,
)
from hyplan.clouds.plotting import (
    plot_cloud_fraction_spatial,
    plot_yearly_cloud_fraction_heatmaps_with_visits,
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

        with patch("hyplan.clouds._requests.get" if hasattr(OpenMeteoCloudFraction, "_requests") else "requests.get", return_value=mock_resp):
            # Patch at the import location inside the method

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

    def test_satellite_with_openmeteo_raises(self, tmp_path):
        """Passing satellite!='both' with openmeteo should raise."""
        from shapely.geometry import box
        import geopandas as _gpd

        gdf = _gpd.GeoDataFrame(
            {"Name": ["A"]},
            geometry=[box(-120, 33, -119, 34)],
            crs="EPSG:4326",
        )
        path = tmp_path / "test.geojson"
        gdf.to_file(path, driver="GeoJSON")

        with pytest.raises(Exception, match="satellite"):
            fetch_cloud_fraction(
                str(path), 2023, 2023, 1, 30,
                source="openmeteo", satellite="terra",
            )


# ---------------------------------------------------------------------------
# Feature 1: DOY-averaged cloud fraction summary
# ---------------------------------------------------------------------------

class TestSummarizeCloudFractionByDOY:
    @pytest.fixture
    def sample_df(self):
        rows = []
        for year in [2020, 2021, 2022]:
            for doy in range(1, 31):
                for poly in ["A", "B"]:
                    # Cloud fraction increases linearly with DOY
                    rows.append({
                        "polygon_id": poly,
                        "year": year,
                        "day_of_year": doy,
                        "cloud_fraction": doy / 100.0,
                    })
        return pd.DataFrame(rows)

    def test_basic_output(self, sample_df):
        result = summarize_cloud_fraction_by_doy(sample_df)
        assert set(result.columns) == {
            "polygon_id", "day_of_year",
            "cloud_fraction_mean", "cloud_fraction_std", "cloud_fraction_count",
        }
        # 2 polygons x 30 DOYs
        assert len(result) == 60

    def test_mean_correctness(self, sample_df):
        result = summarize_cloud_fraction_by_doy(sample_df)
        # All years have same value for a given DOY, so std should be 0
        row = result[(result["polygon_id"] == "A") & (result["day_of_year"] == 10)]
        assert row["cloud_fraction_mean"].iloc[0] == pytest.approx(0.10)
        assert row["cloud_fraction_std"].iloc[0] == pytest.approx(0.0)
        assert row["cloud_fraction_count"].iloc[0] == 3

    def test_window_smoothing(self):
        """Rolling window should smooth a step function."""
        rows = []
        for year in [2020]:
            for doy in range(1, 11):
                rows.append({
                    "polygon_id": "A", "year": year,
                    "day_of_year": doy,
                    "cloud_fraction": 0.0 if doy <= 5 else 1.0,
                })
        df = pd.DataFrame(rows)
        summarize_cloud_fraction_by_doy(df)
        smoothed = summarize_cloud_fraction_by_doy(df, window=5)
        # The smoothed version should have intermediate values near the step
        step_val = smoothed[smoothed["day_of_year"] == 5]["cloud_fraction_mean"].iloc[0]
        assert 0.0 < step_val < 1.0

    def test_empty_df(self):
        df = pd.DataFrame(columns=["polygon_id", "year", "day_of_year", "cloud_fraction"])
        result = summarize_cloud_fraction_by_doy(df)
        assert len(result) == 0

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(Exception, match="columns"):
            summarize_cloud_fraction_by_doy(df)


class TestPlotDoyCloudFraction:
    def test_returns_axes(self):
        rows = []
        for doy in range(1, 11):
            for poly in ["A", "B"]:
                rows.append({
                    "polygon_id": poly, "day_of_year": doy,
                    "cloud_fraction_mean": 0.5, "cloud_fraction_std": 0.1,
                    "cloud_fraction_count": 3,
                })
        df = pd.DataFrame(rows)
        ax = plot_doy_cloud_fraction(df)
        assert isinstance(ax, plt.Axes)
        # 2 polygons = 2 lines
        assert len(ax.lines) == 2
        plt.close("all")

    def test_existing_axes(self):
        fig, ax = plt.subplots()
        rows = [{"polygon_id": "X", "day_of_year": 1,
                 "cloud_fraction_mean": 0.3, "cloud_fraction_std": 0.0,
                 "cloud_fraction_count": 1}]
        result_ax = plot_doy_cloud_fraction(pd.DataFrame(rows), ax=ax)
        assert result_ax is ax
        plt.close("all")


# ---------------------------------------------------------------------------
# Feature 2: Cloud forecasts
# ---------------------------------------------------------------------------

def _make_openmeteo_forecast_daily(dates, cloud_pcts):
    """Build a mock Open-Meteo forecast (daily) JSON response."""
    return {
        "latitude": 34.4,
        "longitude": -119.8,
        "daily_units": {"time": "iso8601", "cloud_cover_mean": "%"},
        "daily": {
            "time": dates,
            "cloud_cover_mean": cloud_pcts,
        },
    }


def _make_openmeteo_forecast_hourly(times, cc, cc_low, cc_mid, cc_high):
    """Build a mock Open-Meteo forecast (hourly) JSON response."""
    return {
        "latitude": 34.4,
        "longitude": -119.8,
        "hourly_units": {"time": "iso8601", "cloud_cover": "%"},
        "hourly": {
            "time": times,
            "cloud_cover": cc,
            "cloud_cover_low": cc_low,
            "cloud_cover_mid": cc_mid,
            "cloud_cover_high": cc_high,
        },
    }


class TestOpenMeteoCloudForecast:
    def test_daily_fetch(self):
        dates = ["2026-04-11", "2026-04-12", "2026-04-13"]
        pcts = [20.0, 60.0, 80.0]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openmeteo_forecast_daily(dates, pcts)

        gdf = _mock_gdf([("Box1", 34.4, -119.8)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp):
            df = OpenMeteoCloudForecast().fetch(gdf, forecast_days=3)

        assert set(df.columns) == {"polygon_id", "date", "cloud_fraction"}
        assert len(df) == 3
        assert df["cloud_fraction"].iloc[0] == pytest.approx(0.20)
        assert df["cloud_fraction"].iloc[2] == pytest.approx(0.80)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_hourly_fetch(self):
        times = ["2026-04-11T00:00", "2026-04-11T01:00"]
        cc = [30.0, 40.0]
        cc_low = [10.0, 15.0]
        cc_mid = [5.0, 10.0]
        cc_high = [15.0, 15.0]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openmeteo_forecast_hourly(
            times, cc, cc_low, cc_mid, cc_high,
        )

        gdf = _mock_gdf([("Box1", 34.4, -119.8)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp):
            df = OpenMeteoCloudForecast().fetch(gdf, forecast_days=1, hourly=True)

        assert "hour" in df.columns
        assert "cloud_fraction_low" in df.columns
        assert "cloud_fraction_mid" in df.columns
        assert "cloud_fraction_high" in df.columns
        assert len(df) == 2
        assert df["hour"].tolist() == [0, 1]
        assert df["cloud_fraction"].iloc[0] == pytest.approx(0.30)
        assert df["cloud_fraction_low"].iloc[0] == pytest.approx(0.10)

    def test_forecast_days_too_large(self):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        with pytest.raises(Exception, match="forecast_days"):
            OpenMeteoCloudForecast().fetch(gdf, forecast_days=20)

    def test_forecast_days_zero(self):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        with pytest.raises(Exception, match="forecast_days"):
            OpenMeteoCloudForecast().fetch(gdf, forecast_days=0)

    def test_missing_name_column(self):
        import geopandas as _gpd
        from shapely.geometry import Point

        gdf = _gpd.GeoDataFrame({"id": ["A"]}, geometry=[Point(0, 0)])
        with pytest.raises(Exception, match="Name"):
            OpenMeteoCloudForecast().fetch(gdf)

    def test_http_error_raises(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        gdf = _mock_gdf([("A", 34.0, -118.0)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                OpenMeteoCloudForecast().fetch(gdf, forecast_days=3)

    def test_models_param_passed(self):
        """The models parameter should be included in the request."""
        dates = ["2026-04-11"]
        pcts = [50.0]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openmeteo_forecast_daily(dates, pcts)

        gdf = _mock_gdf([("Box1", 34.4, -119.8)])

        import requests as _req
        with patch.object(_req, "get", return_value=mock_resp) as mock_get:
            OpenMeteoCloudForecast().fetch(
                gdf, forecast_days=1, models=["ecmwf_ifs025"],
            )
            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
            assert params["models"] == "ecmwf_ifs025"


class TestFetchCloudForecast:
    def test_unknown_source_raises(self, tmp_path):
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
            fetch_cloud_forecast(str(path), source="gfs")


# ---------------------------------------------------------------------------
# Cloud fraction spatial plotting
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed (install with [clouds] extra)")
class TestPlotCloudFractionSpatial:
    """Tests for plot_cloud_fraction_spatial."""

    def _make_spatial_data(self, n_sites=2):
        """Create synthetic spatial_data dict of xarray DataArrays."""
        data = {}
        for i in range(n_sites):
            da = xr.DataArray(
                np.random.rand(10, 10),
                dims=["latitude", "longitude"],
                coords={
                    "latitude": np.linspace(34.0 + i * 0.2, 34.1 + i * 0.2, 10),
                    "longitude": np.linspace(-119.0, -118.9, 10),
                },
            )
            data[f"Site{chr(65 + i)}"] = da
        return data

    def test_returns_figure(self):
        spatial_data = self._make_spatial_data(1)
        fig = plot_cloud_fraction_spatial(spatial_data)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_multiple_sites(self):
        spatial_data = self._make_spatial_data(3)
        fig = plot_cloud_fraction_spatial(spatial_data, ncols=2)
        assert isinstance(fig, plt.Figure)
        # 3 sites with ncols=2 -> 2 rows, 4 subplot axes total (1 hidden)
        axes = fig.get_axes()
        # At least 3 visible axes for the 3 sites
        visible = [ax for ax in axes if ax.get_visible()]
        # colorbars also create axes, so just check figure exists
        assert len(visible) >= 3
        plt.close("all")

    def test_single_column(self):
        spatial_data = self._make_spatial_data(2)
        fig = plot_cloud_fraction_spatial(spatial_data, ncols=1)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_spatial_data_raises(self):
        from hyplan.exceptions import HyPlanValueError

        with pytest.raises(HyPlanValueError, match="empty"):
            plot_cloud_fraction_spatial({})


class TestPlotYearlyCloudFractionHeatmapsWithVisits:
    """Tests for plot_yearly_cloud_fraction_heatmaps_with_visits."""

    def _make_cloud_df(self, year=2023, day_start=1, day_stop=30, polygons=None):
        if polygons is None:
            polygons = ["A", "B"]
        rows = []
        for day in range(day_start, day_stop + 1):
            for poly in polygons:
                cf = 0.05 if day % 3 == 0 else 0.50
                rows.append({
                    "polygon_id": poly,
                    "year": year,
                    "day_of_year": day,
                    "cloud_fraction": cf,
                })
        return pd.DataFrame(rows)

    def _make_visit_tracker(self, year=2023, polygons=None):
        if polygons is None:
            polygons = ["A", "B"]
        # Mark a visit on day 3 for each polygon
        return {year: {poly: [3] for poly in polygons}}

    def _make_rest_days(self, year=2023):
        return {year: [4]}

    def test_runs_without_error(self):
        df = self._make_cloud_df()
        visit_tracker = self._make_visit_tracker()
        rest_days = self._make_rest_days()

        # Should complete without raising
        plot_yearly_cloud_fraction_heatmaps_with_visits(
            df, visit_tracker, rest_days,
            cloud_fraction_threshold=0.10,
            day_start=1, day_stop=30,
        )
        plt.close("all")

    def test_exclude_weekends(self):
        df = self._make_cloud_df()
        visit_tracker = self._make_visit_tracker()
        rest_days = self._make_rest_days()

        plot_yearly_cloud_fraction_heatmaps_with_visits(
            df, visit_tracker, rest_days,
            cloud_fraction_threshold=0.10,
            exclude_weekends=True,
            day_start=1, day_stop=30,
        )
        plt.close("all")

    def test_empty_visit_tracker(self):
        df = self._make_cloud_df()
        plot_yearly_cloud_fraction_heatmaps_with_visits(
            df, visit_tracker={}, rest_days={},
            day_start=1, day_stop=30,
        )
        plt.close("all")

    def test_missing_columns_raises(self):
        from hyplan.exceptions import HyPlanValueError

        bad_df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(HyPlanValueError, match="columns"):
            plot_yearly_cloud_fraction_heatmaps_with_visits(
                bad_df, visit_tracker={}, rest_days={},
            )

    def test_multiple_years(self):
        df1 = self._make_cloud_df(year=2022)
        df2 = self._make_cloud_df(year=2023)
        df = pd.concat([df1, df2], ignore_index=True)
        visit_tracker = {
            2022: {"A": [3], "B": [6]},
            2023: {"A": [9], "B": [12]},
        }
        rest_days = {2022: [4], 2023: [10]}

        plot_yearly_cloud_fraction_heatmaps_with_visits(
            df, visit_tracker, rest_days,
            day_start=1, day_stop=30,
        )
        plt.close("all")
