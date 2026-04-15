"""Tests for hyplan.phenology (no network calls, all earthaccess mocked)."""

import datetime as dt
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from hyplan.phenology import (
    extract_phenology_stages,
    fetch_phenology,
    plot_cloud_phenology_combined,
    plot_phenology_calendar,
    plot_seasonal_profile,
    plot_year_over_year_heatmap,
    summarize_phenology_by_doy,
)
from hyplan.phenology._qa import (
    apply_lai_qa_mask,
    apply_phenology_qa_mask,
    apply_vi_qa_mask,
    convert_mcd12q2_dates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_gdf(names_coords):
    """Build a GeoDataFrame with Name and polygon geometry."""
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for name, lat, lon in names_coords:
        rows.append({
            "Name": name,
            "geometry": box(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _make_vi_df(
    polygons=("A", "B"),
    years=(2020, 2021),
    doy_range=(1, 366, 16),
):
    """Build a synthetic vegetation index DataFrame."""
    rows = []
    for year in years:
        for doy in range(doy_range[0], doy_range[1], doy_range[2]):
            for poly in polygons:
                # Sinusoidal pattern for realistic phenology
                value = 0.3 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
                rows.append({
                    "polygon_id": poly,
                    "date": dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1),
                    "year": year,
                    "day_of_year": doy,
                    "value": value,
                })
    return pd.DataFrame(rows)


def _make_phenology_df(polygons=("A", "B"), years=(2019, 2020, 2021)):
    """Build a synthetic MCD12Q2 phenology DataFrame."""
    rows = []
    for year in years:
        for poly in polygons:
            rows.append({
                "polygon_id": poly,
                "year": year,
                "greenup_doy": 80.0 + np.random.uniform(-5, 5),
                "midgreenup_doy": 100.0 + np.random.uniform(-5, 5),
                "peak_doy": 160.0 + np.random.uniform(-5, 5),
                "maturity_doy": 180.0 + np.random.uniform(-5, 5),
                "midgreendown_doy": 250.0 + np.random.uniform(-5, 5),
                "senescence_doy": 280.0 + np.random.uniform(-5, 5),
                "dormancy_doy": 320.0 + np.random.uniform(-5, 5),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QA filtering tests
# ---------------------------------------------------------------------------

class TestApplyViQAMask:
    def test_good_pixels_pass(self):
        data = np.array([5000, 6000, 7000], dtype=np.int16)
        qa = np.array([0, 0, 1], dtype=np.int8)
        result = apply_vi_qa_mask(data, qa, max_reliability=1)
        assert result.count() == 3

    def test_cloudy_pixels_masked(self):
        data = np.array([5000, 6000, 7000], dtype=np.int16)
        qa = np.array([0, 3, 2], dtype=np.int8)
        result = apply_vi_qa_mask(data, qa, max_reliability=1)
        assert result.count() == 1
        assert result[0] == 5000

    def test_fill_pixels_masked(self):
        data = np.array([5000, 6000], dtype=np.int16)
        qa = np.array([0, -1], dtype=np.int8)
        result = apply_vi_qa_mask(data, qa)
        assert result.count() == 1

    def test_strict_mode(self):
        data = np.array([5000, 6000], dtype=np.int16)
        qa = np.array([0, 1], dtype=np.int8)
        result = apply_vi_qa_mask(data, qa, max_reliability=0)
        assert result.count() == 1


class TestApplyLaiQAMask:
    def test_clear_pass(self):
        data = np.array([50, 60, 70], dtype=np.uint8)
        qa = np.array([0b00000000, 0b00000000, 0b00000000], dtype=np.uint8)
        result = apply_lai_qa_mask(data, qa)
        assert result.count() == 3

    def test_cloud_masked(self):
        data = np.array([50, 60], dtype=np.uint8)
        # bit 0 = 0 (good), but bits 5-7 = 001 (cloudy)
        qa = np.array([0b00000000, 0b00100000], dtype=np.uint8)
        result = apply_lai_qa_mask(data, qa)
        assert result.count() == 1

    def test_bad_algo_masked(self):
        data = np.array([50, 60], dtype=np.uint8)
        qa = np.array([0b00000000, 0b00000001], dtype=np.uint8)
        result = apply_lai_qa_mask(data, qa)
        assert result.count() == 1

    def test_fill_value_masked(self):
        data = np.array([50, 255], dtype=np.uint8)
        qa = np.array([0b00000000, 0b00000000], dtype=np.uint8)
        result = apply_lai_qa_mask(data, qa)
        assert result.count() == 1


class TestApplyPhenologyQAMask:
    def test_best_good_pass(self):
        data_dict = {"greenup": np.array([18000, 18100, 18200])}
        qa = np.array([0b00, 0b01, 0b00], dtype=np.uint8)
        result = apply_phenology_qa_mask(data_dict, qa, max_quality=1)
        assert result["greenup"].count() == 3

    def test_poor_masked(self):
        data_dict = {"greenup": np.array([18000, 18100])}
        qa = np.array([0b00, 0b11], dtype=np.uint8)
        result = apply_phenology_qa_mask(data_dict, qa, max_quality=1)
        assert result["greenup"].count() == 1


class TestConvertMCD12Q2Dates:
    def test_known_date(self):
        # 2020-03-21 = day 81 of 2020
        # Days from 1970-01-01 to 2020-03-21 = 18342
        epoch = dt.date(1970, 1, 1)
        target = dt.date(2020, 3, 21)
        days_since = (target - epoch).days

        result = convert_mcd12q2_dates(np.array([days_since]))
        assert result[0] == pytest.approx(81.0)

    def test_fill_value_becomes_nan(self):
        result = convert_mcd12q2_dates(np.array([32767]))
        assert np.isnan(result[0])

    def test_zero_becomes_nan(self):
        result = convert_mcd12q2_dates(np.array([0]))
        assert np.isnan(result[0])

    def test_mixed(self):
        epoch = dt.date(1970, 1, 1)
        day100 = (dt.date(2020, 1, 1) + dt.timedelta(days=99) - epoch).days
        result = convert_mcd12q2_dates(np.array([day100, 32767, 0]))
        assert result[0] == pytest.approx(100.0)
        assert np.isnan(result[1])
        assert np.isnan(result[2])


# ---------------------------------------------------------------------------
# fetch_phenology validation tests
# ---------------------------------------------------------------------------

class TestFetchPhenologyValidation:
    def test_unknown_product_raises(self, tmp_path):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        f = tmp_path / "poly.geojson"
        gdf.to_file(f, driver="GeoJSON")

        with pytest.raises(Exception, match="Unknown phenology product"):
            fetch_phenology(str(f), product="bogus")

    def test_missing_name_column_raises(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            [{"label": "A", "geometry": box(-118.1, 33.9, -117.9, 34.1)}],
            crs="EPSG:4326",
        )
        f = tmp_path / "poly.geojson"
        gdf.to_file(f, driver="GeoJSON")

        with pytest.raises(Exception, match="Name"):
            fetch_phenology(str(f), product="ndvi")

    def test_invalid_satellite_raises(self, tmp_path):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        f = tmp_path / "poly.geojson"
        gdf.to_file(f, driver="GeoJSON")

        with pytest.raises(Exception, match="Unknown satellite"):
            fetch_phenology(str(f), satellite="sentinel")

    def test_satellite_on_lai_raises(self, tmp_path):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        f = tmp_path / "poly.geojson"
        gdf.to_file(f, driver="GeoJSON")

        with pytest.raises(Exception, match="only supported for ndvi/evi"):
            fetch_phenology(str(f), product="lai", satellite="aqua")

    def test_invalid_spatial_mode_raises(self, tmp_path):
        gdf = _mock_gdf([("A", 34.0, -118.0)])
        f = tmp_path / "poly.geojson"
        gdf.to_file(f, driver="GeoJSON")

        with pytest.raises(Exception, match="Unknown spatial_mode"):
            fetch_phenology(str(f), spatial_mode="detailed")


# ---------------------------------------------------------------------------
# summarize_phenology_by_doy tests
# ---------------------------------------------------------------------------

class TestSummarizePhenologyByDOY:
    @pytest.fixture
    def sample_df(self):
        rows = []
        for year in [2020, 2021, 2022]:
            for doy in range(1, 31):
                for poly in ["A", "B"]:
                    rows.append({
                        "polygon_id": poly,
                        "year": year,
                        "day_of_year": doy,
                        "value": doy / 100.0,
                    })
        return pd.DataFrame(rows)

    def test_basic_output(self, sample_df):
        result = summarize_phenology_by_doy(sample_df)
        assert set(result.columns) == {
            "polygon_id", "day_of_year",
            "value_mean", "value_std", "value_count",
        }
        # 2 polygons x 30 DOYs
        assert len(result) == 60

    def test_mean_correctness(self, sample_df):
        result = summarize_phenology_by_doy(sample_df)
        row = result[(result["polygon_id"] == "A") & (result["day_of_year"] == 10)]
        assert row["value_mean"].iloc[0] == pytest.approx(0.10)
        assert row["value_std"].iloc[0] == pytest.approx(0.0)
        assert row["value_count"].iloc[0] == 3

    def test_window_smoothing(self):
        rows = []
        for doy in range(1, 11):
            rows.append({
                "polygon_id": "A", "year": 2020,
                "day_of_year": doy,
                "value": 0.0 if doy <= 5 else 1.0,
            })
        df = pd.DataFrame(rows)
        smoothed = summarize_phenology_by_doy(df, window=5)
        step_val = smoothed[smoothed["day_of_year"] == 5]["value_mean"].iloc[0]
        assert 0.0 < step_val < 1.0

    def test_empty_df(self):
        df = pd.DataFrame(columns=["polygon_id", "year", "day_of_year", "value"])
        result = summarize_phenology_by_doy(df)
        assert len(result) == 0

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(Exception, match="columns"):
            summarize_phenology_by_doy(df)


# ---------------------------------------------------------------------------
# extract_phenology_stages tests
# ---------------------------------------------------------------------------

class TestExtractPhenologyStages:
    def test_basic_output(self):
        df = _make_phenology_df(polygons=("A",), years=(2020, 2021))
        result = extract_phenology_stages(df)

        assert "polygon_id" in result.columns
        assert "greenup_doy_mean" in result.columns
        assert "greenup_doy_std" in result.columns
        assert "dormancy_doy_mean" in result.columns
        assert len(result) == 1

    def test_multiple_polygons(self):
        df = _make_phenology_df(polygons=("A", "B"), years=(2020, 2021))
        result = extract_phenology_stages(df)
        assert len(result) == 2

    def test_nan_handling(self):
        rows = [
            {"polygon_id": "A", "year": 2020,
             "greenup_doy": 80, "midgreenup_doy": np.nan,
             "peak_doy": 160, "maturity_doy": 180,
             "midgreendown_doy": 250, "senescence_doy": 280,
             "dormancy_doy": 320},
        ]
        df = pd.DataFrame(rows)
        result = extract_phenology_stages(df)
        assert np.isnan(result["midgreenup_doy_mean"].iloc[0])

    def test_empty_df(self):
        from hyplan.phenology.analysis import _STAGE_COLUMNS
        cols = ["polygon_id", "year"] + _STAGE_COLUMNS
        df = pd.DataFrame(columns=cols)
        result = extract_phenology_stages(df)
        assert len(result) == 0

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"polygon_id": ["A"], "year": [2020]})
        with pytest.raises(Exception, match="columns"):
            extract_phenology_stages(df)


# ---------------------------------------------------------------------------
# Plotting tests
# ---------------------------------------------------------------------------

class TestPlotSeasonalProfile:
    def test_returns_axes(self):
        vi_df = _make_vi_df(polygons=("A",), years=(2020,))
        summary = summarize_phenology_by_doy(vi_df)
        ax = plot_seasonal_profile(summary)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_existing_axes(self):
        vi_df = _make_vi_df(polygons=("A",), years=(2020,))
        summary = summarize_phenology_by_doy(vi_df)
        fig, ax = plt.subplots()
        result = plot_seasonal_profile(summary, ax=ax)
        assert result is ax
        plt.close("all")

    def test_multiple_polygons(self):
        vi_df = _make_vi_df(polygons=("A", "B"), years=(2020,))
        summary = summarize_phenology_by_doy(vi_df)
        ax = plot_seasonal_profile(summary)
        # Should have 2 lines (one per polygon)
        assert len(ax.lines) >= 2
        plt.close("all")

    def test_no_std(self):
        vi_df = _make_vi_df(polygons=("A",), years=(2020,))
        summary = summarize_phenology_by_doy(vi_df)
        ax = plot_seasonal_profile(summary, show_std=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestPlotPhenologyCalendar:
    def test_returns_axes(self):
        df = _make_phenology_df()
        stages = extract_phenology_stages(df)
        ax = plot_phenology_calendar(stages)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_multiple_polygons(self):
        df = _make_phenology_df(polygons=("A", "B", "C"))
        stages = extract_phenology_stages(df)
        ax = plot_phenology_calendar(stages)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestPlotYearOverYearHeatmap:
    def test_returns_axes(self):
        vi_df = _make_vi_df(polygons=("A",), years=(2018, 2019, 2020))
        ax = plot_year_over_year_heatmap(vi_df)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_multiple_polygons_requires_id(self):
        vi_df = _make_vi_df(polygons=("A", "B"), years=(2020,))
        with pytest.raises(Exception, match="polygon_id"):
            plot_year_over_year_heatmap(vi_df)

    def test_explicit_polygon_id(self):
        vi_df = _make_vi_df(polygons=("A", "B"), years=(2020,))
        ax = plot_year_over_year_heatmap(vi_df, polygon_id="B")
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestPlotCloudPhenologyCombined:
    @pytest.fixture
    def cloud_summary(self):
        rows = []
        for doy in range(1, 366, 16):
            rows.append({
                "polygon_id": "A",
                "day_of_year": doy,
                "cloud_fraction_mean": 0.3 + 0.2 * np.sin(2 * np.pi * doy / 365),
                "cloud_fraction_std": 0.05,
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def pheno_summary(self):
        vi_df = _make_vi_df(polygons=("A",), years=(2020,))
        return summarize_phenology_by_doy(vi_df)

    def test_overlay_returns_axes(self, cloud_summary, pheno_summary):
        ax = plot_cloud_phenology_combined(
            cloud_summary, pheno_summary, layout="overlay",
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_side_by_side_returns_figure(self, cloud_summary, pheno_summary):
        fig = plot_cloud_phenology_combined(
            cloud_summary, pheno_summary, layout="side_by_side",
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_invalid_layout_raises(self, cloud_summary, pheno_summary):
        with pytest.raises(Exception, match="Unknown layout"):
            plot_cloud_phenology_combined(
                cloud_summary, pheno_summary, layout="stacked",
            )
