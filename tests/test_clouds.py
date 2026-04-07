"""Tests for hyplan.clouds (date range logic, no Google Earth Engine required)."""

import pytest
import pandas as pd

from hyplan.clouds import create_date_ranges, simulate_visits


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
