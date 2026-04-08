"""Tests for hyplan.sun."""

import pytest
import pandas as pd
from datetime import datetime
from hyplan.sun import solar_threshold_times, solar_azimuth, solar_position_increments


class TestSolarThresholdTimes:
    def test_single_threshold(self):
        df = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-06-21", end_date="2025-06-21",
            thresholds=[30],
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Rise_30" in df.columns
        assert "Set_30" in df.columns
        # Sun should rise above 30° at LA summer solstice
        assert df["Rise_30"].iloc[0] is not None

    def test_two_thresholds(self):
        df = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-06-21", end_date="2025-06-21",
            thresholds=[20, 50],
        )
        assert "Rise_20" in df.columns
        assert "Rise_50" in df.columns
        assert "Set_50" in df.columns
        assert "Set_20" in df.columns

    def test_multi_day(self):
        df = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-06-20", end_date="2025-06-22",
            thresholds=[30],
        )
        assert len(df) == 3

    def test_invalid_threshold_count(self):
        with pytest.raises(ValueError):
            solar_threshold_times(
                latitude=34.05, longitude=-118.25,
                start_date="2025-06-21", end_date="2025-06-21",
                thresholds=[10, 20, 30],
            )

    def test_timezone_offset(self):
        df = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-06-21", end_date="2025-06-21",
            thresholds=[30],
            timezone_offset=-8,
        )
        assert len(df) == 1
        assert df["Rise_30"].iloc[0] is not None

    def test_iana_timezone_matches_fixed_offset_outside_dst(self):
        """In January, America/Los_Angeles is fixed at UTC-8 — so the IANA
        zone should yield the same wall-clock answer as ``timezone_offset=-8``."""
        df_iana = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-01-15", end_date="2025-01-15",
            thresholds=[30],
            timezone="America/Los_Angeles",
        )
        df_fixed = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-01-15", end_date="2025-01-15",
            thresholds=[30],
            timezone_offset=-8,
        )
        assert df_iana["Rise_30"].iloc[0] == df_fixed["Rise_30"].iloc[0]
        assert df_iana["Set_30"].iloc[0] == df_fixed["Set_30"].iloc[0]

    def test_iana_timezone_handles_dst(self):
        """Across the spring-forward boundary the IANA zone must shift from
        UTC-8 to UTC-7. A fixed ``timezone_offset=-8`` would be wrong by an
        hour after the transition."""
        df = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-03-08", end_date="2025-03-10",
            thresholds=[30],
            timezone="America/Los_Angeles",
        )
        # March 8 = PST (UTC-8), March 10 = PDT (UTC-7) — wall-clock times
        # for the same solar event should jump roughly an hour later.
        rise_pst = pd.Timestamp(df["Rise_30"].iloc[0])  # 2025-03-08
        rise_pdt = pd.Timestamp(df["Rise_30"].iloc[2])  # 2025-03-10
        delta_minutes = (rise_pdt - rise_pst).total_seconds() / 60.0
        assert 50 < delta_minutes < 70  # ~60 min jump from DST

    def test_timezone_takes_precedence_over_offset(self):
        """When both ``timezone`` and ``timezone_offset`` are passed, the
        IANA zone wins and the offset is ignored."""
        df_both = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-01-15", end_date="2025-01-15",
            thresholds=[30],
            timezone_offset=5,  # nonsense value, must be ignored
            timezone="America/Los_Angeles",
        )
        df_iana_only = solar_threshold_times(
            latitude=34.05, longitude=-118.25,
            start_date="2025-01-15", end_date="2025-01-15",
            thresholds=[30],
            timezone="America/Los_Angeles",
        )
        assert df_both["Rise_30"].iloc[0] == df_iana_only["Rise_30"].iloc[0]


class TestSolarAzimuth:
    def test_returns_float(self):
        dt = datetime(2025, 6, 21, 18, 0)
        az = solar_azimuth(34.05, -118.25, dt)
        assert isinstance(az, float)
        assert 0 <= az <= 360

    def test_noon_roughly_south(self):
        # At solar noon in northern hemisphere, sun should be roughly south (~180°)
        dt = datetime(2025, 6, 21, 20, 0)  # ~noon PDT in UTC
        az = solar_azimuth(34.05, -118.25, dt)
        assert 90 < az < 270


class TestSolarPositionIncrements:
    def test_returns_dataframe(self):
        df = solar_position_increments(
            latitude=34.05, longitude=-118.25,
            date="2025-06-21", min_elevation=30,
        )
        assert isinstance(df, pd.DataFrame)
        assert "Azimuth" in df.columns
        assert "Elevation" in df.columns
        assert len(df) > 0

    def test_all_above_threshold(self):
        df = solar_position_increments(
            latitude=34.05, longitude=-118.25,
            date="2025-06-21", min_elevation=30,
        )
        # All returned elevations should exceed the threshold
        assert (df["Elevation"] > 30).all()

    def test_string_and_date_input(self):
        from datetime import date
        df1 = solar_position_increments(34.05, -118.25, "2025-06-21", 20)
        df2 = solar_position_increments(34.05, -118.25, date(2025, 6, 21), 20)
        assert len(df1) == len(df2)

    def test_iana_timezone_matches_fixed_offset_outside_dst(self):
        """In January, America/Los_Angeles == UTC-8, so the two parameter
        forms should produce identical local-time labels."""
        df_iana = solar_position_increments(
            34.05, -118.25, "2025-01-15", min_elevation=20,
            timezone="America/Los_Angeles",
        )
        df_fixed = solar_position_increments(
            34.05, -118.25, "2025-01-15", min_elevation=20,
            timezone_offset=-8,
        )
        assert list(df_iana["Time"]) == list(df_fixed["Time"])
