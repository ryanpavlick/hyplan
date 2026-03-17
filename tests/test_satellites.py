"""Tests for hyplan.satellites (registry and helper functions, no network)."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hyplan.satellites import (
    SatelliteInfo,
    SATELLITE_REGISTRY,
    get_satellite,
    _compute_headings,
    _segment_passes,
    _merge_time_windows,
    _empty_overpass_gdf,
)


class TestSatelliteRegistry:
    def test_registry_populated(self):
        assert len(SATELLITE_REGISTRY) > 0
        assert "PACE" in SATELLITE_REGISTRY
        assert "Landsat-8" in SATELLITE_REGISTRY
        assert "Sentinel-2A" in SATELLITE_REGISTRY

    def test_satellite_info_fields(self):
        pace = SATELLITE_REGISTRY["PACE"]
        assert isinstance(pace, SatelliteInfo)
        assert pace.name == "PACE"
        assert pace.norad_id == 58927
        assert pace.swath_width_km > 0
        assert pace.max_sza > 0

    def test_all_entries_have_norad_ids(self):
        for name, sat in SATELLITE_REGISTRY.items():
            assert sat.norad_id > 0, f"{name} has invalid NORAD ID"
            assert sat.swath_width_km > 0, f"{name} has invalid swath width"


class TestGetSatellite:
    def test_lookup_by_name(self):
        sat = get_satellite("PACE")
        assert sat.name == "PACE"

    def test_lookup_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite"):
            get_satellite("NonexistentSat")

    def test_all_registry_entries_retrievable(self):
        for name in SATELLITE_REGISTRY:
            sat = get_satellite(name)
            assert sat.name == SATELLITE_REGISTRY[name].name


class TestComputeHeadings:
    def test_northward_track(self):
        lats = np.array([0.0, 1.0, 2.0])
        lons = np.array([0.0, 0.0, 0.0])
        headings = _compute_headings(lats, lons)
        assert len(headings) == 3
        # Northward heading should be ~0°
        assert headings[0] == pytest.approx(0.0, abs=1.0)
        # Last point reuses second-to-last
        assert headings[2] == pytest.approx(headings[1], abs=0.01)

    def test_eastward_track(self):
        lats = np.array([0.0, 0.0, 0.0])
        lons = np.array([0.0, 1.0, 2.0])
        headings = _compute_headings(lats, lons)
        # Eastward heading should be ~90°
        assert headings[0] == pytest.approx(90.0, abs=1.0)

    def test_single_point(self):
        headings = _compute_headings(np.array([0.0]), np.array([0.0]))
        assert len(headings) == 1


class TestSegmentPasses:
    def test_continuous_track(self):
        """A continuous track should be one pass."""
        n = 10
        lats = np.linspace(0, 10, n)
        timestamps = np.array([
            datetime(2025, 1, 1, 0, 0) + timedelta(seconds=i * 30)
            for i in range(n)
        ])
        passes = _segment_passes(lats, timestamps, time_step_s=30.0)
        assert len(passes) == 1
        assert passes[0] == (0, n)

    def test_gap_creates_two_passes(self):
        """A large time gap should split into two passes."""
        n = 10
        timestamps = []
        for i in range(5):
            timestamps.append(datetime(2025, 1, 1, 0, 0) + timedelta(seconds=i * 30))
        for i in range(5):
            timestamps.append(datetime(2025, 1, 1, 1, 0) + timedelta(seconds=i * 30))  # 1 hour gap
        lats = np.linspace(0, 10, n)
        passes = _segment_passes(lats, np.array(timestamps), time_step_s=30.0)
        assert len(passes) == 2

    def test_single_point(self):
        """A single point should return one pass with start=0, end=1."""
        passes = _segment_passes(
            np.array([0.0]),
            np.array([datetime(2025, 1, 1)]),
            time_step_s=30.0,
        )
        assert len(passes) == 1
        assert passes[0] == (0, 1)


class TestMergeTimeWindows:
    def test_no_timestamps(self):
        assert _merge_time_windows([], margin_s=60) == []

    def test_single_timestamp(self):
        ts = np.array([np.datetime64("2025-01-01T00:00:00")])
        windows = _merge_time_windows(ts, margin_s=60)
        assert len(windows) == 1

    def test_close_timestamps_merge(self):
        base = np.datetime64("2025-01-01T00:00:00")
        ts = np.array([base, base + np.timedelta64(30, "s"), base + np.timedelta64(60, "s")])
        windows = _merge_time_windows(ts, margin_s=120)
        assert len(windows) == 1

    def test_distant_timestamps_separate(self):
        base = np.datetime64("2025-01-01T00:00:00")
        ts = np.array([base, base + np.timedelta64(1, "h")])
        windows = _merge_time_windows(ts, margin_s=60)
        assert len(windows) == 2


class TestEmptyOverpassGdf:
    def test_returns_empty_gdf(self):
        gdf = _empty_overpass_gdf()
        assert len(gdf) == 0
        assert "satellite_name" in gdf.columns
        assert "geometry" in gdf.columns
        assert str(gdf.crs) == "EPSG:4326"
