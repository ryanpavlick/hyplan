"""Tests for hyplan.satellites (registry and helper functions, no network)."""

import os
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from unittest.mock import patch
from shapely.geometry import Point, Polygon, LineString

from hyplan.satellites import (
    SatelliteInfo,
    SATELLITE_REGISTRY,
    get_satellite,
    fetch_tle,
    clear_tle_cache,
    compute_ground_track,
    compute_swath_footprint,
    find_overpasses,
    find_all_overpasses,
    compute_overpass_overlap,
    overpasses_to_kml,
    _compute_headings,
    _segment_passes,
    _merge_time_windows,
    _empty_overpass_gdf,
    _get_tle_cache_dir,
    _tle_cache_path,
    _is_tle_stale,
)

# ---------------------------------------------------------------------------
# Synthetic TLE data (valid ISS-like orbital elements)
# ---------------------------------------------------------------------------
_ISS_TLE_NAME = "ISS (ZARYA)"
_ISS_TLE_LINE1 = "1 25544U 98067A   24170.54791667  .00016717  00000-0  10270-3 0  9993"
_ISS_TLE_LINE2 = "2 25544  51.6400 200.0000 0001234  90.0000 270.0000 15.49000000999990"
_ISS_TLE_TEXT = f"{_ISS_TLE_NAME}\n{_ISS_TLE_LINE1}\n{_ISS_TLE_LINE2}\n"


def _make_earth_satellite():
    """Create a Skyfield EarthSatellite from the synthetic ISS TLE."""
    from skyfield.api import load as sf_load, EarthSatellite
    ts = sf_load.timescale()
    return EarthSatellite(_ISS_TLE_LINE1, _ISS_TLE_LINE2, name=_ISS_TLE_NAME, ts=ts)


def _make_ground_track_gdf(n=20, start=None, step_s=30.0, sat_name="ISS (ZARYA)",
                           norad_id=25544, lat_start=30.0, lat_end=50.0,
                           lon_start=-100.0, lon_end=-80.0):
    """Build a synthetic ground track GeoDataFrame for testing."""
    if start is None:
        start = datetime(2025, 6, 15, 12, 0, 0)
    lats = np.linspace(lat_start, lat_end, n)
    lons = np.linspace(lon_start, lon_end, n)
    timestamps = np.array([start + timedelta(seconds=i * step_s) for i in range(n)])
    alt_km = np.full(n, 420.0)
    sza = np.full(n, 30.0)
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    return gpd.GeoDataFrame(
        {
            "satellite_name": sat_name,
            "norad_id": norad_id,
            "timestamp": timestamps,
            "latitude": lats,
            "longitude": lons,
            "altitude_km": alt_km,
            "solar_zenith": sza,
        },
        geometry=geometry,
        crs="EPSG:4326",
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


# ---------------------------------------------------------------------------
# TLE caching helpers
# ---------------------------------------------------------------------------

class TestTleCacheHelpers:
    def test_get_tle_cache_dir_creates_directory(self, tmp_path):
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            cache_dir = _get_tle_cache_dir()
            assert os.path.isdir(cache_dir)
            assert cache_dir == os.path.join(str(tmp_path), "tle_cache")

    def test_tle_cache_path(self, tmp_path):
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            path = _tle_cache_path(25544)
            assert path.endswith("25544.tle")

    def test_is_tle_stale_missing_file(self):
        assert _is_tle_stale("/nonexistent/path/file.tle") is True

    def test_is_tle_stale_fresh_file(self, tmp_path):
        f = tmp_path / "test.tle"
        f.write_text("data")
        assert _is_tle_stale(str(f), max_age_hours=1.0) is False

    def test_is_tle_stale_old_file(self, tmp_path):
        f = tmp_path / "test.tle"
        f.write_text("data")
        # Set mtime to 48 hours ago
        old_time = os.path.getmtime(str(f)) - 48 * 3600
        os.utime(str(f), (old_time, old_time))
        assert _is_tle_stale(str(f), max_age_hours=24.0) is True


# ---------------------------------------------------------------------------
# fetch_tle
# ---------------------------------------------------------------------------

class TestFetchTle:
    def test_fetch_tle_with_three_line_file(self, tmp_path):
        """Test TLE parsing with a 3-line file (name + line1 + line2)."""
        cache_dir = tmp_path / "tle_cache"
        cache_dir.mkdir()
        tle_file = cache_dir / "25544.tle"
        tle_file.write_text(_ISS_TLE_TEXT)

        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            sat = fetch_tle(iss_info, max_age_hours=999)
            assert sat is not None
            assert sat.name == _ISS_TLE_NAME

    def test_fetch_tle_with_two_line_file(self, tmp_path):
        """Test TLE parsing with a 2-line file (line1 + line2 only)."""
        cache_dir = tmp_path / "tle_cache"
        cache_dir.mkdir()
        tle_file = cache_dir / "25544.tle"
        tle_file.write_text(f"{_ISS_TLE_LINE1}\n{_ISS_TLE_LINE2}\n")

        # Add ISS to registry temporarily for this test
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            sat = fetch_tle(iss_info, max_age_hours=999)
            assert sat is not None

    def test_fetch_tle_downloads_when_stale(self, tmp_path):
        """Test that fetch_tle calls download_file when cache is stale."""
        cache_dir = tmp_path / "tle_cache"
        cache_dir.mkdir()

        def fake_download(filepath, url, replace=False):
            with open(filepath, "w") as f:
                f.write(_ISS_TLE_TEXT)

        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)

        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)), \
             patch("hyplan.satellites.download_file", side_effect=fake_download) as mock_dl:
            sat = fetch_tle(iss_info, max_age_hours=24.0)
            mock_dl.assert_called_once()
            assert sat is not None

    def test_fetch_tle_raises_on_insufficient_lines(self, tmp_path):
        """Test that a TLE file with <2 lines raises RuntimeError."""
        cache_dir = tmp_path / "tle_cache"
        cache_dir.mkdir()
        tle_file = cache_dir / "25544.tle"
        tle_file.write_text("only one line\n")

        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            with pytest.raises(RuntimeError, match="fewer than 2 lines"):
                fetch_tle(iss_info, max_age_hours=999)

    def test_fetch_tle_by_name(self, tmp_path):
        """Test that fetch_tle accepts a string name from the registry."""
        cache_dir = tmp_path / "tle_cache"
        cache_dir.mkdir()
        pace = SATELLITE_REGISTRY["PACE"]
        tle_file = cache_dir / f"{pace.norad_id}.tle"
        # Write a valid TLE using ISS-like orbital elements but with PACE's NORAD ID
        line1 = "1 58927U 24007A   24170.54791667  .00016717  00000-0  10270-3 0  9993"
        line2 = "2 58927  51.6400 200.0000 0001234  90.0000 270.0000 15.49000000999990"
        tle_file.write_text(f"PACE\n{line1}\n{line2}\n")

        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            sat = fetch_tle("PACE", max_age_hours=999)
            assert sat.name == "PACE"


# ---------------------------------------------------------------------------
# clear_tle_cache
# ---------------------------------------------------------------------------

class TestClearTleCache:
    def test_clear_nonexistent_cache(self, tmp_path):
        """Clearing when no cache dir exists should just log and return."""
        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            # tle_cache dir does not exist
            clear_tle_cache(confirm=False)
            # Should not raise

    def test_clear_existing_cache_no_confirm(self, tmp_path):
        """Clear cache without confirmation prompt."""
        tle_dir = tmp_path / "tle_cache"
        tle_dir.mkdir()
        (tle_dir / "25544.tle").write_text("data")
        assert tle_dir.exists()

        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)):
            clear_tle_cache(confirm=False)
            assert not tle_dir.exists()

    def test_clear_cache_confirm_yes(self, tmp_path):
        """Clear cache with confirmation when user says yes."""
        tle_dir = tmp_path / "tle_cache"
        tle_dir.mkdir()
        (tle_dir / "25544.tle").write_text("data")

        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)), \
             patch("builtins.input", return_value="yes"):
            clear_tle_cache(confirm=True)
            assert not tle_dir.exists()

    def test_clear_cache_confirm_no(self, tmp_path):
        """Cache should NOT be cleared when user says no."""
        tle_dir = tmp_path / "tle_cache"
        tle_dir.mkdir()
        (tle_dir / "25544.tle").write_text("data")

        with patch("hyplan.satellites.get_cache_root", return_value=str(tmp_path)), \
             patch("builtins.input", return_value="no"):
            clear_tle_cache(confirm=True)
            assert tle_dir.exists()


# ---------------------------------------------------------------------------
# compute_ground_track
# ---------------------------------------------------------------------------

class TestComputeGroundTrack:
    def test_compute_ground_track_basic(self):
        """Test ground track computation with a mocked fetch_tle."""
        sat_obj = _make_earth_satellite()
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 12, 10, 0)
            gdf = compute_ground_track(iss_info, start, end, time_step_s=60.0)

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) > 0
            assert "latitude" in gdf.columns
            assert "longitude" in gdf.columns
            assert "altitude_km" in gdf.columns
            assert "solar_zenith" in gdf.columns
            assert "satellite_name" in gdf.columns
            assert str(gdf.crs) == "EPSG:4326"
            # ISS altitude should be roughly 400-430 km
            assert gdf["altitude_km"].mean() > 300
            assert gdf["altitude_km"].mean() < 500

    def test_compute_ground_track_by_name(self):
        """Test that passing a string name resolves the satellite."""
        sat_obj = _make_earth_satellite()

        # Register ISS temporarily
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)
        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj), \
             patch("hyplan.satellites.get_satellite", return_value=iss_info):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 12, 5, 0)
            gdf = compute_ground_track("ISS", start, end, time_step_s=60.0)
            assert len(gdf) > 0

    def test_ground_track_time_steps(self):
        """Number of points should match expected time steps."""
        sat_obj = _make_earth_satellite()
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=0.0)

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 12, 5, 0)
            gdf = compute_ground_track(iss_info, start, end, time_step_s=60.0)
            # 5 minutes / 60s step + 1 = 6 points
            assert len(gdf) == 6


# ---------------------------------------------------------------------------
# compute_swath_footprint
# ---------------------------------------------------------------------------

class TestComputeSwathFootprint:
    def test_basic_swath(self):
        """Swath polygon should be created from a ground track."""
        gdf = _make_ground_track_gdf(n=20, step_s=30.0)
        swath_gdf = compute_swath_footprint(gdf, swath_width_km=100.0)

        assert isinstance(swath_gdf, gpd.GeoDataFrame)
        assert len(swath_gdf) >= 1
        assert "satellite_name" in swath_gdf.columns
        assert "pass_start" in swath_gdf.columns
        assert "pass_end" in swath_gdf.columns
        assert "ascending" in swath_gdf.columns
        assert str(swath_gdf.crs) == "EPSG:4326"
        # Geometry should be a polygon
        assert swath_gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_swath_uses_registry_width(self):
        """When swath_width_km is None, should look up from registry."""
        gdf = _make_ground_track_gdf(sat_name="PACE", norad_id=58927)
        swath_gdf = compute_swath_footprint(gdf, swath_width_km=None)
        assert len(swath_gdf) >= 1
        # PACE swath is 2663 km, polygon should be large
        area = swath_gdf.geometry.iloc[0].area
        assert area > 0

    def test_empty_ground_track(self):
        """An empty ground track should return an empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(
            columns=["satellite_name", "norad_id", "timestamp", "latitude",
                     "longitude", "altitude_km", "solar_zenith", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )
        swath_gdf = compute_swath_footprint(empty_gdf, swath_width_km=100.0)
        assert len(swath_gdf) == 0

    def test_ascending_pass(self):
        """Pass with increasing latitude should be marked ascending=True."""
        gdf = _make_ground_track_gdf(lat_start=30.0, lat_end=50.0)
        swath_gdf = compute_swath_footprint(gdf, swath_width_km=100.0)
        assert bool(swath_gdf["ascending"].iloc[0]) is True

    def test_descending_pass(self):
        """Pass with decreasing latitude should be marked ascending=False."""
        gdf = _make_ground_track_gdf(lat_start=50.0, lat_end=30.0)
        swath_gdf = compute_swath_footprint(gdf, swath_width_km=100.0)
        assert bool(swath_gdf["ascending"].iloc[0]) is False

    def test_narrow_swath(self):
        """A very narrow swath should still produce a valid polygon."""
        gdf = _make_ground_track_gdf(n=10, step_s=30.0)
        swath_gdf = compute_swath_footprint(gdf, swath_width_km=0.1)
        assert len(swath_gdf) == 1
        assert swath_gdf.geometry.iloc[0].is_valid


# ---------------------------------------------------------------------------
# find_overpasses
# ---------------------------------------------------------------------------

class TestFindOverpasses:
    def _setup_mocked_overpass(self):
        """Set up common mocks for overpass tests."""
        sat_obj = _make_earth_satellite()
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=100.0, max_sza=90.0)
        return sat_obj, iss_info

    def test_find_overpasses_no_passes(self):
        """A region far from the ground track should yield no overpasses."""
        sat_obj, iss_info = self._setup_mocked_overpass()
        # Region at south pole -- ISS (51.6 deg inclination) won't pass here
        region = Polygon([(-180, -89), (-180, -88), (-179, -88), (-179, -89)])

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 13, 0, 0)
            gdf = find_overpasses(iss_info, region, start, end,
                                  time_step_s=30.0)
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 0

    def test_find_overpasses_with_geodataframe_region(self):
        """Region can be passed as a GeoDataFrame."""
        sat_obj, iss_info = self._setup_mocked_overpass()
        region_poly = Polygon([(-180, -89), (-180, -88), (-179, -88), (-179, -89)])
        region_gdf = gpd.GeoDataFrame(geometry=[region_poly], crs="EPSG:4326")

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 13, 0, 0)
            gdf = find_overpasses(iss_info, region_gdf, start, end,
                                  time_step_s=30.0)
            assert isinstance(gdf, gpd.GeoDataFrame)

    def test_find_overpasses_by_name(self):
        """Test that find_overpasses accepts a string satellite name."""
        sat_obj = _make_earth_satellite()
        region = Polygon([(-180, -89), (-180, -88), (-179, -88), (-179, -89)])

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj), \
             patch("hyplan.satellites.get_satellite",
                   return_value=SatelliteInfo("ISS", 25544, swath_width_km=100.0)):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 13, 0, 0)
            gdf = find_overpasses("ISS", region, start, end, time_step_s=30.0)
            assert isinstance(gdf, gpd.GeoDataFrame)

    def test_find_overpasses_result_columns(self):
        """When overpasses are found, check result columns are present."""
        sat_obj = _make_earth_satellite()
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=100.0, max_sza=90.0)

        # Use a wide equatorial band -- ISS crosses equator every orbit
        region = Polygon([(-180, -10), (-180, 10), (180, 10), (180, -10)])

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 14, 0, 0)
            gdf = find_overpasses(iss_info, region, start, end,
                                  time_step_s=30.0, include_swath=True)
            if len(gdf) > 0:
                expected_cols = {"satellite_name", "norad_id", "pass_start",
                                 "pass_end", "ascending", "ground_track",
                                 "solar_zenith_at_center", "is_usable", "geometry"}
                assert expected_cols.issubset(set(gdf.columns))

    def test_find_overpasses_without_swath(self):
        """When include_swath=False, geometry should be LineString."""
        sat_obj = _make_earth_satellite()
        iss_info = SatelliteInfo("ISS (ZARYA)", 25544, swath_width_km=100.0, max_sza=90.0)
        region = Polygon([(-180, -10), (-180, 10), (180, 10), (180, -10)])

        with patch("hyplan.satellites.fetch_tle", return_value=sat_obj):
            start = datetime(2024, 6, 18, 12, 0, 0)
            end = datetime(2024, 6, 18, 14, 0, 0)
            gdf = find_overpasses(iss_info, region, start, end,
                                  time_step_s=30.0, include_swath=False)
            if len(gdf) > 0:
                assert gdf.geometry.iloc[0].geom_type == "LineString"


# ---------------------------------------------------------------------------
# find_all_overpasses
# ---------------------------------------------------------------------------

class TestFindAllOverpasses:
    def test_find_all_returns_combined(self):
        """find_all_overpasses should concatenate results from multiple satellites."""
        # Mock find_overpasses to return synthetic data
        row_data = {
            "satellite_name": ["SAT-A"],
            "norad_id": [99999],
            "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
            "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
            "pass_duration_s": [300.0],
            "ascending": [True],
            "ground_track": [LineString([(-90, 30), (-80, 40)])],
            "solar_zenith_at_center": [30.0],
            "is_usable": [True],
        }
        geom = [Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])]
        mock_gdf = gpd.GeoDataFrame(row_data, geometry=geom, crs="EPSG:4326")

        with patch("hyplan.satellites.find_overpasses", return_value=mock_gdf):
            region = Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])
            start = datetime(2025, 6, 15, 12, 0, 0)
            end = datetime(2025, 6, 15, 14, 0, 0)
            result = find_all_overpasses(
                satellites=["PACE", "Landsat-8"],
                region=region,
                start_time=start,
                end_time=end,
            )
            assert isinstance(result, gpd.GeoDataFrame)
            # Two satellites, each returning 1 row
            assert len(result) == 2

    def test_find_all_defaults_to_registry(self):
        """When satellites=None, should use all registry entries."""
        empty_gdf = _empty_overpass_gdf()
        with patch("hyplan.satellites.find_overpasses", return_value=empty_gdf):
            region = Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])
            result = find_all_overpasses(
                satellites=None,
                region=region,
                start_time=datetime(2025, 6, 15, 12, 0, 0),
                end_time=datetime(2025, 6, 15, 14, 0, 0),
            )
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 0

    def test_find_all_handles_errors(self):
        """Errors from individual satellites should be caught, not raised."""
        def side_effect(sat, region, start_time, end_time, **kwargs):
            raise RuntimeError("Network error")

        with patch("hyplan.satellites.find_overpasses", side_effect=side_effect):
            region = Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])
            result = find_all_overpasses(
                satellites=["PACE"],
                region=region,
                start_time=datetime(2025, 6, 15, 12, 0, 0),
                end_time=datetime(2025, 6, 15, 14, 0, 0),
            )
            assert len(result) == 0

    def test_find_all_sorts_by_pass_start(self):
        """Result should be sorted by pass_start."""
        call_count = [0]

        def mock_find(sat, region, start_time, end_time, **kwargs):
            call_count[0] += 1
            t = pd.Timestamp("2025-06-15 14:00:00") if call_count[0] == 1 else pd.Timestamp("2025-06-15 12:00:00")
            row_data = {
                "satellite_name": [sat if isinstance(sat, str) else sat.name],
                "norad_id": [99999],
                "pass_start": [t],
                "pass_end": [t + pd.Timedelta(minutes=5)],
                "pass_duration_s": [300.0],
                "ascending": [True],
                "ground_track": [LineString([(-90, 30), (-80, 40)])],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            }
            geom = [Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])]
            return gpd.GeoDataFrame(row_data, geometry=geom, crs="EPSG:4326")

        with patch("hyplan.satellites.find_overpasses", side_effect=mock_find):
            region = Polygon([(-100, 25), (-100, 45), (-70, 45), (-70, 25)])
            result = find_all_overpasses(
                satellites=["PACE", "Landsat-8"],
                region=region,
                start_time=datetime(2025, 6, 15, 12, 0, 0),
                end_time=datetime(2025, 6, 15, 16, 0, 0),
            )
            assert len(result) == 2
            assert result["pass_start"].iloc[0] <= result["pass_start"].iloc[1]


# ---------------------------------------------------------------------------
# compute_overpass_overlap
# ---------------------------------------------------------------------------

class TestComputeOverpassOverlap:
    def test_overlap_empty_inputs(self):
        """Empty inputs should return empty result."""
        empty_overpasses = _empty_overpass_gdf()
        empty_flight = gpd.GeoDataFrame(
            columns=["segment_name", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )
        result = compute_overpass_overlap(
            empty_flight, empty_overpasses,
            flight_time_utc=datetime(2025, 6, 15, 12, 0, 0),
        )
        assert len(result) == 0
        assert "overlap_area_km2" in result.columns

    def test_overlap_non_intersecting(self):
        """Non-overlapping geometries should produce no results."""
        overpass_poly = Polygon([(10, 10), (10, 20), (20, 20), (20, 10)])
        overpasses = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A"],
                "norad_id": [99999],
                "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            },
            geometry=[overpass_poly],
            crs="EPSG:4326",
        )
        flight_poly = Polygon([(-100, -50), (-100, -40), (-90, -40), (-90, -50)])
        flight_plan = gpd.GeoDataFrame(
            {"segment_name": ["seg1"]},
            geometry=[flight_poly],
            crs="EPSG:4326",
        )
        result = compute_overpass_overlap(
            flight_plan, overpasses,
            flight_time_utc=datetime(2025, 6, 15, 12, 0, 0),
        )
        assert len(result) == 0

    def test_overlap_intersecting(self):
        """Overlapping geometries should produce a result with area."""
        overpass_poly = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        overpasses = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A"],
                "norad_id": [99999],
                "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            },
            geometry=[overpass_poly],
            crs="EPSG:4326",
        )
        flight_poly = Polygon([(-92, 29), (-92, 31), (-88, 31), (-88, 29)])
        flight_plan = gpd.GeoDataFrame(
            {"segment_name": ["seg1"]},
            geometry=[flight_poly],
            crs="EPSG:4326",
        )

        result = compute_overpass_overlap(
            flight_plan, overpasses,
            flight_time_utc=datetime(2025, 6, 15, 12, 0, 0),
        )
        assert len(result) >= 1
        assert result["overlap_area_km2"].iloc[0] > 0
        assert result["satellite_name"].iloc[0] == "SAT-A"

    def test_overlap_time_filter(self):
        """Passes with time_offset > max_time_offset_min should be excluded."""
        overpass_poly = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        overpasses = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A"],
                "norad_id": [99999],
                "pass_start": [pd.Timestamp("2025-06-15 18:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 18:05:00")],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            },
            geometry=[overpass_poly],
            crs="EPSG:4326",
        )
        flight_poly = Polygon([(-92, 29), (-92, 31), (-88, 31), (-88, 29)])
        flight_plan = gpd.GeoDataFrame(
            {
                "segment_name": ["seg1"],
                "time_to_segment": [0.0],  # hours from flight start
            },
            geometry=[flight_poly],
            crs="EPSG:4326",
        )
        # Flight starts at 12:00, overpass at 18:00 = 360 min offset
        result = compute_overpass_overlap(
            flight_plan, overpasses,
            flight_time_utc=datetime(2025, 6, 15, 12, 0, 0),
            max_time_offset_min=60.0,
        )
        assert len(result) == 0

    def test_overlap_with_empty_geometry(self):
        """Flight segments with None geometry should be skipped."""
        overpass_poly = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        overpasses = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A"],
                "norad_id": [99999],
                "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            },
            geometry=[overpass_poly],
            crs="EPSG:4326",
        )
        flight_plan = gpd.GeoDataFrame(
            {"segment_name": ["seg1"]},
            geometry=[None],
            crs="EPSG:4326",
        )
        result = compute_overpass_overlap(
            flight_plan, overpasses,
            flight_time_utc=datetime(2025, 6, 15, 12, 0, 0),
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# overpasses_to_kml
# ---------------------------------------------------------------------------

class TestOverpassesToKml:
    def test_kml_with_polygon_geometry(self, tmp_path):
        """KML export with Polygon geometries."""
        poly = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        gdf = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A"],
                "norad_id": [99999],
                "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
                "solar_zenith_at_center": [30.0],
                "is_usable": [True],
            },
            geometry=[poly],
            crs="EPSG:4326",
        )
        kml_path = str(tmp_path / "test_overpasses.kml")
        overpasses_to_kml(gdf, kml_path)
        assert os.path.exists(kml_path)
        assert os.path.getsize(kml_path) > 0

    def test_kml_with_linestring_geometry(self, tmp_path):
        """KML export with LineString geometries."""
        line = LineString([(-95, 28), (-90, 30), (-85, 32)])
        gdf = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-B"],
                "norad_id": [88888],
                "pass_start": [pd.Timestamp("2025-06-15 12:00:00")],
                "pass_end": [pd.Timestamp("2025-06-15 12:05:00")],
                "solar_zenith_at_center": [45.0],
                "is_usable": [True],
            },
            geometry=[line],
            crs="EPSG:4326",
        )
        kml_path = str(tmp_path / "test_linestring.kml")
        overpasses_to_kml(gdf, kml_path)
        assert os.path.exists(kml_path)

    def test_kml_multiple_satellites(self, tmp_path):
        """KML export with multiple satellites gets different colors."""
        poly1 = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        poly2 = Polygon([(-80, 28), (-80, 32), (-70, 32), (-70, 28)])
        gdf = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A", "SAT-B"],
                "norad_id": [99999, 88888],
                "pass_start": [
                    pd.Timestamp("2025-06-15 12:00:00"),
                    pd.Timestamp("2025-06-15 13:00:00"),
                ],
                "pass_end": [
                    pd.Timestamp("2025-06-15 12:05:00"),
                    pd.Timestamp("2025-06-15 13:05:00"),
                ],
                "solar_zenith_at_center": [30.0, 45.0],
                "is_usable": [True, False],
            },
            geometry=[poly1, poly2],
            crs="EPSG:4326",
        )
        kml_path = str(tmp_path / "test_multi.kml")
        overpasses_to_kml(gdf, kml_path)
        assert os.path.exists(kml_path)
        # Read file content to verify both satellites appear
        with open(kml_path) as f:
            content = f.read()
        assert "SAT-A" in content
        assert "SAT-B" in content

    def test_kml_mixed_geometries(self, tmp_path):
        """KML export with a mix of Polygon and LineString."""
        poly = Polygon([(-95, 28), (-95, 32), (-85, 32), (-85, 28)])
        line = LineString([(-80, 28), (-75, 30), (-70, 32)])
        gdf = gpd.GeoDataFrame(
            {
                "satellite_name": ["SAT-A", "SAT-A"],
                "norad_id": [99999, 99999],
                "pass_start": [
                    pd.Timestamp("2025-06-15 12:00:00"),
                    pd.Timestamp("2025-06-15 13:00:00"),
                ],
                "pass_end": [
                    pd.Timestamp("2025-06-15 12:05:00"),
                    pd.Timestamp("2025-06-15 13:05:00"),
                ],
                "solar_zenith_at_center": [30.0, 45.0],
                "is_usable": [True, True],
            },
            geometry=[poly, line],
            crs="EPSG:4326",
        )
        kml_path = str(tmp_path / "test_mixed.kml")
        overpasses_to_kml(gdf, kml_path)
        assert os.path.exists(kml_path)
