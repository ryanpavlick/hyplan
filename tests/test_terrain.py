"""Tests for hyplan.terrain (unit-testable parts, no network)."""
# ruff: noqa: E402

import os
import pytest
import tempfile

import numpy as np

from hyplan.terrain import get_cache_root, clear_cache, _COS_TILT_MIN


# ---------------------------------------------------------------------------
# Synthetic DEM helper
# ---------------------------------------------------------------------------

def _has_rasterio():
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def _write_synthetic_dem(filepath, lat_center, lon_center, elevation_func, size=100):
    """Write a small GeoTIFF DEM for testing.

    elevation_func(row, col) returns elevation in meters for each pixel.
    The raster covers ±(size*pixel_deg/2) degrees around (lat_center, lon_center).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    pixel_deg = 0.001  # ~111m resolution
    x_min = lon_center - size * pixel_deg / 2
    x_max = lon_center + size * pixel_deg / 2
    y_min = lat_center - size * pixel_deg / 2
    y_max = lat_center + size * pixel_deg / 2

    raster = np.zeros((size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            raster[r, c] = elevation_func(r, c)

    transform = from_bounds(x_min, y_min, x_max, y_max, size, size)
    with rasterio.open(
        filepath, "w", driver="GTiff",
        height=size, width=size, count=1,
        dtype=raster.dtype, crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(raster, 1)


class TestGetCacheRoot:
    def test_default_path(self):
        root = get_cache_root()
        assert root.endswith("hyplan")
        assert tempfile.gettempdir() in root

    def test_custom_path(self):
        root = get_cache_root(custom_path="/tmp/custom_hyplan_cache")
        assert root == "/tmp/custom_hyplan_cache"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/tmp/env_cache")
        root = get_cache_root()
        assert root == "/tmp/env_cache"

    def test_custom_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/tmp/env_cache")
        root = get_cache_root(custom_path="/tmp/override")
        assert root == "/tmp/override"


class TestClearCache:
    def test_clear_nonexistent(self, monkeypatch):
        """Clearing a non-existent cache under tempdir should not raise."""
        nonexistent = os.path.join(tempfile.gettempdir(), "hyplan_test_nonexistent")
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", nonexistent)
        clear_cache()  # Should not raise since dir doesn't exist

    def test_clear_existing(self, monkeypatch, tmp_path):
        """Should remove the cache directory."""
        cache_dir = tmp_path / "hyplan"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_text("data")
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", str(cache_dir))
        # clear_cache checks that the path starts with tempdir
        # Since tmp_path is under tempdir, this should work
        # But the safety check requires startswith(tempfile.gettempdir())
        if str(cache_dir).startswith(tempfile.gettempdir()):
            clear_cache()
            assert not cache_dir.exists()

    def test_refuses_unsafe_path(self, monkeypatch):
        """Should refuse to clear a directory outside tempdir."""
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/home/user/important_data")
        with pytest.raises(ValueError, match="unsafe"):
            clear_cache()


class TestConstants:
    def test_cos_tilt_min(self):
        assert _COS_TILT_MIN > 0
        assert _COS_TILT_MIN < 0.01


# ---------------------------------------------------------------------------
# DEMGrid
# ---------------------------------------------------------------------------

from hyplan.terrain import DEMGrid


class TestDEMGrid:
    def test_properties(self):
        arr = np.array([[100.0, 200.0], [300.0, 400.0]])
        dem = DEMGrid(
            array=arr,
            geotransform=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0),
            bounds=(0.0, -1.0, 2.0, 1.0),
        )
        assert dem.raster_min == 100.0
        assert dem.raster_max == 400.0
        assert dem.shape == (2, 2)
        assert dem.nodata is None

    def test_frozen(self):
        dem = DEMGrid(
            array=np.zeros((2, 2)),
            geotransform=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0),
            bounds=(0.0, -1.0, 2.0, 1.0),
        )
        with pytest.raises(AttributeError):
            dem.nodata = -9999.0


# ---------------------------------------------------------------------------
# DEM loading
# ---------------------------------------------------------------------------

from hyplan.terrain import load_dem


class TestLoadDEM:
    @pytest.fixture
    def flat_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "flat.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0)
        return path

    def test_returns_demgrid(self, flat_dem_path):
        dem = load_dem(flat_dem_path)
        assert isinstance(dem, DEMGrid)
        assert dem.shape == (100, 100)
        assert dem.raster_min == pytest.approx(500.0)
        assert dem.raster_max == pytest.approx(500.0)

    def test_geotransform_valid(self, flat_dem_path):
        dem = load_dem(flat_dem_path)
        gt = dem.geotransform
        assert len(gt) == 6
        # pixel width should be positive, pixel height negative (north-up)
        assert gt[1] > 0
        assert gt[5] < 0


# ---------------------------------------------------------------------------
# Elevation sampling
# ---------------------------------------------------------------------------

from hyplan.terrain import get_elevations, get_elevations_from_grid


class TestGetElevations:
    @pytest.fixture
    def gradient_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "gradient.tif")
        # Elevation = 1000 + col*10 (increases eastward)
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 1000.0 + c * 10.0)
        return path

    def test_flat_dem_returns_constant(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "flat.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 750.0)
        lats = np.array([35.0, 35.01, 34.99])
        lons = np.array([-111.0, -111.01, -110.99])
        elevs = get_elevations(lats, lons, path)
        np.testing.assert_allclose(elevs, 750.0, atol=1.0)

    def test_gradient_dem_increases_eastward(self, gradient_dem_path):
        lats = np.array([35.0, 35.0])
        # West and east of center
        lons = np.array([-111.03, -110.97])
        elevs = get_elevations(lats, lons, gradient_dem_path)
        assert elevs[1] > elevs[0]  # east is higher

    def test_from_grid_matches_from_file(self, gradient_dem_path):
        dem = load_dem(gradient_dem_path)
        lats = np.array([35.0, 35.01])
        lons = np.array([-111.0, -111.01])
        from_file = get_elevations(lats, lons, gradient_dem_path)
        from_grid = get_elevations_from_grid(lats, lons, dem)
        np.testing.assert_array_equal(from_file, from_grid)


# ---------------------------------------------------------------------------
# DEM merge
# ---------------------------------------------------------------------------

from hyplan.terrain import merge_tiles


class TestMergeTiles:
    def test_merge_single_tile(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        tile_path = str(tmp_path / "tile.tif")
        _write_synthetic_dem(tile_path, 35.0, -111.0, lambda r, c: 500.0, size=50)
        out_path = str(tmp_path / "merged.tif")
        merge_tiles(out_path, [tile_path])
        assert os.path.exists(out_path)
        dem = load_dem(out_path)
        assert dem.raster_min == pytest.approx(500.0)

    def test_merge_empty_raises(self):
        with pytest.raises(Exception):
            merge_tiles("/tmp/out.tif", [])


# ---------------------------------------------------------------------------
# Ray-terrain intersection
# ---------------------------------------------------------------------------

from hyplan.terrain import ray_terrain_intersection


class TestRayTerrainIntersection:
    @pytest.fixture
    def flat_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "flat500.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0, size=200)
        return path

    def test_nadir_ray_hits_ground(self, flat_dem_path):
        """Straight-down ray from 5000m should hit the 500m surface."""
        lat0 = np.array([35.0])
        lon0 = np.array([-111.0])
        az = np.array([0.0])
        tilt = np.array([0.0])  # nadir = 0 tilt
        ilat, ilon, ialt = ray_terrain_intersection(
            lat0, lon0, 5000.0, az, tilt, precision=5.0, dem_file=flat_dem_path,
        )
        assert not np.isnan(ilat[0])
        assert ialt[0] == pytest.approx(500.0, abs=10.0)
        # Should be close to observer position for nadir
        assert abs(ilat[0] - 35.0) < 0.01
        assert abs(ilon[0] - (-111.0)) < 0.01

    def test_off_nadir_displaces_intersection(self, flat_dem_path):
        """Off-nadir ray should intersect ground displaced from observer."""
        lat0 = np.array([35.0])
        lon0 = np.array([-111.0])
        az = np.array([0.0])  # north
        tilt = np.array([45.0])  # 45 deg off-nadir
        ilat, ilon, ialt = ray_terrain_intersection(
            lat0, lon0, 5000.0, az, tilt, precision=5.0, dem_file=flat_dem_path,
        )
        assert not np.isnan(ilat[0])
        # Intersection should be north of observer
        assert ilat[0] > 35.0

    def test_batch_multiple_observers(self, flat_dem_path):
        """Multiple simultaneous observers should all find intersections."""
        n = 5
        lat0 = np.full(n, 35.0)
        lon0 = np.full(n, -111.0)
        az = np.linspace(0, 180, n)
        tilt = np.full(n, 30.0)
        ilat, ilon, ialt = ray_terrain_intersection(
            lat0, lon0, 5000.0, az, tilt, precision=5.0, dem_file=flat_dem_path,
        )
        assert ilat.shape == (n,)
        assert not np.any(np.isnan(ilat))

    def test_invalid_tilt_raises(self):
        with pytest.raises(Exception):
            ray_terrain_intersection(
                np.array([35.0]), np.array([-111.0]), 5000.0,
                np.array([0.0]), np.array([91.0]),
            )


# ---------------------------------------------------------------------------
# Surface normals
# ---------------------------------------------------------------------------

from hyplan.terrain import surface_normal_at


class TestSurfaceNormalAt:
    @pytest.fixture
    def flat_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "flat.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0)
        return path

    @pytest.fixture
    def sloped_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "slope.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0 + c * 30.0)
        return path

    def test_flat_terrain_vertical_normal(self, flat_dem_path):
        normals = surface_normal_at(
            np.array([35.0]), np.array([-111.0]), flat_dem_path,
        )
        assert normals.shape == (1, 3)
        # Should be approximately [0, 0, 1]
        assert normals[0, 2] == pytest.approx(1.0, abs=0.01)
        assert abs(normals[0, 0]) < 0.01
        assert abs(normals[0, 1]) < 0.01

    def test_sloped_terrain_tilted_normal(self, sloped_dem_path):
        normals = surface_normal_at(
            np.array([35.0]), np.array([-111.0]), sloped_dem_path,
        )
        # East-facing slope: normal should have negative east component
        # (normal points away from the rising surface)
        assert normals[0, 0] < -0.01

    def test_unit_vector(self, flat_dem_path):
        lats = np.array([35.0, 35.01, 34.99])
        lons = np.array([-111.0, -111.01, -110.99])
        normals = surface_normal_at(lats, lons, flat_dem_path)
        magnitudes = np.sqrt(np.sum(normals ** 2, axis=1))
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# terrain_elevation_along_track
# ---------------------------------------------------------------------------

from hyplan.terrain import terrain_elevation_along_track
from hyplan.flight_line import FlightLine
from hyplan.units import ureg


class TestTerrainElevationAlongTrack:
    @pytest.fixture
    def flat_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "flat_track.tif")
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0, size=200)
        return path

    @pytest.fixture
    def sloped_dem_path(self, tmp_path):
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "sloped_track.tif")
        # Elevation ramps from 100 to ~1090 across rows (north to south)
        _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 100.0 + r * 10.0, size=200)
        return path

    @pytest.fixture
    def flight_line_ns(self):
        """A short north-south flight line near the DEM center."""
        return FlightLine.start_length_azimuth(
            lat1=35.02, lon1=-111.0,
            length=ureg.Quantity(5000, "meter"),
            az=180,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="NS_line",
        )

    def test_flat_dem_returns_constant(self, flat_dem_path, flight_line_ns):
        result = terrain_elevation_along_track(flight_line_ns, flat_dem_path)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"min", "mean", "max"}
        assert result["min"] == pytest.approx(500.0, abs=5.0)
        assert result["max"] == pytest.approx(500.0, abs=5.0)
        assert result["mean"] == pytest.approx(500.0, abs=5.0)

    def test_sloped_dem_min_less_than_max(self, sloped_dem_path, flight_line_ns):
        result = terrain_elevation_along_track(flight_line_ns, sloped_dem_path)
        assert result["min"] < result["max"]
        assert result["min"] <= result["mean"] <= result["max"]

    def test_custom_precision(self, flat_dem_path, flight_line_ns):
        result = terrain_elevation_along_track(flight_line_ns, flat_dem_path, precision=50.0)
        assert isinstance(result, dict)
        assert result["min"] == pytest.approx(500.0, abs=5.0)


# ---------------------------------------------------------------------------
# terrain_aspect_azimuth
# ---------------------------------------------------------------------------

from hyplan.terrain import terrain_aspect_azimuth
from shapely.geometry import box


class TestTerrainAspectAzimuth:
    @pytest.fixture
    def east_slope_dem_path(self, tmp_path):
        """DEM that slopes upward to the east with quadratic curvature.

        Uses c**2 so that gradient magnitude increases eastward, ensuring
        the 25th-percentile threshold in terrain_aspect_azimuth retains
        the majority of pixels (those with the strongest eastward gradient).
        """
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "east_slope.tif")
        _write_synthetic_dem(
            path, 35.0, -111.0,
            lambda r, c: 500.0 + (c ** 2) * 0.1,
            size=100,
        )
        return path

    @pytest.fixture
    def south_slope_dem_path(self, tmp_path):
        """DEM that slopes upward to the south with quadratic curvature.

        In a north-up raster, rows increase southward, so row-increasing
        elevation means the surface rises toward the south.  The quadratic
        term ensures varying gradient magnitudes across the raster.
        """
        if not _has_rasterio():
            pytest.skip("rasterio not available")
        path = str(tmp_path / "south_slope.tif")
        _write_synthetic_dem(
            path, 35.0, -111.0,
            lambda r, c: 500.0 + (r ** 2) * 0.1,
            size=100,
        )
        return path

    @pytest.fixture
    def survey_polygon(self):
        """Small polygon around the DEM center."""
        return box(-111.02, 34.98, -110.98, 35.02)

    def test_east_slope_downslope_points_west(self, east_slope_dem_path, survey_polygon):
        """Surface rises eastward, so downslope should point roughly west (~270)."""
        azimuth = terrain_aspect_azimuth(survey_polygon, dem_file=east_slope_dem_path)
        assert isinstance(azimuth, float)
        # West is 270, allow some tolerance
        assert 240.0 < azimuth < 300.0, f"Expected ~270 but got {azimuth}"

    def test_south_slope_downslope_points_north(self, south_slope_dem_path, survey_polygon):
        """Surface rises southward (rows increase), so downslope should point roughly north (~0/360)."""
        azimuth = terrain_aspect_azimuth(survey_polygon, dem_file=south_slope_dem_path)
        assert isinstance(azimuth, float)
        # North is 0/360, so azimuth should be close to 0 or 360
        assert azimuth < 30.0 or azimuth > 330.0, f"Expected ~0/360 but got {azimuth}"

    def test_returns_float_in_range(self, east_slope_dem_path, survey_polygon):
        azimuth = terrain_aspect_azimuth(survey_polygon, dem_file=east_slope_dem_path)
        assert 0.0 <= azimuth < 360.0
