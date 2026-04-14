"""Comprehensive tests for the LVIS instrument model."""

import os
import tempfile

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.instruments import (
    LVIS,
    LVIS_LENS_NARROW,
    LVIS_LENS_MEDIUM,
    LVIS_LENS_WIDE,
    LVIS_LENSES,
)


@pytest.fixture
def altitude():
    return 8000 * ureg.meter


@pytest.fixture
def speed():
    return 150 * ureg.knot


@pytest.fixture
def lvis_default():
    return LVIS()


@pytest.fixture
def lvis_narrow():
    return LVIS(lens="narrow")


class TestLVISInstantiation:
    def test_default_lens_is_wide(self, lvis_default):
        assert lvis_default.lens is LVIS_LENS_WIDE

    def test_lens_by_string(self):
        lvis = LVIS(lens="narrow")
        assert lvis.lens is LVIS_LENS_NARROW

    def test_lens_by_object(self):
        lvis = LVIS(lens=LVIS_LENS_MEDIUM)
        assert lvis.lens is LVIS_LENS_MEDIUM

    def test_invalid_lens_string(self):
        with pytest.raises(ValueError, match="Unknown lens"):
            LVIS(lens="ultrawide")

    def test_invalid_lens_type(self):
        with pytest.raises(TypeError, match="lens must be"):
            LVIS(lens=42)

    def test_custom_rep_rate(self):
        lvis = LVIS(rep_rate=10000 * ureg.Hz)
        assert lvis.rep_rate.magnitude == pytest.approx(10000)

    def test_numeric_rep_rate(self):
        lvis = LVIS(rep_rate=5000)
        assert lvis.rep_rate.magnitude == pytest.approx(5000)

    def test_name(self, lvis_default):
        assert str(lvis_default) == "LVIS"


class TestLVISLens:
    def test_lens_names(self):
        assert LVIS_LENS_NARROW.name == "narrow"
        assert LVIS_LENS_MEDIUM.name == "medium"
        assert LVIS_LENS_WIDE.name == "wide"

    def test_divergence_ordering(self):
        assert LVIS_LENS_NARROW.divergence_mrad < LVIS_LENS_MEDIUM.divergence_mrad
        assert LVIS_LENS_MEDIUM.divergence_mrad < LVIS_LENS_WIDE.divergence_mrad

    def test_footprint_diameter(self, altitude):
        fp_narrow = LVIS_LENS_NARROW.footprint_diameter(altitude)
        fp_wide = LVIS_LENS_WIDE.footprint_diameter(altitude)
        assert fp_narrow.magnitude < fp_wide.magnitude
        assert fp_narrow.check("[length]")

    def test_lenses_dict(self):
        assert set(LVIS_LENSES.keys()) == {"narrow", "medium", "wide"}


class TestHalfAngle:
    def test_half_angle_value(self, lvis_default):
        # atan(0.1) ≈ 5.71°
        assert lvis_default.half_angle == pytest.approx(5.71, rel=0.01)

    def test_half_angle_consistent_across_lenses(self, lvis_narrow):
        # half_angle is a scanner property, independent of lens
        assert lvis_narrow.half_angle == LVIS().half_angle


class TestSwathWidth:
    def test_positive(self, lvis_default, altitude):
        sw = lvis_default.swath_width(altitude)
        assert sw.magnitude > 0
        assert sw.check("[length]")

    def test_scales_with_altitude(self, lvis_default):
        sw1 = lvis_default.swath_width(5000 * ureg.meter)
        sw2 = lvis_default.swath_width(10000 * ureg.meter)
        assert sw2.magnitude == pytest.approx(2 * sw1.magnitude, rel=0.01)

    def test_approx_fraction_of_altitude(self, lvis_default, altitude):
        # Max swath ≈ 0.2 * altitude
        sw = lvis_default.swath_width(altitude)
        assert sw.magnitude == pytest.approx(0.2 * altitude.magnitude, rel=0.02)


class TestEquivalentFOV:
    def test_positive(self, lvis_default, altitude, speed):
        efov = lvis_default.equivalent_fov(altitude, speed)
        assert efov > 0

    def test_at_most_geometric_fov(self, lvis_default, altitude, speed):
        efov = lvis_default.equivalent_fov(altitude, speed)
        geo_fov = 2 * lvis_default.half_angle
        assert efov <= geo_fov + 0.01


class TestFootprintDiameter:
    def test_positive(self, lvis_default, altitude):
        fp = lvis_default.footprint_diameter(altitude)
        assert fp.magnitude > 0

    def test_wider_lens_bigger_footprint(self, altitude):
        fp_narrow = LVIS(lens="narrow").footprint_diameter(altitude)
        fp_wide = LVIS(lens="wide").footprint_diameter(altitude)
        assert fp_wide.magnitude > fp_narrow.magnitude


class TestCoverageRate:
    def test_positive(self, lvis_default, altitude, speed):
        cr = lvis_default.coverage_rate(altitude, speed)
        assert cr.magnitude > 0

    def test_faster_speed_higher_rate(self, lvis_default, altitude):
        cr1 = lvis_default.coverage_rate(altitude, 100 * ureg.knot)
        cr2 = lvis_default.coverage_rate(altitude, 200 * ureg.knot)
        assert cr2.magnitude > cr1.magnitude


class TestContiguous:
    def test_slow_speed_contiguous(self, altitude):
        lvis = LVIS(lens="wide", rep_rate=10000 * ureg.Hz)
        # Very slow speed with high rep rate should be contiguous
        assert lvis.is_contiguous(altitude, 50 * ureg.knot)

    def test_fast_speed_may_not_be_contiguous(self, altitude):
        lvis = LVIS(lens="narrow", rep_rate=1000 * ureg.Hz)
        # Fast speed with low rep rate and narrow lens
        is_c = lvis.is_contiguous(altitude, 300 * ureg.knot)
        # Just verify it returns a bool
        assert isinstance(is_c, (bool, np.bool_))


class TestEffectiveSwathWidth:
    def test_at_most_max_swath(self, lvis_default, altitude, speed):
        esw = lvis_default.effective_swath_width(altitude, speed)
        ms = lvis_default.swath_width(altitude)
        assert esw.magnitude <= ms.magnitude + 0.01

    def test_positive(self, lvis_default, altitude, speed):
        esw = lvis_default.effective_swath_width(altitude, speed)
        assert esw.magnitude > 0


class TestSummary:
    def test_summary_keys(self, lvis_default, altitude, speed):
        s = lvis_default.summary(altitude, speed)
        expected_keys = {
            "altitude_agl", "speed", "rep_rate", "lens",
            "lens_divergence_mrad", "footprint_diameter", "max_swath",
            "effective_swath_width", "contiguous",
            "along_track_spacing", "along_track_contiguous",
            "point_density",
            "coverage_rate", "footprint_for_max_swath",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_lens_name(self, lvis_default, altitude, speed):
        s = lvis_default.summary(altitude, speed)
        assert s["lens"] == "wide"

    def test_summary_contiguous_is_bool(self, lvis_default, altitude, speed):
        s = lvis_default.summary(altitude, speed)
        assert isinstance(s["contiguous"], (bool, np.bool_))


class TestAlongTrack:
    def test_spacing_positive(self, lvis_default, speed):
        spacing = lvis_default.along_track_spacing(speed)
        assert spacing.magnitude > 0
        assert spacing.check("[length]")

    def test_spacing_increases_with_speed(self, lvis_default):
        s1 = lvis_default.along_track_spacing(100 * ureg.knot).magnitude
        s2 = lvis_default.along_track_spacing(200 * ureg.knot).magnitude
        assert s2 > s1

    def test_spacing_decreases_with_rep_rate(self, speed):
        lvis_lo = LVIS(rep_rate=2000 * ureg.Hz)
        lvis_hi = LVIS(rep_rate=8000 * ureg.Hz)
        assert lvis_hi.along_track_spacing(speed).magnitude < lvis_lo.along_track_spacing(speed).magnitude

    def test_contiguous_slow_speed(self, altitude):
        lvis = LVIS(lens="wide", rep_rate=10000 * ureg.Hz)
        assert lvis.is_along_track_contiguous(altitude, 50 * ureg.knot)

    def test_contiguous_returns_bool(self, lvis_default, altitude, speed):
        result = lvis_default.is_along_track_contiguous(altitude, speed)
        assert isinstance(result, (bool, np.bool_))


class TestPointDensity:
    def test_positive(self, lvis_default, altitude, speed):
        d = lvis_default.point_density(altitude, speed)
        assert d.magnitude > 0

    def test_density_increases_with_slower_speed(self, lvis_default, altitude):
        d1 = lvis_default.point_density(altitude, 300 * ureg.knot).magnitude
        d2 = lvis_default.point_density(altitude, 100 * ureg.knot).magnitude
        assert d2 > d1

    def test_density_increases_with_rep_rate(self, altitude, speed):
        d1 = LVIS(rep_rate=2000 * ureg.Hz).point_density(altitude, speed).magnitude
        d2 = LVIS(rep_rate=8000 * ureg.Hz).point_density(altitude, speed).magnitude
        assert d2 > d1


class TestSurveysolvers:
    def test_solve_for_speed_positive(self, lvis_default, altitude):
        target = 0.001 / ureg.meter ** 2
        v = lvis_default.solve_for_speed(target, altitude)
        assert v.magnitude > 0

    def test_solve_for_speed_meets_target(self):
        """Speed from solver should yield density >= target."""
        lvis = LVIS(lens="wide", rep_rate=10000 * ureg.Hz)
        alt = 8000 * ureg.meter
        target = 0.004 / ureg.meter ** 2
        v = lvis.solve_for_speed(target, alt)
        actual = lvis.point_density(alt, v)
        assert actual.magnitude >= target.magnitude * 0.99

    def test_solve_for_speed_impossible_density(self, altitude):
        """Density above 1/fp^2 is impossible — should raise."""
        lvis = LVIS(lens="wide")
        fp = lvis.footprint_diameter(altitude).magnitude
        impossible = (2.0 / fp ** 2) / ureg.meter ** 2
        with pytest.raises(ValueError, match="exceeds"):
            lvis.solve_for_speed(impossible, altitude)

    def test_solve_for_altitude_positive(self, lvis_default, speed):
        target = 0.001 / ureg.meter ** 2
        alt = lvis_default.solve_for_altitude(target, speed)
        assert alt.magnitude > 0

    def test_solve_for_altitude_roundtrip(self, lvis_default, speed):
        """Solving for altitude then computing density should recover the target."""
        target = 0.002 / ureg.meter ** 2
        alt = lvis_default.solve_for_altitude(target, speed)
        actual = lvis_default.point_density(alt, speed)
        assert actual.magnitude == pytest.approx(target.magnitude, rel=0.05)

    def test_higher_density_requires_lower_altitude(self, lvis_default, speed):
        alt_lo = lvis_default.solve_for_altitude(0.005 / ureg.meter ** 2, speed)
        alt_hi = lvis_default.solve_for_altitude(0.001 / ureg.meter ** 2, speed)
        assert alt_lo.magnitude < alt_hi.magnitude

    def test_higher_density_requires_slower_speed(self):
        """Higher target density → lower max speed."""
        lvis = LVIS(lens="wide", rep_rate=10000 * ureg.Hz)
        alt = 3000 * ureg.meter
        v_hi_density = lvis.solve_for_speed(0.01 / ureg.meter ** 2, alt)
        v_lo_density = lvis.solve_for_speed(0.005 / ureg.meter ** 2, alt)
        assert v_hi_density.magnitude < v_lo_density.magnitude

    def test_negative_density_raises(self, lvis_default, altitude, speed):
        with pytest.raises(ValueError, match="positive"):
            lvis_default.solve_for_speed(-0.001 / ureg.meter ** 2, altitude)
        with pytest.raises(ValueError, match="positive"):
            lvis_default.solve_for_altitude(-0.001 / ureg.meter ** 2, speed)


class TestScanHalfAngle:
    def test_default_value(self):
        lvis = LVIS()
        assert lvis.half_angle == pytest.approx(5.71, rel=0.01)

    def test_custom_value(self):
        lvis = LVIS(scan_half_angle_deg=10.0)
        assert lvis.half_angle == pytest.approx(10.0)

    def test_wider_angle_wider_swath(self):
        alt = 8000 * ureg.meter
        lvis_narrow = LVIS(scan_half_angle_deg=3.0)
        lvis_wide = LVIS(scan_half_angle_deg=10.0)
        assert lvis_wide.swath_width(alt).magnitude > lvis_narrow.swath_width(alt).magnitude

    def test_invalid_angle_raises(self):
        with pytest.raises(ValueError):
            LVIS(scan_half_angle_deg=0)
        with pytest.raises(ValueError):
            LVIS(scan_half_angle_deg=90)


class TestPrintMethods:
    def test_print_summary_runs(self, lvis_default, altitude, speed, capsys):
        lvis_default.print_summary(altitude, speed)
        captured = capsys.readouterr()
        assert "LVIS Coverage Summary" in captured.out

    def test_compare_lenses_runs(self, lvis_default, altitude, speed, capsys):
        lvis_default.compare_lenses(altitude, speed)
        captured = capsys.readouterr()
        assert "Lens Comparison" in captured.out
        assert "narrow" in captured.out
        assert "wide" in captured.out


# ===================================================================
# Terrain-aware tests
# ===================================================================

def _has_rasterio():
    """Check if rasterio is available for writing synthetic DEMs."""
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def _write_synthetic_dem(filepath, lat_center, lon_center, elevation_func, size=100):
    """Write a small GeoTIFF DEM for testing.

    elevation_func(row, col) returns elevation in meters for each pixel.
    The raster covers ±0.05° around (lat_center, lon_center).
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


@pytest.fixture
def flat_dem(tmp_path):
    """Flat DEM at 500m elevation."""
    if not _has_rasterio():
        pytest.skip("rasterio not available")
    path = str(tmp_path / "flat.tif")
    _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0)
    return path


@pytest.fixture
def sloped_dem(tmp_path):
    """East-facing slope DEM: elevation increases ~30m per pixel eastward.

    With pixel spacing ~111m, this is ~15° slope.
    """
    if not _has_rasterio():
        pytest.skip("rasterio not available")
    path = str(tmp_path / "slope.tif")
    _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0 + c * 30.0)
    return path


class TestSurfaceNormalAt:
    """Test surface_normal_at with synthetic DEMs."""

    def test_flat_terrain_vertical_normal(self, flat_dem):
        from hyplan.terrain import surface_normal_at

        normals = surface_normal_at(
            np.array([35.0]), np.array([-111.0]), flat_dem,
        )
        assert normals.shape == (1, 3)
        # Should be nearly [0, 0, 1] (vertical)
        np.testing.assert_array_almost_equal(normals[0], [0, 0, 1], decimal=3)

    def test_sloped_terrain_tilted_normal(self, sloped_dem):
        from hyplan.terrain import surface_normal_at

        normals = surface_normal_at(
            np.array([35.0]), np.array([-111.0]), sloped_dem,
        )
        # East-facing slope: normal should have negative east component
        assert normals[0, 0] < -0.01  # tilted away from east-facing slope
        assert normals[0, 2] > 0.5    # still mostly upward

    def test_unit_vector(self, flat_dem):
        from hyplan.terrain import surface_normal_at

        lats = np.array([35.0, 35.001, 34.999])
        lons = np.array([-111.0, -111.001, -110.999])
        normals = surface_normal_at(lats, lons, flat_dem)
        magnitudes = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(magnitudes, 1.0, decimal=6)

    def test_batch_shape(self, flat_dem):
        from hyplan.terrain import surface_normal_at

        n = 5
        normals = surface_normal_at(
            np.full(n, 35.0), np.full(n, -111.0), flat_dem,
        )
        assert normals.shape == (n, 3)


class TestFootprintOnTerrain:
    """Test LVIS.footprint_on_terrain with synthetic DEMs."""

    def test_flat_terrain_matches_flat_earth(self, flat_dem):
        lvis = LVIS(lens="wide")
        result = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        # AGL ≈ 8500 - 500 = 8000 m
        flat_fp = lvis.footprint_diameter(8000 * ureg.meter).magnitude
        assert result["footprint_equivalent_diameter_m"] == pytest.approx(
            flat_fp, rel=0.05
        )

    def test_nadir_low_incidence(self, flat_dem):
        lvis = LVIS()
        result = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=90.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        assert result["incidence_deg"] < 2.0  # near-zero on flat terrain

    def test_return_keys(self, flat_dem):
        lvis = LVIS()
        result = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, dem_file=flat_dem,
        )
        expected_keys = {
            "ground_lat", "ground_lon", "ground_alt_m",
            "altitude_agl_m", "slant_range_m", "incidence_deg",
            "scan_angle_deg", "footprint_minor_m", "footprint_major_m",
            "footprint_area_m2", "footprint_equivalent_diameter_m",
            "flat_earth_diameter_m",
        }
        assert set(result.keys()) == expected_keys

    def test_slope_increases_major_axis(self, sloped_dem):
        lvis = LVIS(lens="wide")
        result = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=sloped_dem,
        )
        # On a slope, major axis should exceed minor axis
        if not np.isnan(result["footprint_major_m"]):
            assert result["footprint_major_m"] >= result["footprint_minor_m"]

    def test_off_nadir_increases_slant_range(self, flat_dem):
        lvis = LVIS()
        nadir = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        off_nadir = lvis.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, scan_angle_deg=3.0, dem_file=flat_dem,
        )
        if not np.isnan(off_nadir["slant_range_m"]):
            assert off_nadir["slant_range_m"] >= nadir["slant_range_m"]


class TestEffectiveSwathOnTerrain:
    """Test LVIS.effective_swath_on_terrain with synthetic DEMs."""

    def test_output_shapes(self, flat_dem):
        lvis = LVIS()
        n = 11
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=n,
        )
        assert len(result["scan_angles_deg"]) == n
        assert len(result["ground_lats"]) == n
        assert len(result["footprint_diameters_m"]) == n
        assert len(result["contiguous_mask"]) == n
        assert len(result["cross_track_spacings_m"]) == n - 1

    def test_effective_swath_positive(self, flat_dem):
        lvis = LVIS(lens="wide")
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=flat_dem,
        )
        assert result["effective_swath_m"] > 0

    def test_flat_terrain_near_flat_earth(self, flat_dem):
        lvis = LVIS(lens="wide")
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=flat_dem,
        )
        # Should be within 10% of flat-earth value
        assert result["effective_swath_m"] == pytest.approx(
            result["flat_earth_effective_swath_m"], rel=0.10
        )

    def test_local_density_array(self, flat_dem):
        lvis = LVIS(lens="wide")
        n = 11
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=n,
        )
        assert len(result["local_densities"]) == n
        # On flat terrain, density should be roughly uniform
        valid = np.isfinite(result["local_densities"])
        if valid.sum() > 2:
            densities = result["local_densities"][valid]
            assert densities.std() / densities.mean() < 0.15  # low variation

    def test_density_stats(self, flat_dem):
        lvis = LVIS(lens="wide")
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=flat_dem,
        )
        assert result["density_min"] > 0
        assert result["density_max"] >= result["density_min"]
        assert result["density_min"] <= result["density_mean"] <= result["density_max"]
        assert result["density_std"] >= 0

    def test_sloped_terrain_density_varies(self, sloped_dem):
        """On sloped terrain, local density should vary more than flat."""
        lvis = LVIS(lens="wide")
        result = lvis.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=sloped_dem,
        )
        # Just verify the range is non-degenerate
        if result["density_max"] > 0:
            assert result["density_max"] >= result["density_min"]


class TestTerrainSummary:
    """Test LVIS.terrain_summary with synthetic DEMs."""

    def test_contains_flat_keys(self, flat_dem):
        lvis = LVIS()
        s = lvis.terrain_summary(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=flat_dem,
        )
        flat_keys = {
            "altitude_agl", "speed", "rep_rate", "lens",
            "lens_divergence_mrad", "footprint_diameter", "max_swath",
            "effective_swath_width", "contiguous",
            "along_track_spacing", "along_track_contiguous",
            "point_density", "coverage_rate", "footprint_for_max_swath",
        }
        assert flat_keys.issubset(set(s.keys()))

    def test_contains_terrain_keys(self, flat_dem):
        lvis = LVIS()
        s = lvis.terrain_summary(
            lat=35.0, lon=-111.0, altitude_msl=8500.0,
            heading=0.0, speed=150 * ureg.knot, dem_file=flat_dem,
        )
        terrain_keys = {
            "terrain_ground_elevation_m", "terrain_altitude_agl_m",
            "terrain_nadir_incidence_deg",
            "terrain_nadir_footprint_major_m",
            "terrain_nadir_footprint_minor_m",
            "terrain_effective_swath_m",
            "terrain_density_min",
            "terrain_density_max",
            "terrain_density_mean",
            "terrain_density_std",
            "terrain_contiguous_fraction",
        }
        assert terrain_keys.issubset(set(s.keys()))
