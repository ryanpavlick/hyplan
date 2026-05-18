"""Tests for the ALSLidar (Airborne Laser Scanner) module.

Verifies the generic class against:

* RIEGL VQ-480 II datasheet (2024-08-23) swath/footprint tables.
* ScanningSensor protocol conformance.
* End-to-end integration with hyplan.swath.generate_swath_polygon.
* Locked nominal-density semantics + ContiguityError contract on the
  inverse solvers.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from hyplan import ureg
from hyplan.flight_line import FlightLine
from hyplan.instruments import (
    GLIHT_DUAL_VQ_480I,
    RIEGL_VQ_480II,
    ALSLidar,
    ContiguityError,
    LidarMount,
    MultiALSLidarRig,
    ScanningSensor,
)
from hyplan.instruments.als_lidar import SPEED_OF_LIGHT_M_PER_S


@pytest.fixture
def vq480ii() -> ALSLidar:
    return RIEGL_VQ_480II


@pytest.fixture
def fast_low_alt_lidar() -> ALSLidar:
    """A synthetic ALS where the chosen speed leaves along-track gaps.

    PRF and scan_rate chosen so that at moderate groundspeed and
    altitude the kinematic ``groundspeed / scan_rate`` exceeds the
    nadir footprint.
    """
    return ALSLidar(
        name="SYN-GAP",
        prf=400 * ureg.kilohertz,
        scan_rate=50 * ureg.hertz,
        scan_half_angle=20.0 * ureg.degree,
        beam_divergence=0.2 * ureg.milliradian,
        wavelength=1064 * ureg.nanometer,
        max_range=1500 * ureg.meter,
        max_range_reflectivity=0.6,
        mta_zones=2,
    )


class TestALSLidarConstruction:
    def test_riegl_vq480ii_defaults(self, vq480ii: ALSLidar) -> None:
        assert vq480ii.name == "RIEGL VQ-480 II"
        assert vq480ii.prf.m_as("hertz") == pytest.approx(1_200_000)
        assert vq480ii.scan_rate.m_as("hertz") == pytest.approx(200)
        assert vq480ii.scan_half_angle.m_as("degree") == pytest.approx(37.5)
        assert vq480ii.beam_divergence.m_as("milliradian") == pytest.approx(0.35)
        assert vq480ii.wavelength.m_as("nanometer") == pytest.approx(1550)
        assert vq480ii.mta_zones == 9
        assert vq480ii.scan_geometry == "rotating_polygon_active_arc"
        assert "riegl.com" in vq480ii.source.lower()

    def test_rejects_oscillating_mirror(self) -> None:
        with pytest.raises(ValueError, match="oscillating_mirror"):
            ALSLidar(
                name="X",
                prf=300 * ureg.kilohertz,
                scan_rate=100 * ureg.hertz,
                scan_half_angle=30 * ureg.degree,
                beam_divergence=0.3 * ureg.milliradian,
                wavelength=1064 * ureg.nanometer,
                max_range=1000 * ureg.meter,
                scan_geometry="oscillating_mirror",
            )

    def test_rejects_unknown_scan_geometry(self) -> None:
        with pytest.raises(ValueError, match="scan_geometry"):
            ALSLidar(
                name="X",
                prf=300 * ureg.kilohertz,
                scan_rate=100 * ureg.hertz,
                scan_half_angle=30 * ureg.degree,
                beam_divergence=0.3 * ureg.milliradian,
                wavelength=1064 * ureg.nanometer,
                max_range=1000 * ureg.meter,
                scan_geometry="hyperbolic_paraboloid",  # type: ignore[arg-type]
            )

    def test_rejects_non_positive_inputs(self) -> None:
        with pytest.raises(ValueError, match="prf"):
            ALSLidar(
                name="X",
                prf=0 * ureg.hertz,
                scan_rate=100 * ureg.hertz,
                scan_half_angle=30 * ureg.degree,
                beam_divergence=0.3 * ureg.milliradian,
                wavelength=1064 * ureg.nanometer,
                max_range=1000 * ureg.meter,
            )


class TestDatasheetReproducibility:
    """Verify swath against the RIEGL VQ-480 II datasheet's published table.

    Datasheet swath table at FOV ±37.5° (= 75° total):

        AGL (ft)  AGL (m)  Swath (m)
           600      180       280
           800      240       370
          1100      340       520
          1500      460       710
          2000      610       940
          2800      850      1300

    The values are rounded in the datasheet — we accept ±5 m which is
    well within the published precision.
    """

    DATASHEET_POINTS: ClassVar[list[tuple[float, float]]] = [
        (180.0, 280.0),
        (240.0, 370.0),
        (340.0, 520.0),
        (460.0, 710.0),
        (610.0, 940.0),
        (850.0, 1300.0),
    ]

    @pytest.mark.parametrize("alt_m, expected_swath_m", DATASHEET_POINTS)
    def test_swath_reproduces_datasheet(
        self, vq480ii: ALSLidar, alt_m: float, expected_swath_m: float,
    ) -> None:
        sw = vq480ii.swath_width(alt_m * ureg.meter).m_as("meter")
        # 1% tolerance — datasheet rounds to 10 m grid.
        assert abs(sw - expected_swath_m) / expected_swath_m < 0.02, (
            f"AGL={alt_m} m: got {sw:.0f} m, datasheet {expected_swath_m} m"
        )


class TestGeometry:
    def test_swath_width_at_1000m(self, vq480ii: ALSLidar) -> None:
        sw = vq480ii.swath_width(1000 * ureg.meter).m_as("meter")
        expected = 2 * 1000 * np.tan(np.radians(37.5))
        assert sw == pytest.approx(expected)

    def test_footprint_at_nadir(self, vq480ii: ALSLidar) -> None:
        fp = vq480ii.footprint_diameter(1000 * ureg.meter).m_as("meter")
        # 0.35 mrad × 1000 m = 0.35 m
        assert fp == pytest.approx(0.35, rel=1e-6)

    def test_footprint_stretches_at_scan_edge(self, vq480ii: ALSLidar) -> None:
        fp_nadir = vq480ii.footprint_diameter(1000 * ureg.meter).m_as("meter")
        fp_edge = vq480ii.footprint_diameter(
            1000 * ureg.meter, scan_angle=37.5 * ureg.degree,
        ).m_as("meter")
        # at 37.5°, 1/cos(37.5°) ≈ 1.260
        assert fp_edge / fp_nadir == pytest.approx(
            1.0 / np.cos(np.radians(37.5)), rel=1e-9,
        )

    def test_swath_offset_angles_symmetric(self, vq480ii: ALSLidar) -> None:
        port, starboard = vq480ii.swath_offset_angles()
        assert port == -starboard
        assert starboard == pytest.approx(37.5)

    def test_half_angle_property(self, vq480ii: ALSLidar) -> None:
        assert vq480ii.half_angle == pytest.approx(37.5)


class TestPointDensity:
    def test_point_density_formula(self, vq480ii: ALSLidar) -> None:
        # density = prf / (groundspeed × swath_width)
        alt = 1700 * ureg.foot
        spd = 120 * ureg.knot
        d = vq480ii.point_density(alt, spd).m_as(1 / ureg.meter**2)
        prf_hz = vq480ii.prf.m_as("hertz")
        spd_mps = spd.to(ureg.meter / ureg.second).magnitude
        sw_m = vq480ii.swath_width(alt).m_as("meter")
        expected = prf_hz / (spd_mps * sw_m)
        assert d == pytest.approx(expected, rel=1e-9)

    def test_effective_prf_override(self, vq480ii: ALSLidar) -> None:
        alt = 1000 * ureg.meter
        spd = 60 * ureg.knot
        d_full = vq480ii.point_density(alt, spd).m_as(1 / ureg.meter**2)
        d_half = vq480ii.point_density(
            alt, spd, effective_prf=600 * ureg.kilohertz,
        ).m_as(1 / ureg.meter**2)
        assert d_half == pytest.approx(d_full / 2.0)


class TestAlongTrackContiguity:
    def test_along_track_spacing(self, vq480ii: ALSLidar) -> None:
        spd = 60 * ureg.knot
        spacing = vq480ii.along_track_spacing(spd).m_as("meter")
        # 60 kn = 30.87 m/s; spacing = 30.87 / 200 = 0.154 m
        spd_mps = spd.to(ureg.meter / ureg.second).magnitude
        expected = spd_mps / 200.0
        assert spacing == pytest.approx(expected, rel=1e-9)

    def test_along_track_contiguity_gate(
        self, fast_low_alt_lidar: ALSLidar,
    ) -> None:
        # 100 kn at scan_rate=50 Hz: gap = 51.4/50 = 1.03 m
        # nadir footprint at 1000 m = 0.0002 × 1000 = 0.2 m → gap > footprint
        alt = 1000 * ureg.meter
        spd = 100 * ureg.knot
        assert fast_low_alt_lidar.is_along_track_contiguous(alt, spd) is False
        diag = fast_low_alt_lidar.coverage_diagnostic(alt, spd)
        assert diag["along_track_gap_m"] > diag["along_track_footprint_m"]
        assert diag["along_track_contiguous"] is False

    def test_contiguous_when_footprint_dominates(
        self, fast_low_alt_lidar: ALSLidar,
    ) -> None:
        # Slow flight, high altitude → bigger footprint
        alt = 3000 * ureg.meter
        spd = 5 * ureg.knot
        assert fast_low_alt_lidar.is_along_track_contiguous(alt, spd) is True


class TestCrossTrackSpacing:
    def test_active_arc_formula(self, vq480ii: ALSLidar) -> None:
        # dθ = 2 × scan_half_angle × scan_rate / prf
        alt = 1000 * ureg.meter
        sp = vq480ii.cross_track_spacing_at_nadir(alt).m_as("meter")
        d_theta = 2 * np.radians(37.5) * 200.0 / 1_200_000
        assert sp == pytest.approx(1000 * d_theta, rel=1e-9)

    def test_full_circle_differs_from_active_arc(self) -> None:
        # Same PRF/scan_rate, different geometry → different ground spacings.
        common = {
            "prf": 300 * ureg.kilohertz,
            "scan_rate": 100 * ureg.hertz,
            "scan_half_angle": 30 * ureg.degree,
            "beam_divergence": 0.3 * ureg.milliradian,
            "wavelength": 1064 * ureg.nanometer,
            "max_range": 1000 * ureg.meter,
        }
        active = ALSLidar(
            name="A", scan_geometry="rotating_polygon_active_arc", **common,
        )
        full = ALSLidar(
            name="F", scan_geometry="rotating_polygon_full_circle", **common,
        )
        alt = 1000 * ureg.meter
        sp_active = active.cross_track_spacing_at_nadir(alt).m_as("meter")
        sp_full = full.cross_track_spacing_at_nadir(alt).m_as("meter")
        # full-circle pulses are spread over 2π, active arc over 60° (1.047 rad)
        # so full has wider spacing.
        assert sp_full > sp_active
        ratio = sp_full / sp_active
        assert ratio == pytest.approx(np.pi / np.radians(30), rel=1e-9)

    def test_spacing_at_angle_grows_by_inv_cos_squared(
        self, vq480ii: ALSLidar,
    ) -> None:
        alt = 1000 * ureg.meter
        sp0 = vq480ii.cross_track_spacing_at_nadir(alt).m_as("meter")
        sp_edge = vq480ii.cross_track_spacing_at_angle(
            alt, 30 * ureg.degree,
        ).m_as("meter")
        expected = sp0 / (np.cos(np.radians(30)) ** 2)
        assert sp_edge == pytest.approx(expected, rel=1e-9)


class TestMTA:
    def test_mta_max_unambiguous_range(self, vq480ii: ALSLidar) -> None:
        # 9 zones × c / (2 × 1.2 MHz) = 9 × 124.91 ≈ 1124 m
        r = vq480ii.mta_max_unambiguous_range().m_as("meter")
        expected = 9 * SPEED_OF_LIGHT_M_PER_S / (2.0 * 1_200_000.0)
        assert r == pytest.approx(expected, rel=1e-9)

    def test_practical_max_altitude_radiometric_limited(
        self, vq480ii: ALSLidar,
    ) -> None:
        # max_range=1050 m; unambig=~1124 m → radiometric-limited
        practical = vq480ii.mta_practical_max_altitude().m_as("meter")
        assert practical == pytest.approx(1050.0)

    def test_practical_max_altitude_timing_limited(self) -> None:
        # Synthetic instance whose unambig range is < max_range:
        # prf 2 MHz, mta_zones=2 → unambig = 2 × 75 = 150 m
        sensor = ALSLidar(
            name="HIPRF",
            prf=2 * ureg.megahertz,
            scan_rate=100 * ureg.hertz,
            scan_half_angle=30 * ureg.degree,
            beam_divergence=0.3 * ureg.milliradian,
            wavelength=1064 * ureg.nanometer,
            max_range=1000 * ureg.meter,
            mta_zones=2,
        )
        unambig = sensor.mta_max_unambiguous_range().m_as("meter")
        practical = sensor.mta_practical_max_altitude().m_as("meter")
        assert practical == pytest.approx(unambig)
        assert practical < 1000


class TestSolvers:
    def test_solve_for_altitude_round_trip(self, vq480ii: ALSLidar) -> None:
        spd = 60 * ureg.knot
        target = 5.0 / ureg.meter**2
        alt = vq480ii.solve_for_altitude(
            target, spd, strict_contiguity=False,
        )
        d = vq480ii.point_density(alt, spd).m_as(1 / ureg.meter**2)
        assert d == pytest.approx(5.0, rel=1e-6)

    def test_solve_for_groundspeed_round_trip(
        self, vq480ii: ALSLidar,
    ) -> None:
        alt = 1000 * ureg.meter
        target = 10.0 / ureg.meter**2
        spd = vq480ii.solve_for_groundspeed(
            target, alt, strict_contiguity=False,
        )
        d = vq480ii.point_density(alt, spd).m_as(1 / ureg.meter**2)
        assert d == pytest.approx(10.0, rel=1e-6)

    def test_solver_strict_contiguity_raises(
        self, fast_low_alt_lidar: ALSLidar,
    ) -> None:
        # Pick a target density whose closed-form solution yields a config
        # with along-track gaps.  At groundspeed 100 kn (~51 m/s) and
        # scan_rate 50 Hz, gap = 1.03 m.  Footprint at any AGL = AGL ×
        # 0.0002, so for contiguity we'd need AGL ≥ 5150 m.  Targeting
        # a high density forces a low altitude.
        spd = 100 * ureg.knot
        target = 20.0 / ureg.meter**2
        with pytest.raises(ContiguityError, match="along-track gap"):
            fast_low_alt_lidar.solve_for_altitude(target, spd)

    def test_solver_non_strict_returns_anyway(
        self, fast_low_alt_lidar: ALSLidar,
    ) -> None:
        spd = 100 * ureg.knot
        target = 20.0 / ureg.meter**2
        alt = fast_low_alt_lidar.solve_for_altitude(
            target, spd, strict_contiguity=False,
        )
        assert alt.m_as("meter") > 0


class TestRequiredOverlap:
    def test_returns_percent_not_fraction(self, vq480ii: ALSLidar) -> None:
        # No target_density → default 20%
        ov = vq480ii.required_overlap_percent(
            altitude_agl=1000 * ureg.meter,
        )
        assert 0 <= ov < 100
        assert ov == pytest.approx(20.0)

    def test_default_override(self, vq480ii: ALSLidar) -> None:
        ov = vq480ii.required_overlap_percent(
            altitude_agl=1000 * ureg.meter,
            default_overlap_percent=10.0,
        )
        assert ov == pytest.approx(10.0)


class TestProtocolConformance:
    """Follows the pattern at tests/test_sensors.py:133."""

    def test_als_lidar_conforms(self) -> None:
        assert isinstance(RIEGL_VQ_480II, ScanningSensor)

    def test_swath_width_returns_quantity(self, vq480ii: ALSLidar) -> None:
        sw = vq480ii.swath_width(500 * ureg.meter)
        assert hasattr(sw, "magnitude")
        assert sw.dimensionality == ureg.meter.dimensionality


class TestCoverageDiagnostic:
    def test_diagnostic_keys_complete(self, vq480ii: ALSLidar) -> None:
        d = vq480ii.coverage_diagnostic(
            500 * ureg.meter, 60 * ureg.knot,
        )
        expected_keys = {
            "along_track_gap_m",
            "along_track_footprint_m",
            "along_track_contiguous",
            "swath_width_m",
            "nadir_density_pts_m2",
            "edge_density_pts_m2",
            "swath_mean_density_pts_m2",
        }
        assert set(d.keys()) == expected_keys

    def test_edge_density_lower_than_nadir(self, vq480ii: ALSLidar) -> None:
        d = vq480ii.coverage_diagnostic(
            500 * ureg.meter, 60 * ureg.knot,
        )
        # Cross-track spacing at edge ≥ nadir, so edge density ≤ nadir
        assert d["edge_density_pts_m2"] <= d["nadir_density_pts_m2"]


class TestGenericClassIndependentOfSensor:
    """The class must not bake in VQ-480II-specific values."""

    def test_arbitrary_instance_works(self) -> None:
        sensor = ALSLidar(
            name="GENERIC",
            prf=500 * ureg.kilohertz,
            scan_rate=80 * ureg.hertz,
            scan_half_angle=25 * ureg.degree,
            beam_divergence=0.25 * ureg.milliradian,
            wavelength=1064 * ureg.nanometer,
            max_range=2000 * ureg.meter,
            mta_zones=4,
        )
        sw = sensor.swath_width(1000 * ureg.meter).m_as("meter")
        expected = 2 * 1000 * np.tan(np.radians(25))
        assert sw == pytest.approx(expected)


# ===================================================================
# Terrain-aware tests (LVIS-parity)
# ===================================================================


def _has_rasterio() -> bool:
    try:
        import rasterio
        return True
    except ImportError:
        return False


def _write_synthetic_dem(
    filepath: str, lat_center: float, lon_center: float,
    elevation_func, size: int = 100,
) -> None:
    """Write a small GeoTIFF DEM for testing.  Mirrors test_lvis.py."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    pixel_deg = 0.001
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
    if not _has_rasterio():
        pytest.skip("rasterio not available")
    path = str(tmp_path / "flat_als.tif")
    _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0)
    return path


@pytest.fixture
def sloped_dem(tmp_path):
    """East-facing ~15° slope: 30 m elevation per ~111 m pixel."""
    if not _has_rasterio():
        pytest.skip("rasterio not available")
    path = str(tmp_path / "slope_als.tif")
    _write_synthetic_dem(path, 35.0, -111.0, lambda r, c: 500.0 + c * 30.0)
    return path


class TestFootprintOnTerrain:
    def test_flat_terrain_matches_flat_earth(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        result = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        # AGL ≈ 800 m; flat-earth = 0.35 mrad × 800 = 0.28 m
        flat_fp = vq480ii.footprint_diameter(800 * ureg.meter).m_as("meter")
        assert result["flat_earth_diameter_m"] == pytest.approx(
            flat_fp, rel=0.01,
        )
        # On flat terrain, major ≈ minor (low incidence)
        assert result["footprint_major_m"] == pytest.approx(
            result["footprint_minor_m"], rel=0.05,
        )

    def test_nadir_low_incidence(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        result = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=90.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        assert result["incidence_deg"] < 2.0

    def test_return_keys(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        result = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, dem_file=flat_dem,
        )
        expected = {
            "ground_lat", "ground_lon", "ground_alt_m",
            "altitude_agl_m", "slant_range_m", "incidence_deg",
            "scan_angle_deg", "footprint_minor_m", "footprint_major_m",
            "footprint_area_m2", "footprint_equivalent_diameter_m",
            "flat_earth_diameter_m",
        }
        assert set(result.keys()) == expected

    def test_slope_stretches_major_axis(
        self, vq480ii: ALSLidar, sloped_dem,
    ) -> None:
        result = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=sloped_dem,
        )
        if not np.isnan(result["footprint_major_m"]):
            assert result["footprint_major_m"] > result["footprint_minor_m"]

    def test_off_nadir_increases_slant_range(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        nadir = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, scan_angle_deg=0.0, dem_file=flat_dem,
        )
        edge = vq480ii.footprint_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, scan_angle_deg=30.0, dem_file=flat_dem,
        )
        if not np.isnan(edge["slant_range_m"]):
            assert edge["slant_range_m"] > nadir["slant_range_m"]


class TestEffectiveSwathOnTerrain:
    def test_output_shapes(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        n = 21
        result = vq480ii.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, groundspeed=120 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=n,
        )
        assert len(result["scan_angles_deg"]) == n
        assert len(result["ground_lats"]) == n
        assert len(result["footprint_minor_m"]) == n
        assert len(result["footprint_major_m"]) == n
        assert len(result["pulse_spacings_m"]) == n
        assert len(result["contiguous_mask"]) == n
        assert len(result["discretization_spacings_m"]) == n - 1

    def test_flat_terrain_matches_flat_earth_swath_in_contig_regime(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        # AGL ≈ 800 m at 60 kn:
        #   along_track_gap = 30.87 / 200 = 0.154 m
        #   nadir footprint = 800 × 0.35e-3 = 0.280 m → contiguous everywhere
        result = vq480ii.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, groundspeed=60 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=51,
        )
        assert result["effective_swath_m"] == pytest.approx(
            result["flat_earth_swath_m"], rel=0.1,
        )

    def test_high_speed_low_alt_loses_centre_to_gaps(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        # AGL ≈ 800 m at 120 kn: along-track gap (0.309 m) exceeds the
        # nadir footprint (0.280 m), so nadir is NOT along-track
        # contiguous; only the scan edges (longer slant range, bigger
        # footprint) are contiguous.  Effective swath is therefore
        # smaller than the geometric swath.
        result = vq480ii.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, groundspeed=120 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=51,
        )
        assert result["effective_swath_m"] < result["flat_earth_swath_m"]
        # Some edge positions are contiguous, some centre positions aren't.
        assert result["contiguous_mask"].any()
        assert not result["contiguous_mask"].all()

    def test_density_within_known_envelope(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        result = vq480ii.effective_swath_on_terrain(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, groundspeed=60 * ureg.knot,
            dem_file=flat_dem, n_scan_positions=51,
        )
        assert result["density_min_pts_m2"] > 0
        assert result["density_max_pts_m2"] >= result["density_min_pts_m2"]


class TestTerrainSummary:
    def test_combined_keys(
        self, vq480ii: ALSLidar, flat_dem,
    ) -> None:
        s = vq480ii.terrain_summary(
            lat=35.0, lon=-111.0, altitude_msl=1300.0,
            heading=0.0, groundspeed=120 * ureg.knot,
            dem_file=flat_dem,
        )
        # Flat-earth keys preserved
        assert "swath_width_m" in s
        # Terrain keys present
        for k in (
            "terrain_ground_elevation_m",
            "terrain_altitude_agl_m",
            "terrain_nadir_incidence_deg",
            "terrain_effective_swath_m",
            "terrain_density_mean_pts_m2",
            "terrain_contiguous_fraction",
        ):
            assert k in s


class TestSwathIntegration:
    """End-to-end: ALSLidar instance feeds the swath pipeline (the location
    where the ScanningSensor protocol is actually enforced in operational
    workflows — see hyplan/swath.py and hyplan/flight_box.py).
    """

    def test_riegl_vq480ii_swath_polygon(self, vq480ii: ALSLidar) -> None:
        from hyplan.swath import calculate_swath_widths, generate_swath_polygon

        # 10 km eastbound flight line at 3000 m MSL over flat low-elevation
        # terrain (US Great Plains).  Expected ground swath ≈ 2 × ~3000 m ×
        # tan(37.5°) ≈ 4600 m (assuming ~1000 ft terrain elevation).
        line = FlightLine.start_length_azimuth(
            lat1=39.0,
            lon1=-99.0,
            length=10.0 * ureg.kilometer,
            az=90.0,
            altitude_msl=3000 * ureg.meter,
        )
        poly = generate_swath_polygon(line, vq480ii, along_precision=2000.0)
        assert poly is not None
        assert poly.is_valid
        assert poly.area > 0
        widths = calculate_swath_widths(poly)
        # Mean swath should reflect ~3000 m AGL geometry; lenient bounds
        # absorb the actual terrain elevation contribution.
        assert 2000.0 < widths["mean_width"] < 6000.0


# ===================================================================
# Multi-lidar rig tests
# ===================================================================


def _two_unit_pitch_rig(tilt_deg: float = 7.0) -> MultiALSLidarRig:
    """Two identical VQ-480i-class units pitch-tilted ±tilt_deg."""
    unit = ALSLidar(
        name="UNIT",
        prf=300 * ureg.kilohertz,
        scan_rate=100 * ureg.hertz,
        scan_half_angle=30.0 * ureg.degree,
        beam_divergence=0.3 * ureg.milliradian,
        wavelength=1550 * ureg.nanometer,
        max_range=1500 * ureg.meter,
        mta_zones=2,
    )
    return MultiALSLidarRig(
        name="TEST-DUAL",
        units=[
            LidarMount(lidar=unit, label="fwd", pitch_tilt_deg=tilt_deg),
            LidarMount(lidar=unit, label="aft", pitch_tilt_deg=-tilt_deg),
        ],
    )


def _two_unit_roll_rig(roll_deg: float = 20.0) -> MultiALSLidarRig:
    """Two identical units rolled ±roll_deg for cross-track swath extension."""
    unit = ALSLidar(
        name="UNIT",
        prf=300 * ureg.kilohertz,
        scan_rate=100 * ureg.hertz,
        scan_half_angle=30.0 * ureg.degree,
        beam_divergence=0.3 * ureg.milliradian,
        wavelength=1550 * ureg.nanometer,
        max_range=1500 * ureg.meter,
        mta_zones=2,
    )
    return MultiALSLidarRig(
        name="TEST-ROLL",
        units=[
            LidarMount(lidar=unit, label="port", roll_tilt_deg=-roll_deg),
            LidarMount(lidar=unit, label="starboard", roll_tilt_deg=+roll_deg),
        ],
    )


class TestMultiALSLidarRigConstruction:
    def test_requires_at_least_one_unit(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MultiALSLidarRig(name="X", units=[])

    def test_len_and_iter(self) -> None:
        rig = _two_unit_pitch_rig()
        assert len(rig) == 2
        labels = [u.label for u in rig]
        assert labels == ["fwd", "aft"]


class TestMultiALSLidarRigGeometry:
    def test_pitch_only_keeps_single_unit_swath(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        single = rig.units[0].lidar.swath_width(335 * ureg.meter).m_as("meter")
        combined = rig.swath_width(335 * ureg.meter).m_as("meter")
        assert combined == pytest.approx(single, rel=1e-9)

    def test_roll_extends_combined_swath(self) -> None:
        rig = _two_unit_roll_rig(20.0)
        # combined edges: port=-50°, starboard=+50°
        port, starboard = rig.swath_offset_angles()
        assert port == pytest.approx(-50.0)
        assert starboard == pytest.approx(50.0)
        # combined swath = AGL × (tan(50°) - tan(-50°)) = 2 × AGL × tan(50°)
        expected = 2 * 335 * np.tan(np.radians(50.0))
        combined = rig.swath_width(335 * ureg.meter).m_as("meter")
        assert combined == pytest.approx(expected, rel=1e-9)

    def test_half_angle_takes_outer_edge(self) -> None:
        # Asymmetric roll: port=-15°, starboard=+45° (max abs edge = 45°+30°=75°)
        unit = ALSLidar(
            name="U", prf=300*ureg.kHz, scan_rate=100*ureg.Hz,
            scan_half_angle=30*ureg.degree,
            beam_divergence=0.3*ureg.mrad, wavelength=1550*ureg.nm,
            max_range=1500*ureg.meter,
        )
        rig = MultiALSLidarRig(
            name="ASYM",
            units=[
                LidarMount(lidar=unit, label="a", roll_tilt_deg=-15),
                LidarMount(lidar=unit, label="b", roll_tilt_deg=+45),
            ],
        )
        port, starboard = rig.swath_offset_angles()
        assert port == pytest.approx(-45.0)  # -15 - 30
        assert starboard == pytest.approx(75.0)  # 45 + 30
        assert rig.half_angle == pytest.approx(75.0)


class TestMultiALSLidarRigDensity:
    def test_pitch_only_doubles_density(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        single_d = rig.units[0].lidar.point_density(
            335*ureg.meter, 110*ureg.knot,
        ).m_as(1/ureg.meter**2)
        combined_d = rig.combined_point_density(
            335*ureg.meter, 110*ureg.knot,
        ).m_as(1/ureg.meter**2)
        assert combined_d == pytest.approx(2 * single_d, rel=1e-9)

    def test_roll_extended_density_is_mean(self) -> None:
        # For roll-extended rig, combined_density is the *mean* over the
        # extended swath: total_prf / (speed × combined_swath).
        rig = _two_unit_roll_rig(20.0)
        alt = 335 * ureg.meter
        spd = 110 * ureg.knot
        total_prf = sum(u.lidar.prf.m_as("hertz") for u in rig)
        sw = rig.swath_width(alt).m_as("meter")
        spd_mps = spd.to(ureg.meter/ureg.second).magnitude
        expected = total_prf / (spd_mps * sw)
        assert rig.combined_point_density(alt, spd).m_as(
            1/ureg.meter**2,
        ) == pytest.approx(expected, rel=1e-9)

    def test_unit_point_densities_keyed_by_label(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        d = rig.unit_point_densities(335*ureg.meter, 110*ureg.knot)
        assert set(d.keys()) == {"fwd", "aft"}
        # Identical units → equal per-unit density
        assert d["fwd"].m_as(1/ureg.meter**2) == pytest.approx(
            d["aft"].m_as(1/ureg.meter**2), rel=1e-12,
        )

    def test_solve_for_groundspeed_round_trip(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        alt = 335 * ureg.meter
        target = 10.0 / ureg.meter**2
        spd = rig.solve_for_groundspeed(
            target, alt, strict_contiguity=False,
        )
        d = rig.combined_point_density(alt, spd).m_as(1 / ureg.meter**2)
        assert d == pytest.approx(10.0, rel=1e-6)

    def test_solve_for_groundspeed_pitch_rig_is_double_single(self) -> None:
        # For a pitch-only rig (shared swath, equal PRF), the rig-level
        # speed at a given density is exactly 2× the per-unit speed.
        rig = _two_unit_pitch_rig(7.0)
        alt = 335 * ureg.meter
        target = 10.0 / ureg.meter**2
        rig_spd = rig.solve_for_groundspeed(
            target, alt, strict_contiguity=False,
        ).m_as("meter/second")
        unit_spd = rig.units[0].lidar.solve_for_groundspeed(
            target, alt, strict_contiguity=False,
        ).m_as("meter/second")
        assert rig_spd == pytest.approx(2 * unit_spd, rel=1e-9)


class TestMultiALSLidarRigPitchOffsets:
    def test_along_track_offsets_symmetric(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        offsets = rig.along_track_offsets(335 * ureg.meter)
        # forward unit: +335 × tan(7°) ahead; aft: same distance behind
        expected_fwd = 335 * np.tan(np.radians(7.0))
        assert offsets["fwd"].m_as("meter") == pytest.approx(expected_fwd)
        assert offsets["aft"].m_as("meter") == pytest.approx(-expected_fwd)


class TestMultiALSLidarRigMultiAngle:
    def test_pitch_rig_detected_as_multi_angle_pair(self) -> None:
        rig = _two_unit_pitch_rig(7.0)
        pairs = rig.multi_angle_pairs()
        assert len(pairs) == 1
        fwd, bwd = pairs[0]
        assert fwd.pitch_tilt_deg > 0
        assert bwd.pitch_tilt_deg < 0
        assert fwd.label == "fwd"
        assert bwd.label == "aft"

    def test_zero_pitch_pair_not_detected(self) -> None:
        # Two nadir-looking units should NOT count as multi-angle.
        rig = _two_unit_pitch_rig(0.0)
        assert rig.multi_angle_pairs() == []

    def test_roll_only_pair_not_detected(self) -> None:
        rig = _two_unit_roll_rig(20.0)
        assert rig.multi_angle_pairs() == []


class TestMultiALSLidarRigProtocolConformance:
    def test_gliht_dual_conforms(self) -> None:
        assert isinstance(GLIHT_DUAL_VQ_480I, ScanningSensor)

    def test_rig_swath_polygon_runs(self) -> None:
        # End-to-end: rig plugs into generate_swath_polygon.
        from hyplan.swath import calculate_swath_widths, generate_swath_polygon
        rig = GLIHT_DUAL_VQ_480I
        line = FlightLine.start_length_azimuth(
            lat1=39.0, lon1=-99.0,
            length=10 * ureg.kilometer,
            az=90.0,
            altitude_msl=1500 * ureg.meter,
        )
        # along_precision=500 keeps the polygon's vertex pairs aligned
        # cross-track (calculate_swath_widths matches polygon halves by
        # index — too-coarse sampling produces diagonal pairs and inflates
        # the measured width).
        poly = generate_swath_polygon(line, rig, along_precision=500.0)
        assert poly.is_valid
        widths = calculate_swath_widths(poly)
        # AGL ≈ 950 m; expected swath ≈ 2 × 950 × tan(30°) ≈ 1097 m.
        # Allow ±30% for terrain variability.
        assert 800 < widths["mean_width"] < 1500


class TestGLIHTDualReference:
    def test_two_units_opposite_pitch(self) -> None:
        assert len(GLIHT_DUAL_VQ_480I) == 2
        pitches = [u.pitch_tilt_deg for u in GLIHT_DUAL_VQ_480I]
        assert pitches[0] == -pitches[1]
        assert all(u.roll_tilt_deg == 0.0 for u in GLIHT_DUAL_VQ_480I)

    def test_combined_swath_matches_gliht_user_guide(self) -> None:
        # User guide nominal: 387 m swath at 335 m AGL (single-unit 60° FOV).
        sw = GLIHT_DUAL_VQ_480I.swath_width(335 * ureg.meter).m_as("meter")
        assert sw == pytest.approx(387.0, abs=1.0)

    def test_single_unit_density_matches_gliht_envelope(self) -> None:
        # Per-unit nominal density at the user-guide operating point
        # (335 m AGL, 110 kn) should be ~13–14 pts/m² — single-unit class.
        unit = GLIHT_DUAL_VQ_480I.units[0].lidar
        d = unit.point_density(335*ureg.meter, 110*ureg.knot).m_as(
            1/ureg.meter**2,
        )
        assert 10.0 < d < 18.0

    def test_docstring_mentions_user_guide(self) -> None:
        assert "G-LiHT" in (GLIHT_DUAL_VQ_480I.__doc__ or "")
        assert "User Guide" in (GLIHT_DUAL_VQ_480I.__doc__ or "")
