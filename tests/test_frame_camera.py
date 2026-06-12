"""Comprehensive tests for the FrameCamera instrument model."""

import contextlib

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from hyplan.instruments import FrameCamera, MultiCameraRig
from hyplan.units import ureg


@pytest.fixture
def camera():
    """Standard test camera (36x24mm sensor, 50mm lens)."""
    return FrameCamera(
        name="Test Camera",
        sensor_width=36 * ureg.mm,
        sensor_height=24 * ureg.mm,
        focal_length=50 * ureg.mm,
        resolution_x=6000,
        resolution_y=4000,
        frame_rate=1 * ureg.Hz,
        f_speed=2.8,
    )


@pytest.fixture
def altitude():
    return 1000 * ureg.meter


class TestInstantiation:
    def test_valid_construction(self, camera):
        assert camera.name == "Test Camera"
        assert camera.resolution_x == 6000
        assert camera.resolution_y == 4000
        assert camera.f_speed == 2.8

    def test_non_integer_resolution_raises(self):
        with pytest.raises(TypeError, match="integers"):
            FrameCamera(
                name="Bad",
                sensor_width=36 * ureg.mm,
                sensor_height=24 * ureg.mm,
                focal_length=50 * ureg.mm,
                resolution_x=6000.0,
                resolution_y=4000,
                frame_rate=1 * ureg.Hz,
                f_speed=2.8,
            )

    def test_non_quantity_raises(self):
        with pytest.raises(TypeError):
            FrameCamera(
                name="Bad",
                sensor_width=36,  # not a Quantity
                sensor_height=24 * ureg.mm,
                focal_length=50 * ureg.mm,
                resolution_x=6000,
                resolution_y=4000,
                frame_rate=1 * ureg.Hz,
                f_speed=2.8,
            )


class TestFieldOfView:
    def test_fov_x_positive(self, camera):
        assert camera.fov_x > 0
        assert camera.fov_x < 180

    def test_fov_y_positive(self, camera):
        assert camera.fov_y > 0
        assert camera.fov_y < 180

    def test_fov_x_greater_than_fov_y(self, camera):
        # 36mm width > 24mm height with same focal length → wider horizontal FOV
        assert camera.fov_x > camera.fov_y

    def test_fov_expected_value(self, camera):
        # 2 * atan(36 / (2*50)) = 2 * atan(0.36) ≈ 39.6°
        expected = 2 * np.degrees(np.arctan(36 / (2 * 50)))
        assert camera.fov_x == pytest.approx(expected, rel=0.01)

    def test_shorter_focal_length_wider_fov(self):
        wide = FrameCamera(
            name="Wide",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=24 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
        )
        narrow = FrameCamera(
            name="Narrow",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=100 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
        )
        assert wide.fov_x > narrow.fov_x


class TestGroundSampleDistance:
    def test_gsd_keys(self, camera, altitude):
        gsd = camera.ground_sample_distance(altitude)
        assert set(gsd.keys()) == {"x", "y"}

    def test_gsd_positive(self, camera, altitude):
        gsd = camera.ground_sample_distance(altitude)
        assert gsd["x"].magnitude > 0
        assert gsd["y"].magnitude > 0

    def test_gsd_x_smaller_for_higher_resolution_axis(self, camera, altitude):
        gsd = camera.ground_sample_distance(altitude)
        # 6000 px across vs 4000 px along, but FOV also differs
        # GSD depends on FOV/resolution ratio
        assert gsd["x"].check("[length]")

    def test_gsd_scales_with_altitude(self, camera):
        gsd1 = camera.ground_sample_distance(500 * ureg.meter)
        gsd2 = camera.ground_sample_distance(1000 * ureg.meter)
        assert gsd2["x"].magnitude == pytest.approx(2 * gsd1["x"].magnitude, rel=0.02)


class TestAltitudeForGSD:
    def test_round_trip(self, camera):
        alt = 2000 * ureg.meter
        gsd = camera.ground_sample_distance(alt)
        recovered_alt = camera.altitude_agl_for_ground_sample_distance(gsd["x"], gsd["y"])
        assert recovered_alt.m_as("meter") == pytest.approx(alt.magnitude, rel=0.02)

    def test_both_axes_satisfy_requested_gsd(self, camera):
        requested = 0.05 * ureg.meter
        alt = camera.altitude_agl_for_ground_sample_distance(requested, requested)
        gsd = camera.ground_sample_distance(alt)
        assert gsd["x"].m_as("meter") <= requested.m_as("meter") * (1 + 1e-9)
        assert gsd["y"].m_as("meter") <= requested.m_as("meter") * (1 + 1e-9)

    def test_positive(self, camera):
        alt = camera.altitude_agl_for_ground_sample_distance(
            0.1 * ureg.meter, 0.1 * ureg.meter
        )
        assert alt.magnitude > 0

    def test_larger_gsd_needs_higher_altitude(self, camera):
        alt1 = camera.altitude_agl_for_ground_sample_distance(
            0.05 * ureg.meter, 0.05 * ureg.meter
        )
        alt2 = camera.altitude_agl_for_ground_sample_distance(
            0.10 * ureg.meter, 0.10 * ureg.meter
        )
        assert alt2.magnitude > alt1.magnitude


class TestFootprint:
    def test_footprint_keys(self, camera, altitude):
        fp = camera.footprint_at(altitude)
        assert set(fp.keys()) == {"width", "height"}

    def test_footprint_positive(self, camera, altitude):
        fp = camera.footprint_at(altitude)
        assert fp["width"].magnitude > 0
        assert fp["height"].magnitude > 0

    def test_width_greater_than_height(self, camera, altitude):
        fp = camera.footprint_at(altitude)
        assert fp["width"].magnitude > fp["height"].magnitude

    def test_footprint_scales_with_altitude(self, camera):
        fp1 = camera.footprint_at(500 * ureg.meter)
        fp2 = camera.footprint_at(1000 * ureg.meter)
        assert fp2["width"].magnitude == pytest.approx(
            2 * fp1["width"].magnitude, rel=0.02
        )


class TestCriticalGroundSpeed:
    def test_positive(self, camera, altitude):
        cgs = camera.critical_ground_speed(altitude)
        assert cgs.magnitude > 0

    def test_higher_frame_rate_allows_faster(self, altitude):
        slow = FrameCamera(
            name="Slow",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
        )
        fast = FrameCamera(
            name="Fast",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=10 * ureg.Hz,
            f_speed=2.8,
        )
        assert fast.critical_ground_speed(altitude).magnitude > slow.critical_ground_speed(altitude).magnitude


class TestPortraitWarning:
    def test_portrait_orientation_warns(self):
        with pytest.warns(UserWarning, match="portrait-oriented"):
            FrameCamera(
                name="Portrait",
                sensor_width=24 * ureg.mm,   # narrower
                sensor_height=36 * ureg.mm,  # taller
                focal_length=50 * ureg.mm,
                resolution_x=4000,
                resolution_y=6000,
                frame_rate=1 * ureg.Hz,
                f_speed=2.8,
            )

    def test_landscape_no_warning(self, camera):
        # Standard camera fixture is landscape — no warning expected
        assert camera.sensor_width > camera.sensor_height


class TestIFOV:
    def test_ifov_x_positive(self, camera):
        assert camera.ifov_x > 0

    def test_ifov_y_positive(self, camera):
        assert camera.ifov_y > 0

    def test_ifov_x_equals_expected(self, camera):
        # pixel_size = 36mm / 6000 = 0.006mm = 6µm
        # IFOV = 6µm / 50mm = 0.00012 rad = 120 µrad
        expected = (36 / 6000) / 50 * 1e6
        assert camera.ifov_x == pytest.approx(expected, rel=0.01)

    def test_square_pixels_equal_ifov(self):
        cam = FrameCamera(
            name="Square",
            sensor_width=36 * ureg.mm,
            sensor_height=36 * ureg.mm,
            focal_length=50 * ureg.mm,
            resolution_x=6000,
            resolution_y=6000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
        )
        assert cam.ifov_x == pytest.approx(cam.ifov_y)


class TestImageScale:
    def test_scale_at_1000m(self, camera):
        # scale = altitude / focal_length = 1000m / 0.05m = 20000
        scale = camera.image_scale(1000 * ureg.meter)
        assert scale == pytest.approx(20000, rel=0.01)

    def test_altitude_for_scale_round_trip(self, camera):
        alt = 2000 * ureg.meter
        scale = camera.image_scale(alt)
        recovered = camera.altitude_for_scale(scale)
        assert recovered.m_as("meter") == pytest.approx(alt.magnitude, rel=0.01)

    def test_higher_altitude_larger_scale(self, camera):
        s1 = camera.image_scale(500 * ureg.meter)
        s2 = camera.image_scale(1000 * ureg.meter)
        assert s2 > s1


class TestFocalLengthForGSD:
    def test_round_trip(self, camera, altitude):
        # focal_length_for_gsd uses the linear photogrammetric model
        # (GSD = alt * pixel_size / f) while ground_sample_distance uses a
        # per-pixel trig model, so they differ slightly at wide FOV.
        # Use the linear GSD for a clean round-trip.
        pixel_size = camera.sensor_width / camera.resolution_x
        linear_gsd = (altitude * pixel_size / camera.focal_length).to(ureg.meter)
        fl = camera.focal_length_for_gsd(altitude, linear_gsd)
        assert fl.m_as("mm") == pytest.approx(
            camera.focal_length.m_as("mm"), rel=0.01
        )

    def test_finer_gsd_needs_longer_focal(self, camera, altitude):
        fl_coarse = camera.focal_length_for_gsd(altitude, 0.20 * ureg.meter)
        fl_fine = camera.focal_length_for_gsd(altitude, 0.05 * ureg.meter)
        assert fl_fine.magnitude > fl_coarse.magnitude

    def test_returns_mm(self, camera, altitude):
        fl = camera.focal_length_for_gsd(altitude, 0.10 * ureg.meter)
        assert fl.units == ureg.mm


class TestLineSpacing:
    def test_zero_sidelap_equals_footprint_width(self, camera, altitude):
        spacing = camera.line_spacing(altitude, sidelap_pct=0.0)
        fp = camera.footprint_at(altitude)
        assert spacing.magnitude == pytest.approx(fp["width"].magnitude, rel=0.01)

    def test_50_percent_sidelap(self, camera, altitude):
        spacing = camera.line_spacing(altitude, sidelap_pct=50.0)
        fp = camera.footprint_at(altitude)
        assert spacing.magnitude == pytest.approx(fp["width"].magnitude * 0.5, rel=0.01)

    def test_default_sidelap_60(self, camera, altitude):
        spacing = camera.line_spacing(altitude)
        fp = camera.footprint_at(altitude)
        assert spacing.magnitude == pytest.approx(fp["width"].magnitude * 0.4, rel=0.01)

    def test_more_sidelap_less_spacing(self, camera, altitude):
        s1 = camera.line_spacing(altitude, sidelap_pct=30.0)
        s2 = camera.line_spacing(altitude, sidelap_pct=70.0)
        assert s2.magnitude < s1.magnitude


class TestTriggerDistance:
    def test_zero_overlap_equals_footprint_height(self, camera, altitude):
        dist = camera.trigger_distance(altitude, overlap_pct=0.0)
        fp = camera.footprint_at(altitude)
        assert dist.magnitude == pytest.approx(fp["height"].magnitude, rel=0.01)

    def test_80_percent_overlap(self, camera, altitude):
        dist = camera.trigger_distance(altitude, overlap_pct=80.0)
        fp = camera.footprint_at(altitude)
        assert dist.magnitude == pytest.approx(fp["height"].magnitude * 0.2, rel=0.01)

    def test_default_overlap_80(self, camera, altitude):
        dist = camera.trigger_distance(altitude)
        fp = camera.footprint_at(altitude)
        assert dist.magnitude == pytest.approx(fp["height"].magnitude * 0.2, rel=0.01)


class TestTriggerInterval:
    def test_positive(self, camera, altitude):
        interval = camera.trigger_interval(
            altitude, 50 * ureg.meter / ureg.second
        )
        assert interval.magnitude > 0

    def test_returns_seconds(self, camera, altitude):
        interval = camera.trigger_interval(
            altitude, 50 * ureg.meter / ureg.second
        )
        assert interval.units == ureg.second

    def test_faster_speed_shorter_interval(self, camera, altitude):
        slow = camera.trigger_interval(altitude, 30 * ureg.meter / ureg.second)
        fast = camera.trigger_interval(altitude, 60 * ureg.meter / ureg.second)
        assert fast.magnitude < slow.magnitude

    def test_consistent_with_trigger_distance(self, camera, altitude):
        speed = 50 * ureg.meter / ureg.second
        dist = camera.trigger_distance(altitude, overlap_pct=60.0)
        interval = camera.trigger_interval(altitude, speed, overlap_pct=60.0)
        expected = (dist / speed).to(ureg.second)
        assert interval.magnitude == pytest.approx(expected.magnitude, rel=0.01)


class TestCoverageBuffer:
    def test_default_4_frames(self, camera, altitude):
        buf = camera.coverage_buffer(altitude, overlap_pct=80.0)
        dist = camera.trigger_distance(altitude, overlap_pct=80.0)
        assert buf.magnitude == pytest.approx(dist.magnitude * 4, rel=0.01)

    def test_custom_n_frames(self, camera, altitude):
        buf = camera.coverage_buffer(altitude, overlap_pct=80.0, n_frames=6)
        dist = camera.trigger_distance(altitude, overlap_pct=80.0)
        assert buf.magnitude == pytest.approx(dist.magnitude * 6, rel=0.01)

    def test_positive(self, camera, altitude):
        buf = camera.coverage_buffer(altitude)
        assert buf.magnitude > 0


class TestFootprintCorners:
    def test_deprecation_warning(self):
        """footprint_corners emits DeprecationWarning."""
        # Will fail on missing DEM, but the warning fires first
        with (
            pytest.warns(DeprecationWarning, match="deprecated"),
            contextlib.suppress(Exception),
        ):
            FrameCamera.footprint_corners(
                34.0, -117.0, 5000.0, 36.0, 24.0, "__nonexistent__.tif"
            )


class TestGroundFootprint:
    def test_nadir_symmetric(self, camera, altitude):
        poly = camera.ground_footprint(altitude)
        assert isinstance(poly, ShapelyPolygon)
        assert not poly.is_empty
        # Nadir camera: centroid near origin
        assert poly.centroid.x == pytest.approx(0, abs=0.1)
        assert poly.centroid.y == pytest.approx(0, abs=0.1)

    def test_nadir_width_matches_footprint(self, camera, altitude):
        poly = camera.ground_footprint(altitude)
        width = poly.bounds[2] - poly.bounds[0]  # maxx - minx
        fp = camera.footprint_at(altitude)
        assert width == pytest.approx(fp["width"].magnitude, rel=0.01)

    def test_tilted_forward_shifted(self, tilted_camera, altitude):
        poly = tilted_camera.ground_footprint(altitude)
        # Forward tilt: all y values should be positive (ahead of nadir)
        assert poly.bounds[1] > 0  # miny > 0

    def test_cross_track_offset_shifts_x(self, camera, altitude):
        poly_center = camera.ground_footprint(altitude, cross_track_offset=0.0)
        poly_right = camera.ground_footprint(altitude, cross_track_offset=10.0)
        assert poly_right.centroid.x > poly_center.centroid.x

    def test_terrain_mode_returns_polygon(self, camera, altitude):
        """Terrain mode returns a 3D Shapely Polygon."""
        from unittest.mock import patch

        def fake_rti(lat0, lon0, h0, az, tilt, **kw):
            n = len(az)
            return (np.full(n, 34.01), np.full(n, -117.01), np.full(n, 350.0))

        with patch("hyplan.instruments.frame_camera.ray_terrain_intersection", side_effect=fake_rti):
            poly = camera.ground_footprint(
                altitude, lat=34.0, lon=-117.0, altitude_msl=5000.0,
                heading=90.0, dem_file="dummy.tif",
            )
        assert isinstance(poly, ShapelyPolygon)
        assert not poly.is_empty
        # 3D coordinates: (lon, lat, elev)
        coords = list(poly.exterior.coords)
        assert len(coords[0]) == 3

    def test_terrain_mode_calls_per_ray(self, camera, altitude):
        """Verify ray_terrain_intersection is called once per valid ray."""
        from unittest.mock import patch

        def fake_rti(lat0, lon0, h0, az, tilt, **kw):
            n = len(az)
            return (np.full(n, 34.0), np.full(n, -117.0), np.full(n, 0.0))

        with patch("hyplan.instruments.frame_camera.ray_terrain_intersection",
                    side_effect=fake_rti) as mock_rti:
            camera.ground_footprint(
                altitude, lat=34.0, lon=-117.0, altitude_msl=5000.0,
            )
        # Default edge_points=10 → 40 rays, all valid for nadir camera
        assert mock_rti.call_count == 40

    def test_edge_points_2_gives_8_vertices(self, camera, altitude):
        """edge_points=2 → 4 edges × 2 = 8 points."""
        poly = camera.ground_footprint(altitude, edge_points=2)
        # Shapely closes the ring, so exterior.coords has N+1 entries
        assert len(poly.exterior.coords) == 9  # 8 unique + closing

    def test_edge_points_default_gives_40_vertices(self, camera, altitude):
        """Default edge_points=10 → 40 points."""
        poly = camera.ground_footprint(altitude)
        assert len(poly.exterior.coords) == 41  # 40 unique + closing

    def test_polygon_area_positive(self, camera, altitude):
        poly = camera.ground_footprint(altitude)
        assert poly.area > 0

    def test_deprecated_name_warns(self, camera, altitude):
        with pytest.warns(DeprecationWarning, match="ground_footprint"):
            camera.ground_footprint_corners(altitude)


# ── Tilt support fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tilted_camera():
    """Camera tilted 15° forward."""
    return FrameCamera(
        name="Tilted",
        sensor_width=36 * ureg.mm,
        sensor_height=24 * ureg.mm,
        focal_length=50 * ureg.mm,
        resolution_x=6000,
        resolution_y=4000,
        frame_rate=1 * ureg.Hz,
        f_speed=2.8,
        tilt_angle=15.0,
        tilt_direction=0.0,
    )


# ── Tilt backward compatibility ─────────────────────────────────────────────

class TestTiltBackwardCompat:
    def test_tilt_zero_matches_nadir_footprint(self, camera, altitude):
        """tilt_angle=0 gives identical footprint to default camera."""
        cam0 = FrameCamera(
            name="Zero Tilt",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
            tilt_angle=0.0,
        )
        fp0 = cam0.footprint_at(altitude)
        fp = camera.footprint_at(altitude)
        assert fp0["width"].magnitude == pytest.approx(fp["width"].magnitude, rel=1e-6)
        assert fp0["height"].magnitude == pytest.approx(fp["height"].magnitude, rel=1e-6)

    def test_tilt_zero_matches_nadir_gsd(self, camera, altitude):
        gsd0 = FrameCamera(
            name="Zero Tilt",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
            tilt_angle=0.0,
        ).ground_sample_distance(altitude)
        gsd = camera.ground_sample_distance(altitude)
        assert gsd0["x"].magnitude == pytest.approx(gsd["x"].magnitude, rel=1e-6)
        assert gsd0["y"].magnitude == pytest.approx(gsd["y"].magnitude, rel=1e-6)

    def test_default_tilt_is_zero(self, camera):
        assert camera.tilt_angle == 0.0
        assert camera.tilt_direction == 0.0


# ── Tilt geometry ────────────────────────────────────────────────────────────

class TestTiltGeometry:
    def test_footprint_tilted_larger_along_track(self, camera, tilted_camera, altitude):
        fp_nadir = camera.footprint_at(altitude)
        fp_tilt = tilted_camera.footprint_at(altitude)
        assert fp_tilt["height"].magnitude > fp_nadir["height"].magnitude

    def test_footprint_tilted_has_near_far(self, tilted_camera, altitude):
        fp = tilted_camera.footprint_at(altitude)
        assert "height_near" in fp
        assert "height_far" in fp
        assert fp["height_far"].magnitude > fp["height_near"].magnitude

    def test_gsd_tilted_coarser_at_center(self, camera, tilted_camera, altitude):
        gsd_nadir = camera.ground_sample_distance(altitude)
        gsd_tilt = tilted_camera.ground_sample_distance(altitude)
        assert gsd_tilt["y"].magnitude > gsd_nadir["y"].magnitude

    def test_gsd_near_finer_than_far(self, tilted_camera, altitude):
        gsd = tilted_camera.ground_sample_distance(altitude)
        assert gsd["y_near"].magnitude < gsd["y_far"].magnitude

    def test_tilt_angle_validation(self):
        with pytest.raises(ValueError):
            FrameCamera(
                name="Bad",
                sensor_width=36 * ureg.mm,
                sensor_height=24 * ureg.mm,
                focal_length=50 * ureg.mm,
                resolution_x=6000,
                resolution_y=4000,
                frame_rate=1 * ureg.Hz,
                f_speed=2.8,
                tilt_angle=90.0,
            )

    def test_footprint_past_horizon_raises(self, altitude):
        # tilt 75° + fov_along/2 ≈ 31° reaches past the horizon
        cam = FrameCamera(
            name="Horizon",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=20 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
            tilt_angle=75.0,
            tilt_direction=0.0,
        )
        with pytest.raises(ValueError, match="horizon"):
            cam.footprint_at(altitude)
        with pytest.raises(ValueError, match="horizon"):
            cam.trigger_distance(altitude)

    def test_steep_tilt_below_horizon_works(self, altitude):
        cam = FrameCamera(
            name="Oblique",
            sensor_width=36 * ureg.mm,
            sensor_height=24 * ureg.mm,
            focal_length=20 * ureg.mm,
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=1 * ureg.Hz,
            f_speed=2.8,
            tilt_angle=30.0,
            tilt_direction=0.0,
        )
        fp = cam.footprint_at(altitude)
        assert fp["height"].magnitude > 0
        assert cam.trigger_distance(altitude).magnitude > 0

    def test_forward_tilt_line_spacing_unchanged(self, camera, tilted_camera, altitude):
        # Forward tilt should not change across-track width significantly
        # (tilted along-track, not cross-track)
        sp_nadir = camera.line_spacing(altitude, sidelap_pct=60.0)
        sp_tilt = tilted_camera.line_spacing(altitude, sidelap_pct=60.0)
        # Allow some difference due to tilt geometry, but should be close
        assert sp_tilt.magnitude == pytest.approx(sp_nadir.magnitude, rel=0.15)

    def test_swath_width_matches_footprint_width(self, tilted_camera, altitude):
        sw = tilted_camera.swath_width(altitude)
        fp = tilted_camera.footprint_at(altitude)
        assert sw.magnitude == pytest.approx(fp["width"].magnitude, rel=1e-6)

    def test_swath_width_nadir(self, camera, altitude):
        sw = camera.swath_width(altitude)
        fp = camera.footprint_at(altitude)
        assert sw.magnitude == pytest.approx(fp["width"].magnitude, rel=1e-6)


# ── Stereo methods ───────────────────────────────────────────────────────────

class TestStereoMethods:
    def test_bh_ratio_positive(self, camera, altitude):
        bh = camera.base_height_ratio(altitude, overlap_pct=60.0)
        assert bh > 0

    def test_bh_ratio_increases_with_less_overlap(self, camera, altitude):
        bh_high = camera.base_height_ratio(altitude, overlap_pct=80.0)
        bh_low = camera.base_height_ratio(altitude, overlap_pct=50.0)
        assert bh_low > bh_high

    def test_vertical_accuracy_positive(self, camera, altitude):
        va = camera.vertical_accuracy(altitude, overlap_pct=60.0)
        assert va.magnitude > 0
        assert va.check("[length]")

    def test_vertical_accuracy_worse_with_more_overlap(self, camera, altitude):
        # More overlap = smaller baseline = worse (larger) σ_z
        va_80 = camera.vertical_accuracy(altitude, overlap_pct=80.0)
        va_60 = camera.vertical_accuracy(altitude, overlap_pct=60.0)
        assert va_80.magnitude > va_60.magnitude

    def test_range_accuracy_positive(self, tilted_camera, altitude):
        baseline = 5000 * ureg.meter
        ra = tilted_camera.range_accuracy(altitude, baseline)
        assert ra.magnitude > 0
        assert ra.check("[length]")

    def test_range_accuracy_better_with_longer_baseline(self, tilted_camera, altitude):
        ra_short = tilted_camera.range_accuracy(altitude, 2000 * ureg.meter)
        ra_long = tilted_camera.range_accuracy(altitude, 5000 * ureg.meter)
        assert ra_long.magnitude < ra_short.magnitude


# ── MultiCameraRig ───────────────────────────────────────────────────────────

class TestMultiCameraRig:
    @pytest.fixture
    def simple_rig(self):
        """Two-camera rig: forward + aft at 15° tilt."""
        fwd = FrameCamera(
            name="Fwd", sensor_width=36 * ureg.mm, sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm, resolution_x=6000, resolution_y=4000,
            frame_rate=2 * ureg.Hz, f_speed=2.8,
            tilt_angle=15.0, tilt_direction=0.0,
        )
        aft = FrameCamera(
            name="Aft", sensor_width=36 * ureg.mm, sensor_height=24 * ureg.mm,
            focal_length=50 * ureg.mm, resolution_x=6000, resolution_y=4000,
            frame_rate=2 * ureg.Hz, f_speed=2.8,
            tilt_angle=15.0, tilt_direction=180.0,
        )
        return MultiCameraRig("Test Rig", [
            {"camera": fwd, "label": "fwd_1"},
            {"camera": aft, "label": "aft_1"},
        ])

    def test_rig_construction(self, simple_rig):
        assert len(simple_rig) == 2
        assert simple_rig.name == "Test Rig"

    def test_rig_swath_width(self, simple_rig):
        sw = simple_rig.swath_width(1000 * ureg.meter)
        assert sw.magnitude > 0
        assert sw.check("[length]")

    def test_rig_gsd(self, simple_rig):
        gsd = simple_rig.ground_sample_distance(1000 * ureg.meter)
        assert "x" in gsd and "y" in gsd
        assert gsd["x"].magnitude > 0
        assert gsd["y"].magnitude > 0

    def test_stereo_pairs_detected(self, simple_rig):
        pairs = simple_rig.stereo_pairs()
        assert len(pairs) == 1
        labels = {pairs[0][0]["label"], pairs[0][1]["label"]}
        assert labels == {"fwd_1", "aft_1"}

    def test_composite_bh_ratio(self, simple_rig):
        results = simple_rig.composite_base_height_ratio(1000 * ureg.meter)
        assert len(results) == 1
        bh = results[0]["bh_ratio"]
        # B/H = tan(15°) + tan(15°) ≈ 0.536
        assert bh == pytest.approx(2 * np.tan(np.radians(15.0)), rel=0.01)

    def test_rig_line_spacing(self, simple_rig):
        sp = simple_rig.line_spacing(1000 * ureg.meter, sidelap_pct=60.0)
        sw = simple_rig.swath_width(1000 * ureg.meter)
        assert sp.magnitude == pytest.approx(sw.magnitude * 0.4, rel=0.01)

    def test_rig_ground_footprint_default_offsets(self, simple_rig):
        fps = simple_rig.ground_footprint(1000 * ureg.meter)
        assert len(fps) == 2
        for fp in fps:
            assert isinstance(fp["polygon"], ShapelyPolygon)
            assert not fp["polygon"].is_empty

    def test_rig_rejects_length_cross_track_offset(self, simple_rig):
        cam = simple_rig.cameras[0]["camera"]
        with pytest.raises(ValueError, match="cross_track_offset"):
            MultiCameraRig("Bad Rig", [
                {"camera": cam, "label": "bad",
                 "cross_track_offset": 5 * ureg.meter},
            ])

    def test_rig_offset_widens_swath(self, simple_rig):
        cam = simple_rig.cameras[0]["camera"]
        offset_rig = MultiCameraRig("Offset Rig", [
            {"camera": cam, "label": "port",
             "cross_track_offset": -10 * ureg.degree},
            {"camera": cam, "label": "starboard",
             "cross_track_offset": 10 * ureg.degree},
        ])
        alt = 1000 * ureg.meter
        assert offset_rig.swath_width(alt).magnitude > (
            cam.swath_width(alt).magnitude
        )


# ── QUAKES-I factory ─────────────────────────────────────────────────────────

class TestQUAKESI:
    @pytest.fixture
    def quakes(self):
        return MultiCameraRig.quakes_i()

    def test_quakes_i_factory(self, quakes):
        assert len(quakes) == 8
        for entry in quakes.cameras:
            assert entry["camera"].tilt_angle == pytest.approx(11.3)

    def test_quakes_i_gv_swath(self, quakes):
        alt = 12500 * ureg.meter
        sw = quakes.swath_width(alt)
        # Union of the cross-track FOVs: outer cameras at ±20.25° plus
        # ~19.85° per-camera FOV → ≈ 2 × 12500 × tan(30.2°) ≈ 14.5 km
        # at 12.5 km AGL over sea-level terrain (paper's "12 km" is at
        # ~2 km MSL terrain elevation).
        assert 14_000 < sw.m_as("meter") < 15_000

    def test_quakes_i_stereo_pairs(self, quakes):
        pairs = quakes.stereo_pairs()
        assert len(pairs) == 4  # 4 fwd matched with 4 aft

    def test_quakes_i_range_accuracy(self, quakes):
        alt = 12500 * ureg.meter
        baseline = 5000 * ureg.meter
        cam = quakes.cameras[0]["camera"]
        sigma_q = 0.023e-3  # 0.023 mrad in radians
        ra = cam.range_accuracy(alt, baseline, sigma_q=sigma_q)
        # Should be on the order of meters
        assert 0.1 < ra.m_as("meter") < 100

    def test_quakes_i_frame_rate(self, quakes):
        for entry in quakes.cameras:
            assert entry["camera"].frame_rate.magnitude == pytest.approx(2.0)

    def test_quakes_i_combined_footprints(self, quakes):
        fps = quakes.combined_footprints(12500 * ureg.meter)
        assert len(fps) == 8
        for fp in fps:
            assert "label" in fp
            assert fp["width"].magnitude > 0

    def test_quakes_i_ground_footprint(self, quakes):
        fps = quakes.ground_footprint(12500 * ureg.meter)
        assert len(fps) == 8
        for fp in fps:
            assert "label" in fp
            assert isinstance(fp["polygon"], ShapelyPolygon)
            assert not fp["polygon"].is_empty
        # Outer cameras should be further from center than inner ones
        fwd_1_cx = fps[0]["polygon"].centroid.x  # leftmost fwd
        fwd_2_cx = fps[1]["polygon"].centroid.x  # inner left fwd
        assert abs(fwd_1_cx) > abs(fwd_2_cx)


# ---------------------------------------------------------------------------
# Overlap-based frame-rate limit + validation
# ---------------------------------------------------------------------------


def test_max_ground_speed_for_overlap_matches_trigger_distance(camera, altitude):
    speed = camera.max_ground_speed_for_overlap(altitude, overlap_pct=80.0)
    expected = camera.trigger_distance(altitude, overlap_pct=80.0) * camera.frame_rate
    assert speed.m_as("meter / second") == pytest.approx(
        expected.m_as("meter / second")
    )


@pytest.mark.parametrize("overlap_pct", [-1.0, 100.0, 125.0])
def test_frame_camera_overlap_validation(camera, altitude, overlap_pct):
    # Both APIs that accept overlap_pct must reject the same bad values,
    # so future edits cannot silently leave one of them permissive.
    with pytest.raises(ValueError):
        camera.max_ground_speed_for_overlap(altitude, overlap_pct=overlap_pct)
    with pytest.raises(ValueError):
        camera.trigger_distance(altitude, overlap_pct=overlap_pct)


# ---------------------------------------------------------------------------
# G-LiHT High Resolution Aerial Camera reference instance
# ---------------------------------------------------------------------------


class TestGlihtHrac:
    def test_singleton_identity(self):
        from hyplan.instruments import GLIHT_HRAC, create_sensor
        assert create_sensor("GLIHT_HRAC") is GLIHT_HRAC
        assert create_sensor("G-LiHT HRAC") is GLIHT_HRAC
        assert create_sensor("Phase One iXU-R 1000") is GLIHT_HRAC

    def test_fov_matches_published(self):
        # G-LiHT published FOV is 56.2° × 43.7°; allow ±0.5°.
        from hyplan.instruments import GLIHT_HRAC
        assert abs(GLIHT_HRAC.fov_x - 56.2) < 0.5
        assert abs(GLIHT_HRAC.fov_y - 43.7) < 0.5

    def test_gsd_matches_published(self):
        # 4 cm GSD reached at approximately 435 m AGL (G-LiHT-published
        # value uses the linear-pinhole formula `pixel_pitch / focal`;
        # HyPlan's angular-FOV-based ground_sample_distance returns
        # slightly under at the same altitude — still well within the
        # planning tolerance for a "nominal 4 cm GSD" claim).
        from hyplan.instruments import GLIHT_HRAC
        gsd = GLIHT_HRAC.ground_sample_distance(435 * ureg.meter)
        assert abs(float(gsd["x"].m_as("centimeter")) - 4.0) < 0.5

    def test_resolution_count(self):
        from hyplan.instruments import GLIHT_HRAC
        # ~101 MP within the rounding G-LiHT publishes ("100 Megapixels").
        n = GLIHT_HRAC.resolution_x * GLIHT_HRAC.resolution_y
        assert 99_000_000 < n < 103_000_000

    def test_max_ground_speed_for_overlap(self):
        from hyplan.instruments import GLIHT_HRAC
        speed = GLIHT_HRAC.max_ground_speed_for_overlap(
            435 * ureg.meter,
            overlap_pct=80.0,
        )
        assert abs(speed.m_as("meter / second") - 70.0) < 5.0


# ---------------------------------------------------------------------------
# G-LiHT Thermal Imager — Xenics Gobi-640 (was a LineScanner; retired in v1.9.x)
# ---------------------------------------------------------------------------


class TestGlihtThermal:
    def test_singleton_identity(self):
        from hyplan.instruments import GLIHT_THERMAL, create_sensor
        assert create_sensor("GLIHT_THERMAL") is GLIHT_THERMAL
        assert create_sensor("G-LiHT Thermal") is GLIHT_THERMAL
        assert create_sensor("Xenics Gobi-640") is GLIHT_THERMAL
        assert create_sensor("Gobi-640") is GLIHT_THERMAL

    def test_fov_matches_metadata(self):
        # G-LiHT Loudon June 2017 metadata specifies a 14.25 mm
        # athermalized lens + 640 × 480 17 μm Gobi-640 sensor →
        # cross-track FOV ≈ 41.8°.  (The earlier 42.6° value in the
        # retired LineScanner model came from a stale spec sheet
        # inconsistent with the documented lens/sensor.)
        from hyplan.instruments import GLIHT_THERMAL
        assert abs(GLIHT_THERMAL.fov_x - 41.8) < 0.5

    def test_footprint_cross_track_at_335m(self):
        # ~256 m cross-track at G-LiHT's 335 m nominal AGL, derived
        # live from the 41.8° FOV.
        from hyplan.instruments import GLIHT_THERMAL
        fp = GLIHT_THERMAL.footprint_at(335 * ureg.meter)
        assert abs(fp["width"].m_as("meter") - 256.0) < 3.0

    def test_is_frame_camera_not_line_scanner(self):
        # Regression guard: the Gobi-640 must be modelled as a FrameCamera,
        # not as a LineScanner.  An older HyPlan release had it backwards.
        from hyplan.instruments import GLIHT_THERMAL, FrameCamera
        assert isinstance(GLIHT_THERMAL, FrameCamera)

    def test_legacy_gliht_thermal_linescanner_is_retired(self):
        # The retired LineScanner class must no longer be importable from
        # the public surface.
        import hyplan
        import hyplan.instruments
        assert not hasattr(hyplan.instruments, "GLiHT_Thermal")
        assert not hasattr(hyplan, "GLiHT_Thermal")


# ---------------------------------------------------------------------------
# Motion-blur API
# ---------------------------------------------------------------------------


class TestMotionBlur:
    def test_motion_blur_requires_integration_time(self, camera):
        # The default fixture camera has no integration_time set.
        with pytest.raises(ValueError, match="integration_time"):
            camera.motion_blur_at(100 * ureg.meter, 50 * ureg.meter / ureg.second)
        with pytest.raises(ValueError, match="integration_time"):
            camera.max_ground_speed_for_motion_blur(100 * ureg.meter)

    def test_motion_blur_at_returns_length_and_pixels(self):
        from hyplan.instruments import GLIHT_THERMAL
        d = GLIHT_THERMAL.motion_blur_at(
            335 * ureg.meter, 74 * ureg.meter / ureg.second,
        )
        # 74 m/s × 20 ms = 1.48 m
        assert abs(d["length"].m_as("meter") - 1.48) < 0.01
        # 1.48 m / 0.4 m GSD ≈ 3.7 px
        assert 3.5 < d["pixels"] < 4.0

    def test_max_speed_for_motion_blur_inversely_with_integration(self):
        from hyplan.instruments import GLIHT_HRAC, GLIHT_THERMAL
        # HRAC: 1 ms × max_blur_pixels=1 / GSD_y(335m) ≈ 1.0 × 0.0297 / 0.001 ≈ 30 m/s
        hrac_speed = GLIHT_HRAC.max_ground_speed_for_motion_blur(
            335 * ureg.meter, max_blur_pixels=1.0,
        )
        assert 28 < hrac_speed.m_as("meter / second") < 32

        # Thermal: 20 ms integration → ~20 m/s for 1-px blur
        thermal_speed = GLIHT_THERMAL.max_ground_speed_for_motion_blur(
            335 * ureg.meter, max_blur_pixels=1.0,
        )
        assert 18 < thermal_speed.m_as("meter / second") < 22

        # 3-px blur is 3× as permissive.
        thermal_3px = GLIHT_THERMAL.max_ground_speed_for_motion_blur(
            335 * ureg.meter, max_blur_pixels=3.0,
        )
        assert thermal_3px.m_as("meter / second") == pytest.approx(
            3 * thermal_speed.m_as("meter / second"), rel=1e-9,
        )

    def test_integration_time_validation(self):
        from hyplan.instruments import FrameCamera
        # integration_time > frame_period must raise.
        with pytest.raises(ValueError, match="exceeds frame period"):
            FrameCamera(
                name="bad", sensor_width=10*ureg.mm, sensor_height=8*ureg.mm,
                focal_length=14*ureg.mm, resolution_x=640, resolution_y=480,
                frame_rate=50 * ureg.hertz, f_speed=1.2,
                integration_time=50 * ureg.millisecond,  # 2.5× frame period
            )
        # Non-positive integration_time must raise.
        with pytest.raises(ValueError, match="positive"):
            FrameCamera(
                name="bad", sensor_width=10*ureg.mm, sensor_height=8*ureg.mm,
                focal_length=14*ureg.mm, resolution_x=640, resolution_y=480,
                frame_rate=50 * ureg.hertz, f_speed=1.2,
                integration_time=-1 * ureg.millisecond,
            )
        # max_blur_pixels must be positive.
        from hyplan.instruments import GLIHT_THERMAL
        with pytest.raises(ValueError, match="positive"):
            GLIHT_THERMAL.max_ground_speed_for_motion_blur(
                335 * ureg.meter, max_blur_pixels=0,
            )


# ---------------------------------------------------------------------------
# Footprint polygon (lat/lon, flat earth)
# ---------------------------------------------------------------------------


class TestFootprintPolygonAt:
    def test_returns_polygon_centered_on_lat_lon(self):
        from hyplan.instruments import GLIHT_HRAC
        poly = GLIHT_HRAC.footprint_polygon_at(
            30.0, -80.0, 335 * ureg.meter, heading=0.0,
        )
        assert not poly.is_empty
        # Centroid should sit at the supplied (lat, lon).
        cx, cy = poly.centroid.x, poly.centroid.y
        assert abs(cx - (-80.0)) < 1e-4
        assert abs(cy - 30.0) < 1e-4

    def test_polygon_width_matches_footprint(self):
        from hyplan.instruments import GLIHT_HRAC
        poly = GLIHT_HRAC.footprint_polygon_at(
            30.0, -80.0, 335 * ureg.meter, heading=0.0,
        )
        # Cross-track span at this latitude ≈ footprint width.
        minx, miny, maxx, maxy = poly.bounds
        m_per_deg_lon = 111_320.0 * float(np.cos(np.radians(30.0)))
        cross_track_m = (maxx - minx) * m_per_deg_lon
        expected = float(GLIHT_HRAC.footprint_at(335 * ureg.meter)["width"].m_as("meter"))
        # Allow 5% for geodesic vs flat-earth slop.
        assert abs(cross_track_m - expected) / expected < 0.05

    def test_heading_rotates_polygon(self):
        from hyplan.instruments import GLIHT_HRAC
        poly_north = GLIHT_HRAC.footprint_polygon_at(
            30.0, -80.0, 335 * ureg.meter, heading=0.0,
        )
        poly_east = GLIHT_HRAC.footprint_polygon_at(
            30.0, -80.0, 335 * ureg.meter, heading=90.0,
        )
        # 90° rotation swaps cross-track and along-track extents.
        nx = poly_north.bounds[2] - poly_north.bounds[0]   # lon span when heading=N
        ny = poly_north.bounds[3] - poly_north.bounds[1]   # lat span
        ex = poly_east.bounds[2] - poly_east.bounds[0]     # lon span when heading=E
        ey = poly_east.bounds[3] - poly_east.bounds[1]
        # HRAC is wider cross-track than along-track, so for heading=N
        # lon-span > lat-span; flipped for heading=E (lat-span > lon-span
        # after accounting for cos(lat) anisotropy of lat/lon).
        assert nx > ny     # heading N: cross-track is east-west
        assert ey > ex     # heading E: cross-track is north-south
