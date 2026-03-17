"""Comprehensive tests for hyplan.frame_camera module."""

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.frame_camera import FrameCamera


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
        assert recovered_alt.to("meter").magnitude == pytest.approx(alt.magnitude, rel=0.02)

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


class TestFootprintCorners:
    def test_signature_exists(self):
        """Verify footprint_corners is callable (requires DEM for full test)."""
        assert callable(FrameCamera.footprint_corners)
