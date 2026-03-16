"""Tests for hyplan.sensors, hyplan.frame_camera, hyplan.lvis, hyplan.radar."""

import pytest
from hyplan.units import ureg
from hyplan.sensors import (
    AVIRIS3,
    AVIRISNextGen,
    HyTES,
    PRISM,
    MASTER,
    SENSOR_REGISTRY,
    create_sensor,
)
from hyplan.frame_camera import FrameCamera
from hyplan.lvis import LVIS, LVISLens, LVIS_LENS_NARROW, LVIS_LENS_WIDE
from hyplan.radar import UAVSAR_Lband, UAVSAR_Pband, SidelookingRadar


class TestLineScanner:
    def test_aviris3_instantiation(self):
        s = AVIRIS3()
        assert s.name == "AVIRIS 3"
        assert s.fov > 0

    def test_swath_width(self):
        s = AVIRIS3()
        sw = s.swath_width(ureg.Quantity(6000, "meter"))
        assert sw.magnitude > 0
        assert sw.check("[length]")

    def test_ground_sample_distance(self):
        s = AVIRIS3()
        gsd = s.ground_sample_distance(ureg.Quantity(6000, "meter"))
        assert gsd.magnitude > 0

    def test_altitude_for_gsd(self):
        s = AVIRIS3()
        alt = s.altitude_agl_for_ground_sample_distance(ureg.Quantity(5, "meter"))
        assert alt.magnitude > 0

    def test_critical_ground_speed(self):
        s = AVIRIS3()
        speed = s.critical_ground_speed(ureg.Quantity(6000, "meter"))
        assert speed.magnitude > 0

    def test_half_angle(self):
        s = AVIRIS3()
        assert 0 < s.half_angle < 90

    def test_hytes(self):
        s = HyTES()
        assert s.swath_width(ureg.Quantity(5000, "meter")).magnitude > 0

    def test_prism(self):
        s = PRISM()
        assert s.swath_width(ureg.Quantity(8000, "meter")).magnitude > 0


class TestSensorRegistry:
    def test_registry_populated(self):
        assert len(SENSOR_REGISTRY) > 0
        assert "AVIRIS3" in SENSOR_REGISTRY

    def test_create_sensor(self):
        s = create_sensor("AVIRIS3")
        assert isinstance(s, AVIRIS3)


class TestFrameCamera:
    def test_instantiation(self):
        cam = FrameCamera(
            name="Test Camera",
            sensor_width=ureg.Quantity(36, "mm"),
            sensor_height=ureg.Quantity(24, "mm"),
            focal_length=ureg.Quantity(50, "mm"),
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=ureg.Quantity(1, "Hz"),
            f_speed=2.8,
        )
        assert cam.fov_x > 0
        assert cam.fov_y > 0

    def test_gsd(self):
        cam = FrameCamera(
            name="Test Camera",
            sensor_width=ureg.Quantity(36, "mm"),
            sensor_height=ureg.Quantity(24, "mm"),
            focal_length=ureg.Quantity(50, "mm"),
            resolution_x=6000,
            resolution_y=4000,
            frame_rate=ureg.Quantity(1, "Hz"),
            f_speed=2.8,
        )
        gsd = cam.ground_sample_distance(ureg.Quantity(1000, "meter"))
        assert "x" in gsd
        assert gsd["x"].magnitude > 0


class TestLVIS:
    def test_instantiation(self):
        lvis = LVIS()
        assert lvis.half_angle > 0

    def test_swath_width(self):
        lvis = LVIS()
        sw = lvis.swath_width(ureg.Quantity(8000, "meter"))
        assert sw.magnitude > 0

    def test_with_lens(self):
        lvis = LVIS(lens=LVIS_LENS_NARROW)
        d = lvis.footprint_diameter(ureg.Quantity(8000, "meter"))
        assert d.magnitude > 0

    def test_effective_swath(self):
        lvis = LVIS()
        esw = lvis.effective_swath_width(
            ureg.Quantity(8000, "meter"),
            ureg.Quantity(150, "knot"),
        )
        assert esw.magnitude > 0


class TestRadar:
    def test_uavsar_lband(self):
        r = UAVSAR_Lband()
        assert r.name == "UAVSAR L-band"
        sw = r.swath_width(ureg.Quantity(12000, "meter"))
        assert sw.magnitude > 0

    def test_wavelength(self):
        r = UAVSAR_Lband()
        wl = r.wavelength
        assert wl.to("cm").magnitude == pytest.approx(23.8, rel=0.1)

    def test_ground_sample_distance(self):
        r = UAVSAR_Lband()
        gsd = r.ground_sample_distance(ureg.Quantity(12000, "meter"))
        assert "azimuth" in gsd
        assert "center" in gsd
