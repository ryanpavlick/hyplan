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


class TestSwathOffsetAngles:
    def test_nadir_symmetric(self):
        """Default cross_track_tilt=0 gives symmetric angles."""
        s = AVIRIS3()
        port, starboard = s.swath_offset_angles()
        assert port == pytest.approx(-s.half_angle)
        assert starboard == pytest.approx(s.half_angle)

    def test_tilted_starboard(self):
        """Starboard tilt shifts both angles positive."""
        from hyplan.sensors import LineScanner
        s = LineScanner("Tilted", fov=30.0, across_track_pixels=600,
                        frame_rate=100.0 * ureg.Hz, cross_track_tilt=10.0)
        port, starboard = s.swath_offset_angles()
        assert port == pytest.approx(-5.0)   # 10 - 15
        assert starboard == pytest.approx(25.0)  # 10 + 15

    def test_tilted_port(self):
        """Port tilt shifts both angles negative."""
        from hyplan.sensors import LineScanner
        s = LineScanner("Tilted", fov=30.0, across_track_pixels=600,
                        frame_rate=100.0 * ureg.Hz, cross_track_tilt=-10.0)
        port, starboard = s.swath_offset_angles()
        assert port == pytest.approx(-25.0)
        assert starboard == pytest.approx(5.0)

    def test_swath_width_unchanged_at_nadir(self):
        """Swath width with tilt=0 matches the original formula."""
        s = AVIRIS3()
        alt = ureg.Quantity(6000, "meter")
        import numpy as np
        expected = 2 * 6000 * np.tan(np.radians(s.fov / 2))
        assert s.swath_width(alt).magnitude == pytest.approx(expected, rel=1e-6)

    def test_swath_width_with_tilt(self):
        """Tilted sensor has different (wider) swath than nadir."""
        from hyplan.sensors import LineScanner
        import numpy as np
        s_nadir = LineScanner("N", fov=30.0, across_track_pixels=600,
                              frame_rate=100.0 * ureg.Hz, cross_track_tilt=0.0)
        s_tilted = LineScanner("T", fov=30.0, across_track_pixels=600,
                               frame_rate=100.0 * ureg.Hz, cross_track_tilt=20.0)
        alt = ureg.Quantity(6000, "meter")
        # Tilted swath should be wider (tan is nonlinear)
        assert s_tilted.swath_width(alt).magnitude > s_nadir.swath_width(alt).magnitude

    def test_lvis_nadir(self):
        """LVIS swath_offset_angles is symmetric about nadir."""
        lvis = LVIS()
        port, starboard = lvis.swath_offset_angles()
        assert port == pytest.approx(-lvis.half_angle)
        assert starboard == pytest.approx(lvis.half_angle)

    def test_radar_left_looking(self):
        """Left-looking radar has both angles negative (port side)."""
        r = UAVSAR_Lband()
        port, starboard = r.swath_offset_angles()
        assert port < 0
        assert starboard < 0
        assert port < starboard  # far edge is more negative

    def test_radar_right_looking(self):
        """Right-looking radar has both angles positive (starboard side)."""
        r = SidelookingRadar(
            name="Test", frequency=1.0 * ureg.GHz, bandwidth=80 * ureg.MHz,
            near_range_angle=20.0, far_range_angle=60.0,
            azimuth_resolution=1.0 * ureg.meter, polarization="HH",
            look_direction="right",
        )
        port, starboard = r.swath_offset_angles()
        assert port > 0
        assert starboard > 0


class TestSensorRegistry:
    def test_registry_populated(self):
        assert len(SENSOR_REGISTRY) > 0
        assert "AVIRIS3" in SENSOR_REGISTRY

    def test_create_sensor(self):
        s = create_sensor("AVIRIS3")
        assert isinstance(s, AVIRIS3)

    def test_create_uavsar(self):
        s = create_sensor("UAVSAR_Lband")
        assert isinstance(s, UAVSAR_Lband)

    def test_create_glistin(self):
        from hyplan.radar import UAVSAR_Kaband
        s = create_sensor("GLISTIN-A")
        assert isinstance(s, UAVSAR_Kaband)


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
