"""Comprehensive tests for the SAR radar instrument models."""

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.instruments import SidelookingRadar, UAVSAR_Lband, UAVSAR_Pband, UAVSAR_Kaband


@pytest.fixture
def lband():
    return UAVSAR_Lband()


@pytest.fixture
def pband():
    return UAVSAR_Pband()


@pytest.fixture
def kaband():
    return UAVSAR_Kaband()


@pytest.fixture
def altitude():
    return 12500 * ureg.meter


class TestSidelookingRadarInstantiation:
    def test_valid_construction(self):
        r = SidelookingRadar(
            name="Test SAR",
            frequency=1.0 * ureg.GHz,
            bandwidth=50 * ureg.MHz,
            near_range_angle=20.0,
            far_range_angle=60.0,
            azimuth_resolution=1.0 * ureg.meter,
            polarization="HH",
        )
        assert r.name == "Test SAR"
        assert r.look_direction == "left"

    def test_right_looking(self):
        r = SidelookingRadar(
            name="Right SAR",
            frequency=1.0 * ureg.GHz,
            bandwidth=50 * ureg.MHz,
            near_range_angle=20.0,
            far_range_angle=60.0,
            azimuth_resolution=1.0 * ureg.meter,
            polarization="HH",
            look_direction="right",
        )
        assert r.look_direction == "right"

    def test_invalid_look_direction(self):
        with pytest.raises(ValueError, match="look_direction"):
            SidelookingRadar(
                name="Bad SAR",
                frequency=1.0 * ureg.GHz,
                bandwidth=50 * ureg.MHz,
                near_range_angle=20.0,
                far_range_angle=60.0,
                azimuth_resolution=1.0 * ureg.meter,
                polarization="HH",
                look_direction="up",
            )

    def test_near_greater_than_far_raises(self):
        with pytest.raises(ValueError, match="near_range_angle must be less"):
            SidelookingRadar(
                name="Bad SAR",
                frequency=1.0 * ureg.GHz,
                bandwidth=50 * ureg.MHz,
                near_range_angle=60.0,
                far_range_angle=20.0,
                azimuth_resolution=1.0 * ureg.meter,
                polarization="HH",
            )


class TestUAVSARVariants:
    def test_lband_instantiation(self, lband):
        assert lband.name == "UAVSAR L-band"
        assert lband.polarization == "quad-pol"
        assert lband.frequency.magnitude == pytest.approx(1.2575)

    def test_pband_instantiation(self, pband):
        assert pband.name == "UAVSAR P-band (AirMOSS)"
        assert pband.frequency.magnitude == pytest.approx(0.430)

    def test_kaband_instantiation(self, kaband):
        assert kaband.name == "UAVSAR Ka-band (GLISTIN-A)"
        assert kaband.frequency.magnitude == pytest.approx(35.66)
        assert kaband.polarization == "HH"

    def test_kaband_no_peak_power(self, kaband):
        assert kaband.peak_power is None


class TestWavelengthAndResolution:
    def test_lband_wavelength(self, lband):
        wl = lband.wavelength
        assert wl.check("[length]")
        # L-band ~ 23.8 cm
        assert wl.m_as("cm") == pytest.approx(23.8, rel=0.05)

    def test_kaband_wavelength(self, kaband):
        wl = kaband.wavelength
        # Ka-band ~ 8.4 mm
        assert wl.m_as("mm") == pytest.approx(8.4, rel=0.05)

    def test_range_resolution(self, lband):
        rr = lband.range_resolution
        assert rr.check("[length]")
        # c / (2 * 80 MHz) ~ 1.87 m
        assert rr.magnitude == pytest.approx(1.87, rel=0.05)

    def test_pband_coarser_range_resolution(self, pband, lband):
        # P-band has 20 MHz BW vs L-band 80 MHz → coarser resolution
        assert pband.range_resolution.magnitude > lband.range_resolution.magnitude


class TestAngles:
    def test_half_angle(self, lband):
        # (65 - 22) / 2 = 21.5
        assert lband.half_angle == pytest.approx(21.5)

    def test_swath_center_angle(self, lband):
        # (22 + 65) / 2 = 43.5
        assert lband.swath_center_angle == pytest.approx(43.5)

    def test_pband_angles(self, pband):
        assert pband.half_angle == pytest.approx(10.0)
        assert pband.swath_center_angle == pytest.approx(35.0)


class TestSwathGeometry:
    def test_swath_width_positive(self, lband, altitude):
        sw = lband.swath_width(altitude)
        assert sw.magnitude > 0
        assert sw.check("[length]")

    def test_swath_width_scales_with_altitude(self, lband):
        sw1 = lband.swath_width(10000 * ureg.meter)
        sw2 = lband.swath_width(20000 * ureg.meter)
        assert sw2.magnitude == pytest.approx(2 * sw1.magnitude, rel=0.01)

    def test_near_range_ground_distance(self, lband, altitude):
        nr = lband.near_range_ground_distance(altitude)
        assert nr.magnitude > 0
        # h * tan(22°)
        expected = 12500 * np.tan(np.radians(22))
        assert nr.magnitude == pytest.approx(expected, rel=0.01)

    def test_far_range_ground_distance(self, lband, altitude):
        fr = lband.far_range_ground_distance(altitude)
        nr = lband.near_range_ground_distance(altitude)
        assert fr.magnitude > nr.magnitude

    def test_swath_equals_far_minus_near(self, lband, altitude):
        sw = lband.swath_width(altitude)
        fr = lband.far_range_ground_distance(altitude)
        nr = lband.near_range_ground_distance(altitude)
        assert sw.magnitude == pytest.approx((fr - nr).magnitude, rel=0.01)


class TestGroundResolution:
    def test_ground_range_resolution_at_center(self, lband, altitude):
        grr = lband.ground_range_resolution(altitude)
        assert grr.magnitude > 0
        assert grr.check("[length]")

    def test_resolution_worse_at_near_range(self, lband, altitude):
        # At smaller angles, sin is smaller → resolution is coarser
        grr_near = lband.ground_range_resolution(altitude, lband.near_range_angle)
        grr_far = lband.ground_range_resolution(altitude, lband.far_range_angle)
        assert grr_near.magnitude > grr_far.magnitude

    def test_gsd_dict_keys(self, lband, altitude):
        gsd = lband.ground_sample_distance(altitude)
        assert set(gsd.keys()) == {"near_range", "center", "far_range", "azimuth"}
        assert gsd["azimuth"] == lband.azimuth_resolution

    def test_gsd_values_positive(self, lband, altitude):
        gsd = lband.ground_sample_distance(altitude)
        for key in ["near_range", "center", "far_range"]:
            assert gsd[key].magnitude > 0


class TestSlantRange:
    def test_slant_range_at_center(self, lband, altitude):
        sr = lband.slant_range(altitude)
        assert sr.magnitude > altitude.magnitude  # slant range > altitude

    def test_slant_range_increases_with_angle(self, lband, altitude):
        sr_near = lband.slant_range(altitude, lband.near_range_angle)
        sr_far = lband.slant_range(altitude, lband.far_range_angle)
        assert sr_far.magnitude > sr_near.magnitude


class TestSwathOffsetAngles:
    def test_returns_tuple(self, lband):
        angles = lband.swath_offset_angles()
        assert isinstance(angles, tuple)
        assert len(angles) == 2

    def test_values_left_looking(self, lband):
        near, far = lband.swath_offset_angles()
        # Left-looking: swath on port side (negative angles)
        assert near == -lband.far_range_angle
        assert far == -lband.near_range_angle

    def test_left_right_differ(self):
        from hyplan.instruments import SidelookingRadar
        left = SidelookingRadar(
            name="test", frequency=1.26 * ureg.GHz, bandwidth=80 * ureg.MHz,
            near_range_angle=25.0, far_range_angle=65.0,
            azimuth_resolution=5.0 * ureg.meter, polarization="HH",
            look_direction="left",
        )
        right = SidelookingRadar(
            name="test", frequency=1.26 * ureg.GHz, bandwidth=80 * ureg.MHz,
            near_range_angle=25.0, far_range_angle=65.0,
            azimuth_resolution=5.0 * ureg.meter, polarization="HH",
            look_direction="right",
        )
        l_angles = left.swath_offset_angles()
        r_angles = right.swath_offset_angles()
        # Left and right should be on opposite sides
        assert l_angles[0] < 0 and l_angles[1] < 0
        assert r_angles[0] > 0 and r_angles[1] > 0


class TestInterferometricLineSpacing:
    def test_no_overlap(self, lband, altitude):
        spacing = lband.interferometric_line_spacing(altitude, overlap_fraction=0.0)
        sw = lband.swath_width(altitude)
        assert spacing.magnitude == pytest.approx(sw.magnitude, rel=0.01)

    def test_with_overlap(self, lband, altitude):
        spacing_full = lband.interferometric_line_spacing(altitude, overlap_fraction=0.0)
        spacing_half = lband.interferometric_line_spacing(altitude, overlap_fraction=0.5)
        assert spacing_half.magnitude == pytest.approx(spacing_full.magnitude * 0.5, rel=0.01)
