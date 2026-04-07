"""Comprehensive tests for the LVIS instrument model."""

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.instruments import (
    LVIS,
    LVISLens,
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


class TestEffectiveFOV:
    def test_positive(self, lvis_default, altitude, speed):
        efov = lvis_default.effective_fov(altitude, speed)
        assert efov > 0

    def test_at_most_geometric_fov(self, lvis_default, altitude, speed):
        efov = lvis_default.effective_fov(altitude, speed)
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
            "effective_swath_width", "contiguous", "coverage_rate",
            "footprint_for_max_swath",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_lens_name(self, lvis_default, altitude, speed):
        s = lvis_default.summary(altitude, speed)
        assert s["lens"] == "wide"

    def test_summary_contiguous_is_bool(self, lvis_default, altitude, speed):
        s = lvis_default.summary(altitude, speed)
        assert isinstance(s["contiguous"], (bool, np.bool_))


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
