"""Tests for hyplan.flight_box."""

import pytest
from shapely.geometry import Polygon
from hyplan.units import ureg
from hyplan.sensors import AVIRIS3
from hyplan.flight_box import box_around_center_line, box_around_polygon


class TestBoxAroundCenterLine:
    def test_generates_lines(self):
        sensor = AVIRIS3()
        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0,
            lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=ureg.Quantity(10000, "meter"),
            box_name="TestBox",
        )
        assert len(lines) > 0
        for fl in lines:
            assert fl.altitude_msl.to("meter").magnitude == pytest.approx(6000)

    def test_overlap_changes_count(self):
        sensor = AVIRIS3()
        kwargs = dict(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0,
            lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=ureg.Quantity(20000, "meter"),
            box_name="TestBox",
        )
        lines_20 = box_around_center_line(overlap=20, **kwargs)
        lines_50 = box_around_center_line(overlap=50, **kwargs)
        # More overlap → more lines
        assert len(lines_50) >= len(lines_20)


class TestBoxAroundPolygon:
    def test_generates_lines(self):
        sensor = AVIRIS3()
        poly = Polygon([
            (-118.1, 33.9), (-117.9, 33.9),
            (-117.9, 34.1), (-118.1, 34.1),
        ])
        lines = box_around_polygon(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            polygon=poly,
            azimuth=0.0,
            box_name="PolyBox",
        )
        assert len(lines) > 0
