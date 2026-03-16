"""Tests for hyplan.swath."""

import pytest
from hyplan.units import ureg
from hyplan.sensors import AVIRIS3
from hyplan.flight_line import FlightLine
from hyplan.swath import generate_swath_polygon, calculate_swath_widths


class TestGenerateSwathPolygon:
    def test_creates_polygon(self, sample_flight_line):
        sensor = AVIRIS3()
        poly = generate_swath_polygon(sample_flight_line, sensor)
        assert poly is not None
        assert poly.area > 0
        assert poly.is_valid

    def test_swath_widths(self, sample_flight_line):
        sensor = AVIRIS3()
        poly = generate_swath_polygon(sample_flight_line, sensor)
        widths = calculate_swath_widths(poly)
        assert "min_width" in widths
        assert "mean_width" in widths
        assert "max_width" in widths
        assert widths["min_width"] > 0
        assert widths["max_width"] >= widths["min_width"]
