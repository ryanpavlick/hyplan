"""Tests for hyplan.swath."""

import pytest
import numpy as np
from shapely.geometry import Polygon

from hyplan.units import ureg
from hyplan.sensors import AVIRIS3
from hyplan.flight_line import FlightLine
from hyplan.swath import generate_swath_polygon, calculate_swath_widths, export_polygon_to_kml


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


class TestCalculateSwathWidths:
    def test_rectangular_polygon(self):
        """A known rectangular polygon should give consistent widths."""
        # ~1 km wide rectangle at the equator
        poly = Polygon([
            (0.0, 0.0), (0.0, 0.01),
            (0.1, 0.01), (0.1, 0.0), (0.0, 0.0)
        ])
        widths = calculate_swath_widths(poly)
        assert widths["min_width"] > 0
        assert widths["mean_width"] > 0
        assert widths["max_width"] >= widths["min_width"]

    def test_empty_result_on_degenerate(self):
        """A degenerate polygon with zero-width sides should handle gracefully."""
        # Line-like polygon (all points on same longitude)
        poly = Polygon([
            (0.0, 0.0), (0.0, 0.01),
            (0.0, 0.02), (0.0, 0.01), (0.0, 0.0)
        ])
        widths = calculate_swath_widths(poly)
        assert isinstance(widths, dict)


class TestExportPolygonToKml:
    def test_export(self, tmp_path):
        poly = Polygon([
            (-118.0, 34.0), (-118.0, 34.1),
            (-117.9, 34.1), (-117.9, 34.0), (-118.0, 34.0)
        ])
        kml_path = str(tmp_path / "test_swath.kml")
        export_polygon_to_kml(poly, kml_path, name="Test")
        assert (tmp_path / "test_swath.kml").exists()
