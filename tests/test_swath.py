"""Tests for hyplan.swath."""

import pytest
import numpy as np
from shapely.geometry import Polygon

from hyplan.units import ureg
from hyplan.instruments import AVIRIS3
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


class TestRadarSwathPolygon:
    def test_radar_polygon_is_valid(self, sample_flight_line):
        from hyplan.instruments import UAVSAR_Lband
        radar = UAVSAR_Lband()
        poly = generate_swath_polygon(sample_flight_line, radar)
        assert poly is not None
        assert poly.area > 0
        assert poly.is_valid

    def test_radar_swath_one_sided(self):
        """Left-looking radar swath should be entirely to the left of the flight track."""
        from hyplan.instruments import UAVSAR_Lband
        radar = UAVSAR_Lband()
        # Northward flight line at 12,500 m (typical UAVSAR altitude)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "kilometer"),
            az=0.0,  # due north
            altitude_msl=ureg.Quantity(12500, "meter"),
            site_name="Radar Test",
        )
        poly = generate_swath_polygon(fl, radar, along_precision=5000.0)
        # For a northward flight, left = west (smaller longitude)
        track_lon = -118.0
        centroid_lon = poly.centroid.x
        # Swath centroid should be west of (less than) the track longitude
        assert centroid_lon < track_lon

    def test_tilted_scanner_shifted(self):
        """A starboard-tilted scanner should have swath centroid to the right of track."""
        from hyplan.instruments import LineScanner
        sensor = LineScanner("Tilted", fov=30.0, across_track_pixels=600,
                             frame_rate=100.0 * ureg.Hz, cross_track_tilt=20.0)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "kilometer"),
            az=0.0,  # due north
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="Tilt Test",
        )
        poly = generate_swath_polygon(fl, sensor, along_precision=5000.0)
        track_lon = -118.0
        centroid_lon = poly.centroid.x
        # Starboard = east for northward flight = larger longitude
        assert centroid_lon > track_lon

    def test_nadir_scanner_regression(self, sample_flight_line):
        """Nadir scanner swath should still be symmetric and valid."""
        sensor = AVIRIS3()
        poly = generate_swath_polygon(sample_flight_line, sensor)
        assert poly.is_valid
        assert poly.area > 0
        widths = calculate_swath_widths(poly)
        assert widths["min_width"] > 0


class TestExportPolygonToKml:
    def test_export(self, tmp_path):
        poly = Polygon([
            (-118.0, 34.0), (-118.0, 34.1),
            (-117.9, 34.1), (-117.9, 34.0), (-118.0, 34.0)
        ])
        kml_path = str(tmp_path / "test_swath.kml")
        export_polygon_to_kml(poly, kml_path, name="Test")
        assert (tmp_path / "test_swath.kml").exists()
