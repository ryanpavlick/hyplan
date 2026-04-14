"""Tests for hyplan.swath."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from hyplan.units import ureg
from hyplan.instruments import AVIRIS3
from hyplan.flight_line import FlightLine
from hyplan.swath import (
    generate_swath_polygon,
    calculate_swath_widths,
    analyze_swath_gaps_overlaps,
    export_polygon_to_kml,
    _resolve_swath_boresight_azimuths,
)


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


class TestAnalyzeSwathGapsOverlaps:
    def test_two_overlapping_lonlat_squares(self):
        # Two ~1 km squares near the equator that overlap by ~half
        a = Polygon([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)])
        b = Polygon([(0.005, 0.0), (0.015, 0.0), (0.015, 0.01), (0.005, 0.01)])
        df = analyze_swath_gaps_overlaps([a, b])
        assert len(df) == 1
        assert df.iloc[0]["overlap_area_m2"] > 0
        assert df.iloc[0]["gap_area_m2"] == 0
        assert df.iloc[0]["overlap_fraction"] > 0

    def test_two_disjoint_lonlat_squares(self):
        a = Polygon([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)])
        b = Polygon([(0.02, 0.0), (0.03, 0.0), (0.03, 0.01), (0.02, 0.01)])
        df = analyze_swath_gaps_overlaps([a, b])
        assert df.iloc[0]["overlap_area_m2"] == 0
        assert df.iloc[0]["gap_area_m2"] > 0
        assert df.iloc[0]["overlap_fraction"] == 0

    def test_single_polygon_returns_empty(self):
        a = Polygon([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)])
        df = analyze_swath_gaps_overlaps([a])
        assert df.empty
        assert list(df.columns) == [
            "pair_index", "overlap_area_m2", "gap_area_m2", "overlap_fraction"
        ]

    def test_empty_returns_empty(self):
        df = analyze_swath_gaps_overlaps([])
        assert df.empty

    def test_three_polygons_two_pairs(self):
        a = Polygon([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)])
        b = Polygon([(0.005, 0.0), (0.015, 0.0), (0.015, 0.01), (0.005, 0.01)])
        c = Polygon([(0.02, 0.0), (0.03, 0.0), (0.03, 0.01), (0.02, 0.01)])
        df = analyze_swath_gaps_overlaps([a, b, c])
        assert len(df) == 2
        assert df.iloc[0]["overlap_area_m2"] > 0  # a-b overlap
        assert df.iloc[1]["gap_area_m2"] > 0      # b-c gap


class TestCrabAwareSwath:
    """Test crab-angle-aware swath generation."""

    @pytest.fixture
    def northward_line(self):
        return FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "kilometer"),
            az=0.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="North Line",
        )

    def test_track_mode_matches_default(self, northward_line):
        """heading_mode='track' should reproduce existing behavior exactly."""
        sensor = AVIRIS3()
        poly_default = generate_swath_polygon(northward_line, sensor,
                                               along_precision=5000.0)
        poly_track = generate_swath_polygon(northward_line, sensor,
                                             along_precision=5000.0,
                                             heading_mode="track")
        # Same polygon
        assert poly_default.equals(poly_track)

    def test_crabbed_with_zero_crab_matches_track(self, northward_line):
        """Crabbed mode with crab=0 should match track mode."""
        sensor = AVIRIS3()
        poly_track = generate_swath_polygon(northward_line, sensor,
                                             along_precision=5000.0)
        poly_crab0 = generate_swath_polygon(northward_line, sensor,
                                             along_precision=5000.0,
                                             heading_mode="crabbed",
                                             crab_angle_deg=0.0)
        # Should be very close (not bit-identical due to float modular arithmetic)
        assert abs(poly_track.area - poly_crab0.area) / poly_track.area < 0.01

    def test_crabbed_rotates_swath(self, northward_line):
        """Nonzero crab should shift the swath centroid relative to track mode."""
        sensor = AVIRIS3()
        poly_track = generate_swath_polygon(northward_line, sensor,
                                             along_precision=5000.0)
        poly_crabbed = generate_swath_polygon(northward_line, sensor,
                                               along_precision=5000.0,
                                               heading_mode="crabbed",
                                               crab_angle_deg=10.0)
        # Centroids should differ
        assert poly_track.centroid.x != pytest.approx(poly_crabbed.centroid.x, abs=1e-6)

    def test_crabbed_with_heading_deg(self, northward_line):
        """heading_deg should override track azimuths."""
        sensor = AVIRIS3()
        poly = generate_swath_polygon(northward_line, sensor,
                                       along_precision=5000.0,
                                       heading_mode="crabbed",
                                       heading_deg=10.0)
        assert poly.is_valid
        assert poly.area > 0

    def test_invalid_heading_mode_raises(self, northward_line):
        sensor = AVIRIS3()
        with pytest.raises(ValueError, match="heading_mode"):
            generate_swath_polygon(northward_line, sensor,
                                    heading_mode="invalid")

    def test_crabbed_without_params_raises(self, northward_line):
        sensor = AVIRIS3()
        with pytest.raises(ValueError, match="crab_angle_deg or heading_deg"):
            generate_swath_polygon(northward_line, sensor,
                                    heading_mode="crabbed")


class TestResolveBoresightAzimuths:
    def test_track_mode_passthrough(self):
        azimuths = np.array([0.0, 10.0, 20.0])
        result = _resolve_swath_boresight_azimuths(azimuths, "track")
        np.testing.assert_array_equal(result, azimuths)

    def test_crabbed_with_angle(self):
        azimuths = np.array([0.0, 90.0, 180.0])
        result = _resolve_swath_boresight_azimuths(azimuths, "crabbed",
                                                     crab_angle_deg=5.0)
        expected = np.array([5.0, 95.0, 185.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_crabbed_with_heading(self):
        azimuths = np.array([0.0, 90.0, 180.0])
        result = _resolve_swath_boresight_azimuths(azimuths, "crabbed",
                                                     heading_deg=45.0)
        np.testing.assert_array_almost_equal(result, [45.0, 45.0, 45.0])


class TestExportPolygonToKml:
    def test_export(self, tmp_path):
        poly = Polygon([
            (-118.0, 34.0), (-118.0, 34.1),
            (-117.9, 34.1), (-117.9, 34.0), (-118.0, 34.0)
        ])
        kml_path = str(tmp_path / "test_swath.kml")
        export_polygon_to_kml(poly, kml_path, name="Test")
        assert (tmp_path / "test_swath.kml").exists()
