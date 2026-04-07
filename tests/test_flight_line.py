"""Tests for hyplan.flight_line."""

import pytest
from hyplan.units import ureg
from hyplan.flight_line import FlightLine, to_gdf


class TestFlightLineCreation:
    def test_start_length_azimuth(self, sample_flight_line):
        fl = sample_flight_line
        assert fl.lat1 == pytest.approx(34.05)
        assert fl.lon1 == pytest.approx(-118.25)
        assert fl.length.to("meter").magnitude == pytest.approx(50000, rel=0.01)
        assert fl.site_name == "Test Line"

    def test_center_length_azimuth(self):
        fl = FlightLine.center_length_azimuth(
            lat=34.0,
            lon=-118.0,
            length=ureg.Quantity(20000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(5000, "meter"),
        )
        # Center should be near (34, -118)
        mid_lat = (fl.lat1 + fl.lat2) / 2
        assert mid_lat == pytest.approx(34.0, abs=0.01)
        assert fl.length.to("meter").magnitude == pytest.approx(20000, rel=0.01)

    def test_altitude_stored(self, sample_flight_line):
        assert sample_flight_line.altitude_msl.to("meter").magnitude == pytest.approx(6000)


class TestFlightLineProperties:
    def test_azimuth(self, sample_flight_line):
        assert sample_flight_line.az12.magnitude == pytest.approx(45.0, abs=1.0)

    def test_reverse_azimuth(self, sample_flight_line):
        assert sample_flight_line.az21.magnitude == pytest.approx(225.0, abs=1.0)

    def test_waypoints(self, sample_flight_line):
        wp1 = sample_flight_line.waypoint1
        wp2 = sample_flight_line.waypoint2
        assert wp1.latitude == pytest.approx(34.05)
        assert wp2.latitude != wp1.latitude  # should have moved


class TestFlightLineOperations:
    def test_reverse(self, sample_flight_line):
        rev = sample_flight_line.reverse()
        assert rev.lat1 == pytest.approx(sample_flight_line.lat2)
        assert rev.lon1 == pytest.approx(sample_flight_line.lon2)
        assert rev.lat2 == pytest.approx(sample_flight_line.lat1)

    def test_split_by_length(self, sample_flight_line):
        # 50 km line split into 10 km segments → 5 parts
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        assert isinstance(parts, list)
        assert len(parts) == 5
        for part in parts:
            assert part.length.to("meter").magnitude == pytest.approx(10000, rel=0.01)

    def test_offset_across(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(1000, "meter"))
        # Should be a parallel line roughly 1 km away
        assert offset_fl.length.to("meter").magnitude == pytest.approx(
            sample_flight_line.length.to("meter").magnitude, rel=0.01
        )

    def test_rotate_around_midpoint(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(90.0)
        # Rotation should change azimuth; just verify it changed
        assert rotated.az12.magnitude != pytest.approx(45.0, abs=5.0)

    def test_to_dict(self, sample_flight_line):
        d = sample_flight_line.to_dict()
        assert "lat1" in d
        assert "lon1" in d
        assert "altitude_msl" in d

    def test_to_geojson(self, sample_flight_line):
        gj = sample_flight_line.to_geojson()
        assert gj["type"] == "Feature"
        assert gj["geometry"]["type"] == "LineString"


class TestToGDF:
    def test_creates_geodataframe(self, sample_flight_line, short_flight_line):
        gdf = to_gdf([sample_flight_line, short_flight_line])
        assert len(gdf) == 2
        assert gdf.crs is not None
