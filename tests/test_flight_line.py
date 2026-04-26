"""Tests for hyplan.flight_line."""

import pytest
from hyplan.units import ureg
from hyplan.flight_line import FlightLine, to_gdf


class TestFlightLineCreation:
    def test_start_length_azimuth(self, sample_flight_line):
        fl = sample_flight_line
        assert fl.lat1 == pytest.approx(34.05)
        assert fl.lon1 == pytest.approx(-118.25)
        assert fl.length.m_as("meter") == pytest.approx(50000, rel=0.01)
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
        assert fl.length.m_as("meter") == pytest.approx(20000, rel=0.01)

    def test_altitude_stored(self, sample_flight_line):
        assert sample_flight_line.altitude_msl.m_as("meter") == pytest.approx(6000)


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
            assert part.length.m_as("meter") == pytest.approx(10000, rel=0.01)

    def test_offset_across(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(1000, "meter"))
        # Should be a parallel line roughly 1 km away
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
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

    def test_from_geojson_roundtrip(self, sample_flight_line):
        gj = sample_flight_line.to_geojson()
        restored = FlightLine.from_geojson(gj)
        assert restored.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert restored.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)
        assert restored.lat2 == pytest.approx(sample_flight_line.lat2, abs=1e-4)
        assert restored.lon2 == pytest.approx(sample_flight_line.lon2, abs=1e-4)
        assert restored.altitude_msl.m_as("meter") == pytest.approx(
            sample_flight_line.altitude_msl.m_as("meter"), rel=1e-3
        )
        assert restored.site_name == sample_flight_line.site_name

    def test_from_geojson_preserves_metadata(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(10, "km"), az=90,
            altitude_msl=ureg.Quantity(3000, "meter"),
            site_name="Test Line",
            site_description="A test",
            investigator="Dr. Smith",
        )
        gj = fl.to_geojson()
        restored = FlightLine.from_geojson(gj)
        assert restored.site_name == "Test Line"
        assert restored.site_description == "A test"
        assert restored.investigator == "Dr. Smith"


class TestSplitByLength:
    def test_line_shorter_than_max_returns_self(self, short_flight_line):
        """A line shorter than max_length should return itself unchanged."""
        parts = short_flight_line.split_by_length(ureg.Quantity(20000, "meter"))
        assert len(parts) == 1
        assert parts[0].length.m_as("meter") == pytest.approx(
            short_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_segments_cover_full_length(self, sample_flight_line):
        """Sum of segment lengths should equal total line length."""
        parts = sample_flight_line.split_by_length(ureg.Quantity(12000, "meter"))
        total = sum(p.length.m_as("meter") for p in parts)
        assert total == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_segments_preserve_altitude(self, sample_flight_line):
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        for part in parts:
            assert part.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_split_with_gap(self, sample_flight_line):
        """Splitting with a gap should produce fewer segments than without."""
        parts_no_gap = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        parts_with_gap = sample_flight_line.split_by_length(
            ureg.Quantity(10000, "meter"),
            gap_length=ureg.Quantity(5000, "meter"),
        )
        # With gaps the segments don't cover the full line, so fewer segments
        assert len(parts_with_gap) <= len(parts_no_gap)
        # Each segment should still be at most max_length
        for part in parts_with_gap:
            assert part.length.m_as("meter") <= 10001  # tolerance

    def test_split_invalid_max_length(self, sample_flight_line):
        """Zero or negative max_length should raise an error."""
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(ureg.Quantity(0, "meter"))
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(ureg.Quantity(-100, "meter"))

    def test_split_negative_gap(self, sample_flight_line):
        """Negative gap_length should raise an error."""
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(
                ureg.Quantity(10000, "meter"),
                gap_length=ureg.Quantity(-500, "meter"),
            )

    def test_segment_names_include_index(self, sample_flight_line):
        """Each segment should get a unique name with its index."""
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        for i, part in enumerate(parts):
            assert f"seg_{i}" in part.site_name


class TestOffsetAcross:
    def test_positive_offset_preserves_length(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(2000, "meter"))
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_negative_offset_preserves_length(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(-2000, "meter"))
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_offset_preserves_altitude(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(500, "meter"))
        assert offset_fl.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_zero_offset_returns_same_position(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(0, "meter"))
        assert offset_fl.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert offset_fl.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)

    def test_plain_float_assumed_meters(self, sample_flight_line):
        """A plain float should be treated as meters."""
        offset_fl = sample_flight_line.offset_across(1000.0)
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_opposite_offsets_are_symmetric(self, sample_flight_line):
        """Positive and negative offsets should move the line to opposite sides."""
        right = sample_flight_line.offset_across(ureg.Quantity(1000, "meter"))
        left = sample_flight_line.offset_across(ureg.Quantity(-1000, "meter"))
        # The midpoints should be on opposite sides of the original
        mid_orig_lat = (sample_flight_line.lat1 + sample_flight_line.lat2) / 2
        mid_right_lat = (right.lat1 + right.lat2) / 2
        mid_left_lat = (left.lat1 + left.lat2) / 2
        # One should be above and one below the original (for az=45)
        assert (mid_right_lat - mid_orig_lat) != pytest.approx(
            mid_left_lat - mid_orig_lat, abs=1e-5
        )


class TestRotateAroundMidpoint:
    def test_rotation_changes_azimuth(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(45.0)
        # Az should have changed significantly from original ~45 degrees
        assert rotated.az12.magnitude != pytest.approx(
            sample_flight_line.az12.magnitude, abs=10.0
        )

    def test_rotation_preserves_length(self, short_flight_line):
        """Rotation should approximately preserve length. Note that
        rotate_around_midpoint operates in planar lon/lat space, so
        some distortion is expected, especially at higher latitudes
        and larger rotation angles."""
        rotated = short_flight_line.rotate_around_midpoint(10.0)
        assert rotated.length.m_as("meter") == pytest.approx(
            short_flight_line.length.m_as("meter"), rel=0.10
        )

    def test_rotation_preserves_altitude(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(90.0)
        assert rotated.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_360_rotation_returns_original(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(360.0)
        assert rotated.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert rotated.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)

    def test_invalid_angle_type_raises(self, sample_flight_line):
        with pytest.raises(Exception):
            sample_flight_line.rotate_around_midpoint("ninety")


class TestTrack:
    def test_track_returns_linestring(self, sample_flight_line):
        from shapely.geometry import LineString
        track = sample_flight_line.track(precision=500.0)
        assert isinstance(track, LineString)
        assert track.is_valid

    def test_track_endpoints_match(self, sample_flight_line):
        track = sample_flight_line.track(precision=1000.0)
        coords = list(track.coords)
        # First point should be near (lon1, lat1)
        assert coords[0][0] == pytest.approx(sample_flight_line.lon1, abs=1e-3)
        assert coords[0][1] == pytest.approx(sample_flight_line.lat1, abs=1e-3)
        # Last point should be near (lon2, lat2)
        assert coords[-1][0] == pytest.approx(sample_flight_line.lon2, abs=1e-3)
        assert coords[-1][1] == pytest.approx(sample_flight_line.lat2, abs=1e-3)

    def test_track_point_count_scales_with_precision(self, sample_flight_line):
        fine = sample_flight_line.track(precision=100.0)
        coarse = sample_flight_line.track(precision=1000.0)
        assert len(list(fine.coords)) > len(list(coarse.coords))

    def test_track_with_quantity_precision(self, sample_flight_line):
        track = sample_flight_line.track(precision=ureg.Quantity(500, "meter"))
        assert len(list(track.coords)) > 2

    def test_track_invalid_precision_raises(self, sample_flight_line):
        with pytest.raises(Exception):
            sample_flight_line.track(precision=0.0)
        with pytest.raises(Exception):
            sample_flight_line.track(precision=-100.0)


class TestErrorPaths:
    def test_negative_altitude_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=ureg.Quantity(10000, "meter"),
                az=90.0,
                altitude_msl=ureg.Quantity(-100, "meter"),
            )

    def test_non_quantity_length_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=10000,  # not a Quantity
                az=90.0,
                altitude_msl=ureg.Quantity(3000, "meter"),
            )

    def test_non_numeric_azimuth_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=ureg.Quantity(10000, "meter"),
                az="north",
                altitude_msl=ureg.Quantity(3000, "meter"),
            )

    def test_non_waypoint_init_raises(self):
        from hyplan.exceptions import HyPlanTypeError
        with pytest.raises(HyPlanTypeError):
            FlightLine(waypoint1="not_a_waypoint", waypoint2="also_not")

    def test_altitude_setter_validates(self, short_flight_line):
        with pytest.raises(Exception):
            short_flight_line.altitude_msl = ureg.Quantity(-500, "meter")

    def test_from_endpoints(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0,
            lat2=34.1, lon2=-117.9,
            altitude_msl=ureg.Quantity(5000, "meter"),
            site_name="EP",
        )
        assert fl.geometry.is_valid
        assert fl.site_name == "EP"


class TestToGDF:
    def test_creates_geodataframe(self, sample_flight_line, short_flight_line):
        gdf = to_gdf([sample_flight_line, short_flight_line])
        assert len(gdf) == 2
        assert gdf.crs is not None
