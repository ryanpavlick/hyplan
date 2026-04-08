"""Tests for hyplan.geometry."""

import pytest
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import transform
from hyplan.geometry import (
    wrap_to_180,
    wrap_to_360,
    haversine,
    get_utm_crs,
    calculate_geographic_mean,
    minimum_rotated_rectangle,
    _validate_polygon,
    get_utm_transforms,
    rotated_rectangle,
    buffer_polygon_along_azimuth,
    process_linestring,
    dd_to_ddms,
    dd_to_nddmm,
    dd_to_ddm,
    dd_to_foreflight_oneline,
    translate_polygon,
    random_points_in_polygon,
    true_to_magnetic,
    get_timezone,
)
from hyplan.exceptions import HyPlanValueError, HyPlanTypeError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_polygon():
    """A small polygon near Los Angeles in WGS84."""
    return Polygon([(-118.3, 34.0), (-118.2, 34.0), (-118.2, 34.1), (-118.3, 34.1)])


@pytest.fixture
def unit_polygon():
    """A 1x1 degree polygon at the origin."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------

class TestWrapAngles:
    def test_wrap_to_180(self):
        assert wrap_to_180(190) == pytest.approx(-170)
        assert wrap_to_180(-190) == pytest.approx(170)
        assert wrap_to_180(0) == pytest.approx(0)
        assert wrap_to_180(180) == pytest.approx(-180)  # 180 and -180 are equivalent

    def test_wrap_to_360(self):
        result = wrap_to_360(-10)
        assert result == pytest.approx(350)
        assert wrap_to_360(370) == pytest.approx(10)
        assert wrap_to_360(0) == pytest.approx(0)


class TestHaversine:
    def test_zero_distance(self):
        assert haversine(0, 0, 0, 0) == pytest.approx(0)

    def test_known_distance(self):
        # LA to NYC: ~3944 km
        dist = haversine(34.05, -118.25, 40.71, -74.01)
        assert dist == pytest.approx(3944000, rel=0.02)

    def test_equator_one_degree(self):
        # 1 degree longitude at equator ~ 111 km
        dist = haversine(0, 0, 0, 1)
        assert dist == pytest.approx(111195, rel=0.01)


class TestUTM:
    def test_get_utm_crs(self):
        crs = get_utm_crs(-118.25, 34.05)
        assert crs is not None
        # EPSG codes for UTM zones are 326xx (north) and 327xx (south)
        assert "326" in crs.to_string() or "327" in crs.to_string()


class TestGeographicMean:
    def test_point_mean(self):
        poly = Polygon([(-118, 34), (-117, 34), (-117, 35), (-118, 35)])
        mean = calculate_geographic_mean(poly)
        assert mean.x == pytest.approx(-117.5, abs=0.2)
        assert mean.y == pytest.approx(34.5, abs=0.2)


class TestMinimumRotatedRectangle:
    def test_square(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        rect = minimum_rotated_rectangle(poly)
        assert rect.area == pytest.approx(1.0, rel=0.01)

    def test_output_contains_input(self, simple_polygon):
        rect = minimum_rotated_rectangle(simple_polygon)
        # The minimum rotated rectangle must contain the original polygon
        assert rect.contains(simple_polygon) or rect.buffer(1e-6).contains(simple_polygon)


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------

class TestValidatePolygon:
    def test_valid_polygon(self, simple_polygon):
        assert _validate_polygon(simple_polygon) is True

    def test_none_returns_none(self):
        assert _validate_polygon(None) is None

    def test_non_polygon_type_raises(self):
        with pytest.raises(HyPlanValueError, match="Input must be a Shapely Polygon"):
            _validate_polygon("not a polygon")

    def test_multipolygon_raises(self):
        mp = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ])
        with pytest.raises(HyPlanValueError, match="MultiPolygon"):
            _validate_polygon(mp)

    def test_empty_polygon_raises(self):
        with pytest.raises(HyPlanValueError, match="empty"):
            _validate_polygon(Polygon())

    def test_point_type_raises(self):
        with pytest.raises(HyPlanValueError, match="Input must be a Shapely Polygon"):
            _validate_polygon(Point(0, 0))


class TestGetUtmTransforms:
    def test_round_trip(self, simple_polygon):
        """Transform WGS84 -> UTM -> WGS84 should recover original coordinates."""
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(simple_polygon)
        poly_utm = transform(wgs84_to_utm, simple_polygon)
        poly_back = transform(utm_to_wgs84, poly_utm)

        orig_coords = np.array(simple_polygon.exterior.coords)
        back_coords = np.array(poly_back.exterior.coords)
        np.testing.assert_allclose(orig_coords, back_coords, atol=1e-6)

    def test_point_input(self):
        pt = Point(-118.25, 34.05)
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(pt)
        pt_utm = transform(wgs84_to_utm, pt)
        pt_back = transform(utm_to_wgs84, pt_utm)
        assert pt_back.x == pytest.approx(pt.x, abs=1e-6)
        assert pt_back.y == pytest.approx(pt.y, abs=1e-6)

    def test_list_of_geometries(self):
        geoms = [Point(-118, 34), Point(-117, 35)]
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(geoms)
        assert callable(wgs84_to_utm)
        assert callable(utm_to_wgs84)

    def test_invalid_input_raises(self):
        with pytest.raises(HyPlanTypeError):
            get_utm_transforms("not a geometry")


class TestRotatedRectangle:
    def test_zero_azimuth_contains_input(self, simple_polygon):
        rect = rotated_rectangle(simple_polygon, azimuth=0.0)
        assert rect.is_valid
        # Rotated bounding box must contain the original polygon
        assert rect.buffer(1e-6).contains(simple_polygon)

    def test_known_azimuth_produces_rectangle(self, simple_polygon):
        rect = rotated_rectangle(simple_polygon, azimuth=45.0)
        assert rect.is_valid
        # Should be a 5-coordinate ring (closed rectangle)
        assert len(rect.exterior.coords) == 5

    def test_azimuth_90_contains_input(self, simple_polygon):
        rect = rotated_rectangle(simple_polygon, azimuth=90.0)
        assert rect.buffer(1e-6).contains(simple_polygon)


class TestBufferPolygonAlongAzimuth:
    def test_output_larger_than_input(self, simple_polygon):
        buffered = buffer_polygon_along_azimuth(
            simple_polygon,
            along_track_distance=5000.0,
            across_track_distance=5000.0,
            azimuth=0.0,
        )
        assert buffered.area > simple_polygon.area

    def test_output_contains_input(self, simple_polygon):
        buffered = buffer_polygon_along_azimuth(
            simple_polygon,
            along_track_distance=1000.0,
            across_track_distance=1000.0,
            azimuth=45.0,
        )
        assert buffered.buffer(1e-6).contains(simple_polygon)

    def test_invalid_distance_raises(self, simple_polygon):
        with pytest.raises(HyPlanValueError):
            buffer_polygon_along_azimuth(
                simple_polygon,
                along_track_distance=-1.0,
                across_track_distance=1000.0,
                azimuth=0.0,
            )


class TestProcessLinestring:
    def test_basic_output(self):
        # A simple north-south line
        ls = LineString([(-118.25, 34.0), (-118.25, 34.1), (-118.25, 34.2)])
        lats, lons, azimuths, distances = process_linestring(ls)

        assert len(lats) == 3
        assert len(lons) == 3
        assert len(azimuths) == 3
        assert len(distances) == 3

        # Latitudes should match input
        np.testing.assert_allclose(lats, [34.0, 34.1, 34.2], atol=1e-6)
        # Longitudes should match input
        np.testing.assert_allclose(lons, [-118.25, -118.25, -118.25], atol=1e-6)

        # First distance should be 0
        assert distances[0] == pytest.approx(0.0)
        # Distances should be monotonically increasing
        assert np.all(np.diff(distances) > 0)

    def test_heading_north(self):
        ls = LineString([(0.0, 0.0), (0.0, 1.0)])
        _, _, azimuths, _ = process_linestring(ls)
        # Heading north -> azimuth ~ 0 degrees
        assert azimuths[0] == pytest.approx(0.0, abs=0.5)

    def test_heading_east(self):
        ls = LineString([(0.0, 0.0), (1.0, 0.0)])
        _, _, azimuths, _ = process_linestring(ls)
        # Heading east -> azimuth ~ 90 degrees
        assert azimuths[0] == pytest.approx(90.0, abs=0.5)

    def test_invalid_input_raises(self):
        with pytest.raises(HyPlanValueError):
            process_linestring("not a linestring")


class TestDdToDdms:
    def test_known_coordinate(self):
        # 37.405 degrees = 37 deg 24 min 18.0 sec
        lat_str, lon_str = dd_to_ddms(37.405, -122.0575)
        assert lat_str == "37 24 18.0"
        assert lon_str == "-122 03 27.0"

    def test_zero(self):
        lat_str, lon_str = dd_to_ddms(0.0, 0.0)
        assert lat_str == "00 00 00.0"
        assert lon_str == "000 00 00.0"

    def test_negative_latitude(self):
        lat_str, _ = dd_to_ddms(-33.8688, 0.0)
        assert lat_str.startswith("-")


class TestDdToNddmm:
    def test_known_coordinate(self):
        lat_str, lon_str = dd_to_nddmm(37.405, -122.0575)
        assert lat_str.startswith("N")
        assert lon_str.startswith("W")
        # 37.405 -> N37 24.30
        assert lat_str == "N37 24.30"
        # -122.0575 -> W122 03.45
        assert lon_str == "W122 03.45"

    def test_southern_hemisphere(self):
        lat_str, _ = dd_to_nddmm(-33.8688, 151.2093)
        assert lat_str.startswith("S")

    def test_eastern_hemisphere(self):
        _, lon_str = dd_to_nddmm(48.8566, 2.3522)
        assert lon_str.startswith("E")


class TestDdToDdm:
    def test_known_coordinate(self):
        lat_str, lon_str = dd_to_ddm(37.405, -122.0575)
        assert lat_str == "37 24.30"
        assert lon_str == "-122 03.45"

    def test_positive_values(self):
        lat_str, lon_str = dd_to_ddm(0.0, 0.0)
        assert lat_str == "00 00.00"
        assert lon_str == "000 00.00"


class TestDdToForeflightOneline:
    def test_known_coordinate(self):
        result = dd_to_foreflight_oneline(37.405, -122.0575)
        assert result.startswith("N")
        assert "/W" in result

    def test_format(self):
        result = dd_to_foreflight_oneline(0.0, 0.0)
        # Should be N00xxx/E000xxx format
        assert "/" in result
        parts = result.split("/")
        assert len(parts) == 2


class TestTranslatePolygon:
    def test_centroid_shifts(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        # Translate 1 unit north (azimuth=0)
        translated = translate_polygon(poly, distance=1.0, azimuth=0.0)
        orig_centroid = poly.centroid
        new_centroid = translated.centroid
        # y should increase by ~1, x should stay the same
        assert new_centroid.y == pytest.approx(orig_centroid.y + 1.0, abs=1e-6)
        assert new_centroid.x == pytest.approx(orig_centroid.x, abs=1e-6)

    def test_translate_east(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        # Translate 2 units east (azimuth=90)
        translated = translate_polygon(poly, distance=2.0, azimuth=90.0)
        orig_centroid = poly.centroid
        new_centroid = translated.centroid
        assert new_centroid.x == pytest.approx(orig_centroid.x + 2.0, abs=1e-6)
        assert new_centroid.y == pytest.approx(orig_centroid.y, abs=1e-6)

    def test_area_preserved(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        translated = translate_polygon(poly, distance=5.0, azimuth=45.0)
        assert translated.area == pytest.approx(poly.area, rel=1e-10)


class TestRandomPointsInPolygon:
    def test_correct_count(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        points = random_points_in_polygon(poly, 50)
        assert len(points) == 50

    def test_points_are_shapely_points(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        points = random_points_in_polygon(poly, 5)
        for pt in points:
            assert isinstance(pt, Point)


class TestTrueToMagnetic:
    def test_zero_declination(self):
        assert true_to_magnetic(90.0, 0.0) == pytest.approx(90.0)

    def test_east_declination(self):
        # East declination (positive) -> magnetic heading is less than true
        assert true_to_magnetic(90.0, 10.0) == pytest.approx(80.0)

    def test_west_declination(self):
        # West declination (negative) -> magnetic heading is greater than true
        assert true_to_magnetic(90.0, -10.0) == pytest.approx(100.0)

    def test_wrap_around(self):
        # True heading 5, declination 10 -> magnetic 355
        assert true_to_magnetic(5.0, 10.0) == pytest.approx(355.0)


class TestCalculateGeographicMeanExtended:
    def test_single_point(self):
        mean = calculate_geographic_mean(Point(-118.0, 34.0))
        assert mean.x == pytest.approx(-118.0, abs=1e-6)
        assert mean.y == pytest.approx(34.0, abs=1e-6)

    def test_linestring(self):
        ls = LineString([(0, 0), (2, 0)])
        mean = calculate_geographic_mean(ls)
        assert mean.x == pytest.approx(1.0, abs=0.1)
        assert mean.y == pytest.approx(0.0, abs=0.1)

    def test_list_of_points(self):
        pts = [Point(0, 0), Point(2, 0)]
        mean = calculate_geographic_mean(pts)
        assert mean.x == pytest.approx(1.0, abs=0.1)

    def test_invalid_input_raises(self):
        with pytest.raises(HyPlanTypeError):
            calculate_geographic_mean("not a geometry")


class TestGetTimezone:
    def test_los_angeles(self):
        assert get_timezone(34.05, -118.25) == "America/Los_Angeles"

    def test_new_york(self):
        assert get_timezone(40.71, -74.00) == "America/New_York"

    def test_sydney(self):
        assert get_timezone(-33.87, 151.21) == "Australia/Sydney"

    def test_invalid_latitude(self):
        with pytest.raises(HyPlanValueError):
            get_timezone(91.0, 0.0)

    def test_invalid_longitude(self):
        with pytest.raises(HyPlanValueError):
            get_timezone(0.0, 200.0)
