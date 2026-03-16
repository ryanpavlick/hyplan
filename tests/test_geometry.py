"""Tests for hyplan.geometry."""

import pytest
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from hyplan.geometry import (
    wrap_to_180,
    wrap_to_360,
    haversine,
    get_utm_crs,
    calculate_geographic_mean,
    minimum_rotated_rectangle,
)


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
        # 1 degree longitude at equator ≈ 111 km
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
