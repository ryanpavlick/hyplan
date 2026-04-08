"""Tests for hyplan.flight_box."""

import pytest
from shapely.geometry import Polygon
from hyplan.units import ureg
from hyplan.instruments import AVIRIS3, UAVSAR_Lband
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_box import box_around_center_line, box_around_polygon, box_around_polygon_terrain


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
            assert fl.altitude_msl.m_as("meter") == pytest.approx(6000)

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


    def test_mixed_units(self):
        """box_width in km and swath in m must produce correct line count."""
        sensor = AVIRIS3()
        altitude = ureg.Quantity(20000, "feet")
        swath = sensor.swath_width(altitude)
        spacing = swath * (1 - 20 / 100)

        box_width_km = ureg.Quantity(15, "km")
        import numpy as np
        expected = max(1, int(np.ceil(
            (box_width_km / spacing).m_as("dimensionless")
        )))

        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=altitude,
            lat0=36.8,
            lon0=-121.9,
            azimuth=0.0,
            box_length=ureg.Quantity(30, "km"),
            box_width=box_width_km,
            box_name="MixedUnits",
            overlap=20.0,
        )
        assert len(lines) == expected
        # Lines should not span more than a fraction of a degree for a 15 km box
        lons = [fl.lon1 for fl in lines]
        assert max(lons) - min(lons) < 1.0

    def test_lines_centered_on_box(self):
        """Flight lines should be symmetric around the box center."""
        sensor = AVIRIS3()
        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0,
            lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=ureg.Quantity(10000, "meter"),
            box_name="Sym",
        )
        lons = sorted([fl.lon1 for fl in lines])
        center_lon = -118.0
        # Mean longitude of lines should be close to the box center
        assert abs(sum(lons) / len(lons) - center_lon) < 0.01


class TestFlightLineCount:
    """Verify that nlines produces exactly nlines flight lines (no off-by-one)."""

    def _make_lines(self, nlines_target):
        """Helper: choose box_width so that nlines == nlines_target exactly."""
        sensor = AVIRIS3()
        altitude = ureg.Quantity(6000, "meter")
        swath = sensor.swath_width(altitude)
        spacing = swath * (1 - 20 / 100)
        # box_width that makes ceil(box_width / spacing) == nlines_target
        box_width = spacing * (nlines_target - 0.5)
        return box_around_center_line(
            instrument=sensor,
            altitude_msl=altitude,
            lat0=34.0,
            lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=box_width,
            box_name="CountTest",
            overlap=20.0,
        )

    def test_even_nlines_count(self):
        """nlines=4 must produce exactly 4 flight lines."""
        lines = self._make_lines(4)
        assert len(lines) == 4

    def test_odd_nlines_count(self):
        """nlines=5 must produce exactly 5 flight lines."""
        lines = self._make_lines(5)
        assert len(lines) == 5

    def test_single_line_centered(self):
        """nlines=1 must produce exactly 1 flight line at the box center."""
        sensor = AVIRIS3()
        altitude = ureg.Quantity(6000, "meter")
        swath = sensor.swath_width(altitude)
        # box_width smaller than one swath spacing → nlines=1
        box_width = swath * 0.5
        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=altitude,
            lat0=34.0,
            lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=box_width,
            box_name="SingleLine",
            overlap=20.0,
        )
        assert len(lines) == 1
        # The single line should pass through the box center
        assert abs(lines[0].lon1 - (-118.0)) < 0.01 or abs(lines[0].lon2 - (-118.0)) < 0.01


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

    def test_line_length_matches_polygon_extent(self):
        """Flight lines should be roughly as long as the polygon's along-track extent."""
        sensor = AVIRIS3()
        # E-W elongated polygon: ~40 km wide, ~12 km tall
        poly = Polygon([
            (-120.20, 34.05), (-120.05, 34.00), (-119.80, 34.00),
            (-119.75, 34.03), (-119.80, 34.08), (-120.05, 34.09),
            (-120.15, 34.08), (-120.20, 34.05),
        ])
        lines = box_around_polygon(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            polygon=poly,
            azimuth=90.0,
            clip_to_polygon=False,
        )
        # Lines at azimuth=90 should be ~40 km long (E-W), not ~12 km
        line_len_km = lines[0].length.m_as(ureg.km)
        assert line_len_km > 30, f"Line length {line_len_km:.1f} km is too short for E-W polygon"
        # Should have few cross-track lines (~12 km / ~3.5 km spacing)
        assert len(lines) < 10, f"Too many lines ({len(lines)}) for a 12 km cross-track extent"


class TestBoxAroundPolygonTerrain:
    # Small, flat polygon over the Mojave Desert (low elevation, minimal terrain variation)
    poly = Polygon([
        (-116.1, 34.8), (-115.9, 34.8),
        (-115.9, 35.0), (-116.1, 35.0),
    ])
    altitude = ureg.Quantity(6000, "meter")

    def test_linescanner(self):
        sensor = AVIRIS3()
        lines = box_around_polygon_terrain(
            instrument=sensor,
            altitude_msl=self.altitude,
            polygon=self.poly,
            azimuth=0.0,
            box_name="TerrainLS",
            overlap=20,
        )
        assert len(lines) > 0
        assert all("TerrainLS" in fl.site_name for fl in lines)

    def test_sar(self):
        lband = UAVSAR_Lband()
        lines = box_around_polygon_terrain(
            instrument=lband,
            altitude_msl=self.altitude,
            polygon=self.poly,
            azimuth=0.0,
            box_name="TerrainSAR",
            overlap=10,
        )
        assert len(lines) > 0
        assert all("TerrainSAR" in fl.site_name for fl in lines)

    def test_invalid_sensor(self):
        with pytest.raises(HyPlanValueError, match="swath_width"):
            box_around_polygon_terrain(
                instrument=object(),
                altitude_msl=self.altitude,
                polygon=self.poly,
            )

    def test_insufficient_clearance(self):
        # Terrain in Mojave is ~600–900 m; 100 m MSL is below it
        with pytest.raises(HyPlanValueError, match="clearance"):
            box_around_polygon_terrain(
                instrument=AVIRIS3(),
                altitude_msl=ureg.Quantity(100, "meter"),
                polygon=self.poly,
                safe_altitude=ureg.Quantity(300, "meter"),
            )
