"""Tests for hyplan.flight_box."""

import pytest
from shapely.geometry import Polygon
from hyplan.units import ureg
from hyplan.instruments import AVIRIS3, UAVSAR_Lband
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_box import (
    box_around_center_line,
    box_around_polygon,
    box_around_polygon_terrain,
    altitude_msl_for_pixel_size,
    _validate_inputs,
)


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


class TestAltitudeMslForPixelSize:
    """Tests for altitude_msl_for_pixel_size."""

    # Small, flat polygon over the Mojave Desert (low elevation)
    poly = Polygon([
        (-116.1, 34.8), (-115.9, 34.8),
        (-115.9, 35.0), (-116.1, 35.0),
    ])

    def test_returns_quantity(self):
        from hyplan.terrain import generate_demfile
        import numpy as np

        sensor = AVIRIS3()
        lats = np.array([34.8, 35.0])
        lons = np.array([-116.1, -115.9])
        dem_file = generate_demfile(lats, lons)

        result = altitude_msl_for_pixel_size(
            instrument=sensor,
            pixel_size=ureg.Quantity(5, "meter"),
            dem_file=dem_file,
        )
        assert hasattr(result, "magnitude")
        assert result.m_as("meter") > 0

    def test_larger_pixel_gives_higher_altitude(self):
        from hyplan.terrain import generate_demfile
        import numpy as np

        sensor = AVIRIS3()
        lats = np.array([34.8, 35.0])
        lons = np.array([-116.1, -115.9])
        dem_file = generate_demfile(lats, lons)

        alt_small = altitude_msl_for_pixel_size(
            instrument=sensor,
            pixel_size=ureg.Quantity(3, "meter"),
            dem_file=dem_file,
        )
        alt_large = altitude_msl_for_pixel_size(
            instrument=sensor,
            pixel_size=ureg.Quantity(10, "meter"),
            dem_file=dem_file,
        )
        # Larger pixel size requires higher altitude
        assert alt_large.m_as("meter") > alt_small.m_as("meter")


class TestBoxAroundPolygonEdgeCases:
    """Edge cases for box_around_polygon."""

    def test_invalid_polygon_type_raises(self):
        sensor = AVIRIS3()
        with pytest.raises(HyPlanValueError, match="Polygon"):
            box_around_polygon(
                instrument=sensor,
                altitude_msl=ureg.Quantity(6000, "meter"),
                polygon="not_a_polygon",
            )

    def test_small_polygon_produces_lines(self):
        """A very small polygon should still produce at least one line."""
        sensor = AVIRIS3()
        tiny_poly = Polygon([
            (-118.001, 33.999), (-117.999, 33.999),
            (-117.999, 34.001), (-118.001, 34.001),
        ])
        lines = box_around_polygon(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            polygon=tiny_poly,
            azimuth=0.0,
            clip_to_polygon=False,
        )
        assert len(lines) >= 1

    def test_no_azimuth_uses_minimum_rotated_rectangle(self):
        """When azimuth=None, the function should auto-detect orientation."""
        sensor = AVIRIS3()
        poly = Polygon([
            (-118.1, 33.9), (-117.9, 33.9),
            (-117.9, 34.1), (-118.1, 34.1),
        ])
        lines = box_around_polygon(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            polygon=poly,
            azimuth=None,
            box_name="AutoAz",
            clip_to_polygon=False,
        )
        assert len(lines) > 0
        # All lines should have site_name containing box_name
        for fl in lines:
            assert "AutoAz" in fl.site_name

    def test_clip_to_polygon_false(self):
        """When clip_to_polygon=False, all lines should have the same length
        (no clipping applied), vs clipped lines which may be shorter."""
        sensor = AVIRIS3()
        poly = Polygon([
            (-118.1, 33.9), (-117.9, 33.9),
            (-117.9, 34.1), (-118.1, 34.1),
        ])
        lines_unclipped = box_around_polygon(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            polygon=poly,
            azimuth=0.0,
            clip_to_polygon=False,
        )
        # Without clipping, all lines should have the same length
        lengths = [fl.length.m_as("meter") for fl in lines_unclipped]
        for length in lengths:
            assert length == pytest.approx(lengths[0], rel=0.01)


class TestValidateInputs:
    """Tests for the _validate_inputs helper."""

    def test_negative_altitude_raises(self):
        with pytest.raises(HyPlanValueError):
            _validate_inputs(altitude=ureg.Quantity(-100, "meter"))

    def test_zero_box_length_raises(self):
        with pytest.raises(HyPlanValueError):
            _validate_inputs(box_length=ureg.Quantity(0, "meter"))

    def test_overlap_out_of_range_raises(self):
        with pytest.raises(HyPlanValueError):
            _validate_inputs(overlap=150)
        with pytest.raises(HyPlanValueError):
            _validate_inputs(overlap=-10)

    def test_invalid_starting_point_raises(self):
        with pytest.raises(HyPlanValueError):
            _validate_inputs(starting_point="middle")

    def test_valid_inputs_pass(self):
        """Valid inputs should not raise."""
        _validate_inputs(
            altitude=ureg.Quantity(6000, "meter"),
            box_length=ureg.Quantity(50000, "meter"),
            overlap=20,
            azimuth=45.0,
            starting_point="center",
        )

    def test_wrong_dimensionality_raises(self):
        """A Quantity with wrong units (e.g. seconds) should raise."""
        with pytest.raises(HyPlanValueError):
            _validate_inputs(altitude=ureg.Quantity(5, "second"))

    def test_invalid_azimuth_type_raises(self):
        with pytest.raises(HyPlanValueError):
            _validate_inputs(azimuth="north")


class TestBoxAroundCenterLineValidation:
    """Input validation edge cases for box_around_center_line."""

    def test_invalid_instrument_raises(self):
        with pytest.raises(HyPlanValueError, match="swath_width"):
            box_around_center_line(
                instrument=object(),
                altitude_msl=ureg.Quantity(6000, "meter"),
                lat0=34.0, lon0=-118.0,
                azimuth=0.0,
                box_length=ureg.Quantity(50000, "meter"),
                box_width=ureg.Quantity(10000, "meter"),
            )

    def test_alternate_direction_false(self):
        """All lines should have the same direction when alternate_direction=False."""
        sensor = AVIRIS3()
        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0, lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=ureg.Quantity(10000, "meter"),
            alternate_direction=False,
        )
        # When not alternating, all forward azimuths should be similar
        # Normalize to [0, 360) to handle wrap-around (e.g. 0 vs 360)
        import numpy as np
        azimuths = [fl.az12.magnitude % 360 for fl in lines]
        for az in azimuths:
            diff = abs(az - azimuths[0])
            # Handle wrap-around: 0 and 360 are equivalent
            diff = min(diff, 360 - diff)
            assert diff < 1.0, f"Azimuth {az} differs from {azimuths[0]} by {diff}"

    def test_starting_point_edge(self):
        """starting_point='edge' should start from the edge, not center."""
        sensor = AVIRIS3()
        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0, lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(50000, "meter"),
            box_width=ureg.Quantity(10000, "meter"),
            starting_point="edge",
        )
        assert len(lines) > 0
