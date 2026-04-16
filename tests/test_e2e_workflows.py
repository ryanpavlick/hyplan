"""End-to-end workflow regression tests.

Each test chains multiple HyPlan subsystems into a realistic planning
workflow.  All tests are deterministic, require no network access, and
use synthetic data only.
"""

from __future__ import annotations


import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from hyplan import (
    Airport,
    AVIRIS3,
    ConstantWindField,
    KingAirB200,
    FlightLine,
    StillAirField,
    box_around_center_line,
    compute_flight_plan,
    generate_swath_polygon,
    calculate_swath_widths,
    initialize_data,
    ureg,
)
from hyplan.exports import (
    to_kml,
    to_gpx,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def airport_db():
    """Initialize the airport database once for all tests."""
    initialize_data(countries=["US"])


def _make_lines(
    n: int = 3,
    lat0: float = 34.4,
    lon0: float = -119.8,
    spacing_m: float = 2000.0,
    length_m: float = 50_000.0,
    altitude_ft: float = 20_000.0,
    azimuth: float = 90.0,
) -> list[FlightLine]:
    """Create *n* parallel flight lines with alternating direction."""
    lines = []
    for i in range(n):
        lat = lat0 + (i - n // 2) * (spacing_m / 111_000.0)
        az = azimuth if i % 2 == 0 else (azimuth + 180) % 360
        fl = FlightLine.center_length_azimuth(
            lat=lat, lon=lon0,
            length=ureg.Quantity(length_m, "meter"),
            az=az,
            altitude_msl=ureg.Quantity(altitude_ft, "feet"),
            site_name=f"Line_{i + 1:02d}",
        )
        lines.append(fl)
    return lines


def _write_flat_dem(filepath: str, lat: float, lon: float, elev: float = 500.0):
    """Write a small flat GeoTIFF DEM centered on (lat, lon)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    size = 100
    pixel_deg = 0.001
    half = size * pixel_deg / 2
    raster = np.full((size, size), elev, dtype=np.float32)
    transform = from_bounds(
        lon - half, lat - half, lon + half, lat + half, size, size,
    )
    with rasterio.open(
        filepath, "w", driver="GTiff",
        height=size, width=size, count=1, dtype="float32",
        crs=CRS.from_epsg(4326), transform=transform, nodata=-9999,
    ) as dst:
        dst.write(raster, 1)


# ===================================================================
# Test 1: Simple survey — flat terrain, no wind
# ===================================================================

class TestSimpleSurvey:
    """Generate flight lines, compute a plan in still air, verify output."""

    def test_plan_segments(self):
        aircraft = KingAirB200()
        lines = _make_lines(n=3)

        plan = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
        )

        # Plan should have rows for transits and flight lines
        assert len(plan) > 0
        seg_types = set(plan["segment_type"].unique())
        assert "flight_line" in seg_types
        assert "transit" in seg_types

        # All flight line segments should be present
        fl_names = set(plan[plan["segment_type"] == "flight_line"]["segment_name"])
        assert fl_names == {"Line_01", "Line_02", "Line_03"}

        # All times should be positive
        assert (plan["time_to_segment"] > 0).all()

        # Total mission time should be reasonable (< 10 hours)
        total_min = plan["time_to_segment"].sum()
        assert 0 < total_min < 600

    def test_swath_geometry(self):
        """Generate swath polygon and verify it has reasonable width."""
        sensor = AVIRIS3()
        line = _make_lines(n=1, altitude_ft=20_000)[0]

        poly = generate_swath_polygon(line, sensor, along_precision=5000.0)
        assert poly.is_valid
        assert poly.area > 0

        widths = calculate_swath_widths(poly)
        assert widths["min_width"] > 0
        assert widths["max_width"] > widths["min_width"] * 0.5  # not degenerate
        assert widths["mean_width"] > 0


# ===================================================================
# Test 2: Wind-aware survey
# ===================================================================

class TestWindAwareSurvey:
    """Plan with a constant wind and verify wind effects on timing."""

    def test_headwind_increases_time(self):
        """East-west lines with westerly wind should take longer than still air."""
        aircraft = KingAirB200()
        lines = _make_lines(n=2, azimuth=90.0)

        plan_still = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
            wind_source=StillAirField(),
        )
        plan_wind = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
            wind_source=ConstantWindField(
                wind_speed=40 * ureg.knot, wind_from_deg=270.0,
            ),
        )

        time_still = plan_still["time_to_segment"].sum()
        time_wind = plan_wind["time_to_segment"].sum()

        # A 40 kt headwind on eastbound legs should increase total time
        assert time_wind > time_still

    def test_wind_populates_crab_angle(self):
        """Crosswind should produce non-zero crab angles."""
        aircraft = KingAirB200()
        lines = _make_lines(n=1, azimuth=0.0)  # north-south line

        plan = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
            wind_source=ConstantWindField(
                wind_speed=30 * ureg.knot, wind_from_deg=270.0,
            ),
        )

        fl_rows = plan[plan["segment_type"] == "flight_line"]
        assert len(fl_rows) > 0
        # Wind from west on a north-south line is pure crosswind
        assert (fl_rows["crab_angle_deg"].abs() > 0.1).any()


# ===================================================================
# Test 3: Terrain-aware swath
# ===================================================================

class TestTerrainAwareSwath:
    """Generate a swath polygon over a synthetic DEM."""

    def test_swath_with_dem(self, tmp_path):
        """Swath over flat terrain should be similar to no-terrain case."""
        sensor = AVIRIS3()
        line = _make_lines(n=1, lat0=34.0, lon0=-118.0, altitude_ft=20_000)[0]

        dem_file = str(tmp_path / "flat.tif")
        _write_flat_dem(dem_file, lat=34.0, lon=-118.0, elev=200.0)

        poly = generate_swath_polygon(
            line, sensor,
            along_precision=5000.0,
            dem_file=dem_file,
        )
        assert poly.is_valid
        assert poly.area > 0

        widths = calculate_swath_widths(poly)
        # Over flat terrain, swath should be reasonably uniform
        ratio = widths["max_width"] / widths["min_width"]
        assert ratio < 2.0, f"Swath width ratio {ratio:.1f} too variable for flat terrain"

    def test_elevated_terrain_reduces_swath(self, tmp_path):
        """Higher terrain (less AGL) should produce narrower swath."""
        sensor = AVIRIS3()
        alt_ft = 20_000
        line_low = _make_lines(n=1, lat0=34.0, lon0=-118.0, altitude_ft=alt_ft)[0]
        line_high = _make_lines(n=1, lat0=34.05, lon0=-118.0, altitude_ft=alt_ft)[0]

        dem_low = str(tmp_path / "low.tif")
        dem_high = str(tmp_path / "high.tif")
        _write_flat_dem(dem_low, lat=34.0, lon=-118.0, elev=100.0)
        _write_flat_dem(dem_high, lat=34.05, lon=-118.0, elev=3000.0)

        poly_low = generate_swath_polygon(
            line_low, sensor, along_precision=5000.0, dem_file=dem_low,
        )
        poly_high = generate_swath_polygon(
            line_high, sensor, along_precision=5000.0, dem_file=dem_high,
        )

        widths_low = calculate_swath_widths(poly_low)
        widths_high = calculate_swath_widths(poly_high)

        # Higher terrain = less AGL = narrower swath
        assert widths_high["mean_width"] < widths_low["mean_width"]


# ===================================================================
# Test 4: Airport departure/return mission
# ===================================================================

class TestAirportMission:
    """Full mission with takeoff, flight lines, and return to airport."""

    def test_departure_return(self, airport_db):
        aircraft = KingAirB200()
        departure = Airport("KSBA")
        lines = _make_lines(n=3)

        plan = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
            takeoff_airport=departure,
            return_airport=departure,
        )

        seg_types = list(plan["segment_type"])

        # Should start with takeoff/climb and end with descent/approach
        assert seg_types[0] in ("takeoff", "climb")
        assert seg_types[-1] in ("approach", "descent")

        # All three flight lines present
        fl_names = set(plan[plan["segment_type"] == "flight_line"]["segment_name"])
        assert fl_names == {"Line_01", "Line_02", "Line_03"}

        # Total time should be reasonable
        total_min = plan["time_to_segment"].sum()
        assert 0 < total_min < 600

    def test_exports(self, airport_db, tmp_path):
        """Verify that a plan can be exported to KML and GPX."""
        aircraft = KingAirB200()
        departure = Airport("KSBA")
        lines = _make_lines(n=2)

        plan = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
            takeoff_airport=departure,
            return_airport=departure,
        )

        kml_path = str(tmp_path / "mission.kml")
        gpx_path = str(tmp_path / "mission.gpx")

        to_kml(plan, kml_path)
        to_gpx(plan, gpx_path)

        # Files should be created and non-empty
        import os
        assert os.path.getsize(kml_path) > 100
        assert os.path.getsize(gpx_path) > 100


# ===================================================================
# Test 5: Cloud-driven scheduling (no network)
# ===================================================================

class TestCloudScheduling:
    """Simulate visit scheduling from synthetic cloud data."""

    def test_simulate_visits(self):
        import pandas as pd
        from hyplan.clouds import simulate_visits

        # Build synthetic cloud data: 3 polygons, 2 years
        rows = []
        for year in [2020, 2021]:
            for doy in range(1, 91):  # Jan-Mar
                for poly in ["SiteA", "SiteB", "SiteC"]:
                    # Clear every 4th day for SiteA, every 5th for B, every 6th for C
                    period = {"SiteA": 4, "SiteB": 5, "SiteC": 6}[poly]
                    cf = 0.05 if doy % period == 0 else 0.60
                    rows.append({
                        "polygon_id": poly,
                        "year": year,
                        "day_of_year": doy,
                        "cloud_fraction": cf,
                    })
        cloud_df = pd.DataFrame(rows)

        result_df, visit_tracker, rest_days = simulate_visits(
            cloud_df,
            day_start=1, day_stop=90,
            year_start=2020, year_stop=2021,
            cloud_fraction_threshold=0.10,
            rest_day_threshold=5,
        )

        # Should produce results for both years
        assert len(result_df) == 2
        assert set(result_df["year"]) == {2020, 2021}

        # All three sites should be visited in at least one year
        all_visited = set()
        for year_visits in visit_tracker.values():
            all_visited.update(year_visits.keys())
        assert all_visited == {"SiteA", "SiteB", "SiteC"}

        # Each visit should have at least one day
        for year, polygons in visit_tracker.items():
            for poly_id, days in polygons.items():
                assert len(days) >= 1


# ===================================================================
# Test 6: Flight box generation + plan + swath
# ===================================================================

class TestFlightBoxWorkflow:
    """Generate a flight box, compute plan, and verify swath coverage."""

    def test_box_to_plan(self):
        sensor = AVIRIS3()
        aircraft = KingAirB200()

        lines = box_around_center_line(
            instrument=sensor,
            altitude_msl=ureg.Quantity(6000, "meter"),
            lat0=34.0, lon0=-118.0,
            azimuth=0.0,
            box_length=ureg.Quantity(30, "km"),
            box_width=ureg.Quantity(8, "km"),
            box_name="Box",
            overlap=20,
        )

        assert len(lines) >= 2, "Box should produce multiple flight lines"

        plan = compute_flight_plan(
            aircraft=aircraft,
            flight_sequence=lines,
        )

        fl_rows = plan[plan["segment_type"] == "flight_line"]
        assert len(fl_rows) == len(lines)

        # Total time should be reasonable for a small box
        total_min = plan["time_to_segment"].sum()
        assert 0 < total_min < 300
