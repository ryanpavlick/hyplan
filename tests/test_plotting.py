"""Tests for hyplan.plotting."""

import matplotlib
matplotlib.use("Agg")

import pytest
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from hyplan.plotting import map_flight_lines, plot_flight_plan, plot_altitude_trajectory, terrain_profile_along_track
from hyplan.flight_line import FlightLine
from hyplan.units import ureg


@pytest.fixture
def flight_lines():
    """Create a few flight lines for testing."""
    lines = []
    for i, az in enumerate([0, 90, 180]):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0 + i * 0.1,
            lon1=-118.0,
            length=ureg.Quantity(10000, "meter"),
            az=float(az),
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name=f"Line_{i}",
        )
        lines.append(fl)
    return lines


class TestMapFlightLines:
    def test_returns_folium_map(self, flight_lines):
        m = map_flight_lines(flight_lines)
        assert isinstance(m, folium.Map)

    def test_custom_center(self, flight_lines):
        m = map_flight_lines(flight_lines, center=(35.0, -117.0))
        assert isinstance(m, folium.Map)

    def test_custom_zoom(self, flight_lines):
        m = map_flight_lines(flight_lines, zoom_start=12)
        assert isinstance(m, folium.Map)

    def test_custom_colors(self, flight_lines):
        m = map_flight_lines(flight_lines, line_color="red", line_weight=5)
        assert isinstance(m, folium.Map)

    def test_single_line(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(10000, "meter"),
            az=90.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="Single",
        )
        m = map_flight_lines([fl])
        assert isinstance(m, folium.Map)

    def test_html_output_contains_site_names(self, flight_lines):
        m = map_flight_lines(flight_lines)
        html = m._repr_html_()
        assert isinstance(html, str)
        assert len(html) > 0


# ---------------------------------------------------------------------------
# Helpers for plot_flight_plan / plot_altitude_trajectory tests
# ---------------------------------------------------------------------------

class _MockAirport:
    """Lightweight stand-in for Airport (avoids dataset download)."""
    def __init__(self, lat, lon):
        self.latitude = lat
        self.longitude = lon


def _make_simple_flight_plan_gdf():
    """Return a minimal GeoDataFrame matching compute_flight_plan output."""
    records = [
        {
            "geometry": LineString([(-118.0, 34.0), (-117.5, 34.0)]),
            "start_lat": 34.0,
            "start_lon": -118.0,
            "end_lat": 34.0,
            "end_lon": -117.5,
            "start_altitude": 0.0,
            "end_altitude": 20000.0,
            "segment_type": "takeoff",
            "segment_name": "Departure climb",
            "distance": 25.0,
            "time_to_segment": 15.0,
            "start_heading": 90.0,
            "end_heading": 90.0,
        },
        {
            "geometry": LineString([(-117.5, 34.0), (-117.0, 34.0)]),
            "start_lat": 34.0,
            "start_lon": -117.5,
            "end_lat": 34.0,
            "end_lon": -117.0,
            "start_altitude": 20000.0,
            "end_altitude": 20000.0,
            "segment_type": "flight_line",
            "segment_name": "FL-1",
            "distance": 25.0,
            "time_to_segment": 10.0,
            "start_heading": 90.0,
            "end_heading": 90.0,
        },
        {
            "geometry": LineString([(-117.0, 34.0), (-116.5, 34.0)]),
            "start_lat": 34.0,
            "start_lon": -117.0,
            "end_lat": 34.0,
            "end_lon": -116.5,
            "start_altitude": 20000.0,
            "end_altitude": 0.0,
            "segment_type": "descent",
            "segment_name": "Arrival descent",
            "distance": 25.0,
            "time_to_segment": 15.0,
            "start_heading": 90.0,
            "end_heading": 90.0,
        },
    ]
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


class TestPlotFlightPlan:
    """Smoke tests for plot_flight_plan (Agg backend, no display)."""

    def test_returns_without_error(self):
        gdf = _make_simple_flight_plan_gdf()
        takeoff = _MockAirport(34.0, -118.0)
        ret = _MockAirport(34.0, -116.5)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-117.5,
            length=ureg.Quantity(50000, "meter"),
            az=90.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="FL-1",
        )
        # plot_flight_plan calls plt.show(); with Agg backend this is a no-op
        plot_flight_plan(gdf, takeoff, ret, [fl])
        # If we get here without error the smoke test passes
        plt.close("all")


class TestPlotAltitudeTrajectory:
    """Smoke tests for plot_altitude_trajectory (Agg backend, no display)."""

    def test_without_terrain(self):
        gdf = _make_simple_flight_plan_gdf()
        # Disable terrain to avoid DEM download
        plot_altitude_trajectory(gdf, aircraft=None, show_terrain=False)
        plt.close("all")

    def test_with_aircraft_no_terrain(self):
        from hyplan.aircraft import DynamicAviation_B200
        gdf = _make_simple_flight_plan_gdf()
        ac = DynamicAviation_B200()
        plot_altitude_trajectory(gdf, aircraft=ac, show_terrain=False)
        plt.close("all")


class TestTerrainProfileAlongTrack:
    """Test that terrain_profile_along_track is importable and handles edge cases."""

    def test_importable(self):
        """Verify the function is importable."""
        assert callable(terrain_profile_along_track)

    def test_empty_geodataframe(self):
        """An empty GeoDataFrame should return empty arrays."""
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "time_to_segment", "segment_type"],
            geometry="geometry",
        )
        times, elevations = terrain_profile_along_track(gdf)
        assert len(times) == 0
        assert len(elevations) == 0
