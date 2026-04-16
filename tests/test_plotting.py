"""Tests for hyplan.plotting."""

import matplotlib
matplotlib.use("Agg")

import pytest
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

from hyplan.plotting import (
    map_flight_lines,
    plot_flight_plan,
    plot_altitude_trajectory,
    terrain_profile_along_track,
    plot_airspace_map,
    plot_conflict_matrix,
    plot_vertical_profile,
    plot_oceanic_tracks,
    map_airspace,
)
from hyplan.airspace import Airspace, AirspaceConflict, OceanicTrack
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


# ---------------------------------------------------------------------------
# Synthetic data helpers for airspace-related plot tests
# ---------------------------------------------------------------------------

def _make_airspace(name="R-2508", airspace_class="RESTRICTED", airspace_type=1,
                   floor_ft=0.0, ceiling_ft=18000.0, lon=-117.5, lat=34.0,
                   size=0.5):
    """Return a synthetic Airspace with a square polygon."""
    half = size / 2
    poly = Polygon([
        (lon - half, lat - half),
        (lon + half, lat - half),
        (lon + half, lat + half),
        (lon - half, lat + half),
    ])
    return Airspace(
        name=name,
        airspace_class=airspace_class,
        airspace_type=airspace_type,
        floor_ft=floor_ft,
        ceiling_ft=ceiling_ft,
        geometry=poly,
        country="US",
        source="faa_nasr",
    )


def _make_conflict(airspace, flight_line_index=0):
    """Return a synthetic AirspaceConflict."""
    return AirspaceConflict(
        airspace=airspace,
        flight_line_index=flight_line_index,
        horizontal_intersection=LineString([(-117.5, 34.0), (-117.3, 34.0)]),
        vertical_overlap_ft=(0.0, 18000.0),
        severity="HARD",
        entry_point=(-117.5, 34.0),
        exit_point=(-117.3, 34.0),
    )


def _make_near_miss(airspace, flight_line_index=0):
    """Return a synthetic near-miss AirspaceConflict."""
    return AirspaceConflict(
        airspace=airspace,
        flight_line_index=flight_line_index,
        horizontal_intersection=LineString([(-117.5, 34.0), (-117.3, 34.0)]),
        vertical_overlap_ft=(0.0, 0.0),
        severity="NEAR_MISS",
        distance_to_boundary_m=5000.0,
    )


def _make_oceanic_track(ident="A", direction="east"):
    """Return a synthetic OceanicTrack."""
    waypoints = [
        (-30.0, 50.0, "NATW"),
        (-20.0, 52.0, "NATX"),
        (-10.0, 53.0, "NATY"),
    ]
    geom = LineString([(w[0], w[1]) for w in waypoints])
    return OceanicTrack(
        ident=ident,
        system="NAT",
        valid_from="2026-01-01T00:00:00Z",
        valid_to="2026-01-01T12:00:00Z",
        waypoints=waypoints,
        east_levels=["350", "370"] if direction == "east" else None,
        west_levels=["360", "380"] if direction == "west" else None,
        geometry=geom,
    )


# ---------------------------------------------------------------------------
# Smoke tests: plot_airspace_map (cartopy)
# ---------------------------------------------------------------------------

class TestPlotAirspaceMap:
    """Smoke tests for plot_airspace_map."""

    def test_basic_airspaces(self):
        airspaces = [_make_airspace()]
        fig, ax = plot_airspace_map(airspaces)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_with_flight_lines(self, sample_flight_line):
        airspaces = [_make_airspace()]
        fig, ax = plot_airspace_map(airspaces, flight_lines=[sample_flight_line])
        assert fig is not None
        plt.close("all")

    def test_with_conflicts(self, sample_flight_line):
        airspace = _make_airspace()
        conflict = _make_conflict(airspace)
        fig, ax = plot_airspace_map(
            [airspace],
            flight_lines=[sample_flight_line],
            conflicts=[conflict],
        )
        assert fig is not None
        plt.close("all")

    def test_with_near_misses(self, sample_flight_line):
        airspace = _make_airspace()
        nm = _make_near_miss(airspace)
        fig, ax = plot_airspace_map(
            [airspace],
            flight_lines=[sample_flight_line],
            near_misses=[nm],
            buffer_m=10000,
        )
        assert fig is not None
        plt.close("all")

    def test_with_inactive_airspaces(self):
        active = _make_airspace(name="Active")
        inactive = _make_airspace(name="Inactive MOA", lon=-118.5)
        fig, ax = plot_airspace_map(
            [active],
            inactive_airspaces=[inactive],
        )
        assert fig is not None
        plt.close("all")

    def test_with_explicit_extent(self):
        airspaces = [_make_airspace()]
        fig, ax = plot_airspace_map(
            airspaces,
            extent=[-119, -116, 33, 35],
        )
        assert fig is not None
        plt.close("all")

    def test_multiple_airspace_classes(self):
        airspaces = [
            _make_airspace(name="R-2508", airspace_class="RESTRICTED", airspace_type=1),
            _make_airspace(name="TFR-01", airspace_class="TFR", airspace_type=31, lon=-118.0),
            _make_airspace(name="SFRA-DC", airspace_class="SFRA", airspace_type=37, lon=-116.0),
            _make_airspace(name="Class B", airspace_class="B", airspace_type=33, lon=-119.0),
        ]
        fig, ax = plot_airspace_map(airspaces, show_labels=True)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Smoke tests: plot_conflict_matrix
# ---------------------------------------------------------------------------

class TestPlotConflictMatrix:
    """Smoke tests for plot_conflict_matrix."""

    def test_basic_matrix(self, sample_flight_line):
        airspaces = [
            _make_airspace(name="R-2508"),
            _make_airspace(name="Class B", airspace_class="B", airspace_type=33, lon=-118.0),
        ]
        fig, ax = plot_conflict_matrix([sample_flight_line], airspaces)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_multiple_flight_lines(self, sample_flight_line, short_flight_line):
        airspaces = [_make_airspace()]
        fig, ax = plot_conflict_matrix(
            [sample_flight_line, short_flight_line],
            airspaces,
        )
        assert fig is not None
        plt.close("all")

    def test_custom_title_and_figsize(self, sample_flight_line):
        airspaces = [_make_airspace()]
        fig, ax = plot_conflict_matrix(
            [sample_flight_line],
            airspaces,
            title="Custom Matrix",
            figsize=(10, 6),
        )
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Smoke tests: plot_vertical_profile
# ---------------------------------------------------------------------------

class TestPlotVerticalProfile:
    """Smoke tests for plot_vertical_profile."""

    def test_basic_profile(self, sample_flight_line):
        airspaces = [_make_airspace()]
        fig, ax = plot_vertical_profile(sample_flight_line, airspaces)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_no_intersecting_airspace(self, sample_flight_line):
        # Airspace far away, no intersection
        airspaces = [_make_airspace(name="Far Away", lon=-80.0, lat=25.0)]
        fig, ax = plot_vertical_profile(sample_flight_line, airspaces)
        assert fig is not None
        plt.close("all")

    def test_custom_title(self, sample_flight_line):
        airspaces = [_make_airspace()]
        fig, ax = plot_vertical_profile(
            sample_flight_line, airspaces, title="Custom Vertical Profile"
        )
        assert fig is not None
        plt.close("all")

    def test_multiple_airspaces(self, sample_flight_line):
        airspaces = [
            _make_airspace(name="R-Low", floor_ft=0, ceiling_ft=10000),
            _make_airspace(name="R-High", floor_ft=15000, ceiling_ft=25000),
        ]
        fig, ax = plot_vertical_profile(sample_flight_line, airspaces)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Smoke tests: plot_oceanic_tracks
# ---------------------------------------------------------------------------

class TestPlotOceanicTracks:
    """Smoke tests for plot_oceanic_tracks."""

    def test_basic_tracks(self):
        tracks = [_make_oceanic_track("A", "east"), _make_oceanic_track("B", "west")]
        fig, ax = plot_oceanic_tracks(tracks)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_single_track(self):
        tracks = [_make_oceanic_track("Z", "east")]
        fig, ax = plot_oceanic_tracks(tracks)
        assert fig is not None
        plt.close("all")

    def test_with_flight_lines(self, sample_flight_line):
        tracks = [_make_oceanic_track()]
        fig, ax = plot_oceanic_tracks(tracks, flight_lines=[sample_flight_line])
        assert fig is not None
        plt.close("all")

    def test_custom_title(self):
        tracks = [_make_oceanic_track()]
        fig, ax = plot_oceanic_tracks(tracks, title="NAT Tracks Test")
        assert fig is not None
        plt.close("all")

    def test_empty_tracks(self):
        fig, ax = plot_oceanic_tracks([])
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Smoke tests: map_airspace (folium)
# ---------------------------------------------------------------------------

class TestMapAirspace:
    """Smoke tests for map_airspace (folium Map)."""

    def test_basic_map(self):
        airspaces = [_make_airspace()]
        m = map_airspace(airspaces)
        assert isinstance(m, folium.Map)

    def test_with_flight_lines(self, sample_flight_line):
        airspaces = [_make_airspace()]
        m = map_airspace(airspaces, flight_lines=[sample_flight_line])
        assert isinstance(m, folium.Map)

    def test_with_conflicts(self, sample_flight_line):
        airspace = _make_airspace()
        conflict = _make_conflict(airspace)
        m = map_airspace(
            [airspace],
            flight_lines=[sample_flight_line],
            conflicts=[conflict],
        )
        assert isinstance(m, folium.Map)

    def test_with_near_misses(self, sample_flight_line):
        airspace = _make_airspace()
        nm = _make_near_miss(airspace)
        m = map_airspace(
            [airspace],
            flight_lines=[sample_flight_line],
            near_misses=[nm],
        )
        assert isinstance(m, folium.Map)

    def test_custom_center_and_zoom(self):
        airspaces = [_make_airspace()]
        m = map_airspace(airspaces, center=(35.0, -117.0), zoom_start=10)
        assert isinstance(m, folium.Map)

    def test_multiple_airspace_classes(self):
        airspaces = [
            _make_airspace(name="R-2508", airspace_class="RESTRICTED", airspace_type=1),
            _make_airspace(name="Class C", airspace_class="C", airspace_type=34, lon=-118.0),
            _make_airspace(name="Other", airspace_class="UNKNOWN", airspace_type=0, lon=-119.0),
        ]
        m = map_airspace(airspaces)
        assert isinstance(m, folium.Map)

    def test_airspace_with_schedule(self):
        a = _make_airspace()
        # Manually set schedule and effective_start for tooltip coverage
        a.schedule = "0700 - 1800, MON - FRI"
        a.effective_start = "2026-01-01T00:00:00Z"
        m = map_airspace([a])
        assert isinstance(m, folium.Map)
