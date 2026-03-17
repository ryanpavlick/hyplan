"""Tests for hyplan.plotting."""

import pytest
import folium

from hyplan.plotting import map_flight_lines
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
