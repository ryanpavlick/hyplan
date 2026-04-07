"""Tests for hyplan.exports module."""

import datetime
import os
import tempfile

import pytest

from hyplan import DynamicAviation_B200, compute_flight_plan
from hyplan.airports import Airport, initialize_data
from hyplan.exports import (
    extract_waypoints,
    generate_wp_names,
    to_er2_csv,
    to_excel,
    to_foreflight_csv,
    to_gpx,
    to_honeywell_fms,
    to_icartt,
    to_kml,
    to_pilot_excel,
    to_trackair,
    to_txt,
)
from hyplan.flight_line import FlightLine
from hyplan.instruments import AVIRIS3
from hyplan.geometry import (
    dd_to_ddm,
    dd_to_ddms,
    dd_to_foreflight_oneline,
    dd_to_nddmm,
    magnetic_declination,
    true_to_magnetic,
)
from hyplan.units import ureg
from hyplan.waypoint import Waypoint


# -------------------------------------------------------------------------
# Coordinate formatting
# -------------------------------------------------------------------------

class TestCoordinateFormatting:

    def test_dd_to_ddm(self):
        lat_s, lon_s = dd_to_ddm(34.4035, -118.0575)
        assert lat_s == "34 24.21"
        assert lon_s == "-118 03.45"

    def test_dd_to_ddms(self):
        lat_s, lon_s = dd_to_ddms(34.4035, -118.0575)
        assert "34 24" in lat_s
        assert "118 03" in lon_s

    def test_dd_to_nddmm(self):
        lat_s, lon_s = dd_to_nddmm(34.4035, -118.0575)
        assert lat_s.startswith("N34")
        assert lon_s.startswith("W118")

    def test_dd_to_nddmm_southern(self):
        lat_s, _ = dd_to_nddmm(-34.0, 0.0)
        assert lat_s.startswith("S")

    def test_dd_to_nddmm_eastern(self):
        _, lon_s = dd_to_nddmm(0.0, 118.0)
        assert lon_s.startswith("E")

    def test_dd_to_foreflight_oneline(self):
        result = dd_to_foreflight_oneline(34.4035, -118.0575)
        assert result.startswith("N")
        assert "/W" in result


# -------------------------------------------------------------------------
# Magnetic declination
# -------------------------------------------------------------------------

class TestMagneticDeclination:

    def test_declination_returns_float(self):
        dec = magnetic_declination(34.0, -118.0)
        assert isinstance(dec, float)
        # LA area declination should be roughly 10-13 degrees east
        assert 5.0 < dec < 20.0

    def test_true_to_magnetic(self):
        mag = true_to_magnetic(90.0, 12.0)
        assert mag == pytest.approx(78.0)

    def test_true_to_magnetic_wrap(self):
        mag = true_to_magnetic(5.0, 12.0)
        assert mag == pytest.approx(353.0)


# -------------------------------------------------------------------------
# Waypoint naming
# -------------------------------------------------------------------------

class TestGenerateWpNames:

    def test_count(self):
        names = generate_wp_names(5)
        assert len(names) == 5

    def test_format(self):
        names = generate_wp_names(3, prefix="B",
                                  date=datetime.date(2025, 3, 21))
        assert names[0] == "B2100"
        assert names[1] == "B2101"
        assert names[2] == "B2102"

    def test_five_chars(self):
        names = generate_wp_names(10, prefix="H",
                                  date=datetime.date(2025, 1, 5))
        for name in names:
            assert len(name) == 5


# -------------------------------------------------------------------------
# Flight plan fixture
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flight_plan():
    """Create a sample flight plan for testing exports."""
    initialize_data(countries=["US"])
    b200 = DynamicAviation_B200()
    kedw = Airport("KEDW")
    wps = [
        Waypoint(34.7, -118.2, 0.0,
                 altitude_msl=ureg.Quantity(10000, "feet"), name="WP1"),
        Waypoint(34.9, -118.0, 45.0,
                 altitude_msl=ureg.Quantity(20000, "feet"), name="WP2"),
        Waypoint(35.1, -118.2, 180.0,
                 altitude_msl=ureg.Quantity(15000, "feet"), name="WP3"),
    ]
    plan = compute_flight_plan(b200, wps,
                               takeoff_airport=kedw, return_airport=kedw)
    return plan, b200


# -------------------------------------------------------------------------
# Extract waypoints
# -------------------------------------------------------------------------

class TestExtractWaypoints:

    def test_count(self, flight_plan):
        plan, _ = flight_plan
        wps = extract_waypoints(plan)
        # n segments + 1 (final endpoint)
        assert len(wps) == len(plan) + 1

    def test_columns(self, flight_plan):
        plan, _ = flight_plan
        wps = extract_waypoints(plan)
        for col in ["wp", "lat", "lon", "alt_m", "alt_kft", "heading",
                     "speed_mps", "speed_kt", "dist_km", "dist_nm",
                     "cum_dist_km", "cum_dist_nm", "leg_time_min",
                     "cum_time_min", "segment_type", "segment_name"]:
            assert col in wps.columns

    def test_cumulative_distance_increases(self, flight_plan):
        plan, _ = flight_plan
        wps = extract_waypoints(plan)
        cum = wps["cum_dist_nm"].values
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))


# -------------------------------------------------------------------------
# File export tests
# -------------------------------------------------------------------------

class TestExportFiles:

    TAKEOFF = datetime.datetime(2025, 6, 15, 15, 0, 0)

    def test_to_excel(self, flight_plan, tmp_path):
        plan, b200 = flight_plan
        path = str(tmp_path / "test.xlsx")
        to_excel(plan, path, aircraft=b200, takeoff_time=self.TAKEOFF,
                 mission_name="TEST")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_to_pilot_excel(self, flight_plan, tmp_path):
        plan, b200 = flight_plan
        path = str(tmp_path / "test_for_pilots.xlsx")
        to_pilot_excel(plan, path, aircraft=b200, takeoff_time=self.TAKEOFF,
                       mission_name="TEST")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_to_pilot_excel_coord_formats(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        for fmt in ["DD MM", "DD MM SS", "NDDD MM.SS"]:
            path = str(tmp_path / f"pilot_{fmt.replace(' ', '_')}.xlsx")
            to_pilot_excel(plan, path, coord_format=fmt)
            assert os.path.exists(path)

    def test_to_foreflight_csv(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test_FOREFLIGHT.csv")
        to_foreflight_csv(plan, path, takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "Waypoint,Description,LAT,LONG"
        assert len(lines) > 1

        # Check companion file
        oneline_path = path.replace(".csv", "_oneline.txt")
        assert os.path.exists(oneline_path)

    def test_to_honeywell_fms(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test_Honeywell.csv")
        to_honeywell_fms(plan, path, takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "E,WPT,FIX,LAT,LON"
        # All data lines start with 'x,'
        for line in lines[1:]:
            if line.strip():
                assert line.startswith("x,")

    def test_to_er2_csv(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test_ER2.csv")
        to_er2_csv(plan, path, takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()
        assert "ID,Description,LAT,LONG" in lines[0]

    def test_to_icartt(self, flight_plan, tmp_path):
        plan, b200 = flight_plan
        path = str(tmp_path / "test.ict")
        to_icartt(plan, path, pi_name="Test PI", mission_name="TEST",
                  flight_date=self.TAKEOFF.date(), aircraft=b200,
                  takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        # Check ICARTT v1001 header
        assert "1001" in content.split("\n")[0]
        # Check data variables are present
        assert "Latitude" in content
        assert "Longitude" in content
        assert "Altitude" in content

    def test_to_kml(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test.kml")
        to_kml(plan, path, takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        # KMZ companion should also be created
        kmz_path = str(tmp_path / "test.kmz")
        assert os.path.exists(kmz_path)

    def test_to_kml_altitude_exaggeration(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test_exag.kml")
        to_kml(plan, path, altitude_exaggeration=10.0)
        assert os.path.exists(path)

    def test_to_kmz_direct(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test.kmz")
        to_kml(plan, path)
        assert os.path.exists(path)

    def test_to_gpx(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test.gpx")
        to_gpx(plan, path, mission_name="TEST", takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "<gpx" in content
        assert "<rte>" in content

    def test_to_txt(self, flight_plan, tmp_path):
        plan, _ = flight_plan
        path = str(tmp_path / "test.txt")
        to_txt(plan, path, takeoff_time=self.TAKEOFF)
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()
        # First line is header starting with #
        assert lines[0].startswith("#WP")
        # Data lines follow
        assert len(lines) > 1

    def test_no_takeoff_time(self, flight_plan, tmp_path):
        """All exports should work without a takeoff time."""
        plan, _ = flight_plan
        to_foreflight_csv(plan, str(tmp_path / "ff.csv"))
        to_honeywell_fms(plan, str(tmp_path / "hw.csv"))
        to_er2_csv(plan, str(tmp_path / "er2.csv"))
        to_txt(plan, str(tmp_path / "out.txt"))
        to_gpx(plan, str(tmp_path / "out.gpx"))
        to_kml(plan, str(tmp_path / "out.kml"))
        to_excel(plan, str(tmp_path / "out.xlsx"))
        to_pilot_excel(plan, str(tmp_path / "pilot.xlsx"))
        to_icartt(plan, str(tmp_path / "out.ict"))
        # All should exist
        for name in ["ff.csv", "hw.csv", "er2.csv", "out.txt", "out.gpx",
                      "out.kml", "out.xlsx", "pilot.xlsx", "out.ict"]:
            assert os.path.exists(str(tmp_path / name))


# -------------------------------------------------------------------------
# Content-validation tests
# -------------------------------------------------------------------------

class TestExportContentValidation:
    """Validate that exported files contain correct data, not just that
    they exist."""

    TAKEOFF = datetime.datetime(2025, 6, 15, 15, 0, 0)

    def test_foreflight_csv_coordinates(self, flight_plan, tmp_path):
        """Verify CSV data rows contain coordinate values matching waypoints."""
        plan, _ = flight_plan
        path = str(tmp_path / "ff_content.csv")
        to_foreflight_csv(plan, path, takeoff_time=self.TAKEOFF)

        wps = extract_waypoints(plan)

        with open(path) as f:
            lines = f.readlines()

        # Header + at least as many data lines as waypoints
        assert len(lines) >= len(wps) + 1

        # Parse each data row and compare lat/lon to extracted waypoints
        for i, line in enumerate(lines[1:]):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            csv_lat = float(parts[2])
            csv_lon = float(parts[3])
            # Match against the corresponding waypoint
            assert csv_lat == pytest.approx(wps.iloc[i]["lat"], abs=1e-6)
            assert csv_lon == pytest.approx(wps.iloc[i]["lon"], abs=1e-6)

    def test_foreflight_csv_altitude_in_description(self, flight_plan, tmp_path):
        """Verify the description field includes altitude info."""
        plan, _ = flight_plan
        path = str(tmp_path / "ff_alt.csv")
        to_foreflight_csv(plan, path, takeoff_time=self.TAKEOFF)

        with open(path) as f:
            lines = f.readlines()

        # Every data line description should contain 'ALT='
        for line in lines[1:]:
            if line.strip():
                assert "ALT=" in line

    def test_gpx_waypoint_count_and_altitudes(self, flight_plan, tmp_path):
        """Parse GPX with gpxpy, verify waypoint count and altitude values."""
        import gpxpy

        plan, _ = flight_plan
        path = str(tmp_path / "content.gpx")
        to_gpx(plan, path, mission_name="TEST", takeoff_time=self.TAKEOFF)

        wps = extract_waypoints(plan)

        with open(path) as f:
            gpx = gpxpy.parse(f)

        assert len(gpx.routes) == 1
        route = gpx.routes[0]
        assert len(route.points) == len(wps)

        for i, rpt in enumerate(route.points):
            expected_alt = wps.iloc[i]["alt_m"] or 0
            assert rpt.latitude == pytest.approx(wps.iloc[i]["lat"], abs=1e-6)
            assert rpt.longitude == pytest.approx(wps.iloc[i]["lon"], abs=1e-6)
            assert rpt.elevation == pytest.approx(expected_alt, abs=1e-1)

    def test_gpx_round_trip(self, flight_plan, tmp_path):
        """Export GPX, reimport with gpxpy, compare lat/lon/alt."""
        import gpxpy

        plan, _ = flight_plan
        path = str(tmp_path / "roundtrip.gpx")
        to_gpx(plan, path, mission_name="ROUNDTRIP", takeoff_time=self.TAKEOFF)

        wps = extract_waypoints(plan)

        with open(path) as f:
            gpx = gpxpy.parse(f)

        route = gpx.routes[0]
        for i, rpt in enumerate(route.points):
            wp = wps.iloc[i]
            assert rpt.latitude == pytest.approx(wp["lat"], abs=1e-6)
            assert rpt.longitude == pytest.approx(wp["lon"], abs=1e-6)
            expected_alt = wp["alt_m"] or 0
            assert rpt.elevation == pytest.approx(expected_alt, abs=0.5)

    def test_kml_placemarks(self, flight_plan, tmp_path):
        """Verify KML output contains Placemark elements for each waypoint."""
        plan, _ = flight_plan
        path = str(tmp_path / "content.kml")
        to_kml(plan, path, takeoff_time=self.TAKEOFF)

        wps = extract_waypoints(plan)

        with open(path) as f:
            content = f.read()

        # Each waypoint should have a Placemark
        placemark_count = content.count("<Placemark")
        # One placemark per waypoint + one for the flight path LineString
        assert placemark_count == len(wps) + 1

        # Verify the KML is valid XML
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        # Namespace-agnostic check for Placemark elements
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        placemarks = root.findall(".//kml:Placemark", ns)
        assert len(placemarks) == len(wps) + 1

    def test_icartt_header_format_and_data_rows(self, flight_plan, tmp_path):
        """Verify ICARTT header format and data row count."""
        plan, b200 = flight_plan
        path = str(tmp_path / "content.ict")
        to_icartt(plan, path, pi_name="Test PI", mission_name="TEST",
                  flight_date=self.TAKEOFF.date(), aircraft=b200,
                  takeoff_time=self.TAKEOFF, interval_seconds=60.0)

        with open(path) as f:
            all_lines = f.readlines()

        # First line: n_header, 1001
        first_parts = all_lines[0].strip().split(",")
        n_header = int(first_parts[0].strip())
        assert first_parts[1].strip() == "1001"

        # The declared header count should match actual header lines
        assert len(all_lines) > n_header

        # Data rows start after header
        data_lines = [l for l in all_lines[n_header:] if l.strip()]
        assert len(data_lines) > 0

        # Each data row should have 7 comma-separated values
        # (Start_UTC, Lat, Lon, Alt, speed, Bearing, SZA)
        for line in data_lines:
            values = line.strip().split(",")
            assert len(values) == 7

        # Verify column header is last header line
        col_header = all_lines[n_header - 1].strip()
        assert "Start_UTC" in col_header
        assert "Latitude" in col_header
        assert "Longitude" in col_header

    def test_icartt_data_values_in_range(self, flight_plan, tmp_path):
        """Verify ICARTT data values are within reasonable ranges."""
        plan, b200 = flight_plan
        path = str(tmp_path / "range.ict")
        to_icartt(plan, path, pi_name="Test PI", mission_name="TEST",
                  flight_date=self.TAKEOFF.date(), aircraft=b200,
                  takeoff_time=self.TAKEOFF)

        with open(path) as f:
            all_lines = f.readlines()

        n_header = int(all_lines[0].strip().split(",")[0].strip())
        data_lines = [l for l in all_lines[n_header:] if l.strip()]

        for line in data_lines:
            vals = [float(v.strip()) for v in line.strip().split(",")]
            utc_s, lat, lon, alt, spd, hdg, sza = vals
            assert 0 <= utc_s <= 86400, f"UTC seconds out of range: {utc_s}"
            assert -90 <= lat <= 90, f"Latitude out of range: {lat}"
            assert -180 <= lon <= 180, f"Longitude out of range: {lon}"
            assert alt >= 0, f"Altitude negative: {alt}"
            assert 0 <= hdg <= 360 or hdg == -9999, f"Heading out of range: {hdg}"

    def test_honeywell_fms_waypoint_count(self, flight_plan, tmp_path):
        """Verify Honeywell FMS data row count matches waypoint count."""
        plan, _ = flight_plan
        path = str(tmp_path / "hw_content.csv")
        to_honeywell_fms(plan, path, takeoff_time=self.TAKEOFF)

        wps = extract_waypoints(plan)

        with open(path) as f:
            lines = f.readlines()

        # Header + data lines
        data_lines = [l for l in lines[1:] if l.strip()]
        assert len(data_lines) == len(wps)

        # Each data line should have the waypoint coordinates in NDDMM format
        for line in data_lines:
            parts = line.strip().split(",")
            assert len(parts) == 5
            assert parts[0] == "x"
            # LAT should start with N or S
            lat_field = parts[3].strip()
            assert lat_field[0] in ("N", "S")
            # LON should start with E or W
            lon_field = parts[4].strip()
            assert lon_field[0] in ("E", "W")


# -------------------------------------------------------------------------
# TrackAir export tests
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flight_plan_with_lines():
    """Flight plan containing FlightLine segments for TrackAir testing."""
    initialize_data(countries=["US"])
    b200 = DynamicAviation_B200()
    kedw = Airport("KEDW")
    fl1 = FlightLine.start_length_azimuth(
        lat1=34.5, lon1=-118.0,
        length=ureg.Quantity(20000, "meter"),
        az=90.0,
        altitude_msl=ureg.Quantity(15000, "feet"),
        site_name="Line1",
    )
    fl2 = FlightLine.start_length_azimuth(
        lat1=34.6, lon1=-118.0,
        length=ureg.Quantity(20000, "meter"),
        az=90.0,
        altitude_msl=ureg.Quantity(15000, "feet"),
        site_name="Line2",
    )
    plan = compute_flight_plan(b200, [fl1, fl2],
                               takeoff_airport=kedw, return_airport=kedw)
    return plan, b200


class TestToTrackAir:

    def test_creates_file(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "plan.txt")
        to_trackair(plan, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_required_sections(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "sections.txt")
        to_trackair(plan, path, mission_name="TEST", author="R. Pavlick")
        text = open(path).read()
        assert "[general]" in text
        assert "[spex]" in text
        assert "[strips]" in text

    def test_mission_and_author(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "meta.txt")
        to_trackair(plan, path, mission_name="SCOAPE2", author="A. Chlus")
        text = open(path).read()
        assert "Flight plan name = SCOAPE2" in text
        assert "Designed by = A. Chlus" in text

    def test_strips_count(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "strips.txt")
        to_trackair(plan, path)
        n_lines = plan[plan["segment_type"] == "flight_line"].shape[0]
        text = open(path).read()
        for i in range(1, n_lines + 1):
            assert f"\n{i}=" in text

    def test_strips_format(self, flight_plan_with_lines, tmp_path):
        """Each strip line should be N=lat,lon,lat,lon with four decimal values."""
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "fmt.txt")
        to_trackair(plan, path)
        text = open(path).read()
        in_strips = False
        for line in text.splitlines():
            if line.strip() == "[strips]":
                in_strips = True
                continue
            if in_strips and "=" in line:
                _, coords = line.split("=", 1)
                parts = coords.split(",")
                assert len(parts) == 4
                for part in parts:
                    float(part)  # should not raise

    def test_with_sensor(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "sensor.txt")
        to_trackair(plan, path, sensor=AVIRIS3())
        text = open(path).read()
        # FOV and swath width should be non-empty numbers
        for line in text.splitlines():
            if line.startswith("Field of view ="):
                assert line.split("=", 1)[1].strip() != ""
            if line.startswith("Swath width (meters) ="):
                assert line.split("=", 1)[1].strip() != ""

    def test_without_sensor_leaves_fields_empty(self, flight_plan_with_lines, tmp_path):
        plan, _ = flight_plan_with_lines
        path = str(tmp_path / "nosensor.txt")
        to_trackair(plan, path)
        text = open(path).read()
        for line in text.splitlines():
            if line.startswith("Field of view ="):
                assert line.split("=", 1)[1].strip() == ""
            if line.startswith("Swath width (meters) ="):
                assert line.split("=", 1)[1].strip() == ""
