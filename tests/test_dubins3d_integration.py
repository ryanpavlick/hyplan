"""Tests for 3D Dubins integration into the flight planning pipeline."""

import pytest

from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.aircraft import (
    NASA_GIII,
    KingAirB200,
)
from hyplan.flight_plan import compute_flight_plan
from hyplan.airports import Airport


@pytest.fixture
def b200():
    return KingAirB200()


@pytest.fixture
def giii():
    return NASA_GIII()


@pytest.fixture
def palmdale():
    return Airport("KPMD")


class TestPitchLimits:
    def test_b200_pitch_limits(self, b200):
        pitch_min, pitch_max = b200.pitch_limits()
        assert pitch_min < 0
        assert pitch_max > 0
        assert -20 < pitch_min < 0
        assert 0 < pitch_max < 20

    def test_giii_pitch_limits(self, giii):
        pitch_min, pitch_max = giii.pitch_limits()
        assert pitch_min < 0
        assert pitch_max > 0

    def test_pitch_limits_with_speed_override(self, b200):
        slow = b200.pitch_limits(ureg.Quantity(100, "knot"))
        fast = b200.pitch_limits(ureg.Quantity(250, "knot"))
        # Slower speed -> steeper pitch angles
        assert slow[1] > fast[1]
        assert slow[0] < fast[0]


class TestTimeToCruise:
    def test_level_cruise(self, b200):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))
        end = Waypoint(34.5, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))

        result = b200.time_to_cruise(start, end)

        assert result["total_time"].magnitude > 0
        assert result["dubins_path"].geometry is not None

    def test_climbing_cruise(self, b200):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(10000, "feet"))
        end = Waypoint(34.3, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))

        result = b200.time_to_cruise(start, end)

        assert result["total_time"].magnitude > 0
        has_climb = any("climb" in k for k in result["phases"])
        assert has_climb

    def test_descending_cruise(self, b200):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(25000, "feet"))
        end = Waypoint(34.3, -118.0, 0.0, altitude_msl=ureg.Quantity(15000, "feet"))

        result = b200.time_to_cruise(start, end)

        assert result["total_time"].magnitude > 0
        has_descent = any("descent" in k for k in result["phases"])
        assert has_descent

    def test_path_has_3d_geometry(self, b200):
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(15000, "feet"))
        end = Waypoint(34.2, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))

        result = b200.time_to_cruise(start, end)
        path = result["dubins_path"]

        assert path.geometry is not None
        assert path.geometry_3d is not None
        assert len(path.geometry.coords) > 2

    def test_phases_cover_total_time(self, b200):
        start = Waypoint(34.0, -118.0, 45.0, altitude_msl=ureg.Quantity(10000, "feet"))
        end = Waypoint(34.3, -117.7, 90.0, altitude_msl=ureg.Quantity(20000, "feet"))

        result = b200.time_to_cruise(start, end)

        phase_time_sum = sum(
            (p["end_time"] - p["start_time"]).m_as(ureg.minute)
            for p in result["phases"].values()
        )
        total = result["total_time"].m_as(ureg.minute)
        assert phase_time_sum == pytest.approx(total, rel=1e-3)


class TestTakeoffAndReturn:
    def test_takeoff(self, b200, palmdale):
        wp = Waypoint(34.8, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))
        result = b200.time_to_takeoff(palmdale, wp)

        assert result["total_time"].magnitude > 0
        assert result["dubins_path"].geometry is not None

    def test_return(self, b200, palmdale):
        wp = Waypoint(34.8, -118.0, 180.0, altitude_msl=ureg.Quantity(20000, "feet"))
        result = b200.time_to_return(wp, palmdale)

        assert result["total_time"].magnitude > 0
        assert result["dubins_path"].geometry is not None


class TestComputeFlightPlan:
    def test_basic_flight_plan(self, b200):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="TestLine",
        )
        gdf = compute_flight_plan(b200, [fl])
        assert len(gdf) > 0
        assert "segment_type" in gdf.columns
        assert "geometry" in gdf.columns

    def test_flight_plan_with_altitude_change(self, b200):
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(15000, "feet"),
                       name="WP1", segment_type="pattern")
        wp2 = Waypoint(34.2, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"),
                       name="WP2", segment_type="pattern")
        wp3 = Waypoint(34.4, -118.0, 180.0,
                       altitude_msl=ureg.Quantity(15000, "feet"),
                       name="WP3", segment_type="pattern")

        gdf = compute_flight_plan(b200, [wp1, wp2, wp3])
        assert len(gdf) > 0

    def test_flight_plan_with_airports(self, b200, palmdale):
        fl = FlightLine.start_length_azimuth(
            lat1=34.8, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=90.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="TestLine",
        )
        gdf = compute_flight_plan(
            b200, [fl],
            takeoff_airport=palmdale,
            return_airport=palmdale,
        )
        assert len(gdf) > 0
        seg_types = set(gdf["segment_type"])
        assert "flight_line" in seg_types
