"""Tests for hybrid-Dubins integration into the flight planning pipeline."""

import pymap3d.vincenty
import pytest

from hyplan.aircraft import (
    NASA_ER2,
    NASA_GIII,
    KingAirB200,
)
from hyplan.airports import Airport
from hyplan.flight_line import FlightLine
from hyplan.flight_plan import compute_flight_plan
from hyplan.units import ureg
from hyplan.waypoint import Waypoint


@pytest.fixture
def b200():
    return KingAirB200()


@pytest.fixture
def giii():
    return NASA_GIII()


@pytest.fixture
def palmdale():
    return Airport("KPMD")


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

    def test_path_has_geometry(self, b200):
        """The hybrid path returns a 2D DubinsPath2D in dubins_path.

        (Previously this asserted geometry_3d, which the legacy 3D-only
        implementation provided.  The hybrid planner uses 2D Dubins for
        plan-view geometry and integrates altitude separately, so a
        single 3D LineString isn't part of the output any more — per-
        phase records carry their own (lon, lat) sublinestrings, and
        altitudes come from the phase records.)
        """
        start = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(15000, "feet"))
        end = Waypoint(34.2, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"))

        result = b200.time_to_cruise(start, end)
        path = result["dubins_path"]

        assert path.geometry is not None
        assert len(path.geometry.coords) >= 2
        # Per-phase explicit geometry replaces the old shared 3D path.
        for phase_record in result["phases"].values():
            assert phase_record["geometry"] is not None

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

    def test_return_heading_is_inbound_course(self, b200, palmdale):
        """The destination heading is the inbound course (direction of
        travel at arrival), not its reciprocal — a reversed heading makes
        the Dubins solver append superfluous turn arcs at the airport."""
        wp = Waypoint(
            palmdale.latitude + 1.0, palmdale.longitude, 180.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        result = b200.time_to_return(wp, palmdale)

        # Due-south leg: the aircraft arrives heading south (~180 deg).
        descent = result["phases"]["descent"]
        assert descent["end_heading"] == pytest.approx(180.0, abs=1.0)

        # With the correct arrival heading the path is near-direct.
        direct_m, _ = pymap3d.vincenty.vdist(
            wp.latitude, wp.longitude,
            palmdale.latitude, palmdale.longitude,
        )
        direct_nmi = float(direct_m) / 1852.0
        length_nmi = result["dubins_path"].length.m_as(ureg.nautical_mile)
        assert length_nmi == pytest.approx(direct_nmi, rel=0.02)

    def test_return_approach_headings_are_inbound_course(self, palmdale):
        """With an ApproachProfile, the FAF waypoint and approach phase
        carry the inbound course, while the FAF itself stays offset back
        along the inbound track."""
        er2 = NASA_ER2()
        wp = Waypoint(
            palmdale.latitude + 1.0, palmdale.longitude, 180.0,
            altitude_msl=ureg.Quantity(60000, "feet"),
        )
        result = er2.time_to_return(wp, palmdale)

        approach = result["phases"]["approach"]
        assert approach["start_heading"] == pytest.approx(180.0, abs=1.0)
        assert approach["end_heading"] == pytest.approx(180.0, abs=1.0)
        assert approach["start_lat"] > palmdale.latitude


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
