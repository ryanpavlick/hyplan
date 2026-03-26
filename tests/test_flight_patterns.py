"""Tests for hyplan.flight_patterns."""

import pytest
import numpy as np
import pymap3d.vincenty

from hyplan.units import ureg
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.flight_patterns import (
    racetrack,
    rosette,
    polygon,
    sawtooth,
    spiral,
    flight_lines_to_waypoint_path,
)


CENTER = (34.0, -118.0)
ALT = ureg.Quantity(20000, "feet")


class TestRacetrack:
    def test_single_leg(self):
        wps = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"))
        assert len(wps) == 2
        assert all(wp.segment_type == "pattern" for wp in wps)

    def test_two_legs_out_and_back(self):
        wps = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"),
                        n_legs=2, offset=ureg.Quantity(5, "km"))
        assert len(wps) == 4
        # Second leg should be reversed (heading ~180 from first)
        h1 = wps[0].heading
        h2 = wps[2].heading
        assert abs((h2 - h1 + 180) % 360 - 180) == pytest.approx(180, abs=1)

    def test_lawnmower(self):
        wps = racetrack(CENTER, 90.0, ALT, ureg.Quantity(30, "km"),
                        n_legs=6, offset=ureg.Quantity(2, "km"))
        assert len(wps) == 12

    def test_offset_list(self):
        offsets = [0, 7000, 14000]
        wps = racetrack(CENTER, 45.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=3, offset=offsets)
        assert len(wps) == 6

    def test_offset_list_wrong_length(self):
        with pytest.raises(ValueError, match="offset list length"):
            racetrack(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_legs=3, offset=[0, 1000])

    def test_vertical_wall(self):
        alts = [ureg.Quantity(a, "feet") for a in [5000, 6000, 7000]]
        wps = racetrack(CENTER, 0.0, alts[0], ureg.Quantity(30, "km"),
                        n_legs=3, offset=0, altitudes=alts)
        assert len(wps) == 6
        # All legs on same track (offset=0) but different altitudes
        for i, alt_expected in enumerate(alts):
            assert wps[2*i].altitude_msl.to(ureg.foot).magnitude == pytest.approx(
                alt_expected.to(ureg.foot).magnitude, rel=1e-3)

    def test_stacked(self):
        stack = [ureg.Quantity(a, "feet") for a in [9000, 7000, 5000]]
        wps = racetrack(CENTER, 0.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=2, offset=ureg.Quantity(3, "km"),
                        stack_altitudes=stack)
        # 2 legs * 3 altitudes = 6 legs, 2 waypoints each = 12
        assert len(wps) == 12

    def test_altitudes_wrong_length(self):
        with pytest.raises(ValueError, match="altitudes length"):
            racetrack(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_legs=2, altitudes=[ALT])

    def test_leg_length_accuracy(self):
        length = ureg.Quantity(50, "km")
        wps = racetrack(CENTER, 0.0, ALT, length)
        dist, _ = pymap3d.vincenty.vdist(
            wps[0].latitude, wps[0].longitude,
            wps[1].latitude, wps[1].longitude
        )
        assert dist == pytest.approx(50000, rel=1e-3)


class TestRosette:
    def test_three_line_default(self):
        wps = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        # 2 * 3 = 6 waypoints (each line = start tip + end tip)
        assert len(wps) == 6
        assert all(wp.segment_type == "pattern" for wp in wps)

    def test_lines_cross_center(self):
        """Each line should cross through center (start and end are on opposite sides)."""
        wps = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        for i in range(0, len(wps), 2):
            start, end = wps[i], wps[i + 1]
            # Midpoint of start and end should be approximately at center
            mid_lat = (start.latitude + end.latitude) / 2
            mid_lon = (start.longitude + end.longitude) / 2
            assert mid_lat == pytest.approx(CENTER[0], abs=0.01)
            assert mid_lon == pytest.approx(CENTER[1], abs=0.01)

    def test_two_line_cross(self):
        wps = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=2)
        assert len(wps) == 4  # 2*2

    def test_custom_angles(self):
        wps = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      angles=[30.0, 120.0])
        assert len(wps) == 4  # 2*2

    def test_single_line(self):
        wps = rosette(CENTER, 90.0, ALT, ureg.Quantity(15, "km"), n_lines=1)
        assert len(wps) == 2  # start tip, end tip

    def test_radius_accuracy(self):
        radius = ureg.Quantity(25, "km")
        wps = rosette(CENTER, 0.0, ALT, radius)
        # Both tips of the first line should be at the radius from center
        for wp in wps[:2]:
            dist, _ = pymap3d.vincenty.vdist(
                CENTER[0], CENTER[1], wp.latitude, wp.longitude
            )
            assert dist == pytest.approx(25000, rel=1e-3)

    def test_line_length(self):
        """Each line should span 2 * radius (full diameter)."""
        radius = ureg.Quantity(25, "km")
        wps = rosette(CENTER, 0.0, ALT, radius)
        dist, _ = pymap3d.vincenty.vdist(
            wps[0].latitude, wps[0].longitude,
            wps[1].latitude, wps[1].longitude
        )
        assert dist == pytest.approx(50000, rel=1e-3)


class TestPolygon:
    def test_square(self):
        wps = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=4)
        # 4 + 1 (closed) = 5
        assert len(wps) == 5
        assert all(wp.segment_type == "pattern" for wp in wps)

    def test_triangle(self):
        wps = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=3)
        assert len(wps) == 4  # 3 + 1 closed

    def test_circle(self):
        wps = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=36)
        assert len(wps) == 37  # 36 + 1 closed

    def test_open(self):
        wps = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_sides=4, closed=False)
        assert len(wps) == 4

    def test_closed_loop(self):
        wps = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=4)
        assert wps[0].latitude == pytest.approx(wps[-1].latitude)
        assert wps[0].longitude == pytest.approx(wps[-1].longitude)

    def test_aspect_ratio(self):
        wps_square = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                             n_sides=4, aspect_ratio=1.0, closed=False)
        wps_rect = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                           n_sides=4, aspect_ratio=2.0, closed=False)
        # Rectangle should be more elongated along heading
        # Check that max lat extent of rect > square
        lats_sq = [wp.latitude for wp in wps_square]
        lats_rect = [wp.latitude for wp in wps_rect]
        assert (max(lats_rect) - min(lats_rect)) > (max(lats_sq) - min(lats_sq))

    def test_vertex_radius(self):
        radius = ureg.Quantity(10, "km")
        wps = polygon(CENTER, 0.0, ALT, radius, n_sides=6, closed=False)
        # All vertices should be at approximately the radius from center
        for wp in wps:
            dist, _ = pymap3d.vincenty.vdist(
                CENTER[0], CENTER[1], wp.latitude, wp.longitude
            )
            assert dist == pytest.approx(10000, rel=1e-2)


class TestSawtooth:
    def test_basic(self):
        wps = sawtooth(CENTER, 0.0,
                       altitude_min=ureg.Quantity(5000, "feet"),
                       altitude_max=ureg.Quantity(10000, "feet"),
                       leg_length=ureg.Quantity(100, "km"),
                       n_cycles=3)
        # 2*3 + 1 = 7
        assert len(wps) == 7
        assert all(wp.segment_type == "pattern" for wp in wps)

    def test_altitude_alternation(self):
        alt_min = ureg.Quantity(5000, "feet")
        alt_max = ureg.Quantity(10000, "feet")
        wps = sawtooth(CENTER, 0.0, alt_min, alt_max,
                       ureg.Quantity(100, "km"), n_cycles=2)
        # Pattern: max, min, max, min, max
        expected_ft = [10000, 5000, 10000, 5000, 10000]
        for wp, exp in zip(wps, expected_ft):
            assert wp.altitude_msl.to(ureg.foot).magnitude == pytest.approx(exp, rel=1e-3)

    def test_track_length(self):
        length = ureg.Quantity(100, "km")
        wps = sawtooth(CENTER, 90.0, ALT, ALT, length, n_cycles=2)
        # Total distance from first to last waypoint should be ~100 km
        total_dist = 0
        for i in range(len(wps) - 1):
            d, _ = pymap3d.vincenty.vdist(
                wps[i].latitude, wps[i].longitude,
                wps[i+1].latitude, wps[i+1].longitude
            )
            total_dist += d
        assert total_dist == pytest.approx(100000, rel=1e-3)

    def test_all_same_heading(self):
        wps = sawtooth(CENTER, 45.0, ALT, ALT, ureg.Quantity(50, "km"), n_cycles=2)
        for wp in wps:
            assert wp.heading == pytest.approx(45.0)


class TestFlightLinesToWaypointPath:
    def test_basic_conversion(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Test",
        )
        wps = flight_lines_to_waypoint_path([fl])
        assert len(wps) == 2
        assert all(wp.segment_type == "pattern" for wp in wps)
        assert wps[0].altitude_msl.to(ureg.foot).magnitude == pytest.approx(20000, rel=1e-2)

    def test_altitude_override(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        new_alt = ureg.Quantity(10000, "feet")
        wps = flight_lines_to_waypoint_path([fl], altitude=new_alt)
        for wp in wps:
            assert wp.altitude_msl.to(ureg.foot).magnitude == pytest.approx(10000, rel=1e-2)

    def test_two_flight_lines(self):
        fl1 = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(30, "km"), az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        fl2 = FlightLine.start_length_azimuth(
            lat1=34.5, lon1=-118.0,
            length=ureg.Quantity(30, "km"), az=90.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        wps = flight_lines_to_waypoint_path([fl1, fl2])
        assert len(wps) == 4


class TestSpiral:
    def test_waypoint_count(self):
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=2, points_per_turn=36)
        assert len(wps) == 2 * 36 + 1

    def test_segment_type(self):
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=1)
        assert all(wp.segment_type == "pattern" for wp in wps)

    def test_radius_accuracy(self):
        radius = ureg.Quantity(5, "km")
        wps = spiral(CENTER, 0.0, ALT, ALT, radius, n_turns=1)
        for wp in wps:
            dist, _ = pymap3d.vincenty.vdist(
                CENTER[0], CENTER[1], wp.latitude, wp.longitude
            )
            assert dist == pytest.approx(5000, rel=1e-2)

    def test_altitude_progression_ascending(self):
        alt_start = ureg.Quantity(5000, "feet")
        alt_end = ureg.Quantity(20000, "feet")
        wps = spiral(CENTER, 0.0, alt_start, alt_end, ureg.Quantity(5, "km"),
                     n_turns=3)
        alts = [wp.altitude_msl.to(ureg.foot).magnitude for wp in wps]
        # Monotonically increasing
        assert all(alts[i] <= alts[i + 1] for i in range(len(alts) - 1))

    def test_altitude_endpoints(self):
        alt_start = ureg.Quantity(5000, "feet")
        alt_end = ureg.Quantity(20000, "feet")
        wps = spiral(CENTER, 0.0, alt_start, alt_end, ureg.Quantity(5, "km"),
                     n_turns=2)
        assert wps[0].altitude_msl.to(ureg.foot).magnitude == pytest.approx(5000, rel=1e-3)
        assert wps[-1].altitude_msl.to(ureg.foot).magnitude == pytest.approx(20000, rel=1e-3)

    def test_heading_tangent_right(self):
        """Headings should be perpendicular to radial bearing (right turn = +90)."""
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1, direction="right", points_per_turn=36)
        # First waypoint is at bearing 0 from center, tangent should be ~90
        assert wps[0].heading == pytest.approx(90.0, abs=1.0)

    def test_heading_tangent_left(self):
        """Left turn: tangent heading should be bearing - 90."""
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1, direction="left", points_per_turn=36)
        # First waypoint at bearing 0, tangent for left turn should be ~270
        assert wps[0].heading == pytest.approx(270.0, abs=1.0)

    def test_direction_right_clockwise(self):
        """Right turns should progress clockwise (increasing bearing)."""
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=0.25, direction="right", points_per_turn=36)
        # After 1/4 turn clockwise from bearing 0, should be near bearing 90
        _, az = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1], wps[-1].latitude, wps[-1].longitude
        )
        assert float(az) == pytest.approx(90.0, abs=5.0)

    def test_direction_left_counterclockwise(self):
        """Left turns should progress counterclockwise (decreasing bearing)."""
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=0.25, direction="left", points_per_turn=36)
        # After 1/4 turn CCW from bearing 0, should be near bearing 270
        _, az = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1], wps[-1].latitude, wps[-1].longitude
        )
        assert float(az) == pytest.approx(270.0, abs=5.0)

    def test_fractional_turns(self):
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1.5, points_per_turn=36)
        assert len(wps) == int(1.5 * 36) + 1

    def test_constant_altitude_orbit(self):
        """altitude_start == altitude_end should produce a constant-altitude orbit."""
        wps = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=2)
        alt_ft = ALT.to(ureg.foot).magnitude
        for wp in wps:
            assert wp.altitude_msl.to(ureg.foot).magnitude == pytest.approx(alt_ft, rel=1e-6)

    def test_invalid_n_turns(self):
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError, match="n_turns"):
            spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=0)

    def test_invalid_points_per_turn(self):
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError, match="points_per_turn"):
            spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), points_per_turn=2)
