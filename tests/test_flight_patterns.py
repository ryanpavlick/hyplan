"""Tests for hyplan.flight_patterns."""

import pytest
import pymap3d.vincenty

from hyplan.units import ureg
from hyplan.flight_line import FlightLine
from hyplan.pattern import Pattern
from hyplan.flight_patterns import (
    racetrack,
    rosette,
    polygon,
    sawtooth,
    spiral,
    glint_arc,
    flight_lines_to_waypoint_path,
    coordinated_line,
)
from hyplan.aircraft import NASA_ER2, NASA_P3


CENTER = (34.0, -118.0)
ALT = ureg.Quantity(20000, "feet")


class TestRacetrack:
    def test_single_leg(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"))
        assert isinstance(pat, Pattern)
        assert pat.kind == "racetrack"
        assert len(pat.lines) == 1
        assert all(isinstance(fl, FlightLine) for fl in pat.lines.values())

    def test_n_legs(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"),
                        n_legs=3, offset=ureg.Quantity(5, "km"))
        assert len(pat.lines) == 3
        assert list(pat.lines.keys()) == ["leg_1", "leg_2", "leg_3"]

    def test_two_legs_out_and_back(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"),
                        n_legs=2, offset=ureg.Quantity(5, "km"))
        legs = list(pat.lines.values())
        h1 = legs[0].waypoint1.heading
        h2 = legs[1].waypoint1.heading
        assert abs((h2 - h1 + 180) % 360 - 180) == pytest.approx(180, abs=1)

    def test_lawnmower(self):
        pat = racetrack(CENTER, 90.0, ALT, ureg.Quantity(30, "km"),
                        n_legs=6, offset=ureg.Quantity(2, "km"))
        assert len(pat.lines) == 6

    def test_offset_list(self):
        pat = racetrack(CENTER, 45.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=3, offset=[0, 7000, 14000])
        assert len(pat.lines) == 3

    def test_offset_list_wrong_length(self):
        with pytest.raises(ValueError, match="offset list length"):
            racetrack(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_legs=3, offset=[0, 1000])

    def test_vertical_wall(self):
        alts = [ureg.Quantity(a, "feet") for a in [5000, 6000, 7000]]
        pat = racetrack(CENTER, 0.0, alts[0], ureg.Quantity(30, "km"),
                        n_legs=3, offset=0, altitudes=alts)
        legs = list(pat.lines.values())
        for leg, alt_expected in zip(legs, alts):
            assert leg.altitude_msl.m_as(ureg.foot) == pytest.approx(
                alt_expected.m_as(ureg.foot), rel=1e-3)

    def test_stacked(self):
        stack = [ureg.Quantity(a, "feet") for a in [9000, 7000, 5000]]
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=2, offset=ureg.Quantity(3, "km"),
                        stack_altitudes=stack)
        assert len(pat.lines) == 6  # 2 legs * 3 altitudes

    def test_altitudes_wrong_length(self):
        with pytest.raises(ValueError, match="altitudes length"):
            racetrack(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_legs=2, altitudes=[ALT])

    def test_leg_length_accuracy(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(50, "km"))
        leg = list(pat.lines.values())[0]
        dist, _ = pymap3d.vincenty.vdist(leg.lat1, leg.lon1, leg.lat2, leg.lon2)
        assert dist == pytest.approx(50000, rel=1e-3)

    def test_site_name(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_legs=2)
        legs = list(pat.lines.values())
        assert legs[0].site_name == "Leg1"
        assert legs[1].site_name == "Leg2"

    def test_params_roundtrip(self):
        pat = racetrack(CENTER, 45.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=3, offset=ureg.Quantity(2, "km"))
        assert pat.params["center_lat"] == pytest.approx(34.0)
        assert pat.params["center_lon"] == pytest.approx(-118.0)
        assert pat.params["heading"] == pytest.approx(45.0)
        assert pat.params["leg_length_m"] == pytest.approx(20000, rel=1e-6)
        assert pat.params["n_legs"] == 3
        assert pat.params["offset_m"] == pytest.approx(2000, rel=1e-6)


class TestRosette:
    def test_three_line_default(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        assert isinstance(pat, Pattern)
        assert pat.kind == "rosette"
        assert len(pat.lines) == 3

    def test_lines_cross_center(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        for line in pat.lines.values():
            mid_lat = (line.lat1 + line.lat2) / 2
            mid_lon = (line.lon1 + line.lon2) / 2
            assert mid_lat == pytest.approx(CENTER[0], abs=0.01)
            assert mid_lon == pytest.approx(CENTER[1], abs=0.01)

    def test_two_line_cross(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=2)
        assert len(pat.lines) == 2

    def test_custom_angles(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      angles=[30.0, 120.0])
        assert len(pat.lines) == 2
        assert pat.params["angles"] == [30.0, 120.0]

    def test_single_line(self):
        pat = rosette(CENTER, 90.0, ALT, ureg.Quantity(15, "km"), n_lines=1)
        assert len(pat.lines) == 1

    def test_radius_accuracy(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        first = list(pat.lines.values())[0]
        for lat, lon in [(first.lat1, first.lon1), (first.lat2, first.lon2)]:
            dist, _ = pymap3d.vincenty.vdist(CENTER[0], CENTER[1], lat, lon)
            assert dist == pytest.approx(25000, rel=1e-3)

    def test_line_length(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(25, "km"))
        first = list(pat.lines.values())[0]
        dist, _ = pymap3d.vincenty.vdist(first.lat1, first.lon1, first.lat2, first.lon2)
        assert dist == pytest.approx(50000, rel=1e-3)

    def test_site_names(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        assert [fl.site_name for fl in pat.lines.values()] == ["L1", "L2", "L3"]


class TestPolygon:
    def test_square(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=4)
        assert isinstance(pat, Pattern)
        assert pat.kind == "polygon"
        assert len(pat.waypoints) == 5  # 4 + 1 closed
        assert all(wp.segment_type == "pattern" for wp in pat.waypoints)

    def test_triangle(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=3)
        assert len(pat.waypoints) == 4  # 3 + 1 closed

    def test_circle(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=36)
        assert len(pat.waypoints) == 37

    def test_open(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_sides=4, closed=False)
        assert len(pat.waypoints) == 4

    def test_closed_loop(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=4)
        wps = pat.waypoints
        assert wps[0].latitude == pytest.approx(wps[-1].latitude)
        assert wps[0].longitude == pytest.approx(wps[-1].longitude)

    def test_aspect_ratio(self):
        wps_sq = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                         n_sides=4, aspect_ratio=1.0, closed=False).waypoints
        wps_rect = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                           n_sides=4, aspect_ratio=2.0, closed=False).waypoints
        lats_sq = [wp.latitude for wp in wps_sq]
        lats_rect = [wp.latitude for wp in wps_rect]
        assert (max(lats_rect) - min(lats_rect)) > (max(lats_sq) - min(lats_sq))

    def test_vertex_radius(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"),
                      n_sides=6, closed=False)
        for wp in pat.waypoints:
            dist, _ = pymap3d.vincenty.vdist(
                CENTER[0], CENTER[1], wp.latitude, wp.longitude
            )
            assert dist == pytest.approx(10000, rel=1e-2)


class TestSawtooth:
    def test_basic(self):
        pat = sawtooth(CENTER, 0.0,
                       altitude_min=ureg.Quantity(5000, "feet"),
                       altitude_max=ureg.Quantity(10000, "feet"),
                       leg_length=ureg.Quantity(100, "km"),
                       n_cycles=3)
        assert isinstance(pat, Pattern)
        assert pat.kind == "sawtooth"
        assert len(pat.waypoints) == 7  # 2*3 + 1
        assert all(wp.segment_type == "pattern_turn" for wp in pat.waypoints)

    def test_altitude_alternation(self):
        alt_min = ureg.Quantity(5000, "feet")
        alt_max = ureg.Quantity(10000, "feet")
        pat = sawtooth(CENTER, 0.0, alt_min, alt_max,
                       ureg.Quantity(100, "km"), n_cycles=2)
        expected_ft = [10000, 5000, 10000, 5000, 10000]
        for wp, exp in zip(pat.waypoints, expected_ft):
            assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(exp, rel=1e-3)

    def test_track_length(self):
        pat = sawtooth(CENTER, 90.0, ALT, ALT,
                       ureg.Quantity(100, "km"), n_cycles=2)
        wps = pat.waypoints
        total_dist = 0
        for i in range(len(wps) - 1):
            d, _ = pymap3d.vincenty.vdist(
                wps[i].latitude, wps[i].longitude,
                wps[i+1].latitude, wps[i+1].longitude
            )
            total_dist += d
        assert total_dist == pytest.approx(100000, rel=1e-3)

    def test_all_same_heading(self):
        pat = sawtooth(CENTER, 45.0, ALT, ALT,
                       ureg.Quantity(50, "km"), n_cycles=2)
        for wp in pat.waypoints:
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
        assert wps[0].segment_type == "pattern"
        assert wps[1].segment_type == "pattern_turn"
        assert wps[0].altitude_msl.m_as(ureg.foot) == pytest.approx(20000, rel=1e-2)

    def test_altitude_override(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        new_alt = ureg.Quantity(10000, "feet")
        wps = flight_lines_to_waypoint_path([fl], altitude=new_alt)
        for wp in wps:
            assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(10000, rel=1e-2)

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
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=2, points_per_turn=36)
        assert pat.kind == "spiral"
        assert len(pat.waypoints) == 2 * 36 + 1

    def test_segment_type(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=1)
        assert all(wp.segment_type == "pattern" for wp in pat.waypoints)

    def test_radius_accuracy(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=1)
        for wp in pat.waypoints:
            dist, _ = pymap3d.vincenty.vdist(
                CENTER[0], CENTER[1], wp.latitude, wp.longitude
            )
            assert dist == pytest.approx(5000, rel=1e-2)

    def test_altitude_progression_ascending(self):
        alt_start = ureg.Quantity(5000, "feet")
        alt_end = ureg.Quantity(20000, "feet")
        pat = spiral(CENTER, 0.0, alt_start, alt_end, ureg.Quantity(5, "km"),
                     n_turns=3)
        alts = [wp.altitude_msl.m_as(ureg.foot) for wp in pat.waypoints]
        assert all(alts[i] <= alts[i + 1] for i in range(len(alts) - 1))

    def test_altitude_endpoints(self):
        alt_start = ureg.Quantity(5000, "feet")
        alt_end = ureg.Quantity(20000, "feet")
        pat = spiral(CENTER, 0.0, alt_start, alt_end, ureg.Quantity(5, "km"),
                     n_turns=2)
        wps = pat.waypoints
        assert wps[0].altitude_msl.m_as(ureg.foot) == pytest.approx(5000, rel=1e-3)
        assert wps[-1].altitude_msl.m_as(ureg.foot) == pytest.approx(20000, rel=1e-3)

    def test_heading_tangent_right(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1, direction="right", points_per_turn=36)
        assert pat.waypoints[0].heading == pytest.approx(90.0, abs=1.0)

    def test_heading_tangent_left(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1, direction="left", points_per_turn=36)
        assert pat.waypoints[0].heading == pytest.approx(270.0, abs=1.0)

    def test_direction_right_clockwise(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=0.25, direction="right", points_per_turn=36)
        last = pat.waypoints[-1]
        _, az = pymap3d.vincenty.vdist(CENTER[0], CENTER[1], last.latitude, last.longitude)
        assert float(az) == pytest.approx(90.0, abs=5.0)

    def test_direction_left_counterclockwise(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=0.25, direction="left", points_per_turn=36)
        last = pat.waypoints[-1]
        _, az = pymap3d.vincenty.vdist(CENTER[0], CENTER[1], last.latitude, last.longitude)
        assert float(az) == pytest.approx(270.0, abs=5.0)

    def test_fractional_turns(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=1.5, points_per_turn=36)
        assert len(pat.waypoints) == int(1.5 * 36) + 1

    def test_constant_altitude_orbit(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=2)
        alt_ft = ALT.m_as(ureg.foot)
        for wp in pat.waypoints:
            assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(alt_ft, rel=1e-6)

    def test_invalid_n_turns(self):
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError, match="n_turns"):
            spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=0)

    def test_invalid_points_per_turn(self):
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError, match="points_per_turn"):
            spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), points_per_turn=2)


class TestGlintArc:
    """glint_arc() wraps GlintArc as a waypoint-based Pattern."""

    import datetime as _dt
    DAYTIME = _dt.datetime(2025, 7, 15, 19, 0, tzinfo=_dt.timezone.utc)  # ~noon PDT
    NIGHT = _dt.datetime(2025, 7, 15, 8, 0, tzinfo=_dt.timezone.utc)     # 1am PDT
    SPEED = ureg.Quantity(120, "m/s")
    GA_ALT = ureg.Quantity(6000, "meter")

    def test_basic_returns_pattern(self):
        pat = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
        )
        assert isinstance(pat, Pattern)
        assert pat.kind == "glint_arc"
        assert pat.is_waypoint_based
        assert len(pat.waypoints) > 30

    def test_waypoints_at_constant_altitude(self):
        pat = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
        )
        for wp in pat.waypoints:
            assert wp.altitude_msl.m_as(ureg.meter) == pytest.approx(6000.0, rel=1e-6)

    def test_params_serializable_and_roundtrip(self):
        import json
        pat = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED, bank_direction="left",
        )
        # All params must be JSON-serializable (no Quantity / datetime objects)
        json.dumps(pat.params)
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "glint_arc"
        assert len(clone.waypoints) == len(pat.waypoints)
        assert clone.waypoints[0].latitude == pytest.approx(pat.waypoints[0].latitude)

    def test_regenerate_with_bank_angle_override(self):
        pat = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
        )
        regen = pat.regenerate(bank_angle=15.0)
        assert regen.params["bank_angle"] == pytest.approx(15.0)
        assert regen.params["effective_bank_angle"] == pytest.approx(15.0)

    def test_regenerate_preserves_auto_when_input_was_auto(self):
        # Originally auto (bank_angle=None) -> stored as None.
        # Regenerating without override must keep auto behavior.
        pat = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
        )
        assert pat.params["bank_angle"] is None  # input was auto
        regen = pat.regenerate()
        assert regen.params["bank_angle"] is None
        # effective should equal solar zenith at the original time
        assert regen.params["effective_bank_angle"] == pytest.approx(
            pat.params["solar_zenith"], rel=1e-3,
        )

    def test_sun_below_horizon_raises(self):
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError):
            glint_arc(
                center=CENTER, observation_datetime=self.NIGHT,
                altitude=self.GA_ALT, speed=self.SPEED,
            )

    def test_collection_length_shortens_arc(self):
        full = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
        )
        short = glint_arc(
            center=CENTER, observation_datetime=self.DAYTIME,
            altitude=self.GA_ALT, speed=self.SPEED,
            collection_length=ureg.Quantity(2, "km"),
        )
        # Shorter collection -> fewer densified waypoints
        assert len(short.waypoints) < len(full.waypoints)


class TestPatternRoundtrip:
    """Pattern.to_dict / from_dict preserves content for all kinds."""

    def test_rosette_roundtrip(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "rosette"
        assert len(clone.lines) == 3
        assert list(clone.lines.keys()) == list(pat.lines.keys())

    def test_racetrack_roundtrip(self):
        pat = racetrack(CENTER, 30.0, ALT, ureg.Quantity(20, "km"),
                        n_legs=2, offset=ureg.Quantity(5, "km"))
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "racetrack"
        assert len(clone.lines) == 2

    def test_spiral_roundtrip(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=2, points_per_turn=18)
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "spiral"
        assert len(clone.waypoints) == len(pat.waypoints)
        assert clone.waypoints[0].latitude == pytest.approx(pat.waypoints[0].latitude)

    def test_sawtooth_roundtrip(self):
        pat = sawtooth(CENTER, 0.0,
                       altitude_min=ureg.Quantity(5000, "feet"),
                       altitude_max=ureg.Quantity(10000, "feet"),
                       leg_length=ureg.Quantity(50, "km"),
                       n_cycles=2)
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "sawtooth"
        assert len(clone.waypoints) == len(pat.waypoints)

    def test_polygon_roundtrip(self):
        pat = polygon(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_sides=5)
        clone = Pattern.from_dict(pat.to_dict())
        assert clone.kind == "polygon"
        assert len(clone.waypoints) == len(pat.waypoints)


class TestPatternRegenerate:
    """Pattern.regenerate re-invokes the generator and produces equivalent output."""

    def test_rosette_regenerate_identity(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        regen = pat.regenerate()
        assert regen.kind == "rosette"
        assert len(regen.lines) == 3

    def test_rosette_regenerate_override(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        regen = pat.regenerate(n_lines=5)
        assert len(regen.lines) == 5

    def test_racetrack_regenerate_override(self):
        pat = racetrack(CENTER, 0.0, ALT, ureg.Quantity(20, "km"), n_legs=2,
                        offset=ureg.Quantity(3, "km"))
        regen = pat.regenerate(n_legs=4)
        assert len(regen.lines) == 4

    def test_spiral_regenerate(self):
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"),
                     n_turns=2, points_per_turn=18)
        regen = pat.regenerate()
        assert len(regen.waypoints) == len(pat.waypoints)


class TestPatternReplaceLine:
    def test_replace_line_preserves_id(self):
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        first_id = list(pat.lines.keys())[0]
        new_fl = FlightLine.start_length_azimuth(
            lat1=40.0, lon1=-100.0,
            length=ureg.Quantity(5, "km"), az=45.0,
            altitude_msl=ureg.Quantity(5000, "meter"),
        )
        pat.replace_line(first_id, new_fl)
        assert list(pat.lines.keys())[0] == first_id
        assert pat.lines[first_id].lat1 == pytest.approx(40.0)

    def test_replace_line_waypoint_pattern_rejected(self):
        from hyplan.exceptions import HyPlanValueError
        pat = spiral(CENTER, 0.0, ALT, ALT, ureg.Quantity(5, "km"), n_turns=1)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(1, "km"), az=0.0,
            altitude_msl=ureg.Quantity(1000, "meter"),
        )
        with pytest.raises(HyPlanValueError, match="waypoint-based"):
            pat.replace_line("x", fl)

    def test_replace_line_unknown_id(self):
        from hyplan.exceptions import HyPlanValueError
        pat = rosette(CENTER, 0.0, ALT, ureg.Quantity(10, "km"), n_lines=3)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(1, "km"), az=0.0,
            altitude_msl=ureg.Quantity(1000, "meter"),
        )
        with pytest.raises(HyPlanValueError, match="not part of"):
            pat.replace_line("does_not_exist", fl)


class TestCoordinatedLine:
    """Tests for the coordinated dual-aircraft line pattern."""

    def _make_result(self, **kwargs):
        defaults = dict(
            center=CENTER, heading=0.0,
            primary_leg_length=ureg.Quantity(200, "km"),
            primary_aircraft=NASA_P3(),
            secondary_aircraft=NASA_ER2(),
            primary_altitude=ureg.Quantity(5000, "feet"),
            secondary_altitude=ureg.Quantity(65000, "feet"),
        )
        defaults.update(kwargs)
        return coordinated_line(**defaults)

    def test_basic(self):
        result = self._make_result(ground_speed_ratio=1.2)
        assert len(result["primary"]) == 2
        assert len(result["secondary"]) == 2
        assert result["center"].latitude == CENTER[0]
        assert result["center"].longitude == CENTER[1]
        assert result["ground_speed_ratio"] == 1.2

    def test_symmetry(self):
        result = self._make_result(ground_speed_ratio=1.3)
        d_p1, _ = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1],
            result["primary"][0].latitude, result["primary"][0].longitude)
        d_p2, _ = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1],
            result["primary"][1].latitude, result["primary"][1].longitude)
        assert float(d_p1) == pytest.approx(float(d_p2), rel=1e-3)
        d_e1, _ = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1],
            result["secondary"][0].latitude, result["secondary"][0].longitude)
        d_e2, _ = pymap3d.vincenty.vdist(
            CENTER[0], CENTER[1],
            result["secondary"][1].latitude, result["secondary"][1].longitude)
        assert float(d_e1) == pytest.approx(float(d_e2), rel=1e-3)

    def test_speed_ratio(self):
        ratio = 1.4
        result = self._make_result(ground_speed_ratio=ratio)
        d_pri, _ = pymap3d.vincenty.vdist(
            result["primary"][0].latitude, result["primary"][0].longitude,
            result["primary"][1].latitude, result["primary"][1].longitude)
        d_sec, _ = pymap3d.vincenty.vdist(
            result["secondary"][0].latitude, result["secondary"][0].longitude,
            result["secondary"][1].latitude, result["secondary"][1].longitude)
        assert float(d_sec) == pytest.approx(float(d_pri) * ratio, rel=1e-3)

    def test_multi_ratio(self):
        result = self._make_result(ground_speed_ratio=[1.2, 1.45])
        assert isinstance(result["secondary"], list)
        assert len(result["secondary"]) == 2
        assert len(result["secondary"][0]) == 2
        assert len(result["secondary"][1]) == 2
        assert result["ground_speed_ratio"] == [1.2, 1.45]
        d1, _ = pymap3d.vincenty.vdist(
            result["secondary"][0][0].latitude, result["secondary"][0][0].longitude,
            result["secondary"][0][1].latitude, result["secondary"][0][1].longitude)
        d2, _ = pymap3d.vincenty.vdist(
            result["secondary"][1][0].latitude, result["secondary"][1][0].longitude,
            result["secondary"][1][1].latitude, result["secondary"][1][1].longitude)
        assert float(d2) > float(d1)

    def test_segment_types(self):
        result = self._make_result(ground_speed_ratio=1.2)
        assert result["primary"][0].segment_type == "pattern"
        assert result["primary"][1].segment_type == "pattern_turn"
        assert result["secondary"][0].segment_type == "pattern"
        assert result["secondary"][1].segment_type == "pattern_turn"

    def test_auto_ratio(self):
        result = self._make_result()
        assert isinstance(result["ground_speed_ratio"], float)
        assert result["ground_speed_ratio"] > 1.0

    def test_altitudes(self):
        pri_alt = ureg.Quantity(5000, "feet")
        sec_alt = ureg.Quantity(65000, "feet")
        result = self._make_result(
            primary_altitude=pri_alt, secondary_altitude=sec_alt,
            ground_speed_ratio=1.2)
        for wp in result["primary"]:
            assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(5000, rel=1e-3)
        for wp in result["secondary"]:
            assert wp.altitude_msl.m_as(ureg.foot) == pytest.approx(65000, rel=1e-3)

    def test_heading(self):
        result = self._make_result(heading=45.0, ground_speed_ratio=1.2)
        for wp in result["primary"] + result["secondary"]:
            assert wp.heading == pytest.approx(45.0, abs=1.0)
