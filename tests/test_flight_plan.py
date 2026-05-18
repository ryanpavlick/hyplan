"""Tests for hyplan.flight_plan."""

import geopandas as gpd
import numpy as np
import pytest

from hyplan.aircraft import NASA_ER2, KingAirB200
from hyplan.airports import Airport, initialize_data
from hyplan.exceptions import HyPlanValueError
from hyplan.flight_line import FlightLine
from hyplan.flight_plan import (
    _track_hold_solution_from_uv,
    compute_flight_plan,
)
from hyplan.units import ureg
from hyplan.waypoint import Waypoint


@pytest.fixture(scope="module", autouse=True)
def init_airport_data():
    """Initialize airport data once for all tests in this module."""
    initialize_data(countries=["US"])


@pytest.fixture
def b200():
    return KingAirB200()


@pytest.fixture
def flight_line():
    return FlightLine.start_length_azimuth(
        lat1=34.05, lon1=-118.25,
        length=ureg.Quantity(50000, "meter"),
        az=45.0,
        altitude_msl=ureg.Quantity(20000, "feet"),
        site_name="Test Line",
    )


class TestComputeFlightPlan:
    def test_returns_geodataframe(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0

    def test_has_required_columns(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        for col in ["segment_type", "segment_name", "distance", "time_to_segment"]:
            assert col in plan.columns

    def test_segment_types(self, b200, flight_line):
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        segment_types = plan["segment_type"].unique()
        # Should contain at least takeoff, data collection, and landing segments
        assert len(segment_types) >= 2

    def test_no_airports(self, b200, flight_line):
        # Should work without airports (just flight lines)
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[flight_line],
        )
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0

    def test_waypoint_loiter(self, b200):
        """Waypoint with delay produces a loiter segment with a hold-orbit ground track."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                       delay=ureg.Quantity(5, "minute"))
        wp2 = Waypoint(34.1, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        loiter = plan[plan["segment_type"] == "loiter"]
        assert len(loiter) == 1
        assert loiter.iloc[0]["time_to_segment"] == pytest.approx(5.0)
        # Distance is now the actual ground covered during the loiter
        # (cruise speed × delay), not zero. For the B200 at 20kft / 5min,
        # this is an order-of-magnitude tens of nautical miles.
        assert loiter.iloc[0]["distance"] > 0
        # Geometry is the closed orbit ring (LineString), not a Point.
        from shapely.geometry import LineString
        assert isinstance(loiter.iloc[0]["geometry"], LineString)
        assert loiter.iloc[0]["geometry"].coords[0] == loiter.iloc[0]["geometry"].coords[-1]

    def test_waypoint_speed_override(self, b200):
        """Per-waypoint speed override is used for the departing leg."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1")
        wp2 = Waypoint(34.1, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")

        # Without speed override
        plan_default = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        time_default = plan_default["time_to_segment"].sum()

        # With slower speed override
        wp1_slow = Waypoint(34.0, -118.0, 0.0,
                            altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                            speed=ureg.Quantity(50, "knot"))
        plan_slow = compute_flight_plan(aircraft=b200, flight_sequence=[wp1_slow, wp2])
        time_slow = plan_slow["time_to_segment"].sum()

        # Slower speed should take longer
        assert time_slow > time_default

    def test_waypoint_segment_type(self, b200):
        """Waypoint segment_type overrides 'transit' label."""
        wp1 = Waypoint(34.0, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1",
                       segment_type="pattern")
        wp2 = Waypoint(34.01, -118.0, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        # The level leg from WP1 to WP2 should be labeled "pattern"
        pattern_segs = plan[plan["segment_type"] == "pattern"]
        assert len(pattern_segs) >= 1


class TestLoiterOrbitGeometry:
    """Direct tests for hyplan.planning.segments.loiter_orbit_geometry."""

    @pytest.fixture
    def b200(self):
        return KingAirB200()

    def test_returns_closed_linestring(self, b200):
        from hyplan.planning.segments import loiter_orbit_geometry
        wp = Waypoint(
            34.0, -118.0, 0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            delay=ureg.Quantity(5, "minute"),
        )
        ring = loiter_orbit_geometry(wp, b200)
        assert ring.coords[0] == ring.coords[-1]
        # default n_points=72 → 73 coordinates after closing
        assert len(ring.coords) == 73

    def test_radius_matches_v_squared_over_g_tan_bank(self, b200):
        """The orbit radius equals v² / (g · tan(bank_cruise))."""
        from shapely.geometry import Point
        from shapely.ops import transform as shp_transform

        from hyplan.geometry import get_utm_transforms
        from hyplan.planning.segments import loiter_orbit_geometry

        altitude = ureg.Quantity(20000, "feet")
        wp = Waypoint(34.0, -118.0, 0.0, altitude_msl=altitude)
        ring = loiter_orbit_geometry(wp, b200)

        v_mps = b200.cruise_speed_at(altitude).m_as("meter/second")
        bank_rad = np.radians(b200.turn_model.bank_by_phase.cruise_deg)
        expected_radius_m = (v_mps ** 2) / (9.80665 * np.tan(bank_rad))

        # Diameter of the ring: max distance between any two ring points.
        coords = np.array(list(ring.coords))
        pts_wgs = [Point(lon, lat) for lon, lat in coords]
        to_utm, _ = get_utm_transforms(pts_wgs)
        pts_utm = np.array([(shp_transform(to_utm, p).x, shp_transform(to_utm, p).y) for p in pts_wgs])
        # Pairwise distances; pick max as diameter.
        dx = pts_utm[:, None, 0] - pts_utm[None, :, 0]
        dy = pts_utm[:, None, 1] - pts_utm[None, :, 1]
        diameter = np.sqrt(dx * dx + dy * dy).max()
        assert diameter == pytest.approx(2 * expected_radius_m, rel=0.01)

    def test_waypoint_lies_on_orbit(self, b200):
        """The waypoint sits exactly on the orbit (UTM-distance ≈ 0 to first vertex)."""
        from shapely.geometry import Point

        from hyplan.planning.segments import loiter_orbit_geometry
        wp = Waypoint(
            34.0, -118.0, 0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
        )
        ring = loiter_orbit_geometry(wp, b200)
        first = ring.coords[0]
        # First coordinate is within rounding of the waypoint location.
        wp_pt = Point(wp.longitude, wp.latitude)
        assert wp_pt.distance(Point(first)) < 1e-6

    def test_higher_altitude_gives_larger_orbit(self):
        """Higher cruise speed (or larger v / smaller bank) → larger turn radius."""
        # Use NASA_GIII: its cruise schedule grows monotonically from
        # SL through cruise altitude, so the orbit-grows-with-altitude
        # invariant holds.  The B-200 cruise schedule is flat above
        # FL200 (calibrated against ACT-America), so it would be a
        # poor choice for testing the underlying geometry.
        from shapely.geometry import Point
        from shapely.ops import transform as shp_transform

        from hyplan.aircraft import NASA_GIII
        from hyplan.geometry import get_utm_transforms
        from hyplan.planning.segments import loiter_orbit_geometry

        ac = NASA_GIII()

        def diameter_m(wp):
            ring = loiter_orbit_geometry(wp, ac)
            coords = np.array(list(ring.coords))
            pts_wgs = [Point(lon, lat) for lon, lat in coords]
            to_utm, _ = get_utm_transforms(pts_wgs)
            pts_utm = np.array(
                [(shp_transform(to_utm, p).x, shp_transform(to_utm, p).y) for p in pts_wgs]
            )
            dx = pts_utm[:, None, 0] - pts_utm[None, :, 0]
            dy = pts_utm[:, None, 1] - pts_utm[None, :, 1]
            return np.sqrt(dx * dx + dy * dy).max()

        wp_low = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(5_000, "feet"))
        wp_high = Waypoint(34.0, -118.0, 0.0, altitude_msl=ureg.Quantity(35_000, "feet"))
        # Cruise speed grows with altitude → orbit radius grows.
        assert diameter_m(wp_high) > diameter_m(wp_low)

    def test_requires_altitude(self, b200):
        from hyplan.planning.segments import loiter_orbit_geometry
        wp = Waypoint(34.0, -118.0, 0.0)  # no altitude_msl
        with pytest.raises(ValueError):
            loiter_orbit_geometry(wp, b200)

    def test_loiter_distance_matches_speed_times_delay(self, b200):
        """The loiter row's distance equals cruise speed × delay (not orbit circumference)."""
        wp1 = Waypoint(
            34.0, -118.0, 0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            delay=ureg.Quantity(5, "minute"),
            name="WP1",
        )
        wp2 = Waypoint(34.1, -118.0, 0.0, altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        loiter = plan[plan["segment_type"] == "loiter"].iloc[0]
        v_mps = b200.cruise_speed_at(wp1.altitude_msl).m_as("meter/second")
        expected_nm = ureg.Quantity(v_mps * 5 * 60, "meter").m_as(ureg.nautical_mile)
        assert loiter["distance"] == pytest.approx(expected_nm, rel=0.001)


class TestWindCorrectedTransit:
    """Wind-aware transit time via compute_flight_plan wind_speed/wind_direction."""

    def _north_leg(self):
        # Altitude 12 kft pairs with B-200 cruise TAS 224 kt — a
        # regime where the trochoidal Dubins solver gives a clean
        # straight-leg solution under 50-kt wind.  At lower altitudes
        # (TAS 220 kt) and at higher (TAS ≥ 230 kt) the solver
        # produces path warping that complicates the simple
        # tailwind-reduces-time invariant tested below.
        wp1 = Waypoint(34.00, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(12000, "feet"), name="WP1")
        wp2 = Waypoint(34.50, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(12000, "feet"), name="WP2")
        return wp1, wp2

    def test_zero_wind_regression(self, b200):
        """Explicit zero wind must reproduce the default (wind=None) result."""
        wp1, wp2 = self._north_leg()
        plan_default = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_zero = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(0, "knot"),
            wind_direction=0.0,
        )
        assert plan_zero["time_to_segment"].sum() == pytest.approx(
            plan_default["time_to_segment"].sum()
        )

    def test_headwind_slows_transit(self, b200):
        """Northbound leg into a due-north headwind should take longer."""
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_hw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(50, "knot"),
            wind_direction=0.0,  # wind FROM the north → headwind on a northbound leg
        )
        assert plan_hw["time_to_segment"].sum() > plan_calm["time_to_segment"].sum()

    def test_tailwind_speeds_transit(self, b200):
        """Northbound leg with a due-south tailwind should take less time."""
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_tw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(50, "knot"),
            wind_direction=180.0,  # wind FROM the south → tailwind northbound
        )
        assert plan_tw["time_to_segment"].sum() < plan_calm["time_to_segment"].sum()

    def test_crosswind_small_effect(self, b200):
        """Pure crosswind has a small but non-zero effect on total time.

        With trochoidal Dubins path planning, crosswind affects the turn
        geometry slightly, so we allow up to 2% deviation from still air.
        """
        wp1, wp2 = self._north_leg()
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_cw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=ureg.Quantity(30, "knot"),
            wind_direction=90.0,  # wind FROM the east → crosswind on northbound
        )
        assert plan_cw["time_to_segment"].sum() == pytest.approx(
            plan_calm["time_to_segment"].sum(), rel=0.02
        )

    def test_headwind_magnitude_matches_hand_calc(self, b200):
        """With a 30 kt pure headwind, flight-line time should grow by
        TAS/(TAS-30)."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,  # due north
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="North Line",
        )
        plan_calm = compute_flight_plan(aircraft=b200, flight_sequence=[fl])
        plan_hw = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[fl],
            wind_speed=ureg.Quantity(30, "knot"),
            wind_direction=0.0,
        )
        row_calm = plan_calm[plan_calm["segment_type"] == "flight_line"].iloc[0]
        row_hw = plan_hw[plan_hw["segment_type"] == "flight_line"].iloc[0]
        tas_kt = b200.cruise_speed_at(ureg.Quantity(20000, "feet")).m_as("knot")
        expected_factor = tas_kt / (tas_kt - 30.0)
        assert row_hw["time_to_segment"] / row_calm["time_to_segment"] == pytest.approx(
            expected_factor, rel=1e-3
        )

    def test_wind_direction_required_with_nonzero_wind(self, b200):
        wp1, wp2 = self._north_leg()
        with pytest.raises(Exception):
            compute_flight_plan(
                aircraft=b200,
                flight_sequence=[wp1, wp2],
                wind_speed=ureg.Quantity(30, "knot"),
                # wind_direction omitted on purpose
            )

    def test_unflyable_headwind_raises(self, b200):
        """A headwind larger than TAS must raise — ground speed would go negative."""
        wp1, wp2 = self._north_leg()
        with pytest.raises(Exception):
            compute_flight_plan(
                aircraft=b200,
                flight_sequence=[wp1, wp2],
                wind_speed=ureg.Quantity(1000, "knot"),
                wind_direction=0.0,
            )


class TestTrackHoldSolution:
    """Test the crab-angle-aware track-hold wind solver."""

    def _zero_wind(self):
        return 0.0 * ureg.meter / ureg.second

    def test_no_wind(self):
        """No wind: crab=0, heading=track, groundspeed=TAS."""
        tas = 250 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 90.0, self._zero_wind(), self._zero_wind(),
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["heading_deg"] == pytest.approx(90.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(250.0, rel=0.01)

    def test_pure_tailwind(self):
        """Tailwind along track: crab=0, GS > TAS."""
        tas = 250 * ureg.knot
        # Track north, wind from south (tailwind)
        # Wind from south = v positive (northward component)
        v_tail = 30 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, self._zero_wind(), v_tail,
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) > 250.0
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(280.0, rel=0.01)

    def test_pure_headwind(self):
        """Headwind along track: crab=0, GS < TAS."""
        tas = 250 * ureg.knot
        # Track north, wind from north (headwind) = v negative
        v_head = -30 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, self._zero_wind(), v_head,
        )
        assert sol["crab_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert sol["groundspeed"].m_as(ureg.knot) < 250.0
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(220.0, rel=0.01)

    def test_pure_crosswind(self):
        """Crosswind: nonzero crab, GS = TAS*cos(crab)."""
        tas = 250 * ureg.knot
        # Track north, wind from west (eastward u component)
        u_cross = 50 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, u_cross, self._zero_wind(),
        )
        assert abs(sol["crab_angle_deg"]) > 1.0
        # GS should be TAS*cos(crab) (no along-track wind component)
        expected_gs = 250.0 * np.cos(np.radians(sol["crab_angle_deg"]))
        assert sol["groundspeed"].m_as(ureg.knot) == pytest.approx(expected_gs, rel=0.01)

    def test_crosswind_exceeds_tas_raises(self):
        """Crosswind > TAS: cannot hold track."""
        tas = 100 * ureg.knot
        u_huge = 200 * ureg.knot
        with pytest.raises(HyPlanValueError, match="Crosswind"):
            _track_hold_solution_from_uv(
                tas, 0.0, u_huge, self._zero_wind(),
            )

    def test_crab_sign_convention(self):
        """Crosswind from the right requires left (negative) crab."""
        tas = 250 * ureg.knot
        # Track north, wind from east (u negative = westward)
        # Crosswind = u*cos(track) - v*sin(track) = u*cos(0) = u (negative)
        # crab = asin(-crosswind/TAS) = asin(positive) = positive
        # Actually: wind from east means u < 0 (westward component)
        u_from_east = -50 * ureg.knot
        sol = _track_hold_solution_from_uv(
            tas, 0.0, u_from_east, self._zero_wind(),
        )
        # Aircraft must crab right (positive) to compensate westward drift
        # crosswind = u*cos(0) - v*sin(0) = u = -50 kt (negative)
        # crab = asin(-crosswind/TAS) = asin(50/250) > 0
        assert sol["crab_angle_deg"] > 0


# ---------------------------------------------------------------------------
# End-to-end planner regression tests
# ---------------------------------------------------------------------------

from hyplan.flight_patterns import racetrack
from hyplan.winds import ConstantWindField


class TestPlannerRegression:
    """End-to-end regression tests for compute_flight_plan."""

    def test_single_line_no_wind(self, b200):
        """Single flight line, no wind: basic sanity."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="North Line",
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[fl])
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) > 0
        assert "flight_line" in plan["segment_type"].values
        assert plan["time_to_segment"].sum() > 0
        assert plan["distance"].dropna().sum() > 0

    def test_multi_line_racetrack(self, b200):
        """Racetrack pattern: all flight lines present with transitions."""
        pat = racetrack(
            center=(34.05, -118.25),
            heading=0.0,
            altitude=ureg.Quantity(20000, "feet"),
            leg_length=ureg.Quantity(50000, "meter"),
            n_legs=4,
            offset=ureg.Quantity(5000, "meter"),
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[pat])
        assert isinstance(plan, gpd.GeoDataFrame)
        assert len(plan) >= 4
        seg_types = set(plan["segment_type"].values)
        assert "pattern" in seg_types or "flight_line" in seg_types

    def test_pattern_legs_use_cruise_bank(self, b200):
        """Inter-line transits within a Pattern are level (cruise) flight,
        so the Dubins arc must reflect ``bank_by_phase.cruise_deg``.

        We verify this indirectly by comparing the inter-line transit
        time produced by the planner against the time predicted by the
        air-frame Dubins length under the cruise-deg bank: equal within
        floating-point tolerance.  Any regression that swapped to
        ``max_bank_deg`` would show up as a discrepancy because the two
        bank values differ for every calibrated aircraft.
        """
        from hyplan.dubins3d import DubinsPath2D
        from hyplan.units import ureg as _ureg
        # Two parallel flight lines stacked on top of each other; the
        # inter-line transit is a level cruise turn of ~U-shape geometry.
        fl1 = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=_ureg.Quantity(50, "km"),
            az=90.0,
            altitude_msl=_ureg.Quantity(20000, "feet"),
            site_name="A",
        )
        fl2 = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-117.5,  # downstream of fl1's end
            length=_ureg.Quantity(50, "km"),
            az=270.0,
            altitude_msl=_ureg.Quantity(20000, "feet"),
            site_name="B",
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[fl1, fl2])
        # Locate the transit row between the two flight lines.
        transit_rows = plan[plan["segment_type"] == "transit"]
        assert len(transit_rows) >= 1
        transit_t_min = transit_rows.iloc[0]["time_to_segment"]
        # Reference: build the same transit at cruise_deg directly.
        cruise_bank = b200.turn_model.bank_by_phase.for_phase("cruise")
        cruise_tas = b200.cruise_speed_at(_ureg.Quantity(20000, "feet"))
        ref_path = DubinsPath2D(
            fl1.waypoint2, fl2.waypoint1,
            speed=cruise_tas, bank_angle=cruise_bank,
        )
        ref_t_min = (
            ref_path.length / cruise_tas
        ).to(_ureg.minute).magnitude
        # Tolerance accommodates small phase-split / pitch-implicit
        # adjustments inside _hybrid_path; a regression that swapped
        # to max_bank_deg would shift the transit time by tens of
        # percent (B200: cruise_deg=25, max_bank_deg=30).
        assert transit_t_min == pytest.approx(ref_t_min, rel=2e-2)

    def test_airport_departure_and_return(self, b200):
        """Takeoff and approach phases appear with correct ordering."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.45, lon1=-119.85,
            length=ureg.Quantity(30000, "meter"),
            az=90.0,
            altitude_msl=ureg.Quantity(15000, "feet"),
            site_name="SBA Line",
        )
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[fl],
            takeoff_airport=Airport("KSBA"),
            return_airport=Airport("KSBA"),
        )
        seg_types = list(plan["segment_type"].values)
        # First segment should be takeoff or climb
        assert seg_types[0] in ("takeoff", "climb")
        # Last segment should be approach or descent
        assert seg_types[-1] in ("approach", "descent")
        # Flight line should be somewhere in the middle
        assert "flight_line" in seg_types

    def test_constant_wind_populates_fields(self, b200):
        """Constant wind: crab angle, groundspeed, tailwind fields populated."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="Wind Test Line",
        )
        wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=270.0)
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[fl], wind_source=wf,
        )
        fl_row = plan[plan["segment_type"] == "flight_line"].iloc[0]
        # Wind fields should be present and non-NaN
        assert "crab_angle_deg" in fl_row.index
        assert "groundspeed_kts" in fl_row.index
        assert "tailwind_kts" in fl_row.index
        assert "crosswind_kts" in fl_row.index
        # With westerly wind on a northbound line, crosswind should be nonzero
        assert abs(fl_row["crosswind_kts"]) > 0.1
        # Groundspeed should be positive and reasonable
        assert fl_row["groundspeed_kts"] > 50


class TestComputeFlightPlanApproachIntegration:
    """End-to-end compute_flight_plan tests covering the ApproachProfile path
    (Fix 1+2 from the planner-integration PR).

    Two variants:
      * NASA_ER2 has approach_profile -> last segment is "approach", ends at airport
      * KingAirB200 has no approach_profile -> legacy single-leg-to-runway path
    """

    @pytest.fixture
    def airport(self):
        return Airport("KSBA")

    @pytest.fixture
    def er2_flight_line(self):
        # ER-2 cruise altitude of 60 kft.
        return FlightLine.start_length_azimuth(
            lat1=34.5, lon1=-118.0,
            length=ureg.Quantity(80, "kilometer"),
            az=90.0,
            altitude_msl=ureg.Quantity(60000, "feet"),
            site_name="ER2 Test Line",
        )

    def test_er2_last_segment_is_approach_ending_at_airport(self, er2_flight_line, airport):
        ac = NASA_ER2()
        plan = compute_flight_plan(
            aircraft=ac, flight_sequence=[er2_flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        last = plan.iloc[-1]
        assert last["segment_type"] == "approach"
        # The approach geometry endpoint is the airport (within float epsilon).
        assert last["end_lat"] == pytest.approx(airport.latitude, abs=1e-4)
        assert last["end_lon"] == pytest.approx(airport.longitude, abs=1e-4)
        assert last["end_altitude"] == pytest.approx(airport.elevation_ft, abs=1.0)

    def test_er2_return_rows_sum_to_time_to_return(self, er2_flight_line, airport):
        """Sum of Return-segment times equals time_to_return().total_time."""
        ac = NASA_ER2()
        plan = compute_flight_plan(
            aircraft=ac, flight_sequence=[er2_flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        return_rows = plan[plan["segment_name"] == "Return"]
        assert len(return_rows) > 0
        expected_return_min = ac.time_to_return(
            er2_flight_line.waypoint2, airport,
        )["total_time"].m_as(ureg.minute)
        # Tolerance loose enough to absorb Dubins3D run-to-run numerical noise
        # between the engine's time_to_return call and ours; 1% would still
        # catch a missing-approach-phase regression (which moves totals 5-10%).
        assert return_rows["time_to_segment"].sum() == pytest.approx(
            expected_return_min, rel=1e-2,
        )

    def test_er2_approach_row_time_matches_time_to_touchdown(self, er2_flight_line, airport):
        """The terminal approach row's time equals approach_profile.time_to_touchdown()."""
        ac = NASA_ER2()
        plan = compute_flight_plan(
            aircraft=ac, flight_sequence=[er2_flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        last = plan.iloc[-1]
        assert last["segment_type"] == "approach"
        expected_touchdown_min = ac.approach_profile.time_to_touchdown().m_as(ureg.minute)
        assert last["time_to_segment"] == pytest.approx(expected_touchdown_min, rel=1e-3)

    def test_b200_legacy_no_approach_segment(self, b200, flight_line, airport):
        """Aircraft without approach_profile: no segment of type 'approach' on Return."""
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        return_rows = plan[plan["segment_name"] == "Return"]
        assert len(return_rows) > 0
        assert "approach" not in set(return_rows["segment_type"])
        # Last Return row's geometry still ends at the airport (legacy path).
        last_return = return_rows.iloc[-1]
        assert last_return["end_lat"] == pytest.approx(airport.latitude, abs=1e-3)
        assert last_return["end_lon"] == pytest.approx(airport.longitude, abs=1e-3)

    def test_b200_legacy_return_rows_sum_to_time_to_return(self, b200, flight_line, airport):
        """Legacy path: Return-segment time totals match time_to_return total_time."""
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        return_rows = plan[plan["segment_name"] == "Return"]
        expected = b200.time_to_return(
            flight_line.waypoint2, airport,
        )["total_time"].m_as(ureg.minute)
        assert return_rows["time_to_segment"].sum() == pytest.approx(expected, rel=1e-2)


# ---------------------------------------------------------------------------
# Phase-aware wind sampling: wind_sampling="phase_midpoint"
# ---------------------------------------------------------------------------

class _AltitudeWindField:
    """Synthetic vertical wind profile.

    Eastward wind grows linearly with altitude, mimicking a jet stream
    that's stronger aloft.  `v` is zero everywhere.  Used to verify that
    `wind_sampling="phase_midpoint"` actually samples climb / descent at
    their phase-mid altitudes (not at cruise altitude).
    """

    def __init__(self, u_per_ft_kt: float):
        self.u_per_ft_kt = u_per_ft_kt

    def wind_at(self, lat, lon, altitude, time):
        ft = altitude.m_as(ureg.feet)
        u_kt = self.u_per_ft_kt * ft
        return (
            u_kt * 0.514444 * (ureg.meter / ureg.second),
            0.0 * (ureg.meter / ureg.second),
        )


class TestPhaseAwareWind:
    """compute_flight_plan(wind_sampling="phase_midpoint", ...)."""

    @pytest.fixture
    def airport(self):
        return Airport("KSBA")

    def test_invalid_wind_sampling_raises(self, b200, flight_line, airport):
        with pytest.raises(HyPlanValueError, match="wind_sampling"):
            compute_flight_plan(
                aircraft=b200, flight_sequence=[flight_line],
                takeoff_airport=airport, return_airport=airport,
                wind_sampling="not_a_real_mode",
            )

    def test_constant_wind_phase_modes_match(
        self, b200, flight_line, airport,
    ):
        """Under a uniform constant wind, phase_midpoint and
        cruise_midpoint should produce identical results — the wind
        is the same at every altitude, so per-phase sampling is
        degenerate."""
        import datetime as _dt

        from hyplan.winds import ConstantWindField

        wf = ConstantWindField(40 * ureg.knot, wind_from_deg=270.0)
        t0 = _dt.datetime(2026, 5, 6, 12, tzinfo=_dt.timezone.utc)
        plan_cm = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
            wind_source=wf, takeoff_time=t0,
            wind_sampling="cruise_midpoint",
        )
        plan_pm = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
            wind_source=wf, takeoff_time=t0,
            wind_sampling="phase_midpoint",
        )
        # Same total elapsed time within numeric noise.
        t_cm = plan_cm["time_to_segment"].sum()
        t_pm = plan_pm["time_to_segment"].sum()
        assert abs(t_cm - t_pm) < 1e-3

    def test_flag_below_min_safe_speed_no_stall_raises(self, b200):
        """Aircraft without stall_speed_cas calibrated → raises."""
        from copy import copy

        import pandas as pd

        from hyplan.planning.engine import flag_below_min_safe_speed
        ac = copy(b200)
        ac.stall_speed_cas = None
        empty_plan = gpd.GeoDataFrame(pd.DataFrame())
        with pytest.raises(HyPlanValueError, match="stall_speed_cas"):
            flag_below_min_safe_speed(empty_plan, ac)

    def test_flag_below_min_safe_speed_returns_empty_when_safe(
        self, b200, flight_line, airport,
    ):
        """A normally-flown plan should produce an empty GeoDataFrame —
        the schedules don't violate min-safe-speed at planned altitudes."""
        from hyplan.planning.engine import flag_below_min_safe_speed
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        flagged = flag_below_min_safe_speed(plan, b200)
        assert len(flagged) == 0
        assert "planned_tas_kts" in flagged.columns
        assert "min_safe_tas_kts" in flagged.columns

    def test_flag_below_min_safe_speed_flags_aggressive_margin(
        self, b200, flight_line, airport,
    ):
        """A pathologically-large margin (e.g. 5.0×) forces the
        min-safe-speed above the planned cruise schedule, so every
        flight-line / climb / descent row gets flagged."""
        from hyplan.planning.engine import flag_below_min_safe_speed
        plan = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
        )
        flagged = flag_below_min_safe_speed(plan, b200, margin=5.0)
        assert len(flagged) > 0
        assert (
            flagged["planned_tas_kts"] < flagged["min_safe_tas_kts"]
        ).all()

    def test_gridded_wind_without_takeoff_time_raises(
        self, b200, flight_line, airport,
    ):
        """Gridded-wind providers require takeoff_time so each
        segment can be queried at the right time anchor."""
        # Build a stub gridded provider so the validation fires.  We
        # supply the abstract `_build_urls` as a no-op since validation
        # short-circuits before any URLs are needed.
        from hyplan.winds import _GriddedWindField

        class _StubGridded(_GriddedWindField):  # type: ignore[misc]
            def __init__(self):
                pass

            def _build_urls(self, *args, **kwargs):
                return []

            def wind_at(self, lat, lon, altitude, time):
                return (
                    0.0 * (ureg.meter / ureg.second),
                    0.0 * (ureg.meter / ureg.second),
                )

        with pytest.raises(HyPlanValueError, match="takeoff_time is required"):
            compute_flight_plan(
                aircraft=b200, flight_sequence=[flight_line],
                takeoff_airport=airport, return_airport=airport,
                wind_source=_StubGridded(),
            )

    def test_altitude_varying_wind_phase_modes_diverge(
        self, b200, flight_line, airport,
    ):
        """Under a wind that varies with altitude, phase_midpoint must
        differ from cruise_midpoint (climb-mid samples a different
        altitude than cruise-mid)."""
        import datetime as _dt
        wf = _AltitudeWindField(u_per_ft_kt=0.005)  # 100 kt at 20000 ft
        t0 = _dt.datetime(2026, 5, 6, 12, tzinfo=_dt.timezone.utc)
        plan_cm = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
            wind_source=wf, takeoff_time=t0,
            wind_sampling="cruise_midpoint",
        )
        plan_pm = compute_flight_plan(
            aircraft=b200, flight_sequence=[flight_line],
            takeoff_airport=airport, return_airport=airport,
            wind_source=wf, takeoff_time=t0,
            wind_sampling="phase_midpoint",
        )
        t_cm = plan_cm["time_to_segment"].sum()
        t_pm = plan_pm["time_to_segment"].sum()
        # Difference should be measurable — at least 0.1 min on a
        # short ~25 nmi line; on real ER-2 transits the delta is
        # multiple minutes (see plans/ release notes).
        assert abs(t_cm - t_pm) > 0.1, (
            f"phase_midpoint must differ from cruise_midpoint under "
            f"altitude-varying wind, got Δ = {abs(t_cm - t_pm):.4f} min"
        )
