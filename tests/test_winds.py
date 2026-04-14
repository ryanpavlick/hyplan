"""Tests for hyplan.winds and wind-source integration in flight_plan."""

import datetime
from unittest.mock import patch

import numpy as np
import pytest

from hyplan.units import ureg
from hyplan.winds import (
    ConstantWindField,
    StillAirField,
    _merra2_stream,
    _merra2_url,
    _gfs_filter_url,
    _gfs_best_cycle,
)
from hyplan.flight_plan import (
    _wind_factor,
    _wind_factor_from_uv,
    _resolve_wind_factor,
    compute_flight_plan,
)
from hyplan.waypoint import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.aircraft import DynamicAviation_B200 as B200


# ---------------------------------------------------------------------------
# ConstantWindField
# ---------------------------------------------------------------------------

class TestConstantWindField:
    """Test the ConstantWindField backward-compatibility wrapper."""

    def test_north_wind_gives_negative_v(self):
        """Wind FROM the north → blows southward → V is negative."""
        wf = ConstantWindField(
            wind_speed=10 * ureg.knot, wind_from_deg=0.0,
        )
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        assert abs(u.m_as("m/s")) < 0.01
        assert v.m_as("m/s") < 0  # southward

    def test_east_wind_gives_negative_u(self):
        """Wind FROM the east → blows westward → U is negative."""
        wf = ConstantWindField(
            wind_speed=10 * ureg.knot, wind_from_deg=90.0,
        )
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        assert u.m_as("m/s") < 0  # westward
        assert abs(v.m_as("m/s")) < 0.01

    def test_south_wind_gives_positive_v(self):
        """Wind FROM the south → blows northward → V is positive."""
        wf = ConstantWindField(
            wind_speed=10 * ureg.knot, wind_from_deg=180.0,
        )
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        assert abs(u.m_as("m/s")) < 0.01
        assert v.m_as("m/s") > 0

    def test_west_wind_gives_positive_u(self):
        """Wind FROM the west → blows eastward → U is positive."""
        wf = ConstantWindField(
            wind_speed=20 * ureg.knot, wind_from_deg=270.0,
        )
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        assert u.m_as("m/s") > 0
        assert abs(v.m_as("m/s")) < 0.01

    def test_ignores_position_and_time(self):
        """ConstantWindField returns the same value regardless of inputs."""
        wf = ConstantWindField(
            wind_speed=30 * ureg.knot, wind_from_deg=45.0,
        )
        u1, v1 = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime(2024, 1, 1))
        u2, v2 = wf.wind_at(50, 100, 40000 * ureg.feet, datetime.datetime(2025, 6, 15))
        assert u1.m_as("m/s") == pytest.approx(u2.m_as("m/s"))
        assert v1.m_as("m/s") == pytest.approx(v2.m_as("m/s"))

    def test_zero_wind(self):
        wf = ConstantWindField(
            wind_speed=0 * ureg.knot, wind_from_deg=0.0,
        )
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        assert u.m_as("m/s") == pytest.approx(0)
        assert v.m_as("m/s") == pytest.approx(0)

    def test_magnitude_preserved(self):
        """U^2 + V^2 should equal wind_speed^2."""
        ws = 50 * ureg.knot
        for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
            wf = ConstantWindField(wind_speed=ws, wind_from_deg=direction)
            u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
            mag = np.sqrt(u.m_as("m/s") ** 2 + v.m_as("m/s") ** 2)
            assert mag == pytest.approx(ws.m_as("m/s"), rel=1e-6), (
                f"Failed at direction={direction}"
            )


# ---------------------------------------------------------------------------
# _wind_factor_from_uv
# ---------------------------------------------------------------------------

class TestWindFactorFromUV:
    """Verify _wind_factor_from_uv matches _wind_factor for equivalent inputs."""

    def _compare(self, wind_speed_kt, wind_from_deg, heading_deg, tas_kt=250):
        """Compare UV-based factor against legacy scalar factor."""
        tas = tas_kt * ureg.knot
        ws = wind_speed_kt * ureg.knot

        # Legacy scalar
        legacy = _wind_factor(tas, heading_deg, ws, wind_from_deg)

        # UV-based
        wf = ConstantWindField(wind_speed=ws, wind_from_deg=wind_from_deg)
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        uv_factor = _wind_factor_from_uv(tas, heading_deg, u, v)

        assert uv_factor == pytest.approx(legacy, rel=1e-6), (
            f"Mismatch: ws={wind_speed_kt}, from={wind_from_deg}, hdg={heading_deg}"
        )

    def test_headwind_north(self):
        self._compare(30, 0, 0)  # heading north, wind from north

    def test_tailwind_north(self):
        self._compare(30, 180, 0)  # heading north, wind from south

    def test_crosswind(self):
        self._compare(30, 90, 0)  # heading north, wind from east

    def test_diagonal(self):
        self._compare(50, 45, 90)

    def test_various_combinations(self):
        for ws in [10, 30, 50]:
            for wdir in [0, 45, 90, 135, 180, 225, 270, 315]:
                for hdg in [0, 45, 90, 180, 270]:
                    self._compare(ws, wdir, hdg)

    def test_no_heading_returns_1(self):
        u = 10 * ureg.meter / ureg.second
        v = 5 * ureg.meter / ureg.second
        assert _wind_factor_from_uv(250 * ureg.knot, None, u, v) == 1.0

    def test_unflyable_raises(self):
        # 1000 kt headwind on a 250 kt aircraft
        wf = ConstantWindField(wind_speed=1000 * ureg.knot, wind_from_deg=0.0)
        u, v = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime.now())
        with pytest.raises(Exception):
            _wind_factor_from_uv(250 * ureg.knot, 0.0, u, v)


# ---------------------------------------------------------------------------
# MERRA-2 URL construction
# ---------------------------------------------------------------------------

class TestMERRA2URL:
    def test_stream_100(self):
        assert _merra2_stream(1985) == 100

    def test_stream_200(self):
        assert _merra2_stream(1995) == 200

    def test_stream_300(self):
        assert _merra2_stream(2005) == 300

    def test_stream_400(self):
        assert _merra2_stream(2020) == 400

    def test_url_format(self):
        url = _merra2_url(datetime.date(2024, 6, 15))
        assert "MERRA2_400" in url
        assert "20240615" in url
        assert "2024/06" in url
        assert "goldsmr5.gesdisc.eosdis.nasa.gov" in url
        assert "M2I3NPASM" in url


# ---------------------------------------------------------------------------
# compute_flight_plan with ConstantWindField
# ---------------------------------------------------------------------------

@pytest.fixture
def b200():
    return B200()


class TestFlightPlanWithWindSource:
    """Verify wind_source integration produces same results as legacy params."""

    def _north_leg(self):
        wp1 = Waypoint(34.00, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1")
        wp2 = Waypoint(34.50, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        return wp1, wp2

    def test_constant_wind_matches_legacy_headwind(self, b200):
        """ConstantWindField should produce identical results to scalar params."""
        wp1, wp2 = self._north_leg()

        plan_legacy = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=50 * ureg.knot,
            wind_direction=0.0,  # from north → headwind on northbound
        )

        wf = ConstantWindField(wind_speed=50 * ureg.knot, wind_from_deg=0.0)
        plan_source = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_source=wf,
        )

        assert plan_source["time_to_segment"].sum() == pytest.approx(
            plan_legacy["time_to_segment"].sum(), rel=1e-6
        )

    def test_constant_wind_matches_legacy_tailwind(self, b200):
        wp1, wp2 = self._north_leg()

        plan_legacy = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=50 * ureg.knot,
            wind_direction=180.0,
        )

        wf = ConstantWindField(wind_speed=50 * ureg.knot, wind_from_deg=180.0)
        plan_source = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_source=wf,
        )

        assert plan_source["time_to_segment"].sum() == pytest.approx(
            plan_legacy["time_to_segment"].sum(), rel=1e-6
        )

    def test_constant_wind_matches_legacy_crosswind(self, b200):
        wp1, wp2 = self._north_leg()

        plan_legacy = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_speed=30 * ureg.knot,
            wind_direction=90.0,
        )

        wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=90.0)
        plan_source = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_source=wf,
        )

        assert plan_source["time_to_segment"].sum() == pytest.approx(
            plan_legacy["time_to_segment"].sum(), rel=1e-6
        )

    def test_constant_wind_no_takeoff_time_ok(self, b200):
        """ConstantWindField should work without takeoff_time."""
        wp1, wp2 = self._north_leg()
        wf = ConstantWindField(wind_speed=20 * ureg.knot, wind_from_deg=0.0)
        plan = compute_flight_plan(
            aircraft=b200,
            flight_sequence=[wp1, wp2],
            wind_source=wf,
        )
        assert len(plan) > 0

    def test_cannot_combine_wind_source_and_wind_speed(self, b200):
        wp1, wp2 = self._north_leg()
        wf = ConstantWindField(wind_speed=20 * ureg.knot, wind_from_deg=0.0)
        with pytest.raises(Exception):
            compute_flight_plan(
                aircraft=b200,
                flight_sequence=[wp1, wp2],
                wind_source=wf,
                wind_speed=10 * ureg.knot,
                wind_direction=0.0,
            )

    def test_headwind_flight_line(self, b200):
        """Wind source should affect flight line timing the same as legacy."""
        fl = FlightLine.start_length_azimuth(
            lat1=34.05, lon1=-118.25,
            length=ureg.Quantity(50000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(20000, "feet"),
            site_name="North Line",
        )

        plan_legacy = compute_flight_plan(
            aircraft=b200, flight_sequence=[fl],
            wind_speed=30 * ureg.knot, wind_direction=0.0,
        )
        wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=0.0)
        plan_source = compute_flight_plan(
            aircraft=b200, flight_sequence=[fl],
            wind_source=wf,
        )

        row_legacy = plan_legacy[plan_legacy["segment_type"] == "flight_line"].iloc[0]
        row_source = plan_source[plan_source["segment_type"] == "flight_line"].iloc[0]
        assert row_source["time_to_segment"] == pytest.approx(
            row_legacy["time_to_segment"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# _resolve_wind_factor
# ---------------------------------------------------------------------------

class TestResolveWindFactor:
    def test_no_wind_returns_1(self):
        f = _resolve_wind_factor(
            250 * ureg.knot, 0.0,
            34.0, -118.0, 20000 * ureg.feet, None,
            None, None, None,
        )
        assert f == 1.0

    def test_legacy_scalar(self):
        f = _resolve_wind_factor(
            250 * ureg.knot, 0.0,
            34.0, -118.0, 20000 * ureg.feet, None,
            None, 30 * ureg.knot, 0.0,
        )
        expected = _wind_factor(250 * ureg.knot, 0.0, 30 * ureg.knot, 0.0)
        assert f == pytest.approx(expected)

    def test_wind_source(self):
        wf = ConstantWindField(wind_speed=30 * ureg.knot, wind_from_deg=0.0)
        f = _resolve_wind_factor(
            250 * ureg.knot, 0.0,
            34.0, -118.0, 20000 * ureg.feet, datetime.datetime.now(),
            wf, None, None,
        )
        expected = _wind_factor(250 * ureg.knot, 0.0, 30 * ureg.knot, 0.0)
        assert f == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# StillAirField
# ---------------------------------------------------------------------------

class TestStillAirField:
    def test_returns_zero_wind(self):
        wf = StillAirField()
        u, v = wf.wind_at(34.0, -118.0, 20000 * ureg.feet, datetime.datetime.now())
        assert u.m_as("m/s") == 0.0
        assert v.m_as("m/s") == 0.0

    def test_ignores_position_and_time(self):
        wf = StillAirField()
        u1, v1 = wf.wind_at(0, 0, 0 * ureg.feet, datetime.datetime(2020, 1, 1))
        u2, v2 = wf.wind_at(60, 100, 40000 * ureg.feet, datetime.datetime(2025, 6, 15))
        assert u1.m_as("m/s") == u2.m_as("m/s") == 0.0
        assert v1.m_as("m/s") == v2.m_as("m/s") == 0.0

    def test_still_air_wind_factor_is_1(self):
        wf = StillAirField()
        f = _resolve_wind_factor(
            250 * ureg.knot, 0.0,
            34.0, -118.0, 20000 * ureg.feet, datetime.datetime.now(),
            wf, None, None,
        )
        assert f == pytest.approx(1.0)

    def test_flight_plan_still_air_matches_no_wind(self, b200):
        wp1 = Waypoint(34.00, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP1")
        wp2 = Waypoint(34.50, -118.00, 0.0,
                       altitude_msl=ureg.Quantity(20000, "feet"), name="WP2")
        plan_none = compute_flight_plan(aircraft=b200, flight_sequence=[wp1, wp2])
        plan_still = compute_flight_plan(
            aircraft=b200, flight_sequence=[wp1, wp2],
            wind_source=StillAirField(),
        )
        assert plan_still["time_to_segment"].sum() == pytest.approx(
            plan_none["time_to_segment"].sum(), rel=1e-6
        )


# ---------------------------------------------------------------------------
# GFS S3 key construction and index parsing
# ---------------------------------------------------------------------------

class TestGFSFilter:
    def test_filter_url_format(self):
        url = _gfs_filter_url(
            datetime.date(2024, 6, 15), cycle=12, fhr=6,
            variables=("UGRD", "VGRD"), levels_hpa=[300, 250],
            lat_min=33, lat_max=35, lon_min=-120, lon_max=-117,
        )
        assert "filter_gfs_0p25.pl" in url
        assert "gfs.20240615" in url
        assert "f006" in url
        assert "var_UGRD=on" in url
        assert "var_VGRD=on" in url
        assert "lev_300_mb=on" in url
        assert "toplat=35" in url

    def test_best_cycle(self):
        # _gfs_best_cycle uses datetime.now(), so we just verify it returns
        # a valid (date, cycle) pair
        dt = datetime.datetime(2024, 6, 15, 18, 0)
        date, cycle = _gfs_best_cycle(dt)
        assert cycle in (0, 6, 12, 18)
        assert isinstance(date, datetime.date)


# ---------------------------------------------------------------------------
# Wind vector conversions (utils)
# ---------------------------------------------------------------------------

from hyplan.winds.utils import wind_uv_from_speed_dir, wind_speed_dir_from_uv


class TestWindVectorConversions:
    """Test meteorological wind <-> U/V component conversions."""

    def test_from_north(self):
        u, v = wind_uv_from_speed_dir(10.0, 0.0)
        assert u == pytest.approx(0.0, abs=1e-10)
        assert v == pytest.approx(-10.0)

    def test_from_east(self):
        u, v = wind_uv_from_speed_dir(10.0, 90.0)
        assert u == pytest.approx(-10.0)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_from_south(self):
        u, v = wind_uv_from_speed_dir(10.0, 180.0)
        assert u == pytest.approx(0.0, abs=1e-10)
        assert v == pytest.approx(10.0)

    def test_from_west(self):
        u, v = wind_uv_from_speed_dir(10.0, 270.0)
        assert u == pytest.approx(10.0)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_zero_wind(self):
        u, v = wind_uv_from_speed_dir(0.0, 123.0)
        assert u == pytest.approx(0.0, abs=1e-10)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_roundtrip(self):
        """speed_dir -> uv -> speed_dir should be identity."""
        for speed in [5.0, 15.0, 50.0]:
            for from_deg in [0, 30, 45, 90, 135, 180, 225, 270, 315]:
                u, v = wind_uv_from_speed_dir(speed, from_deg)
                s2, d2 = wind_speed_dir_from_uv(u, v)
                assert s2 == pytest.approx(speed, rel=1e-10)
                assert d2 == pytest.approx(from_deg, abs=1e-6)

    def test_inverse_zero_wind(self):
        speed, from_deg = wind_speed_dir_from_uv(0.0, 0.0)
        assert speed == pytest.approx(0.0)
        assert from_deg == pytest.approx(0.0)

    def test_inverse_cardinal_directions(self):
        # Positive u (eastward) = wind from west (270)
        speed, from_deg = wind_speed_dir_from_uv(10.0, 0.0)
        assert speed == pytest.approx(10.0)
        assert from_deg == pytest.approx(270.0)

        # Positive v (northward) = wind from south (180)
        speed, from_deg = wind_speed_dir_from_uv(0.0, 10.0)
        assert speed == pytest.approx(10.0)
        assert from_deg == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# _interp_weights
# ---------------------------------------------------------------------------

from hyplan.winds.gridded import _GriddedWindField


class TestInterpWeights:
    """Test the linear interpolation weight computation."""

    def test_single_element(self):
        coords = np.array([500.0])
        result = _GriddedWindField._interp_weights(coords, 500.0)
        assert result == [(0, 1.0)]

    def test_single_element_any_value(self):
        coords = np.array([500.0])
        result = _GriddedWindField._interp_weights(coords, 999.0)
        assert result == [(0, 1.0)]

    def test_below_range_clamps(self):
        coords = np.array([100.0, 200.0, 300.0])
        result = _GriddedWindField._interp_weights(coords, 50.0)
        assert result == [(0, 1.0)]

    def test_above_range_clamps(self):
        coords = np.array([100.0, 200.0, 300.0])
        result = _GriddedWindField._interp_weights(coords, 400.0)
        assert result == [(2, 1.0)]

    def test_exact_grid_point(self):
        coords = np.array([100.0, 200.0, 300.0])
        result = _GriddedWindField._interp_weights(coords, 100.0)
        assert result == [(0, 1.0)]

    def test_exact_last_grid_point(self):
        coords = np.array([100.0, 200.0, 300.0])
        result = _GriddedWindField._interp_weights(coords, 300.0)
        assert result == [(2, 1.0)]

    def test_midpoint(self):
        coords = np.array([100.0, 200.0, 300.0])
        result = _GriddedWindField._interp_weights(coords, 150.0)
        assert len(result) == 2
        assert result[0] == (0, pytest.approx(0.5))
        assert result[1] == (1, pytest.approx(0.5))

    def test_weights_sum_to_one(self):
        coords = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        for val in [15.0, 22.0, 37.5, 48.0]:
            result = _GriddedWindField._interp_weights(coords, val)
            total = sum(w for _, w in result)
            assert total == pytest.approx(1.0)

    def test_quarter_point(self):
        coords = np.array([0.0, 100.0])
        result = _GriddedWindField._interp_weights(coords, 25.0)
        assert result[0] == (0, pytest.approx(0.75))
        assert result[1] == (1, pytest.approx(0.25))


# ---------------------------------------------------------------------------
# _index_range
# ---------------------------------------------------------------------------

class TestIndexRange:

    def test_ascending_interior(self):
        coords = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sl = _GriddedWindField._index_range(coords, 25.0, 35.0)
        # Should include margin: indices 1..3 (values 20..40)
        assert sl.start <= 1
        assert sl.stop >= 4  # slice stop is exclusive in isel

    def test_descending_coords(self):
        coords = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        sl = _GriddedWindField._index_range(coords, 25.0, 35.0)
        selected = coords[sl]
        assert np.min(selected) <= 25.0
        assert np.max(selected) >= 35.0


# ---------------------------------------------------------------------------
# Synthetic gridded interpolation
# ---------------------------------------------------------------------------

class _SyntheticGriddedWind(_GriddedWindField):
    """Minimal concrete subclass with synthetic data for testing."""

    def __init__(self, u_data, v_data, times, levs, lats, lons):
        # Skip the parent __init__ which calls _fetch_slab
        # Instead, set the arrays directly
        self._u_data = u_data
        self._v_data = v_data
        self._times = times
        self._levs = levs
        self._lats = lats
        self._lons = lons

    def _build_urls(self):
        return []


class TestGriddedInterpolation:
    """Test 4D interpolation with synthetic data."""

    def _make_field(self):
        """Create a field where U = lat and V = lon (linear gradients)."""
        lats = np.array([30.0, 32.0, 34.0, 36.0])
        lons = np.array([-120.0, -118.0, -116.0])
        levs = np.array([500.0, 700.0, 850.0])
        # Single timestep, epoch seconds for 2024-01-01
        t = np.datetime64("2024-01-01T00:00:00")
        times = np.array([
            (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        ], dtype=float)

        # Shape: (1, 3, 4, 3) = (time, lev, lat, lon)
        u_data = np.zeros((1, 3, 4, 3))
        v_data = np.zeros((1, 3, 4, 3))
        for ila, la in enumerate(lats):
            for ilo, lo in enumerate(lons):
                u_data[0, :, ila, ilo] = la   # U = latitude
                v_data[0, :, ila, ilo] = lo   # V = longitude

        return _SyntheticGriddedWind(u_data, v_data, times, levs, lats, lons)

    def test_exact_grid_point(self):
        wf = self._make_field()
        u, v = wf.wind_at(
            32.0, -118.0,
            ureg.Quantity(18000, "feet"),  # ~500 hPa
            datetime.datetime(2024, 1, 1),
        )
        assert u.m_as("m/s") == pytest.approx(32.0, abs=0.5)
        assert v.m_as("m/s") == pytest.approx(-118.0, abs=0.5)

    def test_interpolated_point(self):
        wf = self._make_field()
        u, v = wf.wind_at(
            33.0, -119.0,
            ureg.Quantity(18000, "feet"),
            datetime.datetime(2024, 1, 1),
        )
        # U should be ~33 (midpoint between 32 and 34)
        assert u.m_as("m/s") == pytest.approx(33.0, abs=0.5)
        # V should be ~-119 (midpoint between -120 and -118)
        assert v.m_as("m/s") == pytest.approx(-119.0, abs=0.5)

    def test_boundary_clamping(self):
        """Points outside the grid should clamp to boundary values."""
        wf = self._make_field()
        u, v = wf.wind_at(
            28.0, -122.0,  # below/left of grid
            ureg.Quantity(18000, "feet"),
            datetime.datetime(2024, 1, 1),
        )
        # Should clamp to nearest grid corner: lat=30, lon=-120
        assert u.m_as("m/s") == pytest.approx(30.0, abs=0.5)
        assert v.m_as("m/s") == pytest.approx(-120.0, abs=0.5)
