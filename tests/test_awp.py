"""Tests for the AWP profiling lidar model and planning helpers."""

import datetime as dt

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from hyplan import (
    FlightLine,
    awp_profile_locations_for_flight_line,
    awp_profile_locations_for_plan,
    create_sensor,
    flag_awp_stable_segments,
)
from hyplan.instruments import AerosolWindProfiler, ProfilingLidar
from hyplan.units import ureg


class TestAerosolWindProfiler:
    def test_instantiation(self):
        awp = AerosolWindProfiler()
        assert isinstance(awp, ProfilingLidar)
        assert awp.name == "Aerosol Wind Profiler"
        assert awp.off_nadir_angle == pytest.approx(30.0)

    def test_factory_registration(self):
        sensor = create_sensor("AWP")
        assert isinstance(sensor, AerosolWindProfiler)

    def test_vertical_bin_spacing_matches_paper_scale(self):
        awp = AerosolWindProfiler()
        assert awp.vertical_bin_spacing(256).m_as("meter") == pytest.approx(66.5, rel=0.03)
        assert awp.vertical_bin_spacing(512).m_as("meter") == pytest.approx(133.0, rel=0.03)

    def test_max_vertical_range_matches_paper_scale(self):
        awp = AerosolWindProfiler()
        assert awp.max_vertical_range().m_as("kilometer") == pytest.approx(17.0, rel=0.03)

    def test_los_surface_separation_matches_paper_example(self):
        awp = AerosolWindProfiler()
        separation = awp.los_surface_separation(12 * ureg.kilometer)
        assert separation.m_as("kilometer") == pytest.approx(9.8, rel=0.03)

    def test_vector_profile_spacing_matches_paper_example(self):
        awp = AerosolWindProfiler()
        spacing = awp.vector_profile_spacing(225 * ureg.meter / ureg.second)
        assert spacing.m_as("meter") == pytest.approx(1575.0, rel=1e-6)

    def test_first_profile_assignment_offset(self):
        awp = AerosolWindProfiler()
        offset = awp.profile_assignment_offset(225 * ureg.meter / ureg.second)
        assert offset.m_as("meter") == pytest.approx(900.0, rel=1e-6)

    def test_ground_intercepts_orient_with_track(self):
        awp = AerosolWindProfiler()
        intercepts = awp.los_ground_intercepts(34.0, -118.0, 0.0, 12 * ureg.kilometer)
        assert intercepts["los1"]["azimuth"] == pytest.approx(315.0)
        assert intercepts["los2"]["azimuth"] == pytest.approx(45.0)


class TestAwpHelpers:
    @pytest.fixture
    def flight_line(self):
        return FlightLine.start_length_azimuth(
            lat1=34.0,
            lon1=-118.0,
            length=50 * ureg.kilometer,
            az=0.0,
            altitude_msl=12000 * ureg.meter,
            site_name="North Leg",
        )

    def test_profile_locations_for_flight_line(self, flight_line):
        gdf = awp_profile_locations_for_flight_line(
            flight_line,
            ground_speed=225 * ureg.meter / ureg.second,
            start_time=dt.datetime(2025, 1, 1, 12, 0, 0),
            altitude_agl=12 * ureg.kilometer,
        )
        assert not gdf.empty
        assert gdf.iloc[0]["source_segment_name"] == "North Leg"
        assert gdf.iloc[0]["distance_from_start_m"] == pytest.approx(900.0, rel=1e-6)
        assert gdf.iloc[1]["distance_from_start_m"] - gdf.iloc[0]["distance_from_start_m"] == pytest.approx(1575.0, rel=1e-6)
        assert gdf.iloc[0]["time_utc"] == dt.datetime(2025, 1, 1, 12, 0, 4)
        assert gdf.iloc[0]["los_surface_separation_m"] == pytest.approx(9800.0, rel=0.03)

    def test_profile_locations_for_short_leg_return_empty(self):
        short_line = FlightLine.start_length_azimuth(
            lat1=34.0,
            lon1=-118.0,
            length=500 * ureg.meter,
            az=90.0,
            altitude_msl=12000 * ureg.meter,
            site_name="Short",
        )
        gdf = awp_profile_locations_for_flight_line(short_line, altitude_agl=12 * ureg.kilometer)
        assert gdf.empty

    def test_flag_awp_stable_segments(self):
        plan = gpd.GeoDataFrame(
            [
                {
                    "geometry": LineString([(-118.0, 34.0), (-118.0, 34.1)]),
                    "segment_type": "flight_line",
                    "segment_name": "Stable",
                    "start_altitude": 39000.0,
                    "end_altitude": 39000.0,
                    "time_to_segment": 5.0,
                    "distance": 20.0,
                    "groundspeed_kts": 430.0,
                },
                {
                    "geometry": LineString([(-118.0, 34.0), (-117.95, 34.05), (-117.9, 34.05)]),
                    "segment_type": "transit",
                    "segment_name": "Turning",
                    "start_altitude": 39000.0,
                    "end_altitude": 39000.0,
                    "time_to_segment": 5.0,
                    "distance": 20.0,
                    "groundspeed_kts": 430.0,
                },
                {
                    "geometry": LineString([(-118.0, 34.0), (-118.0, 34.1)]),
                    "segment_type": "flight_line",
                    "segment_name": "Climbing",
                    "start_altitude": 39000.0,
                    "end_altitude": 42000.0,
                    "time_to_segment": 5.0,
                    "distance": 20.0,
                    "groundspeed_kts": 430.0,
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        flagged = flag_awp_stable_segments(plan)
        assert bool(flagged.iloc[0]["awp_stable_platform_ok"]) is True
        assert bool(flagged.iloc[1]["awp_stable_platform_ok"]) is False
        assert bool(flagged.iloc[2]["awp_stable_platform_ok"]) is False

    def test_profile_locations_for_plan_filters_unstable_segments(self):
        plan = gpd.GeoDataFrame(
            [
                {
                    "geometry": LineString([(-118.0, 34.0), (-118.0, 34.25)]),
                    "segment_type": "flight_line",
                    "segment_name": "Stable",
                    "start_altitude": 39000.0,
                    "end_altitude": 39000.0,
                    "time_to_segment": 8.0,
                    "distance": 55.0,
                    "groundspeed_kts": 437.0,
                },
                {
                    "geometry": LineString([(-118.0, 34.0), (-117.95, 34.05), (-117.9, 34.05)]),
                    "segment_type": "transit",
                    "segment_name": "Turning",
                    "start_altitude": 39000.0,
                    "end_altitude": 39000.0,
                    "time_to_segment": 4.0,
                    "distance": 15.0,
                    "groundspeed_kts": 220.0,
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        profiles = awp_profile_locations_for_plan(
            plan,
            takeoff_time=dt.datetime(2025, 1, 1, 12, 0, 0),
        )
        assert not profiles.empty
        assert set(profiles["source_segment_name"]) == {"Stable"}
        assert profiles["stable_platform_ok"].all()
        assert profiles.iloc[0]["time_utc"] >= dt.datetime(2025, 1, 1, 12, 0, 0)
