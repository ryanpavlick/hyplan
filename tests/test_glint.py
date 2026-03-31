"""Tests for hyplan.glint."""

import pytest
import numpy as np
import geopandas as gpd
from datetime import datetime, timezone
from shapely.geometry import LineString

from hyplan.glint import (
    glint_angle,
    calculate_target_and_glint_vectorized,
    compute_glint_vectorized,
    GlintArc,
    compute_glint_arc,
)
from hyplan.sensors import AVIRIS3
from hyplan.units import ureg
from hyplan.flight_line import FlightLine


class TestGlintAngle:
    def test_zero_glint_at_specular(self):
        """When view mirrors the sun across nadir, glint angle should be ~0."""
        # Sun at azimuth=180°, zenith=30°; view at azimuth=0° (opposite), zenith=30°
        result = glint_angle(
            solar_azimuth=np.array([180.0]),
            solar_zenith=np.array([30.0]),
            view_azimuth=np.array([0.0]),
            view_zenith=np.array([30.0]),
        )
        assert result[0] == pytest.approx(0.0, abs=0.1)

    def test_nadir_view_equals_solar_zenith(self):
        """Viewing straight down (zenith=0), glint angle equals solar zenith."""
        for sza in [20.0, 45.0, 70.0]:
            result = glint_angle(
                solar_azimuth=np.array([180.0]),
                solar_zenith=np.array([sza]),
                view_azimuth=np.array([0.0]),
                view_zenith=np.array([0.0]),
            )
            assert result[0] == pytest.approx(sza, abs=0.1)

    def test_vectorized_shape(self):
        n = 100
        result = glint_angle(
            solar_azimuth=np.random.uniform(0, 360, n),
            solar_zenith=np.random.uniform(10, 80, n),
            view_azimuth=np.random.uniform(0, 360, n),
            view_zenith=np.random.uniform(0, 45, n),
        )
        assert result.shape == (n,)
        assert np.all(result >= 0)
        assert np.all(result <= 180)

    def test_symmetric_azimuth(self):
        """Glint angle should be the same for symmetric azimuth offsets."""
        g1 = glint_angle(
            np.array([180.0]), np.array([30.0]),
            np.array([170.0]), np.array([30.0]),
        )
        g2 = glint_angle(
            np.array([180.0]), np.array([30.0]),
            np.array([190.0]), np.array([30.0]),
        )
        assert g1[0] == pytest.approx(g2[0], abs=0.01)


class TestCalculateTargetAndGlint:
    def test_returns_correct_shapes(self):
        n = 5
        lat, lon, glint = calculate_target_and_glint_vectorized(
            sensor_lat=np.full(n, 34.0),
            sensor_lon=np.full(n, -118.0),
            sensor_alt=np.full(n, 6000.0),
            viewing_azimuth=np.linspace(0, 360, n),
            tilt_angle=np.full(n, 20.0),
            observation_datetime=np.full(n, datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc)),
        )
        assert lat.shape == (n,)
        assert lon.shape == (n,)
        assert glint.shape == (n,)

    def test_glint_values_reasonable(self):
        lat, lon, glint = calculate_target_and_glint_vectorized(
            sensor_lat=np.array([34.0]),
            sensor_lon=np.array([-118.0]),
            sensor_alt=np.array([6000.0]),
            viewing_azimuth=np.array([180.0]),
            tilt_angle=np.array([10.0]),
            observation_datetime=np.array([datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc)]),
        )
        assert 0 <= float(glint.item()) <= 180
        assert -90 <= float(lat.item()) <= 90


class TestComputeGlintVectorized:
    def test_returns_geodataframe(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(10000, "meter"),
            az=90.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="Glint Test",
        )
        sensor = AVIRIS3()
        obs_time = datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc)

        gdf = compute_glint_vectorized(fl, sensor, obs_time)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "glint_angle" in gdf.columns
        assert "tilt_angle" in gdf.columns
        assert len(gdf) > 0
        assert gdf["glint_angle"].min() >= 0
        assert gdf["glint_angle"].max() <= 180

    def test_along_track_output(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(5000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="AT Test",
        )
        sensor = AVIRIS3()
        obs_time = datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc)

        gdf = compute_glint_vectorized(fl, sensor, obs_time, output_geometry="along_track")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_invalid_output_geometry(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(5000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="Bad Geom",
        )
        sensor = AVIRIS3()
        obs_time = datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Invalid output_geometry"):
            compute_glint_vectorized(fl, sensor, obs_time, output_geometry="invalid")


# --- GlintArc tests ---

# Gulf of Mexico platform, midday UTC (morning local) — SZA around 30-50°
ARC_TARGET_LAT = 28.0
ARC_TARGET_LON = -90.0
ARC_OBS_TIME = datetime(2025, 6, 15, 17, 0, tzinfo=timezone.utc)
ARC_ALTITUDE = ureg.Quantity(10000, "foot")
ARC_SPEED = ureg.Quantity(145, "knot")


class TestGlintArc:
    def test_creation(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        assert isinstance(arc.geometry, LineString)
        assert arc.bank_angle == arc.solar_zenith
        assert arc.bank_direction == "right"
        assert arc.arc_extent == 180.0
        assert 5.0 < arc.solar_zenith < 60.0

    def test_bank_direction_left(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
            bank_direction="left",
        )
        assert arc.bank_direction == "left"
        assert isinstance(arc.geometry, LineString)

    def test_invalid_bank_direction(self):
        with pytest.raises(ValueError, match="bank_direction"):
            GlintArc(
                target_lat=ARC_TARGET_LAT,
                target_lon=ARC_TARGET_LON,
                observation_datetime=ARC_OBS_TIME,
                altitude_msl=ARC_ALTITUDE,
                speed=ARC_SPEED,
                bank_direction="up",
            )

    def test_turn_radius_positive(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        assert arc.turn_radius.magnitude > 0

    def test_arc_length(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        expected = arc.turn_radius.magnitude * np.pi
        assert arc.length.magnitude == pytest.approx(expected, rel=0.01)

    def test_waypoints(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        wp1 = arc.waypoint1
        wp2 = arc.waypoint2
        assert -90 <= wp1.latitude <= 90
        assert -90 <= wp2.latitude <= 90
        assert wp1.altitude_msl is not None
        assert wp2.altitude_msl is not None

    def test_track_returns_linestring(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        track = arc.track(precision=50.0)
        assert isinstance(track, LineString)
        assert len(track.coords) >= 3

    def test_to_dict(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
            site_name="TestPlatform",
        )
        d = arc.to_dict()
        assert d["site_name"] == "TestPlatform"
        assert d["target_lat"] == ARC_TARGET_LAT
        assert d["bank_angle"] == arc.solar_zenith

    def test_to_geojson(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        gj = arc.to_geojson()
        assert gj["type"] == "Feature"
        assert gj["geometry"]["type"] == "LineString"
        assert len(gj["geometry"]["coordinates"]) > 0

    def test_sun_near_zenith_raises(self):
        """Solar zenith < 5° should raise an error."""
        # Near equator at local noon in June — SZA very small
        with pytest.raises(ValueError, match="sun near zenith"):
            GlintArc(
                target_lat=0.0,
                target_lon=0.0,
                observation_datetime=datetime(2025, 3, 20, 12, 0, tzinfo=timezone.utc),
                altitude_msl=ARC_ALTITUDE,
                speed=ARC_SPEED,
            )

    def test_bank_angle_auto_from_sza(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        assert arc.bank_angle == pytest.approx(arc.solar_zenith)

    def test_custom_bank_angle(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
            bank_angle=45.0,
        )
        assert arc.bank_angle == 45.0
        assert arc.arc_extent == 180.0

    def test_high_sza_raises_without_override(self):
        """SZA > 60° without explicit bank_angle should raise."""
        with pytest.raises(ValueError, match="60°"):
            GlintArc(
                target_lat=60.0,
                target_lon=0.0,
                observation_datetime=datetime(2025, 12, 15, 12, 0, tzinfo=timezone.utc),
                altitude_msl=ARC_ALTITUDE,
                speed=ARC_SPEED,
            )

    def test_high_sza_with_bank_override(self):
        """Explicit bank_angle allows flying when SZA > 60°."""
        arc = GlintArc(
            target_lat=60.0,
            target_lon=0.0,
            observation_datetime=datetime(2025, 12, 15, 12, 0, tzinfo=timezone.utc),
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
            bank_angle=45.0,
        )
        assert arc.bank_angle == 45.0

    def test_invalid_bank_angle(self):
        with pytest.raises(ValueError, match="bank_angle"):
            GlintArc(
                target_lat=ARC_TARGET_LAT,
                target_lon=ARC_TARGET_LON,
                observation_datetime=ARC_OBS_TIME,
                altitude_msl=ARC_ALTITUDE,
                speed=ARC_SPEED,
                bank_angle=95.0,
            )


class TestComputeGlintArc:
    def test_returns_geodataframe(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        sensor = AVIRIS3()
        gdf = compute_glint_arc(arc, sensor)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "glint_angle" in gdf.columns
        assert "tilt_angle" in gdf.columns
        assert "view_zenith" in gdf.columns
        assert len(gdf) > 0
        assert gdf["glint_angle"].min() >= 0
        assert gdf["glint_angle"].max() <= 180

    def test_minimum_glint_near_center(self):
        """The minimum glint angle should occur near the arc midpoint."""
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        sensor = AVIRIS3()
        gdf = compute_glint_arc(arc, sensor)

        # Find the row with minimum glint angle
        min_idx = gdf["glint_angle"].idxmin()
        min_glint = gdf.loc[min_idx, "glint_angle"]

        # Glint angle at best point should be very small (< 5°)
        assert min_glint < 5.0

        # The minimum should be near the center of along_track_distance
        atd = gdf["along_track_distance"]
        mid_atd = (atd.max() + atd.min()) / 2.0
        min_atd = gdf.loc[min_idx, "along_track_distance"]
        # Within 20% of the arc length from center
        assert abs(min_atd - mid_atd) < 0.2 * (atd.max() - atd.min())

    def test_along_track_output(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        sensor = AVIRIS3()
        gdf = compute_glint_arc(arc, sensor, output_geometry="along_track")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_invalid_output_geometry(self):
        arc = GlintArc(
            target_lat=ARC_TARGET_LAT,
            target_lon=ARC_TARGET_LON,
            observation_datetime=ARC_OBS_TIME,
            altitude_msl=ARC_ALTITUDE,
            speed=ARC_SPEED,
        )
        sensor = AVIRIS3()
        with pytest.raises(ValueError, match="Invalid output_geometry"):
            compute_glint_arc(arc, sensor, output_geometry="invalid")


class TestGlintArcCollectionLength:
    def test_limits_arc_extent(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            collection_length=ureg.Quantity(5000, "meter"),
        )
        assert 0.0 < arc.arc_extent < 180.0

    def test_caps_at_180(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            collection_length=ureg.Quantity(500_000, "meter"),
        )
        assert arc.arc_extent == pytest.approx(180.0)

    def test_none_gives_180(self):
        arc = GlintArc(ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED)
        assert arc.arc_extent == pytest.approx(180.0)

    def test_plain_float_collection_length(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            collection_length=5000.0,
        )
        assert 0.0 < arc.arc_extent < 180.0

    def test_in_dict(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            collection_length=ureg.Quantity(5000, "meter"),
        )
        d = arc.to_dict()
        assert "collection_length" in d
        assert d["collection_length"] == pytest.approx(5000.0, rel=0.01)

    def test_none_collection_length_in_dict(self):
        arc = GlintArc(ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED)
        assert arc.to_dict()["collection_length"] is None

    def test_in_geojson(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            collection_length=ureg.Quantity(5000, "meter"),
        )
        props = arc.to_geojson()["properties"]
        assert "collection_length" in props
        assert props["collection_length"] == pytest.approx(5000.0, rel=0.01)


class TestGlintArcApproachExitLines:
    @pytest.fixture
    def arc(self):
        return GlintArc(ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED)

    def test_approach_returns_flight_line(self, arc):
        assert isinstance(arc.approach_line(ureg.Quantity(5000, "meter")), FlightLine)

    def test_approach_ends_at_arc_start(self, arc):
        fl = arc.approach_line(ureg.Quantity(5000, "meter"))
        wp1 = arc.waypoint1
        assert fl.waypoint2.latitude  == pytest.approx(wp1.latitude,  abs=1e-3)
        assert fl.waypoint2.longitude == pytest.approx(wp1.longitude, abs=1e-3)

    def test_approach_length(self, arc):
        fl = arc.approach_line(ureg.Quantity(5000, "meter"))
        assert fl.length.to("meter").magnitude == pytest.approx(5000.0, rel=0.01)

    def test_exit_returns_flight_line(self, arc):
        assert isinstance(arc.exit_line(ureg.Quantity(5000, "meter")), FlightLine)

    def test_exit_starts_at_arc_end(self, arc):
        fl = arc.exit_line(ureg.Quantity(5000, "meter"))
        wp2 = arc.waypoint2
        assert fl.waypoint1.latitude  == pytest.approx(wp2.latitude,  abs=1e-3)
        assert fl.waypoint1.longitude == pytest.approx(wp2.longitude, abs=1e-3)

    def test_exit_length(self, arc):
        fl = arc.exit_line(ureg.Quantity(5000, "meter"))
        assert fl.length.to("meter").magnitude == pytest.approx(5000.0, rel=0.01)

    def test_plain_float_length(self, arc):
        fl = arc.approach_line(3000.0)
        assert fl.length.to("meter").magnitude == pytest.approx(3000.0, rel=0.01)

    def test_approach_altitude_matches_arc(self, arc):
        fl = arc.approach_line(ureg.Quantity(5000, "meter"))
        assert fl.altitude_msl.to("meter").magnitude == pytest.approx(
            arc.altitude_msl.magnitude, rel=0.001
        )

    def test_left_bank_approach(self):
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            bank_direction="left",
        )
        fl = arc.approach_line(ureg.Quantity(5000, "meter"))
        wp1 = arc.waypoint1
        assert fl.waypoint2.latitude  == pytest.approx(wp1.latitude,  abs=1e-3)
        assert fl.waypoint2.longitude == pytest.approx(wp1.longitude, abs=1e-3)


class TestGlintArcFootprint:
    @pytest.fixture
    def arc(self):
        return GlintArc(ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED)

    def test_returns_valid_polygon(self, arc):
        from shapely.geometry import Polygon
        poly = arc.footprint(AVIRIS3())
        assert isinstance(poly, Polygon)
        assert poly.is_valid

    def test_contains_target(self, arc):
        from shapely.geometry import Point
        poly = arc.footprint(AVIRIS3())
        assert poly.contains(Point(ARC_TARGET_LON, ARC_TARGET_LAT))

    def test_left_bank_footprint(self):
        from shapely.geometry import Point, Polygon
        arc = GlintArc(
            ARC_TARGET_LAT, ARC_TARGET_LON, ARC_OBS_TIME, ARC_ALTITUDE, ARC_SPEED,
            bank_direction="left",
        )
        poly = arc.footprint(AVIRIS3())
        assert isinstance(poly, Polygon)
        assert poly.is_valid
        assert poly.contains(Point(ARC_TARGET_LON, ARC_TARGET_LAT))
