"""Tests for hyplan.glint."""

import pytest
import numpy as np
import geopandas as gpd
from datetime import datetime, timezone

from hyplan.glint import glint_angle, calculate_target_and_glint_vectorized, compute_glint_vectorized
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
