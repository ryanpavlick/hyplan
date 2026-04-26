"""Tests for the nadir-pointing profiling lidar models (HSRL-2, HALO, CPL)."""

import pytest

from hyplan.exceptions import HyPlanTypeError, HyPlanValueError
from hyplan.instruments import CPL, HALO, HSRL2, ProfilingLidar, create_sensor
from hyplan.units import ureg


class TestHSRL2:
    def test_instantiation_defaults(self):
        sensor = HSRL2()
        assert sensor.name == "HSRL-2"
        assert len(sensor.wavelengths) == 3
        assert sensor.wavelengths[0].m_as("nanometer") == pytest.approx(355.0)
        assert sensor.wavelengths[1].m_as("nanometer") == pytest.approx(532.0)
        assert sensor.wavelengths[2].m_as("nanometer") == pytest.approx(1064.0)
        assert sensor.pulse_rate.m_as("hertz") == pytest.approx(200.0)
        assert sensor.telescope_diameter.m_as("centimeter") == pytest.approx(40.6)
        assert sensor.beam_divergence.m_as("milliradian") == pytest.approx(0.8)
        assert sensor.vertical_resolution.m_as("meter") == pytest.approx(15.0)
        assert sensor.sampling_rate.m_as("hertz") == pytest.approx(2.0)
        assert sensor.native_horizontal_resolution.m_as("meter") == pytest.approx(100.0)

    def test_subclass_of_profiling_lidar(self):
        assert issubclass(HSRL2, ProfilingLidar)

    @pytest.mark.parametrize("alias", ["HSRL-2", "HSRL2", "HSRL"])
    def test_factory_registration(self, alias):
        sensor = create_sensor(alias)
        assert isinstance(sensor, HSRL2)

    def test_footprint_diameter(self):
        sensor = HSRL2()
        # 8 km AGL × 0.8 mrad = 6.4 m
        diameter = sensor.footprint_diameter(8 * ureg.kilometer)
        assert diameter.m_as("meter") == pytest.approx(6.4, rel=1e-6)

    def test_footprint_diameter_accepts_scalar(self):
        sensor = HSRL2()
        # Bare numeric values should be interpreted as meters
        diameter = sensor.footprint_diameter(10000.0)
        assert diameter.m_as("meter") == pytest.approx(8.0, rel=1e-6)

    def test_horizontal_resolution(self):
        sensor = HSRL2()
        # 200 m/s × 10 s = 2000 m (Müller 2014's ~10 s averaging window)
        res = sensor.horizontal_resolution(
            200 * ureg.meter / ureg.second, 10 * ureg.second
        )
        assert res.m_as("meter") == pytest.approx(2000.0, rel=1e-6)

    def test_horizontal_resolution_native_window(self):
        sensor = HSRL2()
        # At a slow King Air cruise (100 m/s) with 0.5 s sampling,
        # native horizontal sampling is ~50 m.
        res = sensor.horizontal_resolution(
            100 * ureg.meter / ureg.second, 0.5 * ureg.second
        )
        assert res.m_as("meter") == pytest.approx(50.0, rel=1e-6)

    def test_pulses_per_profile(self):
        sensor = HSRL2()
        # 200 Hz × 10 s = 2000 pulses (matches Müller 2014's averaging window)
        assert sensor.pulses_per_profile(10 * ureg.second) == 2000

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"wavelengths": (-1 * ureg.nanometer,)},
            {"wavelengths": ()},
            {"pulse_rate": -1 * ureg.hertz},
            {"telescope_diameter": -1 * ureg.centimeter},
            {"beam_divergence": -1 * ureg.milliradian},
            {"vertical_resolution": -1 * ureg.meter},
            {"native_horizontal_resolution": -1 * ureg.meter},
            {"sampling_rate": -1 * ureg.hertz},
            {"pulse_rate": 0 * ureg.hertz},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(HyPlanValueError):
            HSRL2(**kwargs)

    def test_type_error_on_non_numeric(self):
        with pytest.raises(HyPlanTypeError):
            HSRL2(pulse_rate="200 Hz")  # string, not Quantity or numeric

    def test_wavelength_count_can_be_overridden(self):
        # User could model an HSRL-1 (2-wavelength) variant via overrides.
        sensor = HSRL2(wavelengths=(532 * ureg.nanometer, 1064 * ureg.nanometer))
        assert len(sensor.wavelengths) == 2


class TestHALO:
    def test_instantiation_defaults(self):
        sensor = HALO()
        assert sensor.name == "HALO"
        assert len(sensor.wavelengths) == 4
        assert sensor.wavelengths[0].m_as("nanometer") == pytest.approx(532.0)
        assert sensor.wavelengths[1].m_as("nanometer") == pytest.approx(935.0)
        assert sensor.wavelengths[2].m_as("nanometer") == pytest.approx(1064.0)
        assert sensor.wavelengths[3].m_as("nanometer") == pytest.approx(1645.0)
        assert sensor.pulse_rate.m_as("hertz") == pytest.approx(1000.0)
        assert sensor.sampling_rate.m_as("hertz") == pytest.approx(2.0)
        assert sensor.telescope_diameter.m_as("centimeter") == pytest.approx(40.0)
        assert sensor.beam_divergence is None
        assert sensor.vertical_resolution.m_as("meter") == pytest.approx(15.0)
        assert sensor.native_horizontal_resolution is None

    def test_subclass_of_profiling_lidar(self):
        assert issubclass(HALO, ProfilingLidar)

    @pytest.mark.parametrize("alias", ["HALO", "High Altitude Lidar Observatory"])
    def test_factory_registration(self, alias):
        sensor = create_sensor(alias)
        assert isinstance(sensor, HALO)

    def test_horizontal_resolution_act_america_window(self):
        # Carroll 2022's ACT-America averaging: ~15 s window at C-130 cruise (~150 m/s)
        sensor = HALO()
        res = sensor.horizontal_resolution(
            150 * ureg.meter / ureg.second, 15 * ureg.second
        )
        assert res.m_as("meter") == pytest.approx(2250.0, rel=1e-6)

    def test_pulses_per_profile_act_america(self):
        sensor = HALO()
        # 1 kHz × 15 s = 15000 pulses
        assert sensor.pulses_per_profile(15 * ureg.second) == 15000

    def test_footprint_diameter_raises_when_divergence_unset(self):
        sensor = HALO()
        with pytest.raises(HyPlanValueError, match="beam_divergence"):
            sensor.footprint_diameter(8 * ureg.kilometer)

    def test_footprint_diameter_works_when_divergence_supplied(self):
        # Same arithmetic as HSRL-2 if user supplies an HSRL-heritage divergence
        sensor = HALO(beam_divergence=0.8 * ureg.milliradian)
        diameter = sensor.footprint_diameter(8 * ureg.kilometer)
        assert diameter.m_as("meter") == pytest.approx(6.4, rel=1e-6)

    def test_mode_override_ch4_hsrl(self):
        # CH4 + HSRL mode: only 532, 1064, 1645 nm are active
        sensor = HALO(
            wavelengths=(
                532 * ureg.nanometer,
                1064 * ureg.nanometer,
                1645 * ureg.nanometer,
            )
        )
        assert len(sensor.wavelengths) == 3
        assert sensor.wavelengths[2].m_as("nanometer") == pytest.approx(1645.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pulse_rate": -1 * ureg.hertz},
            {"sampling_rate": -1 * ureg.hertz},
            {"telescope_diameter": -1 * ureg.centimeter},
            {"beam_divergence": -1 * ureg.milliradian},
            {"vertical_resolution": -1 * ureg.meter},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(HyPlanValueError):
            HALO(**kwargs)


class TestCPL:
    def test_instantiation_defaults(self):
        sensor = CPL()
        assert sensor.name == "CPL"
        assert len(sensor.wavelengths) == 3
        assert sensor.wavelengths[0].m_as("nanometer") == pytest.approx(355.0)
        assert sensor.wavelengths[1].m_as("nanometer") == pytest.approx(532.0)
        assert sensor.wavelengths[2].m_as("nanometer") == pytest.approx(1064.0)
        assert sensor.pulse_rate.m_as("hertz") == pytest.approx(5000.0)
        assert sensor.sampling_rate.m_as("hertz") == pytest.approx(1.0)
        assert sensor.telescope_diameter.m_as("centimeter") == pytest.approx(20.0)
        assert sensor.beam_divergence.m_as("microradian") == pytest.approx(100.0)
        assert sensor.vertical_resolution.m_as("meter") == pytest.approx(30.0)
        assert sensor.native_horizontal_resolution.m_as("meter") == pytest.approx(200.0)

    def test_subclass_of_profiling_lidar(self):
        assert issubclass(CPL, ProfilingLidar)

    @pytest.mark.parametrize("alias", ["CPL", "Cloud Physics Lidar"])
    def test_factory_registration(self, alias):
        sensor = create_sensor(alias)
        assert isinstance(sensor, CPL)

    def test_footprint_at_er2_altitude(self):
        # 20 km AGL × 100 μrad = 2.0 m — much smaller than HSRL-2's footprint
        # at the same altitude (16 m), demonstrating CPL's tighter beam.
        sensor = CPL()
        diameter = sensor.footprint_diameter(20 * ureg.kilometer)
        assert diameter.m_as("meter") == pytest.approx(2.0, rel=1e-6)

    def test_horizontal_resolution_standard_product(self):
        # 200 m/s × 1 s = 200 m (matches McGill 2002's standard 1 Hz product)
        sensor = CPL()
        res = sensor.horizontal_resolution(
            200 * ureg.meter / ureg.second, 1 * ureg.second
        )
        assert res.m_as("meter") == pytest.approx(200.0, rel=1e-6)

    def test_pulses_per_profile_standard_product(self):
        # 5 kHz × 1 s = 5000 pulses per standard profile
        sensor = CPL()
        assert sensor.pulses_per_profile(1 * ureg.second) == 5000


class TestProfilingLidar:
    """Tests for the ProfilingLidar base class directly."""

    def _minimal_kwargs(self, **overrides):
        kwargs = dict(
            wavelengths=(532 * ureg.nanometer,),
            pulse_rate=200 * ureg.hertz,
            telescope_diameter=40 * ureg.centimeter,
            vertical_resolution=15 * ureg.meter,
            sampling_rate=2 * ureg.hertz,
        )
        kwargs.update(overrides)
        return kwargs

    def test_minimal_construction(self):
        sensor = ProfilingLidar("Generic Profiler", **self._minimal_kwargs())
        assert sensor.name == "Generic Profiler"
        assert sensor.beam_divergence is None
        assert sensor.native_horizontal_resolution is None

    def test_optional_attributes_normalize_when_supplied(self):
        sensor = ProfilingLidar(
            "Generic Profiler",
            **self._minimal_kwargs(
                beam_divergence=1.0 * ureg.milliradian,
                native_horizontal_resolution=50 * ureg.meter,
            ),
        )
        assert sensor.beam_divergence.m_as("milliradian") == pytest.approx(1.0)
        assert sensor.native_horizontal_resolution.m_as("meter") == pytest.approx(50.0)

    def test_horizontal_resolution_units(self):
        sensor = ProfilingLidar("Generic Profiler", **self._minimal_kwargs())
        # Bare numeric arguments accepted (interpreted as m/s and seconds)
        res = sensor.horizontal_resolution(100.0, 5.0)
        assert res.m_as("meter") == pytest.approx(500.0, rel=1e-6)
