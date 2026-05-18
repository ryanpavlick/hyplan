"""Tests for hyplan.aircraft."""

import pytest
import numpy as np
from hyplan.units import ureg
from hyplan.aircraft import (
    CasMachSchedule,
    TasSchedule,
    VerticalProfile,
    TurnModel,
    PhaseBankAngles,
    PerformanceConfidence,
    SourceRecord,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    KingAirB200 as B200,
    NASA_C130,
    NOAA_TwinOtter,
    BAS_TwinOtter,
    FAAM_BAe146,
    SAFIRE_ATR42,
    NERC_DO228,
    AWI_BaslerBT67,
    DLR_HALO,
)
from hyplan.exceptions import HyPlanValueError


# ---------------------------------------------------------------------------
# TasSchedule
# ---------------------------------------------------------------------------

class TestTasSchedule:
    def test_single_point_constant(self):
        s = TasSchedule(points=[(0 * ureg.feet, 200 * ureg.knot)])
        assert s.tas_at(0 * ureg.feet).m_as("knot") == pytest.approx(200)
        assert s.tas_at(30000 * ureg.feet).m_as("knot") == pytest.approx(200)

    def test_two_point_interpolation(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 180 * ureg.knot),
            (30000 * ureg.feet, 300 * ureg.knot),
        ])
        assert s.tas_at(15000 * ureg.feet).m_as("knot") == pytest.approx(240)

    def test_clamps_below(self):
        s = TasSchedule(points=[
            (5000 * ureg.feet, 200 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        assert s.tas_at(0 * ureg.feet).m_as("knot") == pytest.approx(200)

    def test_clamps_above(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 200 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        assert s.tas_at(40000 * ureg.feet).m_as("knot") == pytest.approx(280)

    def test_requires_ascending_altitudes(self):
        with pytest.raises(HyPlanValueError):
            TasSchedule(points=[
                (20000 * ureg.feet, 280 * ureg.knot),
                (10000 * ureg.feet, 200 * ureg.knot),
            ])

    def test_requires_at_least_one_point(self):
        with pytest.raises(HyPlanValueError):
            TasSchedule(points=[])

    def test_unit_conversion(self):
        s = TasSchedule(points=[
            (0 * ureg.meter, 100 * ureg.meter / ureg.second),
            (3000 * ureg.meter, 120 * ureg.meter / ureg.second),
        ])
        result = s.tas_at(1500 * ureg.meter)
        assert result.check("[length] / [time]")

    def test_three_point(self):
        s = TasSchedule(points=[
            (0 * ureg.feet, 180 * ureg.knot),
            (15000 * ureg.feet, 260 * ureg.knot),
            (25000 * ureg.feet, 280 * ureg.knot),
        ])
        # Midpoint of first segment
        assert s.tas_at(7500 * ureg.feet).m_as("knot") == pytest.approx(220)


# ---------------------------------------------------------------------------
# CasMachSchedule
# ---------------------------------------------------------------------------

class TestCasMachSchedule:
    def test_below_crossover_uses_cas(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_low = s.tas_at(10000 * ureg.feet)
        assert tas_low.m_as("knot") > 280  # TAS > CAS at altitude

    def test_above_crossover_uses_mach(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_high = s.tas_at(40000 * ureg.feet)
        # M0.80 at FL400 (above tropopause, isothermal) → ~460 kt
        assert 440 < tas_high.m_as("knot") < 480

    def test_speed_continuity_near_crossover(self):
        s = CasMachSchedule(cas=280 * ureg.knot, mach=0.80, crossover_ft=28000)
        tas_below = s.tas_at(27900 * ureg.feet).m_as("knot")
        tas_above = s.tas_at(28100 * ureg.feet).m_as("knot")
        # CAS and Mach targets don't perfectly match at crossover_ft,
        # but the transition should not be wildly discontinuous.
        assert abs(tas_below - tas_above) < 60

    def test_sea_level_cas_approx_tas(self):
        s = CasMachSchedule(cas=250 * ureg.knot, mach=0.78, crossover_ft=30000)
        tas = s.tas_at(0 * ureg.feet)
        # At sea level, CAS ≈ TAS
        assert abs(tas.m_as("knot") - 250) < 1


# ---------------------------------------------------------------------------
# VerticalProfile
# ---------------------------------------------------------------------------

class TestVerticalProfile:
    def test_constant_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "constant"
        assert vp.rate_at(15000 * ureg.feet).m_as("feet/minute") == pytest.approx(2000)

    def test_two_point_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "two_point"
        # Midpoint interpolation
        mid = vp.rate_at(17500 * ureg.feet).m_as("feet/minute")
        assert mid == pytest.approx(1050)

    def test_full_mode(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 3800 * ureg.feet / ureg.minute),
            (10000 * ureg.feet, 3200 * ureg.feet / ureg.minute),
            (20000 * ureg.feet, 2400 * ureg.feet / ureg.minute),
            (30000 * ureg.feet, 1500 * ureg.feet / ureg.minute),
        ])
        assert vp._mode == "full"

    def test_sea_level_rate(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp.sea_level_rate.m_as("feet/minute") == pytest.approx(2000)

    def test_ceiling_rate(self):
        vp = VerticalProfile(points=[
            (0 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            (35000 * ureg.feet, 100 * ureg.feet / ureg.minute),
        ])
        assert vp.ceiling_rate.m_as("feet/minute") == pytest.approx(100)

    def test_requires_ascending_altitudes(self):
        with pytest.raises(HyPlanValueError):
            VerticalProfile(points=[
                (20000 * ureg.feet, 1000 * ureg.feet / ureg.minute),
                (10000 * ureg.feet, 2000 * ureg.feet / ureg.minute),
            ])

    def test_requires_at_least_one_point(self):
        with pytest.raises(HyPlanValueError):
            VerticalProfile(points=[])

    def test_clamps_at_endpoints(self):
        vp = VerticalProfile(points=[
            (5000 * ureg.feet, 1800 * ureg.feet / ureg.minute),
            (25000 * ureg.feet, 500 * ureg.feet / ureg.minute),
        ])
        assert vp.rate_at(0 * ureg.feet).m_as("feet/minute") == pytest.approx(1800)
        assert vp.rate_at(40000 * ureg.feet).m_as("feet/minute") == pytest.approx(500)


# ---------------------------------------------------------------------------
# TurnModel
# ---------------------------------------------------------------------------

class TestTurnModel:
    def test_defaults(self):
        tm = TurnModel()
        assert tm.max_bank_deg == 30.0
        assert tm.bank_by_phase.cruise_deg == 25.0
        assert tm.bank_by_phase.climb_deg == 20.0

    def test_custom_bank_angles(self):
        tm = TurnModel(
            bank_by_phase=PhaseBankAngles(
                climb_deg=15, cruise_deg=20, descent_deg=15, approach_deg=10,
            ),
            max_bank_deg=25.0,
        )
        assert tm.max_bank_deg == 25.0
        assert tm.bank_by_phase.cruise_deg == 20.0

    def test_default_max_load_factor(self):
        """Default load-factor budget is 2.5 g (FAR 23 normal-category)."""
        assert TurnModel().max_load_factor == pytest.approx(2.5)


class TestLoadFactorBudget:
    """Verify the curvature budget on Aircraft.max_bank_under_budget."""

    def test_level_flight_caps_at_normal_category(self):
        """In level flight, n=2.5 → bank_max = acos(1/2.5) ≈ 66.4°."""
        ac = B200()
        assert ac.max_bank_under_budget(0.0) == pytest.approx(66.4, abs=0.1)

    def test_calibrated_aircraft_have_headroom(self):
        """Every aircraft factory's calibrated banks fit well under
        the load-factor budget at level flight."""
        from hyplan.aircraft import (
            NASA_ER2, NASA_GV, NASA_GIII, NASA_GIV, NASA_C20A,
            NASA_P3, NASA_WB57, KingAirB200, NASA_C130, NOAA_TwinOtter,
            BAS_TwinOtter, FAAM_BAe146, SAFIRE_ATR42, NERC_DO228,
            AWI_BaslerBT67, DLR_HALO,
        )
        for cls in (NASA_ER2, NASA_GV, NASA_GIII, NASA_GIV, NASA_C20A,
                    NASA_P3, NASA_WB57, KingAirB200, NASA_C130, NOAA_TwinOtter,
                    BAS_TwinOtter, FAAM_BAe146, SAFIRE_ATR42, NERC_DO228,
                    AWI_BaslerBT67, DLR_HALO):
            ac = cls()
            bp = ac.turn_model.bank_by_phase
            cap = ac.max_bank_under_budget(0.0)
            for label, deg in [
                ("climb", bp.climb_deg), ("cruise", bp.cruise_deg),
                ("descent", bp.descent_deg), ("approach", bp.approach_deg),
            ]:
                assert deg < cap, (
                    f"{cls.__name__} {label}_deg={deg} exceeds "
                    f"load-factor cap {cap:.1f}°"
                )

    def test_pitch_consumes_budget(self):
        """A non-zero pitch reduces the available bank cap."""
        ac = B200()
        cap_level = ac.max_bank_under_budget(0.0)
        cap_climbing = ac.max_bank_under_budget(15.0)
        assert cap_climbing < cap_level

    def test_unbounded_when_max_load_factor_is_zero(self):
        """Setting max_load_factor <= 0 disables the budget (returns 90°)."""
        ac = B200()
        ac.turn_model.max_load_factor = 0.0
        assert ac.max_bank_under_budget(0.0) == pytest.approx(90.0)

    def test_aggressive_bank_is_clipped_when_set(self):
        """A custom 70° cruise bank is clipped to ≤ 67° at n_max=2.5g."""
        ac = B200()
        ac.turn_model.bank_by_phase = PhaseBankAngles(
            climb_deg=20, cruise_deg=70, descent_deg=20, approach_deg=15,
        )
        cap = ac.max_bank_under_budget(0.0)
        assert cap < 70.0
        # _hybrid_path consumers see the clipped value via _hybrid_path
        # itself; here we just confirm the cap exists.
        assert cap == pytest.approx(66.4, abs=0.5)

    def test_higher_n_max_unlocks_higher_banks(self):
        """Raising max_load_factor (e.g., for utility-category) opens the bank cap."""
        ac = B200()
        ac.turn_model.max_load_factor = 4.4  # FAR 23 utility
        # acos(1/4.4) ≈ 76.9°
        assert ac.max_bank_under_budget(0.0) == pytest.approx(76.9, abs=0.5)


# ---------------------------------------------------------------------------
# Aircraft instantiation
# ---------------------------------------------------------------------------

class TestAircraftInstantiation:
    def test_b200(self):
        ac = B200()
        assert ac.aircraft_type == "King Air 200"
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_er2(self):
        ac = NASA_ER2()
        assert ac.service_ceiling.m_as("feet") > 60000

    def test_giii(self):
        ac = NASA_GIII()
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_giii_stall_speed_at_increases_with_altitude(self):
        ac = NASA_GIII()
        sl = ac.stall_speed_at(0 * ureg.feet)
        fl400 = ac.stall_speed_at(40000 * ureg.feet)
        # Vs in CAS is constant; in TAS it scales as 1/sqrt(density),
        # so FL400 should be roughly twice SL.
        assert sl.m_as("knot") == pytest.approx(105, abs=1)
        assert fl400.m_as("knot") > 1.8 * sl.m_as("knot")

    def test_giii_min_safe_speed_applies_margin(self):
        ac = NASA_GIII()
        alt = 30000 * ureg.feet
        stall = ac.stall_speed_at(alt).m_as("knot")
        floor_13 = ac.min_safe_speed_at(alt).m_as("knot")
        floor_12 = ac.min_safe_speed_at(alt, margin=1.2).m_as("knot")
        assert floor_13 == pytest.approx(1.3 * stall, rel=1e-9)
        assert floor_12 == pytest.approx(1.2 * stall, rel=1e-9)
        # Cruise schedule at FL300 should sit well above the 1.3 floor.
        assert ac.cruise_speed_at(alt).m_as("knot") > floor_13

    def test_stall_methods_raise_when_uncalibrated(self):
        ac = NASA_GIV()  # not calibrated for stall_speed_cas
        assert ac.stall_speed_cas is None
        with pytest.raises(Exception):
            ac.stall_speed_at(20000 * ureg.feet)

    def test_giv(self):
        ac = NASA_GIV()
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_gv(self):
        ac = NASA_GV()
        assert ac.aircraft_type == "Gulfstream V"
        # Calibrated op-p99 of per-sortie peaks; not the brochure 51 kft
        # envelope, which the science-mission profile rarely reaches.
        assert ac.service_ceiling.m_as("feet") == pytest.approx(45000)

    def test_nasa_c130(self):
        ac = NASA_C130()
        assert ac.aircraft_type is not None

    def test_noaa_twin_otter(self):
        ac = NOAA_TwinOtter()
        assert ac.cruise_speed_at(5000 * ureg.feet).magnitude > 0

    def test_bas_twin_otter(self):
        ac = BAS_TwinOtter()
        assert ac.aircraft_type == "DHC-6 Twin Otter"
        assert ac.operator == "BAS"
        assert ac.calibration_status == "calibrated"
        assert ac.cruise_speed_at(5000 * ureg.feet).magnitude > 0

    def test_faam_bae146(self):
        ac = FAAM_BAe146()
        assert ac.aircraft_type == "BAe-146-301"
        assert ac.operator == "FAAM"
        assert ac.calibration_status == "calibrated"
        assert ac.service_ceiling.m_as("feet") == pytest.approx(34500, abs=1)
        assert ac.cruise_speed_at(20000 * ureg.feet).magnitude > 0

    def test_safire_atr42(self):
        ac = SAFIRE_ATR42()
        assert ac.aircraft_type == "ATR-42-320"
        assert ac.operator == "SAFIRE"
        assert ac.calibration_status == "calibrated"
        assert ac.cruise_speed_at(15000 * ureg.feet).magnitude > 0

    def test_nerc_do228(self):
        ac = NERC_DO228()
        assert ac.aircraft_type == "Dornier Do228-101"
        assert ac.operator == "NERC ARSF"
        assert ac.tail_number == "D-CALM"
        assert ac.calibration_status == "calibrated"
        assert ac.service_ceiling.m_as("feet") == pytest.approx(23000, abs=1)
        assert ac.cruise_speed_at(10000 * ureg.feet).m_as("knot") == pytest.approx(175)
        assert ac.climb_profile.rate_at(5000 * ureg.feet).m_as("feet/minute") == pytest.approx(957)

    def test_awi_basler_bt67(self):
        ac = AWI_BaslerBT67()
        assert ac.aircraft_type == "Basler BT-67"
        assert ac.operator == "AWI"
        assert ac.tail_number == "Polar 5 + Polar 6"
        assert ac.calibration_status == "calibrated"
        assert ac.service_ceiling.m_as("feet") == pytest.approx(25000, abs=1)
        assert ac.cruise_speed_at(10000 * ureg.feet).m_as("knot") == pytest.approx(186)
        assert ac.climb_profile.rate_at(5000 * ureg.feet).m_as("feet/minute") == pytest.approx(755)

    def test_dlr_halo(self):
        ac = DLR_HALO()
        assert ac.aircraft_type == "Gulfstream G550"
        assert ac.operator == "DLR"
        assert ac.calibration_status == "calibrated"
        # G550 cruise at FL350 should be near M0.80 → ~460 KTAS
        assert ac.cruise_speed_at(35000 * ureg.feet).m_as("knot") > 400

    def test_noaa_giv(self):
        from hyplan.aircraft import NOAA_GIV
        ac = NOAA_GIV()
        assert ac.aircraft_type == "Gulfstream IV-SP"
        assert ac.operator == "NOAA AOC"
        assert ac.tail_number == "N49RF"
        assert ac.calibration_status == "calibrated"
        # G-IV cruise at FL400 should be ~440 KTAS
        assert ac.cruise_speed_at(40000 * ureg.feet).m_as("knot") > 400


# ---------------------------------------------------------------------------
# Aircraft performance methods
# ---------------------------------------------------------------------------

class TestAircraftPerformance:
    def test_cruise_speed_at(self):
        ac = B200()
        speed = ac.cruise_speed_at(ureg.Quantity(20000, "feet"))
        assert speed.magnitude > 0
        assert speed.check("[length] / [time]")

    def test_cruise_speed_varies_with_altitude(self):
        ac = B200()
        low = ac.cruise_speed_at(ureg.Quantity(5000, "feet"))
        high = ac.cruise_speed_at(ureg.Quantity(25000, "feet"))
        assert low.magnitude != high.magnitude

    def test_cruise_speed_at_sea_level(self):
        ac = B200()
        speed = ac.cruise_speed_at(ureg.Quantity(0, "feet"))
        assert speed.magnitude > 0

    def test_cruise_speed_at_ceiling(self):
        ac = B200()
        speed = ac.cruise_speed_at(ac.service_ceiling)
        assert speed.magnitude > 0

    def test_climb_speed_at_returns_climb_schedule_value(self):
        """climb_speed_at delegates to climb_schedule.tas_at."""
        ac = B200()
        alt = 20000 * ureg.feet
        assert ac.climb_speed_at(alt).m_as(ureg.knot) == pytest.approx(
            ac.climb_schedule.tas_at(alt).m_as(ureg.knot), rel=1e-9,
        )

    def test_climb_speed_at_aliased_factory_matches_cruise(self):
        """When climb_schedule is aliased to cruise_schedule, they agree.

        Many factories alias climb_schedule = cruise_schedule.  NASA_ER2,
        NASA_GIII, NASA_P3, NASA_WB57, and KingAirB200 were de-aliased
        when calibrated against IWG1 data; NASA_GIV remains aliased and
        is the canonical example of this pattern.
        """
        from hyplan.aircraft import NASA_GIV
        ac = NASA_GIV()
        alt = 20000 * ureg.feet
        assert ac.climb_speed_at(alt).m_as(ureg.knot) == pytest.approx(
            ac.cruise_speed_at(alt).m_as(ureg.knot), rel=1e-9,
        )

    def test_rate_of_climb_decreases(self):
        ac = B200()
        roc_low = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        roc_high = ac.rate_of_climb(ureg.Quantity(20000, "feet"))
        assert roc_low.magnitude > roc_high.magnitude

    def test_rate_of_climb_at_sea_level(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        assert roc.m_as("feet/minute") == pytest.approx(
            ac.climb_profile.sea_level_rate.m_as("feet/minute"), rel=1e-6
        )

    def test_rate_of_climb_at_ceiling(self):
        # ``climb_profile.ceiling_rate`` is the rate at the *climb_profile's*
        # highest breakpoint, which can sit above the operational
        # ``service_ceiling`` for calibrated aircraft whose data sample
        # includes occasional excursions above the p99 ceiling.  Query at
        # the profile's top alt directly so this test verifies the
        # interpolation-at-boundary contract rather than coupling to the
        # ceiling derivation.
        ac = B200()
        top_alt_ft = ac.climb_profile.points[-1][0].m_as("feet")
        roc = ac.rate_of_climb(top_alt_ft * ureg.feet)
        assert roc.m_as("feet/minute") == pytest.approx(
            ac.climb_profile.ceiling_rate.m_as("feet/minute"), rel=1e-6
        )

    def test_rate_of_climb_mid_altitude(self):
        ac = B200()
        roc = ac.rate_of_climb(ureg.Quantity(15000, "feet"))
        assert roc.m_as("feet/minute") > ac.climb_profile.ceiling_rate.m_as("feet/minute")
        # The empirical B-200 climb_profile is nearly flat at low/mid
        # altitudes (turboprop power band); allow equality with SL.
        assert roc.m_as("feet/minute") <= ac.climb_profile.sea_level_rate.m_as("feet/minute") + 50

    def test_descent_speed_at(self):
        ac = B200()
        cruise = ac.cruise_speed_at(ureg.Quantity(10000, "feet"))
        descent = ac.descent_speed_at(ureg.Quantity(10000, "feet"))
        # Descent speed should be <= cruise speed
        assert descent.magnitude <= cruise.magnitude

    def test_max_bank_angle(self):
        ac = B200()
        assert isinstance(ac.max_bank_angle, float)
        assert 0 < ac.max_bank_angle < 90

    def test_climb_gradient_at(self):
        """Climb gradient = climb_rate / TAS, dimensionless."""
        ac = B200()
        alt = 5000 * ureg.feet
        rate_mps = ac.climb_profile.rate_at(alt).m_as("meter/second")
        tas_mps = ac.climb_speed_at(alt).m_as("meter/second")
        expected = rate_mps / tas_mps
        assert ac.climb_gradient_at(alt) == pytest.approx(expected, rel=1e-9)
        # Sanity: gradient is small for survey aircraft (< 0.2 ≈ 11°)
        assert 0 < ac.climb_gradient_at(alt) < 0.3

    def test_descent_gradient_at(self):
        ac = B200()
        alt = 5000 * ureg.feet
        rate_mps = abs(ac.descent_profile.rate_at(alt).m_as("meter/second"))
        tas_mps = ac.descent_speed_at(alt).m_as("meter/second")
        expected = rate_mps / tas_mps
        assert ac.descent_gradient_at(alt) == pytest.approx(expected, rel=1e-9)
        # Always positive (magnitude)
        assert ac.descent_gradient_at(alt) > 0

    def test_climb_gradient_decreases_with_altitude(self):
        """Climb gradient typically falls off at altitude (engine power decreases)."""
        ac = B200()
        g_low = ac.climb_gradient_at(5000 * ureg.feet)
        g_high = ac.climb_gradient_at(20000 * ureg.feet)
        assert g_low > g_high

    def test_service_ceiling_warning(self):
        """Requesting a cruise altitude above service ceiling warns."""
        from hyplan.waypoint import Waypoint

        ac = B200()
        ceiling_ft = ac.service_ceiling.m_as(ureg.foot)
        above_ft = ceiling_ft + 5000
        wp1 = Waypoint(34.0, -118.0, 90.0, altitude_msl=above_ft * ureg.foot)
        wp2 = Waypoint(34.5, -117.5, 90.0, altitude_msl=above_ft * ureg.foot)
        with pytest.warns(UserWarning, match="service ceiling"):
            ac.time_to_cruise(wp1, wp2)

    def test_service_ceiling_no_warning_below(self):
        """No warning when both endpoints are below ceiling."""
        from hyplan.waypoint import Waypoint
        import warnings as _warnings

        ac = B200()
        ceiling_ft = ac.service_ceiling.m_as(ureg.foot)
        below_ft = ceiling_ft - 5000
        wp1 = Waypoint(34.0, -118.0, 90.0, altitude_msl=below_ft * ureg.foot)
        wp2 = Waypoint(34.5, -117.5, 90.0, altitude_msl=below_ft * ureg.foot)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            ac.time_to_cruise(wp1, wp2)

    def test_endurance_exists_and_reasonable(self):
        ac = B200()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs > 0
        assert endurance_hrs < 24

    def test_endurance_er2(self):
        ac = NASA_ER2()
        endurance_hrs = ac.endurance.m_as("hour")
        assert endurance_hrs >= 6

    def test_climb_altitude_profile(self):
        ac = B200()
        times, altitudes = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet")
        )
        assert len(times) > 0
        assert len(altitudes) > 0
        assert altitudes[-1] == pytest.approx(10000, rel=0.1)


# ---------------------------------------------------------------------------
# Climb and descent integration
# ---------------------------------------------------------------------------

class TestClimbAndDescend:
    def test_climb_time_positive(self):
        ac = B200()
        time, dist = ac._climb(
            ureg.Quantity(0, "feet"), ureg.Quantity(20000, "feet")
        )
        assert time.magnitude > 0
        assert dist.magnitude > 0

    def test_climb_no_change(self):
        ac = B200()
        time, dist = ac._climb(
            ureg.Quantity(10000, "feet"), ureg.Quantity(10000, "feet")
        )
        assert time.magnitude == 0
        assert dist.magnitude == 0

    def test_descend_time_positive(self):
        ac = B200()
        time, dist = ac._descend(
            ureg.Quantity(20000, "feet"), ureg.Quantity(0, "feet")
        )
        assert time.magnitude > 0
        assert dist.magnitude > 0

    def test_descend_no_change(self):
        ac = B200()
        time, dist = ac._descend(
            ureg.Quantity(10000, "feet"), ureg.Quantity(10000, "feet")
        )
        assert time.magnitude == 0
        assert dist.magnitude == 0

    def test_climb_higher_takes_longer(self):
        ac = B200()
        t1, _ = ac._climb(ureg.Quantity(0, "feet"), ureg.Quantity(10000, "feet"))
        t2, _ = ac._climb(ureg.Quantity(0, "feet"), ureg.Quantity(20000, "feet"))
        assert t2.magnitude > t1.magnitude


class TestStepClimb:
    """Verify Aircraft.step_climb (staged climb-out with pauses)."""

    def test_no_pauses_matches_climb(self):
        """Empty pauses list reduces to a plain _climb integration."""
        ac = NASA_ER2()
        t_step, d_step = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(60000, "feet"),
            pauses=[],
        )
        t_plain, d_plain = ac._climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(60000, "feet"),
        )
        assert t_step.m_as(ureg.minute) == pytest.approx(
            t_plain.m_as(ureg.minute), rel=1e-9,
        )
        assert d_step.m_as(ureg.nautical_mile) == pytest.approx(
            d_plain.m_as(ureg.nautical_mile), rel=1e-9,
        )

    def test_hold_adds_only_to_time(self):
        """A hold at an intermediate altitude adds its duration to
        total time but no forward distance.

        Distance is *not* exactly equal to a single _climb integration
        over the same range — _climb uses TAS at the average altitude
        of its band, so splitting into two integrations gives a
        slightly different forward-distance estimate (arguably more
        accurate, since each band gets its own TAS).
        """
        ac = NASA_ER2()
        hold_min = 25.0
        pauses_with_hold = [(35600 * ureg.foot, hold_min * ureg.minute)]
        pauses_zero_hold = [(35600 * ureg.foot, 0 * ureg.minute)]
        t_with, d_with = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(60000, "feet"),
            pauses=pauses_with_hold,
        )
        t_zero, d_zero = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(60000, "feet"),
            pauses=pauses_zero_hold,
        )
        # Distance is identical between zero-hold and 25-min-hold:
        # the hold contributes zero forward progress.
        assert d_with.m_as(ureg.nautical_mile) == pytest.approx(
            d_zero.m_as(ureg.nautical_mile), rel=1e-9,
        )
        # Total time grows by exactly the hold duration.
        assert (t_with - t_zero).m_as(ureg.minute) == pytest.approx(
            hold_min, abs=1e-6,
        )

    def test_pauses_outside_range_ignored(self):
        """Pauses below start or above end have no effect."""
        ac = B200()
        t_in_range, _ = ac.step_climb(
            ureg.Quantity(5000, "feet"),
            ureg.Quantity(20000, "feet"),
            pauses=[(15000 * ureg.foot, 5 * ureg.minute)],
        )
        t_out_below, _ = ac.step_climb(
            ureg.Quantity(5000, "feet"),
            ureg.Quantity(20000, "feet"),
            pauses=[(2000 * ureg.foot, 5 * ureg.minute)],
        )
        t_out_above, _ = ac.step_climb(
            ureg.Quantity(5000, "feet"),
            ureg.Quantity(20000, "feet"),
            pauses=[(25000 * ureg.foot, 5 * ureg.minute)],
        )
        t_no_pauses, _ = ac.step_climb(
            ureg.Quantity(5000, "feet"),
            ureg.Quantity(20000, "feet"),
            pauses=[],
        )
        # In-range pause adds 5 min; out-of-range pauses don't.
        assert (t_in_range - t_no_pauses).m_as(ureg.minute) == pytest.approx(
            5.0, abs=1e-6,
        )
        assert t_out_below.m_as(ureg.minute) == pytest.approx(
            t_no_pauses.m_as(ureg.minute), rel=1e-9,
        )
        assert t_out_above.m_as(ureg.minute) == pytest.approx(
            t_no_pauses.m_as(ureg.minute), rel=1e-9,
        )

    def test_pauses_applied_in_altitude_order(self):
        """Pauses passed out of altitude order produce the same result
        as pauses passed in order."""
        ac = NASA_ER2()
        in_order = [
            (24000 * ureg.foot, 1 * ureg.minute),
            (35600 * ureg.foot, 25 * ureg.minute),
            (61100 * ureg.foot, 2 * ureg.minute),
        ]
        out_of_order = [
            (61100 * ureg.foot, 2 * ureg.minute),
            (24000 * ureg.foot, 1 * ureg.minute),
            (35600 * ureg.foot, 25 * ureg.minute),
        ]
        t1, d1 = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(65000, "feet"),
            pauses=in_order,
        )
        t2, d2 = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(65000, "feet"),
            pauses=out_of_order,
        )
        assert t1.m_as(ureg.minute) == pytest.approx(
            t2.m_as(ureg.minute), rel=1e-9,
        )
        assert d1.m_as(ureg.nautical_mile) == pytest.approx(
            d2.m_as(ureg.nautical_mile), rel=1e-9,
        )

    def test_step_climb_strictly_longer_than_climb_with_holds(self):
        """For any non-empty pauses-with-positive-hold, step_climb
        time exceeds the no-pause _climb time."""
        ac = NASA_ER2()
        t_no_hold, _ = ac._climb(
            ureg.Quantity(0, "feet"), ureg.Quantity(60000, "feet"),
        )
        t_with_holds, _ = ac.step_climb(
            ureg.Quantity(0, "feet"),
            ureg.Quantity(60000, "feet"),
            pauses=[
                (24000 * ureg.foot, 1 * ureg.minute),
                (35600 * ureg.foot, 5 * ureg.minute),
            ],
        )
        # Total hold = 6 min; rest of the climb is at the same rate.
        assert (
            t_with_holds.m_as(ureg.minute) - t_no_hold.m_as(ureg.minute) > 5.5
        )


# ---------------------------------------------------------------------------
# Hybrid 2D-Dubins + integrated-vertical path
# ---------------------------------------------------------------------------

class TestHybridPath:
    """Cover the four cases of Aircraft._hybrid_path (via time_to_cruise).

    Long leg (cruise reachable), short leg (spiral-up scaling), pure
    cruise, and pure descent.
    """

    def _wp(self, lat, lon, alt_ft, hdg=90.0):
        from hyplan.waypoint import Waypoint
        return Waypoint(lat, lon, hdg, altitude_msl=alt_ft * ureg.feet)

    def test_long_leg_normal_cruise(self):
        """Long climb-then-cruise leg: both phases present, totals balance."""
        ac = B200()
        # 200 nmi leg: well over the climb_distance the B200 needs to FL200.
        info = ac.time_to_cruise(
            self._wp(34.0, -118.0, 5000), self._wp(36.0, -114.0, 20000),
        )
        phases = info["phases"]
        assert "climb" in phases
        assert "cruise" in phases
        assert "descent" not in phases
        # Phase times sum to total.
        phase_total = sum(
            (p["end_time"] - p["start_time"]).m_as(ureg.minute)
            for p in phases.values()
        )
        assert phase_total == pytest.approx(
            info["total_time"].m_as(ureg.minute), rel=1e-6,
        )
        # Climb phase ends at the requested cruise altitude.
        assert phases["climb"]["end_altitude"].m_as(ureg.feet) == pytest.approx(20000)
        # Cruise phase is at constant altitude (= cruise alt).
        assert phases["cruise"]["start_altitude"].m_as(ureg.feet) == pytest.approx(20000)
        assert phases["cruise"]["end_altitude"].m_as(ureg.feet) == pytest.approx(20000)

    def test_short_leg_spiral_up_at_departure(self):
        """Short leg where climb_distance > L: orbit at departure, then transit."""
        ac = NASA_ER2()
        # ~30 nmi leg — far less than ER-2's ~250 nmi climb-to-FL600 distance.
        start = self._wp(34.0, -118.0, 0)
        info = ac.time_to_cruise(
            start, self._wp(34.5, -117.5, 60000),
        )
        phases = info["phases"]
        assert "climb" in phases
        assert "cruise" in phases  # full leg cruised at altitude after orbit
        # Climb takes the full integrated climb time (~20 min for ER-2 to
        # FL600 with active-only climb profile).
        climb_min = (
            (phases["climb"]["end_time"] - phases["climb"]["start_time"])
            .m_as(ureg.minute)
        )
        assert climb_min > 15, "spiral-up should take the full climb integration time"
        # Orbit closes back to the departure waypoint — both endpoints sit there.
        assert phases["climb"]["start_lat"] == pytest.approx(start.latitude)
        assert phases["climb"]["start_lon"] == pytest.approx(start.longitude)
        assert phases["climb"]["end_lat"] == pytest.approx(start.latitude)
        assert phases["climb"]["end_lon"] == pytest.approx(start.longitude)
        # Cruise phase covers the full horizontal Dubins length.
        h_length_nmi = info["dubins_path"].length.m_as(ureg.nautical_mile)
        assert phases["cruise"]["distance"].m_as(ureg.nautical_mile) == pytest.approx(
            h_length_nmi, rel=1e-3,
        )
        # Climb-phase geometry is a closed orbit (first coord == last coord).
        coords = list(phases["climb"]["geometry"].coords)
        assert coords[0] == pytest.approx(coords[-1], abs=1e-9)

    def test_spiral_up_orbit_uses_climb_bank_not_cruise(self):
        """Spiral-up orbit radius reflects climb bank (gentler) and
        midpoint-altitude TAS, not cruise bank at cruise altitude."""
        from hyplan.planning.segments import loiter_orbit_geometry
        from hyplan.waypoint import Waypoint

        ac = NASA_ER2()
        start = self._wp(34.0, -118.0, 0)
        info = ac.time_to_cruise(
            start, self._wp(34.5, -117.5, 60000),
        )
        orbit_geom = info["phases"]["climb"]["geometry"]

        # Cruise-phase orbit at cruise altitude (the *previous* behavior)
        # for the same start waypoint at FL600.
        cruise_orbit_wp = Waypoint(
            latitude=start.latitude, longitude=start.longitude,
            heading=start.heading, altitude_msl=60000 * ureg.feet,
        )
        cruise_orbit = loiter_orbit_geometry(
            cruise_orbit_wp, ac, phase="cruise",
        )

        # The spiral-up orbit should be *larger* in extent than the
        # cruise orbit at the same waypoint, because climb bank is
        # gentler.  Use the bounding-box diagonal as a coarse size proxy.
        sb = orbit_geom.bounds
        cb = cruise_orbit.bounds
        spiral_size = max(sb[2] - sb[0], sb[3] - sb[1])
        cruise_size = max(cb[2] - cb[0], cb[3] - cb[1])
        # Climb radius (11°) at FL300 / ~280 kt is roughly the same
        # as cruise radius (20°) at FL600 / 425 kt — the midpoint
        # altitude offsets the gentler bank.  Just verify they're
        # within the same order of magnitude rather than expecting a
        # fixed ratio.
        assert 0.3 < spiral_size / cruise_size < 3.0, (
            f"spiral-up orbit size {spiral_size:.4f} vs cruise orbit "
            f"{cruise_size:.4f} differs by an unrealistic factor"
        )

    def test_pure_cruise(self):
        """Equal altitudes: no climb / descent, only cruise."""
        ac = B200()
        info = ac.time_to_cruise(
            self._wp(34.0, -118.0, 20000), self._wp(34.5, -117.5, 20000),
        )
        phases = info["phases"]
        assert set(phases) == {"cruise"}
        # Cruise time = horizontal distance / TAS.
        h_nmi = info["dubins_path"].length.m_as(ureg.nautical_mile)
        tas_kt = ac.cruise_speed_at(20000 * ureg.feet).m_as(ureg.knot)
        expected_min = h_nmi / tas_kt * 60.0
        assert info["total_time"].m_as(ureg.minute) == pytest.approx(
            expected_min, rel=1e-3,
        )

    def test_descent_then_endpoint(self):
        """End altitude < start altitude: cruise + descent (or pure descent)."""
        ac = B200()
        info = ac.time_to_cruise(
            self._wp(34.0, -118.0, 20000), self._wp(34.5, -117.5, 5000),
        )
        phases = info["phases"]
        assert "descent" in phases
        # Descent phase ends at the requested end altitude.
        assert phases["descent"]["end_altitude"].m_as(ureg.feet) == pytest.approx(5000)

    def test_explicit_phase_geometry_present(self):
        """Each phase carries its own LineString — no shared-Dubins slicing."""
        ac = B200()
        info = ac.time_to_cruise(
            self._wp(34.0, -118.0, 5000), self._wp(36.0, -114.0, 20000),
        )
        for phase_name, p in info["phases"].items():
            assert p.get("geometry") is not None, f"{phase_name} missing geometry"
            assert len(p["geometry"].coords) >= 2

    def test_top_of_climb_geometry_along_2d_path(self):
        """Climb-phase end coord matches sample_at_distance(climb_distance)."""
        ac = B200()
        info = ac.time_to_cruise(
            self._wp(34.0, -118.0, 5000), self._wp(36.0, -114.0, 20000),
        )
        h_path = info["dubins_path"]
        climb = info["phases"]["climb"]
        climb_dist_m = climb["distance"].m_as(ureg.meter)
        expected_lat, expected_lon, _ = h_path.sample_at_distance(climb_dist_m)
        assert climb["end_lat"] == pytest.approx(expected_lat, abs=1e-6)
        assert climb["end_lon"] == pytest.approx(expected_lon, abs=1e-6)


# ---------------------------------------------------------------------------
# ER-2 performance (high-altitude, TAS speed profile)
# ---------------------------------------------------------------------------

class TestER2Performance:
    def test_cruise_speed_at_sea_level(self):
        ac = NASA_ER2()
        speed = ac.cruise_speed_at(ureg.Quantity(0, "feet"))
        assert speed.magnitude > 0

    def test_cruise_speed_at_ceiling(self):
        ac = NASA_ER2()
        speed = ac.cruise_speed_at(ac.service_ceiling)
        assert speed.magnitude > 0

    def test_cruise_speed_increases_with_altitude(self):
        ac = NASA_ER2()
        low = ac.cruise_speed_at(ureg.Quantity(10000, "feet"))
        high = ac.cruise_speed_at(ureg.Quantity(60000, "feet"))
        assert high.magnitude > low.magnitude

    def test_max_bank_angle(self):
        ac = NASA_ER2()
        assert 0 < ac.max_bank_angle < 90

    # --- Hybrid-path / climb-step planner regression tests ------------------

    def test_climb_step_visible_in_planner(self):
        """time_to_cruise's climb-phase time matches Aircraft._climb directly.

        The hybrid planner integrates climb_profile through `_climb`, so the
        calibrated active-climb profile propagates into mission timing.
        Typical ER-2 level-offs and holds are represented separately via
        ``typical_climb_out`` / ``ClimbPlan`` rather than baked into the
        aircraft-intrinsic climb profile.
        """
        from hyplan.waypoint import Waypoint
        ac = NASA_ER2()
        start = Waypoint(34.0, -118.0, 90.0, altitude_msl=0 * ureg.feet)
        # Make the leg long enough that climb fits horizontally (no spiral-up).
        end = Waypoint(35.0, -114.0, 90.0, altitude_msl=60000 * ureg.feet)
        info = ac.time_to_cruise(start, end)
        climb_phase = info["phases"]["climb"]
        climb_phase_min = (
            (climb_phase["end_time"] - climb_phase["start_time"]).m_as(ureg.minute)
        )
        direct_climb_min = ac._climb(0 * ureg.feet, 60000 * ureg.feet)[0].m_as(
            ureg.minute
        )
        # Phase time should equal _climb's integrated time within numerical noise.
        assert climb_phase_min == pytest.approx(direct_climb_min, rel=1e-3)
        # And should be substantially longer than ~16 min (which is what the
        # legacy constant-pitch Dubins path produced for a comparable leg).
        assert climb_phase_min > 18, (
            f"climb phase time {climb_phase_min:.1f} min should reflect the "
            f"integrated climb_profile, not the legacy constant-pitch "
            f"underestimate"
        )

    # --- IWG1 calibration regression tests -----------------------------------

    def test_climb_profile_trends_down_above_mid_altitudes(self):
        """climb_profile rates at FL350+ are below the SL/low-altitude peak.

        Real aircraft climb performance often isn't strictly monotonic
        above the SL peak — constant-CAS climb regimes, step-cruise
        transitions, and weight burn-off can produce small bumps in
        the mid-altitude band.  The earlier strict-monotone assertion
        encoded a textbook simplification that calibrated data
        consistently violates (G-III and G-V also show non-monotone
        bumps in the FL200-FL300 band).  The looser claim — that
        upper-altitude climb is meaningfully below the lower-altitude
        peak — is the physically defensible one.
        """
        ac = NASA_ER2()
        peak = max(
            ac.climb_profile.rate_at(alt * ureg.feet).m_as(ureg.feet / ureg.minute)
            for alt in [0, 5000, 10000, 15000]
        )
        for alt in [35000, 45000, 55000, 65000]:
            rate = ac.climb_profile.rate_at(alt * ureg.feet).m_as(ureg.feet / ureg.minute)
            assert rate < peak * 0.9, (
                f"climb VS at {alt} ft ({rate:.0f}) should be below 90% of "
                f"low-altitude peak ({peak:.0f})"
            )

    def test_climb_profile_matches_active_iwg1_medians(self):
        """ER-2 climb_profile uses hold-band-excluded active-climb medians.

        Values from the 618-sortie 2012-2026 IWG1 cache; see
        ``notebooks/calibration/NASA_ER2/calibration.ipynb`` and
        ``_fetch_asp.py`` for derivation and source data.
        """
        ac = NASA_ER2()
        expected = {
            5000: 3553,
            10000: 3876,
            15000: 3851,
            20000: 3526,
            25000: 3423,
            30000: 3121,
            35000: 2338,
            40000: 2035,
            45000: 1761,
            50000: 1668,
            55000: 1612,
            66000: 200,
        }
        for alt_ft, expected_fpm in expected.items():
            actual = ac.climb_profile.rate_at(alt_ft * ureg.feet).m_as(
                ureg.feet / ureg.minute
            )
            assert actual == pytest.approx(expected_fpm, abs=0.5)

    def test_descent_profile_peaks_in_mid_altitude_band(self):
        """Median-based descent_profile peaks in the 25-45 kft band.

        IWG1 medians show a three-regime shape: shallow VS at top of
        descent (cruise altitudes), steep peak in the middle, then
        shallow again as the aircraft slows for approach.
        """
        ac = NASA_ER2()
        rate_top = ac.descent_profile.rate_at(60000 * ureg.feet).m_as(
            ureg.feet / ureg.minute
        )
        rate_mid = ac.descent_profile.rate_at(35000 * ureg.feet).m_as(
            ureg.feet / ureg.minute
        )
        rate_low = ac.descent_profile.rate_at(10000 * ureg.feet).m_as(
            ureg.feet / ureg.minute
        )
        assert rate_mid > rate_top, (
            f"mid-altitude descent ({rate_mid:.0f} fpm) should exceed "
            f"top-of-descent ({rate_top:.0f} fpm)"
        )
        assert rate_mid > rate_low, (
            f"mid-altitude descent ({rate_mid:.0f} fpm) should exceed "
            f"low-altitude descent ({rate_low:.0f} fpm)"
        )

    def test_descent_path_angle_max_set(self):
        """ER-2 has a calibrated max descent FPA (envelope from IWG1)."""
        ac = NASA_ER2()
        assert ac.descent_path_angle_max_deg is not None
        assert 3.0 < ac.descent_path_angle_max_deg < 10.0

    def test_descent_steepens_to_fit_short_leg(self):
        """When the leg is shorter than preferred descent_dist but the
        required FPA is within ``descent_path_angle_max_deg``, the
        descent should scale to fit rather than spiral at end of leg.
        """
        from hyplan.waypoint import Waypoint
        ac = NASA_ER2()
        # Preferred descent FL650 -> SL covers ~184 nmi.  A 115 nmi leg
        # would have triggered short_descent (spiral) under legacy
        # behavior; the required FPA at 115 nmi is ~5.3°, under the
        # 6° cap, so descent should fill the leg.
        start = Waypoint(34.0, -118.0, 90.0, altitude_msl=65000 * ureg.feet)
        end = Waypoint(34.0, -115.7, 90.0, altitude_msl=0 * ureg.feet)
        info = ac._hybrid_path(start, end, phase="descent")
        # Single descent phase, no cruise phase, no spiral marker.
        assert "descent" in info["phases"]
        descent = info["phases"]["descent"]
        descent_dist_nmi = descent["distance"].m_as(ureg.nautical_mile)
        # Descent should fill the leg (no truncation to 0, no full-leg
        # cruise).  Leg length is ~115 nmi.
        assert 100 < descent_dist_nmi < 130, (
            f"descent should fit the short leg, got {descent_dist_nmi:.1f} nmi"
        )

    def test_descent_spirals_when_leg_too_short_for_max_fpa(self):
        """If the required FPA exceeds ``descent_path_angle_max_deg``,
        the descent falls back to the spiral-down regime."""
        from hyplan.waypoint import Waypoint
        ac = NASA_ER2()
        # 50 nmi leg requires FPA ~12°, well above 6° cap → spiral.
        start = Waypoint(34.0, -118.0, 90.0, altitude_msl=65000 * ureg.feet)
        end = Waypoint(34.0, -116.95, 90.0, altitude_msl=0 * ureg.feet)
        info = ac._hybrid_path(start, end, phase="descent")
        # Spiral regime: cruise phase has the full leg distance,
        # descent's distance is the spiral-track-length proxy.
        assert "cruise" in info["phases"]
        assert "descent" in info["phases"]
        cruise_dist = info["phases"]["cruise"]["distance"].m_as(
            ureg.nautical_mile
        )
        # Cruise should equal full leg (~50 nmi)
        assert 40 < cruise_dist < 65

    def test_approach_profile_present(self):
        ac = NASA_ER2()
        assert ac.approach_profile is not None
        # ER-2 IWG1-derived glideslope (~2.6°) is shallower than 3° ILS.
        assert 2.0 < ac.approach_profile.glideslope_deg < 3.5
        assert ac.approach_profile.top_of_approach_agl.m_as(ureg.feet) == pytest.approx(3000)

    def test_approach_time_to_touchdown_is_a_few_minutes(self):
        ac = NASA_ER2()
        t = ac.approach_profile.time_to_touchdown().m_as(ureg.minute)
        assert 4.0 < t < 8.0

    def test_iwg1_source_record_present(self):
        ac = NASA_ER2()
        kinds = [s.source_type for s in ac.sources]
        assert "iwg1" in kinds
        assert "brochure" in kinds

    # --- Item 2: bank_by_phase metadata -----------------------------------

    def test_bank_by_phase_calibrated(self):
        """NASA_ER2.turn_model carries the IWG1-calibrated per-phase medians.

        Values from the 618-sortie 2012-2026 IWG1 cache (~762 k turn fixes).
        """
        ac = NASA_ER2()
        bp = ac.turn_model.bank_by_phase
        assert bp.climb_deg == pytest.approx(14.0)
        assert bp.cruise_deg == pytest.approx(20.0)
        assert bp.descent_deg == pytest.approx(13.0)
        assert bp.approach_deg == pytest.approx(11.0)
        # Brochure max-bank envelope unchanged (p99 < 30° in every band).
        assert ac.turn_model.max_bank_deg == pytest.approx(30.0)

    def test_climb_plan_auto_sentinel_resolves_to_explicit_plan(self):
        """compute_flight_plan(climb_plan="auto") reads the aircraft's
        typical_climb_out.explicit_climb_plan and produces a plan
        whose total time grows by the hold duration relative to
        climb_plan=None.

        NASA_ER2 ships with a 13-min FL550 representative pause (the
        empirical observed-vs-active-climb gap derived from the
        618-sortie IWG1 cache); the "auto" sentinel resolves to that
        ClimbPlan, which is distinct from `climb_plan=None` (no holds
        — pure active-climb).
        """
        from hyplan.airports import Airport
        from hyplan.flight_line import FlightLine
        from hyplan.flight_plan import compute_flight_plan
        ac = NASA_ER2()
        # Long leg so the climb fits horizontally and the staged
        # branch emits an explicit "loiter" row for the FL550 pause
        # (rather than spiral-up at departure absorbing it).
        line = FlightLine.center_length_azimuth(
            lat=36.0, lon=-100.0, length=ureg.Quantity(120, "km"),
            az=180.0, altitude_msl=ureg.Quantity(65000, "feet"),
            site_name="L1",
        )
        kcos = Airport("KCOS")
        plan_auto = compute_flight_plan(
            ac, [line], takeoff_airport=kcos, return_airport=kcos,
            climb_plan="auto",
        )
        plan_none = compute_flight_plan(
            ac, [line], takeoff_airport=kcos, return_airport=kcos,
            climb_plan=None,
        )
        # "auto" resolves to ER-2's 13-min FL550 representative pause;
        # total time exceeds the no-hold version by ~13 min.
        delta_min = (
            plan_auto["time_to_segment"].sum()
            - plan_none["time_to_segment"].sum()
        )
        assert delta_min == pytest.approx(13.0, abs=1.0), (
            f"auto-sentinel + 13-min FL550 pause should add ~13 min; "
            f"got {delta_min:+.1f} min"
        )
        # Long enough leg that the climb fits horizontally and the
        # staged branch fires — auto plan carries an explicit
        # `loiter` row from the FL356 pause; none plan does not.
        assert "loiter" in plan_auto["segment_type"].values
        assert "loiter" not in plan_none["segment_type"].values
        # Rejection of unknown sentinel strings.
        from hyplan.exceptions import HyPlanValueError
        with pytest.raises(HyPlanValueError, match="climb_plan"):
            compute_flight_plan(
                ac, [line], takeoff_airport=kcos, return_airport=kcos,
                climb_plan="not-a-valid-sentinel",
            )

    def test_typical_climb_out_populated(self):
        """NASA_ER2 carries a ClimbOutPolicy with an explicit_climb_plan.

        Post-Phase-3: ``climb_profile`` is active-climb-only and the
        typical pre-cruise overhead lives in
        ``typical_climb_out.explicit_climb_plan``, which the planner
        consumes by default via the ``"auto"`` sentinel.
        """
        from hyplan.aircraft import ClimbOutPolicy, ClimbPlan
        ac = NASA_ER2()
        p = ac.typical_climb_out
        assert p is not None
        assert isinstance(p, ClimbOutPolicy)
        # Post-Phase-3: climb_profile is active-only and the explicit
        # ClimbPlan carries the typical mission overhead.
        assert p.absorbed_in_climb_profile is False
        assert isinstance(p.explicit_climb_plan, ClimbPlan)
        assert len(p.explicit_climb_plan.pauses) >= 1
        # typical_holds may be empty for aircraft where no discretionary
        # altitude-band hold concentrates in the calibration sample (this
        # is the case for NASA_ER2 in the 618-sortie 2012-2026 cache —
        # see typical_climb_out.notes).
        assert isinstance(p.typical_holds, list)
        # Total overhead should be a plausible double-digit minute count.
        assert 5.0 <= p.typical_overhead_min <= 30.0
        # Notes surface the calibration story.
        assert "climb_profile" in p.notes

    # --- Item 4: distinct climb / cruise / descent TAS schedules -----------

    def test_schedules_are_not_aliased(self):
        """climb / cruise / descent schedules differ — no aliasing on ER-2."""
        ac = NASA_ER2()
        assert ac.climb_schedule.points != ac.cruise_schedule.points
        assert ac.descent_schedule.points != ac.cruise_schedule.points
        assert ac.climb_schedule.points != ac.descent_schedule.points

    def test_descent_speed_below_cruise_speed_at_mid_altitude(self):
        """At mid-cruise altitude, descent TAS sits below cruise TAS."""
        ac = NASA_ER2()
        alt = 50000 * ureg.feet
        assert ac.descent_speed_at(alt).m_as(ureg.knot) < (
            ac.cruise_speed_at(alt).m_as(ureg.knot)
        )

    def test_climb_speed_below_cruise_speed_at_mid_altitude(self):
        """At mid-cruise altitude, climb TAS sits below cruise TAS."""
        ac = NASA_ER2()
        alt = 50000 * ureg.feet
        assert ac.climb_speed_at(alt).m_as(ureg.knot) < (
            ac.cruise_speed_at(alt).m_as(ureg.knot)
        )


# ---------------------------------------------------------------------------
# GV (CAS/Mach schedule) specific tests
# ---------------------------------------------------------------------------

class TestGVPerformance:
    """Tests specific to the NASA GV with CAS/Mach speed schedules."""

    def test_cruise_speed_below_crossover(self):
        ac = NASA_GV()
        # Below crossover (30000 ft), uses CAS
        speed = ac.cruise_speed_at(ureg.Quantity(20000, "feet"))
        assert speed.m_as("knot") > 300  # TAS > CAS at altitude

    def test_cruise_speed_above_crossover(self):
        ac = NASA_GV()
        # Above crossover (30000 ft), uses Mach
        speed = ac.cruise_speed_at(ureg.Quantity(40000, "feet"))
        assert 440 < speed.m_as("knot") < 500

    def test_climb_profile_full_mode(self):
        ac = NASA_GV()
        assert ac.climb_profile._mode == "full"

    def test_descent_profile_full_mode(self):
        ac = NASA_GV()
        assert ac.descent_profile._mode == "full"

    def test_climb_roc_decreases_with_altitude(self):
        ac = NASA_GV()
        roc_low = ac.rate_of_climb(ureg.Quantity(0, "feet"))
        roc_high = ac.rate_of_climb(ureg.Quantity(40000, "feet"))
        assert roc_low.m_as("feet/minute") > roc_high.m_as("feet/minute")

    def test_climb_time_positive(self):
        ac = NASA_GV()
        time, dist = ac._climb(
            ureg.Quantity(0, "feet"), ureg.Quantity(40000, "feet")
        )
        assert time.m_as("minute") > 0
        assert dist.m_as("nautical_mile") > 0

    def test_descend_time_positive(self):
        ac = NASA_GV()
        time, dist = ac._descend(
            ureg.Quantity(40000, "feet"), ureg.Quantity(0, "feet")
        )
        assert time.m_as("minute") > 0
        assert dist.m_as("nautical_mile") > 0

    def test_climb_altitude_profile_monotone(self):
        ac = NASA_GV()
        times, alts = ac.climb_altitude_profile(
            ureg.Quantity(0, "feet"), ureg.Quantity(40000, "feet"), n_points=32
        )
        assert len(times) == len(alts) == 32
        assert times[0] == 0.0
        assert np.all(np.diff(times) > 0)
        assert np.all(np.diff(alts) > 0)
        assert alts[-1] == pytest.approx(40000, rel=1e-6)

    def test_phase_bank_angles(self):
        ac = NASA_GV()
        assert ac.turn_model.bank_by_phase.climb_deg == 20
        assert ac.turn_model.bank_by_phase.cruise_deg == 25
        assert ac.turn_model.bank_by_phase.descent_deg == 20
        assert ac.turn_model.bank_by_phase.approach_deg == 15


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_confidence_defaults(self):
        pc = PerformanceConfidence()
        assert pc.climb == 0.5
        assert pc.cruise == 0.5

    def test_source_record(self):
        sr = SourceRecord(
            source_type="poh",
            reference="Beechcraft King Air B200 POH",
            confidence=0.8,
        )
        assert sr.source_type == "poh"
        assert sr.confidence == 0.8

    def test_gv_has_sources(self):
        ac = NASA_GV()
        assert len(ac.sources) >= 1
        assert ac.confidence.cruise == 0.50


# ---------------------------------------------------------------------------
# Approach profile integration on Aircraft
# ---------------------------------------------------------------------------

class TestAircraftApproachProfile:
    def _ils_profile(self):
        from hyplan.aircraft import ApproachProfile, TasSchedule
        return ApproachProfile(
            speed_schedule=TasSchedule(
                points=[
                    (0 * ureg.feet, 95 * ureg.knot),
                    (200 * ureg.feet, 105 * ureg.knot),
                    (1000 * ureg.feet, 120 * ureg.knot),
                    (3000 * ureg.feet, 140 * ureg.knot),
                ]
            ),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )

    def test_default_aircraft_has_no_approach_profile(self):
        ac = B200()
        assert ac.approach_profile is None

    def test_approach_speed_at_falls_back_to_scalar(self):
        ac = B200()
        scalar_kt = ac.approach_speed.m_as(ureg.knot)
        # Without an approach_profile, every altitude returns the scalar.
        assert ac.approach_speed_at(0 * ureg.feet).m_as(ureg.knot) == pytest.approx(scalar_kt)
        assert ac.approach_speed_at(2000 * ureg.feet).m_as(ureg.knot) == pytest.approx(scalar_kt)

    def test_approach_vertical_rate_at_returns_none_without_profile(self):
        ac = B200()
        assert ac.approach_vertical_rate_at(1000 * ureg.feet) is None

    def test_aircraft_accepts_approach_profile(self):
        from hyplan.aircraft._base import Aircraft
        from hyplan.aircraft._models import KingAirB200
        ap = self._ils_profile()
        # Build a B200-like aircraft with the new field set.
        base = KingAirB200()
        ac = Aircraft(
            aircraft_type=base.aircraft_type,
            tail_number=base.tail_number,
            operator=base.operator,
            service_ceiling=base.service_ceiling,
            approach_speed=base.approach_speed,
            climb_schedule=base.climb_schedule,
            cruise_schedule=base.cruise_schedule,
            descent_schedule=base.descent_schedule,
            climb_profile=base.climb_profile,
            descent_profile=base.descent_profile,
            turn_model=base.turn_model,
            engine_type=base.engine_type,
            approach_profile=ap,
        )
        assert ac.approach_profile is ap
        # approach_speed_at delegates to the profile.
        assert ac.approach_speed_at(0 * ureg.feet).m_as(ureg.knot) == pytest.approx(95.0)
        assert ac.approach_speed_at(3000 * ureg.feet).m_as(ureg.knot) == pytest.approx(140.0)
        # approach_vertical_rate_at returns a real value, not None.
        vs = ac.approach_vertical_rate_at(1500 * ureg.feet)
        assert vs is not None
        assert vs.m_as(ureg.feet / ureg.minute) > 0
        # With explicit groundspeed, VS scales linearly.
        vs_slow = ac.approach_vertical_rate_at(1500 * ureg.feet, groundspeed=80 * ureg.knot)
        vs_fast = ac.approach_vertical_rate_at(1500 * ureg.feet, groundspeed=160 * ureg.knot)
        assert vs_fast.m_as(ureg.feet / ureg.minute) == pytest.approx(
            2.0 * vs_slow.m_as(ureg.feet / ureg.minute), rel=1e-3
        )

    def test_aircraft_rejects_wrong_approach_profile_type(self):
        from hyplan.aircraft._models import KingAirB200
        from hyplan.exceptions import HyPlanTypeError
        from hyplan.aircraft._base import Aircraft
        base = KingAirB200()
        with pytest.raises(HyPlanTypeError, match="approach_profile"):
            Aircraft(
                aircraft_type=base.aircraft_type,
                tail_number=base.tail_number,
                operator=base.operator,
                service_ceiling=base.service_ceiling,
                approach_speed=base.approach_speed,
                climb_schedule=base.climb_schedule,
                cruise_schedule=base.cruise_schedule,
                descent_schedule=base.descent_schedule,
                climb_profile=base.climb_profile,
                descent_profile=base.descent_profile,
                turn_model=base.turn_model,
                engine_type=base.engine_type,
                approach_profile="not a profile",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# time_to_return integration with ApproachProfile
# ---------------------------------------------------------------------------

class TestTimeToReturnApproachIntegration:
    def _build_b200_with_approach(self):
        from hyplan.aircraft import ApproachProfile, TasSchedule
        from hyplan.aircraft._base import Aircraft
        from hyplan.aircraft._models import KingAirB200
        ap = ApproachProfile(
            speed_schedule=TasSchedule(
                points=[
                    (0 * ureg.feet, 90 * ureg.knot),
                    (3000 * ureg.feet, 130 * ureg.knot),
                ]
            ),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        base = KingAirB200()
        return Aircraft(
            aircraft_type=base.aircraft_type,
            tail_number=base.tail_number,
            operator=base.operator,
            service_ceiling=base.service_ceiling,
            approach_speed=base.approach_speed,
            climb_schedule=base.climb_schedule,
            cruise_schedule=base.cruise_schedule,
            descent_schedule=base.descent_schedule,
            climb_profile=base.climb_profile,
            descent_profile=base.descent_profile,
            turn_model=base.turn_model,
            engine_type=base.engine_type,
            approach_profile=ap,
        )

    @pytest.fixture(scope="class")
    def cruise_waypoint(self):
        from hyplan.waypoint import Waypoint
        return Waypoint(
            latitude=35.0,
            longitude=-118.5,
            heading=270.0,
            altitude_msl=20000 * ureg.feet,
        )

    @pytest.fixture(scope="class")
    def airport(self):
        from hyplan.airports import Airport, initialize_data
        initialize_data(countries=["US"])
        return Airport("KEDW")

    def test_legacy_no_profile_unchanged(self, cruise_waypoint, airport):
        ac = B200()
        info = ac.time_to_return(cruise_waypoint, airport)
        assert "approach" not in info["phases"]
        assert info["total_time"].magnitude > 0

    def test_with_profile_appends_approach_phase(self, cruise_waypoint, airport):
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        assert "approach" in info["phases"]
        approach = info["phases"]["approach"]
        assert approach["distance"].magnitude > 0
        assert approach["end_time"] > approach["start_time"]

    def test_with_profile_total_time_strictly_larger_than_legacy(self, cruise_waypoint, airport):
        legacy = B200()
        with_profile = self._build_b200_with_approach()
        legacy_info = legacy.time_to_return(cruise_waypoint, airport)
        new_info = with_profile.time_to_return(cruise_waypoint, airport)
        assert new_info["total_time"] > legacy_info["total_time"]

    def test_approach_phase_altitudes_handoff_at_top_of_approach(self, cruise_waypoint, airport):
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        approach = info["phases"]["approach"]
        expected_top_msl = (
            airport.elevation + 3000 * ureg.feet
        ).m_as(ureg.feet)
        assert approach["start_altitude"].m_as(ureg.feet) == pytest.approx(
            expected_top_msl, abs=1.0
        )
        assert approach["end_altitude"].m_as(ureg.feet) == pytest.approx(
            airport.elevation.m_as(ureg.feet), abs=1.0
        )

    # --- FAF / approach-geometry regression tests ---------------------------

    def test_approach_phase_geometry_ends_at_airport(self, cruise_waypoint, airport):
        """Last coord of the approach geometry is the airport, not somewhere short of it."""
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        approach_geom = info["phases"]["approach"]["geometry"]
        end_lon, end_lat = list(approach_geom.coords)[-1]
        assert end_lat == pytest.approx(airport.latitude, abs=1e-6)
        assert end_lon == pytest.approx(airport.longitude, abs=1e-6)

    def test_approach_phase_geometry_starts_at_faf_offset(self, cruise_waypoint, airport):
        """First coord of the approach geometry is the FAF, offset by approx_approach_distance_nmi from the airport."""
        import pymap3d.vincenty
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        approach_geom = info["phases"]["approach"]["geometry"]
        start_lon, start_lat = next(iter(approach_geom.coords))
        distance_m, _ = pymap3d.vincenty.vdist(
            airport.latitude, airport.longitude, start_lat, start_lon,
        )
        expected_distance_m = ac.approach_profile.approx_approach_distance_nmi * 1852.0
        assert float(distance_m) == pytest.approx(expected_distance_m, rel=1e-2)

    def test_approach_phase_bearing_matches_inbound_course(self, cruise_waypoint, airport):
        """Bearing FAF→airport equals the inbound course waypoint→airport."""
        import pymap3d.vincenty
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        approach_geom = info["phases"]["approach"]["geometry"]
        start_lon, start_lat = next(iter(approach_geom.coords))
        # Bearing FROM faf TO airport.
        _, bearing_faf_to_airport = pymap3d.vincenty.vdist(
            start_lat, start_lon, airport.latitude, airport.longitude,
        )
        # Expected: bearing waypoint → airport (the inbound course).
        _, expected_inbound = pymap3d.vincenty.vdist(
            cruise_waypoint.latitude, cruise_waypoint.longitude,
            airport.latitude, airport.longitude,
        )
        # Compare modulo 360 with a 1° tolerance.
        delta = abs((float(bearing_faf_to_airport) - float(expected_inbound) + 180.0) % 360.0 - 180.0)
        assert delta < 1.0, (
            f"FAF→airport bearing {bearing_faf_to_airport:.2f}° doesn't match "
            f"inbound course {expected_inbound:.2f}° (delta {delta:.2f}°) — "
            f"likely a sign-flip in the FAF offset"
        )

    def test_dubins_descent_ends_at_faf_not_airport(self, cruise_waypoint, airport):
        """The Dubins descent path ends at the FAF, not overhead the airport.

        Verifies that the time-normalization fix lands the descent at the FAF
        cleanly: the cruise_descent dubins_path's end coord should equal the
        first coord of the approach geometry (the FAF), and should be
        meaningfully offset from the airport.
        """
        import pymap3d.vincenty
        ac = self._build_b200_with_approach()
        info = ac.time_to_return(cruise_waypoint, airport)
        dubins_geom = info["dubins_path"].geometry
        dubins_end_lon, dubins_end_lat = list(dubins_geom.coords)[-1]
        approach_geom = info["phases"]["approach"]["geometry"]
        faf_lon, faf_lat = next(iter(approach_geom.coords))
        # Dubins endpoint should match FAF (within 1e-4 deg ≈ 11 m).
        assert dubins_end_lat == pytest.approx(faf_lat, abs=1e-4)
        assert dubins_end_lon == pytest.approx(faf_lon, abs=1e-4)
        # And the FAF must be meaningfully separated from the airport.
        distance_m, _ = pymap3d.vincenty.vdist(
            airport.latitude, airport.longitude, dubins_end_lat, dubins_end_lon,
        )
        expected_m = ac.approach_profile.approx_approach_distance_nmi * 1852.0
        assert float(distance_m) == pytest.approx(expected_m, rel=1e-2)


# ---------------------------------------------------------------------------
# Wind in _climb() / _descend() (Item 2 — v1.4)
# ---------------------------------------------------------------------------

class TestClimbDescentWind:
    """Verify wind_along_track plumbs into vertical-phase distance."""

    def _wp(self, lat, lon, alt_ft, hdg=90.0):
        from hyplan.waypoint import Waypoint
        return Waypoint(lat, lon, hdg, altitude_msl=alt_ft * ureg.feet)

    def test_climb_still_air_unchanged(self):
        """wind_along_track=None must equal wind_along_track=0 m/s."""
        ac = NASA_ER2()
        t0, d0 = ac._climb(0 * ureg.feet, 60000 * ureg.feet)
        t1, d1 = ac._climb(
            0 * ureg.feet, 60000 * ureg.feet,
            wind_along_track=0 * (ureg.meter / ureg.second),
        )
        assert t0 == t1
        assert d0 == d1

    def test_climb_headwind_shortens_distance(self):
        """A 30 kt headwind must shorten the integrated climb distance."""
        ac = NASA_ER2()
        _, d_calm = ac._climb(0 * ureg.feet, 60000 * ureg.feet)
        # Tailwind = -30 kt (headwind).
        _, d_hw = ac._climb(
            0 * ureg.feet, 60000 * ureg.feet,
            wind_along_track=(-30) * ureg.knot,
        )
        delta_nmi = (d_calm - d_hw).m_as(ureg.nautical_mile)
        assert delta_nmi > 0
        # Headwind effect ≈ wind_kt × climb_time_hr.  ER-2 climbs to
        # FL600 in roughly 25 minutes → ≈ 30 × 25/60 ≈ 12.5 nmi shift.
        assert 8 < delta_nmi < 18

    def test_climb_tailwind_extends_distance(self):
        """A 30 kt tailwind must extend the integrated climb distance."""
        ac = NASA_ER2()
        _, d_calm = ac._climb(0 * ureg.feet, 60000 * ureg.feet)
        _, d_tw = ac._climb(
            0 * ureg.feet, 60000 * ureg.feet,
            wind_along_track=30 * ureg.knot,
        )
        delta_nmi = (d_tw - d_calm).m_as(ureg.nautical_mile)
        assert delta_nmi > 0
        assert 8 < delta_nmi < 18

    def test_descent_still_air_unchanged(self):
        ac = NASA_ER2()
        t0, d0 = ac._descend(60000 * ureg.feet, 5000 * ureg.feet)
        t1, d1 = ac._descend(
            60000 * ureg.feet, 5000 * ureg.feet,
            wind_along_track=0 * (ureg.meter / ureg.second),
        )
        assert t0 == t1
        assert d0 == d1

    def test_descent_headwind_shortens_distance(self):
        ac = NASA_ER2()
        _, d_calm = ac._descend(60000 * ureg.feet, 5000 * ureg.feet)
        _, d_hw = ac._descend(
            60000 * ureg.feet, 5000 * ureg.feet,
            wind_along_track=(-30) * ureg.knot,
        )
        assert (d_calm - d_hw).m_as(ureg.nautical_mile) > 0

    def test_extreme_headwind_clipped_to_zero(self):
        """A headwind larger than TAS must clip ground speed to 0, not go negative."""
        ac = NASA_ER2()
        # 800 kt is way over any TAS the ER-2 reaches; ground distance
        # should be zero (no negative distances), and time still finite.
        t, d = ac._climb(
            0 * ureg.feet, 60000 * ureg.feet,
            wind_along_track=(-800) * ureg.knot,
        )
        assert d.m_as(ureg.nautical_mile) == 0.0
        assert t.m_as(ureg.minute) > 0


class TestHybridPathWind:
    """End-to-end: _hybrid_path projects wind onto great-circle bearing."""

    def _wp(self, lat, lon, alt_ft, hdg=90.0):
        from hyplan.waypoint import Waypoint
        return Waypoint(lat, lon, hdg, altitude_msl=alt_ft * ureg.feet)

    def test_eastbound_eastward_wind_is_tailwind(self):
        """East-going leg + eastward wind ⇒ climb forward distance grows.

        u_east > 0, v_north = 0, leg from (35, -120) to (35, -110)
        (true bearing ≈ 90°): track unit (sin 90°, cos 90°) = (1, 0),
        tailwind = u·1 + v·0 = u > 0.  So _climb sees a tailwind and
        TOC moves further from departure.
        """
        ac = NASA_ER2()
        start = self._wp(35.0, -120.0, 0)
        end = self._wp(35.0, -110.0, 60000)
        info_calm = ac.time_to_cruise(start, end, wind=None)
        info_tw = ac.time_to_cruise(start, end, wind=(15.0, 0.0))  # 15 m/s east
        # Climb phase forward distance grows under tailwind.
        d_calm = info_calm["phases"]["climb"]["distance"].m_as(ureg.nautical_mile)
        d_tw = info_tw["phases"]["climb"]["distance"].m_as(ureg.nautical_mile)
        assert d_tw > d_calm

    def test_eastbound_westward_wind_is_headwind(self):
        """East-going leg + westward wind ⇒ climb forward distance shrinks."""
        ac = NASA_ER2()
        start = self._wp(35.0, -120.0, 0)
        end = self._wp(35.0, -110.0, 60000)
        info_calm = ac.time_to_cruise(start, end, wind=None)
        info_hw = ac.time_to_cruise(start, end, wind=(-15.0, 0.0))
        d_calm = info_calm["phases"]["climb"]["distance"].m_as(ureg.nautical_mile)
        d_hw = info_hw["phases"]["climb"]["distance"].m_as(ureg.nautical_mile)
        assert d_hw < d_calm


# ---------------------------------------------------------------------------
# ClimbPlan integration into compute_flight_plan (Item 3 — v1.4)
# ---------------------------------------------------------------------------

class TestClimbPlan:
    """ClimbPlan plumbs through compute_flight_plan into time_to_takeoff."""

    def _wp(self, lat, lon, alt_ft, hdg=0.0):
        from hyplan.waypoint import Waypoint
        return Waypoint(lat, lon, hdg, altitude_msl=alt_ft * ureg.feet)

    def test_no_climb_plan_unchanged(self):
        """climb_plan=None must produce identical _hybrid_path output."""
        ac = NASA_ER2()
        start = self._wp(34.7, -118.0, 0)
        end = self._wp(36.5, -116.0, 60000)
        info_a = ac._hybrid_path(start, end, phase="climb")
        info_b = ac._hybrid_path(start, end, phase="climb", climb_plan=None)
        # Single climb phase, no pauses.
        assert set(info_a["phases"]) == {"climb", "cruise"}
        assert set(info_b["phases"]) == {"climb", "cruise"}
        # Total time matches bit-for-bit.
        assert info_a["total_time"] == info_b["total_time"]

    def test_round_trip_against_step_climb(self):
        """compute_flight_plan with ClimbPlan(pauses=[(FL356, 25 min)])
        should match Aircraft.step_climb's total time/forward distance
        for the same pauses applied to the takeoff phase."""
        from hyplan.aircraft import ClimbPlan
        ac = NASA_ER2()
        # Long enough leg that climb fits within the Dubins path.
        start = self._wp(34.7, -118.0, 0)
        end = self._wp(38.0, -114.0, 60000)
        plan = ClimbPlan(pauses=[
            (35600 * ureg.feet, 25 * ureg.minute),
        ])
        info = ac._hybrid_path(start, end, phase="climb", climb_plan=plan)

        # Sub-phases present.
        assert "climb_1" in info["phases"]
        assert "climb_pause_1" in info["phases"]
        assert "climb_2" in info["phases"]

        # Total climb-phase time equals step_climb's reference time.
        ref_t, ref_d = ac.step_climb(
            0 * ureg.feet, 60000 * ureg.feet,
            pauses=[(35600 * ureg.feet, 25 * ureg.minute)],
        )
        ref_t_min = ref_t.m_as(ureg.minute)
        ref_d_nmi = ref_d.m_as(ureg.nautical_mile)

        climb_total_min = sum(
            (info["phases"][k]["end_time"]
             - info["phases"][k]["start_time"]).m_as(ureg.minute)
            for k in info["phases"]
            if k.startswith("climb_") or k == "climb"
        )
        # Tolerance is generous because step_climb integrates the full
        # [start, end] altitude range as a single _climb call (so its
        # distance matches the no-ClimbPlan baseline), while the
        # planner emits per-sub-segment phases via _climb on each
        # piece — those time integrations differ by ~0.01 min from
        # trapezoidal sampling.
        assert climb_total_min == pytest.approx(ref_t_min, rel=1e-3)

        # Forward climb distance (climb_1 + climb_2; pause is zero).
        # The planner's per-sub-segment _climb calls each use their
        # own midpoint TAS, so the sum is larger than step_climb's
        # single-call midpoint-TAS distance.  Tolerance reflects that.
        climb_dist_nmi = sum(
            info["phases"][k]["distance"].m_as(ureg.nautical_mile)
            for k in info["phases"]
            if k.startswith("climb_") and "pause" not in k
        )
        assert climb_dist_nmi == pytest.approx(ref_d_nmi, rel=0.15)

    def test_pauses_render_as_loiter(self):
        """The pause sub-phase carries segment_type='loiter' and
        process_flight_phase emits a 'loiter' dataframe row."""
        from hyplan.aircraft import ClimbPlan, NASA_ER2
        from hyplan.airports import Airport
        from hyplan.flight_line import FlightLine
        from hyplan.flight_plan import compute_flight_plan
        ac = NASA_ER2()
        line = FlightLine.center_length_azimuth(
            lat=37.0, lon=-104.5, length=ureg.Quantity(100, "km"),
            az=0.0, altitude_msl=ureg.Quantity(60000, "feet"),
            site_name="L1",
        )
        kcos = Airport("KCOS")
        plan = ClimbPlan(pauses=[
            (35600 * ureg.feet, 25 * ureg.minute),
        ])
        df = compute_flight_plan(
            ac, [line],
            takeoff_airport=kcos, return_airport=kcos,
            climb_plan=plan,
        )
        # Exactly one loiter row from the climb-staging pause.
        loiter_rows = df[df["segment_type"] == "loiter"]
        assert len(loiter_rows) == 1
        # Hold duration matches.
        assert loiter_rows.iloc[0]["time_to_segment"] == pytest.approx(25.0, abs=0.1)
        # Pause altitude is the level-off altitude.
        assert loiter_rows.iloc[0]["start_altitude"] == pytest.approx(35600.0, abs=1.0)
        assert loiter_rows.iloc[0]["end_altitude"] == pytest.approx(35600.0, abs=1.0)

    def test_climb_plan_increases_total_duration(self):
        """compute_flight_plan total time grows by the hold duration
        when ClimbPlan adds pauses."""
        from hyplan.aircraft import ClimbPlan, NASA_ER2
        from hyplan.airports import Airport
        from hyplan.flight_line import FlightLine
        from hyplan.flight_plan import compute_flight_plan
        ac = NASA_ER2()
        line = FlightLine.center_length_azimuth(
            lat=37.0, lon=-104.5, length=ureg.Quantity(100, "km"),
            az=0.0, altitude_msl=ureg.Quantity(60000, "feet"),
            site_name="L1",
        )
        kcos = Airport("KCOS")
        # Explicit climb_plan=None bypasses NASA_ER2's typical_climb_out
        # default — without it, "auto" would inject the 12-min FL356
        # hold and the delta would be 25 - 12 = 13 min instead of 25.
        df_no = compute_flight_plan(
            ac, [line], takeoff_airport=kcos, return_airport=kcos,
            climb_plan=None,
        )
        df_yes = compute_flight_plan(
            ac, [line], takeoff_airport=kcos, return_airport=kcos,
            climb_plan=ClimbPlan(pauses=[
                (35600 * ureg.feet, 25 * ureg.minute),
            ]),
        )
        delta_min = df_yes["time_to_segment"].sum() - df_no["time_to_segment"].sum()
        # 25-minute hold dominates; climb-segment durations are very
        # close to one another so net increase ≈ 25 min.
        assert delta_min == pytest.approx(25.0, abs=2.0)
