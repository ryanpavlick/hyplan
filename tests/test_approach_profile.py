"""Tests for hyplan.aircraft.ApproachProfile."""

import math

import pytest

from hyplan.aircraft import ApproachProfile, TasSchedule
from hyplan.exceptions import HyPlanValueError
from hyplan.units import ureg

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ils_schedule_3000ft() -> TasSchedule:
    """A typical 4-point speed schedule from touchdown to a 3000 ft AGL IAF."""
    return TasSchedule(
        points=[
            (0 * ureg.feet, 95 * ureg.knot),
            (200 * ureg.feet, 105 * ureg.knot),
            (1000 * ureg.feet, 120 * ureg.knot),
            (3000 * ureg.feet, 140 * ureg.knot),
        ]
    )


def _flat_schedule(top_alt_ft: float, speed_kt: float) -> TasSchedule:
    """A constant-speed schedule from 0 ft to top_alt_ft."""
    return TasSchedule(
        points=[
            (0 * ureg.feet, speed_kt * ureg.knot),
            (top_alt_ft * ureg.feet, speed_kt * ureg.knot),
        ]
    )


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructs_with_defaults(self):
        ap = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
        )
        assert ap.glideslope_deg == 3.0
        assert ap.top_of_approach_agl.m_as(ureg.feet) == pytest.approx(3000.0)

    def test_rejects_zero_top_altitude(self):
        with pytest.raises(HyPlanValueError, match="strictly positive"):
            ApproachProfile(
                speed_schedule=_ils_schedule_3000ft(),
                top_of_approach_agl=0 * ureg.feet,
            )

    def test_rejects_negative_top_altitude(self):
        with pytest.raises(HyPlanValueError, match="strictly positive"):
            ApproachProfile(
                speed_schedule=_ils_schedule_3000ft(),
                top_of_approach_agl=-100 * ureg.feet,
            )

    def test_rejects_top_altitude_mismatch_with_schedule(self):
        # Schedule maxes at 3000 ft but caller declares top_of_approach=5000 ft.
        with pytest.raises(HyPlanValueError, match="must equal top_of_approach_agl"):
            ApproachProfile(
                speed_schedule=_ils_schedule_3000ft(),
                top_of_approach_agl=5000 * ureg.feet,
            )

    def test_accepts_within_one_foot_tolerance(self):
        # Floats from unit-conversion roundtrips can drift sub-foot.
        ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=(3000.0 + 0.4) * ureg.feet,
        )

    def test_rejects_glideslope_zero(self):
        with pytest.raises(HyPlanValueError, match="glideslope_deg"):
            ApproachProfile(
                speed_schedule=_ils_schedule_3000ft(),
                top_of_approach_agl=3000 * ureg.feet,
                glideslope_deg=0.0,
            )

    def test_rejects_glideslope_at_or_above_90(self):
        with pytest.raises(HyPlanValueError, match="glideslope_deg"):
            ApproachProfile(
                speed_schedule=_ils_schedule_3000ft(),
                top_of_approach_agl=3000 * ureg.feet,
                glideslope_deg=90.0,
            )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestTouchdownSpeed:
    def test_returns_schedule_tas_at_zero(self):
        ap = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
        )
        assert ap.touchdown_speed.m_as(ureg.knot) == pytest.approx(95.0)


class TestApproxApproachDistance:
    def test_3deg_from_3000ft_is_about_94nmi(self):
        # 3000 ft / tan(3°) = 57228 ft = ~9.42 nmi
        ap = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        assert ap.approx_approach_distance_nmi == pytest.approx(9.42, abs=0.05)

    def test_steeper_glideslope_gives_shorter_distance(self):
        ap_shallow = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=2.5,
        )
        ap_steep = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=4.0,
        )
        assert ap_steep.approx_approach_distance_nmi < ap_shallow.approx_approach_distance_nmi

    def test_distance_scales_linearly_with_top_altitude(self):
        ap_3k = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 130),
            top_of_approach_agl=3000 * ureg.feet,
        )
        ap_6k = ApproachProfile(
            speed_schedule=_flat_schedule(6000, 130),
            top_of_approach_agl=6000 * ureg.feet,
        )
        ratio = ap_6k.approx_approach_distance_nmi / ap_3k.approx_approach_distance_nmi
        assert ratio == pytest.approx(2.0, abs=0.001)


# ---------------------------------------------------------------------------
# tas_at + approx_vertical_rate_at
# ---------------------------------------------------------------------------


class TestTasAt:
    def test_delegates_to_schedule(self):
        ap = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
        )
        assert ap.tas_at(0 * ureg.feet).m_as(ureg.knot) == pytest.approx(95.0)
        assert ap.tas_at(1000 * ureg.feet).m_as(ureg.knot) == pytest.approx(120.0)
        assert ap.tas_at(3000 * ureg.feet).m_as(ureg.knot) == pytest.approx(140.0)


class TestApproxVerticalRateAt:
    def test_default_uses_scheduled_tas(self):
        ap = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 120),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        # VS = 120 kt × tan(3°) = 120 × 6076.115 / 60 × tan(3°) fpm
        expected_fpm = 120 * 6076.115485564 / 60 * math.tan(math.radians(3.0))
        assert ap.approx_vertical_rate_at(1500 * ureg.feet).m_as(
            ureg.feet / ureg.minute
        ) == pytest.approx(expected_fpm, rel=1e-4)

    def test_explicit_groundspeed_overrides_tas(self):
        ap = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 120),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        gs = 100 * ureg.knot  # tailwind-equivalent slower-over-ground speed
        expected_fpm = 100 * 6076.115485564 / 60 * math.tan(math.radians(3.0))
        actual_fpm = ap.approx_vertical_rate_at(
            1500 * ureg.feet, groundspeed=gs
        ).m_as(ureg.feet / ureg.minute)
        assert actual_fpm == pytest.approx(expected_fpm, rel=1e-4)

    def test_steeper_glideslope_gives_higher_vs(self):
        ap_a = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 120),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=2.5,
        )
        ap_b = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 120),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=4.0,
        )
        vs_a = ap_a.approx_vertical_rate_at(1500 * ureg.feet)
        vs_b = ap_b.approx_vertical_rate_at(1500 * ureg.feet)
        assert vs_b > vs_a


# ---------------------------------------------------------------------------
# time_to_touchdown
# ---------------------------------------------------------------------------


class TestTimeToTouchdown:
    def test_constant_speed_matches_closed_form(self):
        # For constant speed v at glideslope g, time = h_top / (v × sin(g)) where
        # h_top = top altitude. With our convention VS = v × tan(g), that's
        # time = h_top / (v × tan(g)). Let's verify.
        ap = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 120),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        # 120 kt = 12152.23 fpm; VS = 12152.23 × tan(3°) = 637.1 fpm
        # time = 3000 ft / 637.1 fpm = 4.71 min
        v_fpm = 120 * 6076.115485564 / 60
        vs_fpm = v_fpm * math.tan(math.radians(3.0))
        expected_min = 3000.0 / vs_fpm
        actual_min = ap.time_to_touchdown().m_as(ureg.minute)
        assert actual_min == pytest.approx(expected_min, rel=1e-3)

    def test_explicit_groundspeed_constant(self):
        ap = ApproachProfile(
            speed_schedule=_flat_schedule(3000, 200),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        gs = 100 * ureg.knot
        gs_fpm = 100 * 6076.115485564 / 60
        vs_fpm = gs_fpm * math.tan(math.radians(3.0))
        expected_min = 3000.0 / vs_fpm
        actual_min = ap.time_to_touchdown(groundspeed=gs).m_as(ureg.minute)
        assert actual_min == pytest.approx(expected_min, rel=1e-3)

    def test_real_ils_approach_is_about_ten_minutes(self):
        # The 4-point ER-2-style schedule from 95 kt at touchdown to 140 kt at IAF
        # on a 3° glideslope should take roughly 10-12 minutes.
        ap = ApproachProfile(
            speed_schedule=_ils_schedule_3000ft(),
            top_of_approach_agl=3000 * ureg.feet,
            glideslope_deg=3.0,
        )
        t = ap.time_to_touchdown().m_as(ureg.minute)
        assert 4.0 < t < 6.5  # actually closer to 4.7 with these speeds — see below

    def test_steeper_glideslope_gives_shorter_time(self):
        sched = _flat_schedule(3000, 120)
        slow = ApproachProfile(
            speed_schedule=sched, top_of_approach_agl=3000 * ureg.feet, glideslope_deg=2.5
        )
        steep = ApproachProfile(
            speed_schedule=sched, top_of_approach_agl=3000 * ureg.feet, glideslope_deg=4.0
        )
        assert steep.time_to_touchdown() < slow.time_to_touchdown()
