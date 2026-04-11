"""Tests for the ISA atmosphere model and airspeed conversions."""

import numpy as np
import pytest

from hyplan.atmosphere import (
    temperature_at,
    pressure_at,
    density_at,
    speed_of_sound,
    cas_to_tas,
    tas_to_cas,
    mach_to_tas,
    tas_to_mach,
)
from hyplan.units import ureg


# ---------------------------------------------------------------------------
# ISA standard values (1976 US Standard Atmosphere / ICAO ISA)
# Reference: ICAO Doc 7488, NASA TN-D-7488
# ---------------------------------------------------------------------------

class TestISAProfile:
    """Validate ISA temperature, pressure, density against standard tables."""

    def test_sea_level_temperature(self):
        t = temperature_at(0 * ureg.feet)
        assert abs(t.m_as(ureg.kelvin) - 288.15) < 0.01

    def test_sea_level_pressure(self):
        p = pressure_at(0 * ureg.feet)
        assert abs(p.m_as(ureg.pascal) - 101325.0) < 1.0

    def test_sea_level_density(self):
        rho = density_at(0 * ureg.feet)
        assert abs(rho.m_as(ureg.kilogram / ureg.meter**3) - 1.225) < 0.001

    def test_sea_level_speed_of_sound(self):
        a = speed_of_sound(0 * ureg.feet)
        # a₀ ≈ 340.29 m/s ≈ 661.5 kt
        assert abs(a.m_as(ureg.knot) - 661.5) < 0.5

    def test_fl350_temperature(self):
        """FL350 = 35,000 ft = 10,668 m → T ≈ 218.81 K."""
        t = temperature_at(35000 * ureg.feet)
        assert abs(t.m_as(ureg.kelvin) - 218.81) < 0.5

    def test_fl350_pressure(self):
        """FL350 → P ≈ 23,842 Pa (ICAO table)."""
        p = pressure_at(35000 * ureg.feet)
        assert abs(p.m_as(ureg.pascal) - 23842) < 100

    def test_fl350_density(self):
        """FL350 → ρ ≈ 0.3796 kg/m³."""
        rho = density_at(35000 * ureg.feet)
        assert abs(rho.m_as(ureg.kilogram / ureg.meter**3) - 0.3796) < 0.01

    def test_tropopause_temperature(self):
        """At 36,089 ft (11,000 m) → T = 216.65 K exactly."""
        t = temperature_at(11000 * ureg.meter)
        assert abs(t.m_as(ureg.kelvin) - 216.65) < 0.01

    def test_above_tropopause_isothermal(self):
        """Temperature is constant (216.65 K) above the tropopause."""
        t40 = temperature_at(40000 * ureg.feet)
        t50 = temperature_at(50000 * ureg.feet)
        assert abs(t40.m_as(ureg.kelvin) - 216.65) < 0.01
        assert abs(t50.m_as(ureg.kelvin) - 216.65) < 0.01

    def test_fl450_pressure(self):
        """FL450 = 45,000 ft = 13,716 m → P ≈ 14,748 Pa."""
        p = pressure_at(45000 * ureg.feet)
        assert abs(p.m_as(ureg.pascal) - 14748) < 200

    def test_pressure_decreases_with_altitude(self):
        p0 = pressure_at(0 * ureg.feet).m_as(ureg.pascal)
        p1 = pressure_at(10000 * ureg.feet).m_as(ureg.pascal)
        p2 = pressure_at(30000 * ureg.feet).m_as(ureg.pascal)
        p3 = pressure_at(50000 * ureg.feet).m_as(ureg.pascal)
        assert p0 > p1 > p2 > p3

    def test_density_decreases_with_altitude(self):
        rho0 = density_at(0 * ureg.feet).magnitude
        rho1 = density_at(20000 * ureg.feet).magnitude
        rho2 = density_at(40000 * ureg.feet).magnitude
        assert rho0 > rho1 > rho2


# ---------------------------------------------------------------------------
# Airspeed conversions
# ---------------------------------------------------------------------------

class TestCasToTas:
    """Validate CAS↔TAS conversion against known values."""

    def test_sea_level_cas_equals_tas(self):
        """At sea level, CAS ≈ TAS (standard conditions)."""
        cas = 250 * ureg.knot
        tas = cas_to_tas(cas, 0 * ureg.feet)
        assert abs(tas.m_as(ureg.knot) - 250.0) < 0.5

    def test_fl350_250kcas(self):
        """250 KCAS at FL350 ≈ 430 KTAS (well-known reference value)."""
        cas = 250 * ureg.knot
        tas = cas_to_tas(cas, 35000 * ureg.feet)
        assert abs(tas.m_as(ureg.knot) - 430) < 10

    def test_tas_increases_with_altitude_for_same_cas(self):
        """For a given CAS, TAS increases with altitude (lower density)."""
        cas = 250 * ureg.knot
        tas_low = cas_to_tas(cas, 5000 * ureg.feet).m_as(ureg.knot)
        tas_high = cas_to_tas(cas, 35000 * ureg.feet).m_as(ureg.knot)
        assert tas_high > tas_low

    def test_round_trip_cas_tas(self):
        """cas_to_tas → tas_to_cas should recover original CAS."""
        cas_orig = 280 * ureg.knot
        alt = 25000 * ureg.feet
        tas = cas_to_tas(cas_orig, alt)
        cas_recovered = tas_to_cas(tas, alt)
        assert abs(cas_recovered.m_as(ureg.knot) - 280.0) < 0.01

    def test_round_trip_at_multiple_altitudes(self):
        """Round-trip at several altitudes."""
        cas_orig = 300 * ureg.knot
        for alt_ft in [0, 10000, 25000, 35000, 45000]:
            alt = alt_ft * ureg.feet
            tas = cas_to_tas(cas_orig, alt)
            cas_back = tas_to_cas(tas, alt)
            assert abs(cas_back.m_as(ureg.knot) - 300.0) < 0.01, (
                f"Round-trip failed at FL{alt_ft // 100}"
            )


class TestMachConversions:
    """Validate Mach↔TAS conversions."""

    def test_mach_1_at_sea_level(self):
        """Mach 1.0 at sea level ≈ 661.5 kt."""
        tas = mach_to_tas(1.0, 0 * ureg.feet)
        assert abs(tas.m_as(ureg.knot) - 661.5) < 0.5

    def test_mach_080_at_fl350(self):
        """Mach 0.80 at FL350 ≈ 461 KTAS."""
        tas = mach_to_tas(0.80, 35000 * ureg.feet)
        # a at FL350 ≈ 576.4 kt → M0.80 ≈ 461 kt
        assert abs(tas.m_as(ureg.knot) - 461) < 5

    def test_mach_080_at_fl450(self):
        """Mach 0.80 at FL450 → same TAS as FL350 (isothermal)."""
        tas_350 = mach_to_tas(0.80, 35000 * ureg.feet)
        tas_450 = mach_to_tas(0.80, 45000 * ureg.feet)
        # Above tropopause, temperature is constant, so speed of sound
        # and therefore TAS at constant Mach should be nearly identical.
        # FL350 is just below tropopause, FL450 is above, so slight difference.
        assert abs(tas_450.m_as(ureg.knot) - tas_350.m_as(ureg.knot)) < 5

    def test_round_trip_mach_tas(self):
        """mach_to_tas → tas_to_mach should recover original Mach."""
        mach_orig = 0.82
        alt = 39000 * ureg.feet
        tas = mach_to_tas(mach_orig, alt)
        mach_recovered = tas_to_mach(tas, alt)
        assert abs(mach_recovered - 0.82) < 0.001

    def test_tas_to_mach_at_sea_level(self):
        """300 KTAS at sea level → M ≈ 0.454."""
        mach = tas_to_mach(300 * ureg.knot, 0 * ureg.feet)
        assert abs(mach - 300 / 661.5) < 0.01


class TestCrossoverConsistency:
    """Verify CAS and Mach schedules agree at crossover altitude."""

    def test_crossover_point(self):
        """At crossover altitude, CAS and Mach should give similar TAS."""
        # Typical jet crossover: 280 KCAS / M0.78 ≈ FL280
        cas = 280 * ureg.knot
        alt = 28000 * ureg.feet

        tas_from_cas = cas_to_tas(cas, alt).m_as(ureg.knot)
        mach_at_crossover = tas_to_mach(
            cas_to_tas(cas, alt), alt
        )
        tas_from_mach = mach_to_tas(mach_at_crossover, alt).m_as(ureg.knot)

        assert abs(tas_from_cas - tas_from_mach) < 0.1
