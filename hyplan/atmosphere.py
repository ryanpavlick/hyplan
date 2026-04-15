"""International Standard Atmosphere (ISA) model and airspeed conversions.

Provides the 1976 US Standard Atmosphere (identical to ICAO ISA below 32 km)
and compressible-flow airspeed conversions accurate through FL510+.

All public functions accept and return :class:`pint.Quantity` objects.
"""

from __future__ import annotations

import numpy as np
from pint import Quantity

from .units import ureg

# ---------------------------------------------------------------------------
# ISA constants
# ---------------------------------------------------------------------------
_T0 = 288.15  # sea-level temperature [K]
_P0 = 101_325.0  # sea-level pressure [Pa]
_RHO0 = 1.225  # sea-level density [kg/m³]
_L = -0.0065  # troposphere lapse rate [K/m]
_G = 9.80665  # gravitational acceleration [m/s²]
_R = 287.05287  # specific gas constant for dry air [J/(kg·K)]
_GAMMA = 1.4  # ratio of specific heats for air
_TROPOPAUSE_M = 11_000.0  # tropopause altitude [m]
_T_TROPOPAUSE = _T0 + _L * _TROPOPAUSE_M  # 216.65 K
_P_TROPOPAUSE = _P0 * (_T_TROPOPAUSE / _T0) ** (-_G / (_L * _R))
_A0 = np.sqrt(_GAMMA * _R * _T0)  # sea-level speed of sound [m/s] ≈ 340.29


# ---------------------------------------------------------------------------
# ISA profile functions (internal, operate on floats in SI)
# ---------------------------------------------------------------------------

def _temperature_k(altitude_m: float) -> float:
    """ISA temperature in Kelvin at geometric altitude in metres."""
    if altitude_m <= _TROPOPAUSE_M:
        return _T0 + _L * altitude_m
    return _T_TROPOPAUSE  # isothermal above tropopause


def _pressure_pa(altitude_m: float) -> float:
    """ISA pressure in Pascals at geometric altitude in metres."""
    if altitude_m <= _TROPOPAUSE_M:
        t = _temperature_k(altitude_m)
        return _P0 * (t / _T0) ** (-_G / (_L * _R))  # type: ignore[no-any-return]
    # Isothermal layer above tropopause
    return _P_TROPOPAUSE * np.exp(  # type: ignore[no-any-return]
        -_G / (_R * _T_TROPOPAUSE) * (altitude_m - _TROPOPAUSE_M)
    )


def _density_kgm3(altitude_m: float) -> float:
    """ISA air density in kg/m³ at geometric altitude in metres."""
    return _pressure_pa(altitude_m) / (_R * _temperature_k(altitude_m))


def _speed_of_sound_ms(altitude_m: float) -> float:
    """Speed of sound in m/s at geometric altitude in metres."""
    return np.sqrt(_GAMMA * _R * _temperature_k(altitude_m))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Public ISA functions (Quantity in, Quantity out)
# ---------------------------------------------------------------------------

def temperature_at(altitude: Quantity) -> Quantity:
    """ISA static temperature at *altitude*.

    Covers the troposphere (lapse rate −6.5 K/km) and the lower
    stratosphere (isothermal at 216.65 K above 11 km).
    """
    alt_m = altitude.m_as(ureg.meter)
    return _temperature_k(alt_m) * ureg.kelvin  # type: ignore[no-any-return]


def pressure_at(altitude: Quantity) -> Quantity:
    """ISA static pressure at *altitude*."""
    alt_m = altitude.m_as(ureg.meter)
    return _pressure_pa(alt_m) * ureg.pascal  # type: ignore[no-any-return]


def density_at(altitude: Quantity) -> Quantity:
    """ISA air density at *altitude*."""
    alt_m = altitude.m_as(ureg.meter)
    return _density_kgm3(alt_m) * (ureg.kilogram / ureg.meter**3)  # type: ignore[no-any-return]


def speed_of_sound(altitude: Quantity) -> Quantity:
    """Speed of sound at *altitude* from ISA temperature."""
    alt_m = altitude.m_as(ureg.meter)
    return _speed_of_sound_ms(alt_m) * (ureg.meter / ureg.second)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Airspeed conversions (compressible, subsonic, ICAO standard)
# ---------------------------------------------------------------------------
#
# The conversions use the isentropic compressible-flow equations:
#
#   qc = P₀ · ((1 + 0.2·(CAS/a₀)²)^3.5 − 1)     impact pressure from CAS
#   M  = √(5·((qc/P + 1)^(2/7) − 1))              Mach from qc and static P
#   TAS = M · a                                      TAS from Mach and local a
#
# These are exact for subsonic isentropic flow and accurate through FL510+.

def cas_to_tas(cas: Quantity, altitude: Quantity) -> Quantity:
    """Convert calibrated airspeed to true airspeed at *altitude* (ISA).

    Uses the compressible (isentropic) form, not the incompressible
    approximation.  Accurate for subsonic flight through FL510+.
    """
    cas_ms = cas.m_as(ureg.meter / ureg.second)
    alt_m = altitude.m_as(ureg.meter)

    p = _pressure_pa(alt_m)
    a = _speed_of_sound_ms(alt_m)

    # Impact pressure from CAS (using sea-level conditions)
    qc = _P0 * ((1.0 + 0.2 * (cas_ms / _A0) ** 2) ** 3.5 - 1.0)

    # Mach number from impact pressure and local static pressure
    mach = np.sqrt(5.0 * ((qc / p + 1.0) ** (2.0 / 7.0) - 1.0))

    tas_ms = mach * a
    return (tas_ms * ureg.meter / ureg.second).to(cas.units)  # type: ignore[no-any-return]


def tas_to_cas(tas: Quantity, altitude: Quantity) -> Quantity:
    """Convert true airspeed to calibrated airspeed at *altitude* (ISA).

    Inverse of :func:`cas_to_tas`.
    """
    tas_ms = tas.m_as(ureg.meter / ureg.second)
    alt_m = altitude.m_as(ureg.meter)

    p = _pressure_pa(alt_m)
    a = _speed_of_sound_ms(alt_m)

    # Mach from TAS and local speed of sound
    mach = tas_ms / a

    # Impact pressure from Mach and local static pressure
    qc = p * ((1.0 + 0.2 * mach**2) ** 3.5 - 1.0)

    # CAS from impact pressure (using sea-level conditions)
    cas_ms = _A0 * np.sqrt(5.0 * ((qc / _P0 + 1.0) ** (2.0 / 7.0) - 1.0))
    return (cas_ms * ureg.meter / ureg.second).to(tas.units)  # type: ignore[no-any-return]


def mach_to_tas(mach: float, altitude: Quantity) -> Quantity:
    """Convert Mach number to true airspeed at *altitude* (ISA).

    Returns TAS in knots.
    """
    alt_m = altitude.m_as(ureg.meter)
    a = _speed_of_sound_ms(alt_m)
    tas_ms = mach * a
    return (tas_ms * ureg.meter / ureg.second).to(ureg.knot)  # type: ignore[no-any-return]


def tas_to_mach(tas: Quantity, altitude: Quantity) -> float:
    """Convert true airspeed to Mach number at *altitude* (ISA)."""
    tas_ms = tas.m_as(ureg.meter / ureg.second)
    alt_m = altitude.m_as(ureg.meter)
    a = _speed_of_sound_ms(alt_m)
    return tas_ms / a  # type: ignore[no-any-return]
