"""Dropsonde sensor model and reference instances.

A :class:`DropsondeSystem` describes the descent physics of a
ballistic-parachute dropsonde: its terminal-velocity profile, minimum
release altitude AGL, deployment time, and nominal mass.  The class
extends :class:`hyplan.instruments._base.Sensor` for naming / registry
uniformity but does **not** implement the ``ScanningSensor`` Protocol
— dropsondes have no swath; the science footprint is the slant column
from release to splash, computed by the trajectory simulator in
:mod:`hyplan.instruments.dropsondes.simulate`.

Module-level constants :data:`AVAPS_NRD41` and :data:`RD94` are
**shared singletons** configured with the published Vaisala NRD41 /
RD94 terminal-velocity curve.  To customise a parameter, build a new
:class:`DropsondeSystem` instance rather than mutating these
references — mutation would alter the singleton for every subsequent
factory call.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pint import Quantity

from ...exceptions import HyPlanTypeError, HyPlanValueError
from ...units import ureg
from .._base import Sensor

__all__ = [
    "AVAPS_NRD41",
    "AXCTD",
    "DropsondeSystem",
    "RD94",
    "terminal_velocity_nrd41",
    "terminal_velocity_sippican_axctd",
]


def _as_quantity(value: object, unit: str, label: str) -> Quantity:
    """Normalise *value* to a pint Quantity in *unit*."""
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(
        f"{label} must be numeric or a pint.Quantity, got {type(value)}"
    )


# ---------------------------------------------------------------------------
# Default fall-rate model: Vaisala NRD41 / RD94 terminal-velocity curve
# ---------------------------------------------------------------------------

# Published terminal-velocity values used by NCAR EOL AVAPS operational
# planning.  Linear interpolation between these points; constant
# extrapolation outside.
_NRD41_ALT_M = np.array([0.0, 2000.0, 5000.0, 8000.0, 10000.0, 13000.0, 16000.0, 20000.0])
_NRD41_W_MPS = np.array([11.0, 11.5, 12.0, 14.0, 16.0, 22.0, 28.0, 36.0])


def terminal_velocity_nrd41(altitude_msl: Quantity) -> Quantity:
    """Vaisala NRD41 / RD94 terminal fall velocity, positive downward.

    Linearly interpolates between published Vaisala values (~11 m/s at
    sea level, rising to ~22 m/s at 13 km MSL).  Outside the tabulated
    range the curve clamps to the nearest endpoint.
    """
    alt_m = float(_as_quantity(altitude_msl, "meter", "altitude_msl").magnitude)
    w_mps = float(np.interp(alt_m, _NRD41_ALT_M, _NRD41_W_MPS))
    return w_mps * (ureg.meter / ureg.second)


# Lockheed-Martin Sippican AXCTD air-phase terminal velocity.
# The probe deploys a small drogue chute (~0.3 m diameter) that brings
# the splash-impact velocity to ~10 m/s.  Above the boundary layer the
# rate climbs modestly with altitude as air density falls.  Public
# product literature cites ~10 m/s near the surface rising to ~12-14 m/s
# above ~8 km MSL.  Users wanting tighter splash prediction should
# supply a calibrated `descent_rate_model` callable instead.
_AXCTD_ALT_M = np.array([0.0, 2000.0, 5000.0, 8000.0, 12000.0])
_AXCTD_W_MPS = np.array([10.0, 10.5, 11.5, 12.5, 13.5])


def terminal_velocity_sippican_axctd(altitude_msl: Quantity) -> Quantity:
    """Air-phase terminal fall velocity for a Lockheed-Martin Sippican AXCTD.

    Approximate planning model from public product literature: ~10 m/s
    at the surface rising to ~13-14 m/s at the upper end of operational
    release altitudes.  The water-phase descent through the ocean
    column is **not** modelled — :class:`DropsondeSystem` terminates
    the trajectory at splash.
    """
    alt_m = float(_as_quantity(altitude_msl, "meter", "altitude_msl").magnitude)
    w_mps = float(np.interp(alt_m, _AXCTD_ALT_M, _AXCTD_W_MPS))
    return w_mps * (ureg.meter / ureg.second)


_DEFAULT_MAX_STEPS = 3600


class DropsondeSystem(Sensor):
    """Generic ballistic-parachute dropsonde with configurable descent model.

    Parameters
    ----------
    name : str
        Display name (e.g. ``"Vaisala NRD41"``).
    descent_rate_model : Callable[[Quantity], Quantity]
        Function mapping altitude MSL → terminal fall speed (positive
        downward).  Default: :func:`terminal_velocity_nrd41`.
    min_release_altitude : Quantity
        Minimum height AGL at which the chute can deploy reliably
        (default 300 m).
    deployment_time : Quantity
        Time from canister exit to stable terminal-velocity descent
        (default 5 s).
    nominal_mass : Quantity
        Sonde mass; informational only (default 0.39 kg for NRD41).
    source : str
        Provenance string (URL + retrieval date for the datasheet).
    """

    def __init__(
        self,
        name: str = "Generic Dropsonde",
        *,
        descent_rate_model: Callable[[Quantity], Quantity] = terminal_velocity_nrd41,
        min_release_altitude: Quantity = 300 * ureg.meter,
        deployment_time: Quantity = 5 * ureg.second,
        nominal_mass: Quantity = 0.39 * ureg.kilogram,
        source: str = "",
    ) -> None:
        super().__init__(name=name)
        self.descent_rate_model = descent_rate_model
        self.min_release_altitude = _as_quantity(
            min_release_altitude, "meter", "min_release_altitude",
        )
        self.deployment_time = _as_quantity(
            deployment_time, "second", "deployment_time",
        )
        self.nominal_mass = _as_quantity(nominal_mass, "kilogram", "nominal_mass")
        self.source = str(source)

        for label, q in (
            ("min_release_altitude", self.min_release_altitude),
            ("deployment_time", self.deployment_time),
            ("nominal_mass", self.nominal_mass),
        ):
            if q.magnitude <= 0:
                raise HyPlanValueError(f"{label} must be positive")

    def fall_time(
        self,
        release_altitude_agl: Quantity,
        *,
        dt: Quantity = 1 * ureg.second,
    ) -> Quantity:
        """Still-air time from release to surface (no wind)."""
        z = _as_quantity(release_altitude_agl, "meter", "release_altitude_agl")
        z_m = float(z.magnitude)
        if z_m <= 0:
            return 0.0 * ureg.second
        dt_s = float(_as_quantity(dt, "second", "dt").magnitude)
        t = 0.0
        while z_m > 0.0:
            w = float(self.descent_rate_model(z_m * ureg.meter).m_as("meter / second"))
            z_m -= w * dt_s
            t += dt_s
            if t > _DEFAULT_MAX_STEPS * dt_s:
                break
        return t * ureg.second


# ---------------------------------------------------------------------------
# Reference instances (shared singletons — see module docstring)
# ---------------------------------------------------------------------------

AVAPS_NRD41 = DropsondeSystem(
    name="Vaisala NRD41",
    descent_rate_model=terminal_velocity_nrd41,
    min_release_altitude=300 * ureg.meter,
    deployment_time=5 * ureg.second,
    nominal_mass=0.39 * ureg.kilogram,
    source=(
        "Vaisala NRD41 / AVAPS dropsonde, public product page retrieved "
        "2026-05-15 from "
        "https://www.eol.ucar.edu/observing_facilities/avaps-dropsonde-system "
        "(terminal velocity values from NCAR EOL operational planning "
        "tables; mass and deployment_time from Vaisala datasheet)."
    ),
)
"""NASA / NCAR-EOL standard dropsonde for hurricane / airborne campaigns."""

RD94 = DropsondeSystem(
    name="Vaisala RD94 (legacy)",
    descent_rate_model=terminal_velocity_nrd41,
    min_release_altitude=300 * ureg.meter,
    deployment_time=5 * ureg.second,
    nominal_mass=0.4 * ureg.kilogram,
    source=(
        "Vaisala RD94 / NCAR legacy AVAPS dropsonde (predecessor to "
        "NRD41).  Sharing the NRD41 fall-rate curve as a planning "
        "approximation; the RD94's actual curve is within ~5%.  "
        "Reference: Hock and Franklin (1999), BAMS 80(3), 407-420."
    ),
)
"""Legacy AVAPS dropsonde; shares the NRD41 fall-rate model as an approximation."""

AXCTD = DropsondeSystem(
    name="Sippican AXCTD (air phase)",
    descent_rate_model=terminal_velocity_sippican_axctd,
    min_release_altitude=150 * ureg.meter,
    deployment_time=3 * ureg.second,
    nominal_mass=1.0 * ureg.kilogram,
    source=(
        "Lockheed Martin Sippican AXCTD, public product literature "
        "(approximate planning values).  Air-phase model only; the "
        "water-column descent through the ocean and ocean-current "
        "drift are NOT modelled.  The published splash-impact velocity "
        "is ~10 m/s; the altitude profile is a coarse interpolation "
        "and should be calibrated against in-situ data for "
        "splash-position-critical work."
    ),
)
"""Lockheed-Martin Sippican AXCTD — air-phase descent only.

Shares HyPlan's :class:`DropsondeSystem` abstraction because the
air-phase physics is identical to a parachute-decelerated dropsonde.
The kernel terminates at splash; subsurface (water-column) modelling
is out of scope — that would require an ocean-current field analog
of :class:`hyplan.winds.WindField`, deferred to a future release.
Use this for release-pattern planning, splash-point prediction, and
inverse targeting only.
"""
