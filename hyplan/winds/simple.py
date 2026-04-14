"""Simple wind field implementations (no data fetching)."""

from __future__ import annotations

import datetime
from typing import Tuple

from pint import Quantity

from ..units import ureg
from .base import WindField
from .utils import wind_uv_from_speed_dir

_ZERO_MPS = 0.0 * (ureg.meter / ureg.second)


class StillAirField(WindField):
    """Zero-wind field — always returns U=0, V=0.

    Use this as an explicit "no wind" baseline for comparison or when
    wind data is unavailable.
    """

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        return _ZERO_MPS, _ZERO_MPS


class ConstantWindField(WindField):
    """Constant wind field — same U/V everywhere.

    Useful for backward compatibility with the scalar ``wind_speed`` /
    ``wind_direction`` parameters.

    Args:
        wind_speed: Wind speed magnitude.
        wind_from_deg: Direction the wind is blowing *from* in degrees
            true (meteorological convention: 0 = from north, 90 = from east).
    """

    def __init__(self, wind_speed: Quantity, wind_from_deg: float):
        ws = wind_speed.m_as(ureg.meter / ureg.second)
        u, v = wind_uv_from_speed_dir(ws, wind_from_deg)
        self._u = u * (ureg.meter / ureg.second)
        self._v = v * (ureg.meter / ureg.second)

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        return self._u, self._v
