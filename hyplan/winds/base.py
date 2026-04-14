"""Abstract base class for wind field models."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Tuple

from pint import Quantity


class WindField(ABC):
    """Abstract base for wind data providers.

    All subclasses must implement :meth:`wind_at`, which returns eastward
    (U) and northward (V) wind components as ``pint.Quantity`` in m/s.
    """

    @abstractmethod
    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> Tuple[Quantity, Quantity]:
        """Return (u, v) wind components at the given point.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            altitude: Geometric altitude as a :class:`pint.Quantity`.
            time: UTC datetime.

        Returns:
            Tuple of (u, v) as :class:`pint.Quantity` in m/s.
            u is eastward (positive = from west),
            v is northward (positive = from south).
        """
