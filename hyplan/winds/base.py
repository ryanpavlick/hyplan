"""Abstract base class for wind field models."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from pint import Quantity


class WindField(ABC):
    """Abstract base for wind data providers.

    All subclasses must implement :meth:`wind_at`, which returns eastward
    (U) and northward (V) wind components as ``pint.Quantity`` in m/s.

    Subclasses whose wind values do not depend on the ``time`` argument
    (e.g. still air, constant wind) should override the class attribute
    :attr:`is_time_dependent` to ``False``.  Consumers that need a
    timestamp to sample the field (e.g. dropsonde release simulation)
    use this flag to decide whether a timestamp is required.
    """

    is_time_dependent: bool = True

    @abstractmethod
    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime,
    ) -> tuple[Quantity, Quantity]:
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
