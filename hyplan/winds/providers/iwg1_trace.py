"""IWG1 trace wind field — measured winds from a NASA in-situ flight log."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from pint import Quantity

from ...units import ureg
from ..base import WindField
from typing import Any

_M_PER_S_PER_KT = 0.514444


class IWG1TraceWindField(WindField):
    """:class:`WindField` backed by an IWG1 sortie trace.

    Returns the measured wind at the nearest IWG1 fix in
    ``(lat, lon, altitude)`` space.  ``time`` is accepted (and ignored)
    so this class satisfies the engine's gridded-wind interface;
    nearest-in-space is sufficient when the modeled trajectory and the
    actual sortie share the same flight grid (the typical replay case).

    The lat/lon distance contribution uses nautical miles
    (1° lat ≈ 60 nmi); altitude error contributes via a 100 ft / nmi
    penalty so a 1,000 ft altitude error is weighted the same as 10 nmi
    of horizontal error.  That biases the lookup toward fixes at the
    same flight level, which is the right behavior for a step-cruise
    sortie where different flight lines fly at different altitudes.

    Args:
        iwg1_df: A DataFrame in IWG1 schema with columns ``latitude``,
            ``longitude``, ``altitude`` (feet), ``wind_speed_kt``, and
            ``wind_direction_deg``.  Rows with NaN in any of those
            columns are silently dropped.
    """

    def __init__(self, iwg1_df: pd.DataFrame):
        s_mps = iwg1_df["wind_speed_kt"].astype(float) * _M_PER_S_PER_KT
        d_rad = np.radians(iwg1_df["wind_direction_deg"].astype(float))
        u = -s_mps * np.sin(d_rad)
        v = -s_mps * np.cos(d_rad)
        valid = (
            (~u.isna())
            & (~v.isna())
            & (~iwg1_df["altitude"].isna())
            & (~iwg1_df["latitude"].isna())
            & (~iwg1_df["longitude"].isna())
        )
        df = iwg1_df.loc[valid]
        self._u: np.ndarray[Any, np.dtype[Any]] = u[valid].to_numpy()
        self._v: np.ndarray[Any, np.dtype[Any]] = v[valid].to_numpy()
        self._lat: np.ndarray[Any, np.dtype[Any]] = df["latitude"].to_numpy()
        self._lon: np.ndarray[Any, np.dtype[Any]] = df["longitude"].to_numpy()
        self._alt_ft: np.ndarray[Any, np.dtype[Any]] = df["altitude"].to_numpy()
        self._n: int = len(self._u)
        if self._n == 0:
            raise ValueError(
                "IWG1TraceWindField got zero valid fixes — every row was "
                "missing one of latitude / longitude / altitude / "
                "wind_speed_kt / wind_direction_deg."
            )

    def wind_at(
        self,
        lat: float,
        lon: float,
        altitude: Quantity,
        time: datetime.datetime | None = None,
    ) -> tuple[Quantity, Quantity]:
        alt_ft = altitude.m_as("feet")
        lat_cos = np.cos(np.radians(lat))
        d2 = ((self._lat - lat) * 60.0) ** 2
        d2 += ((self._lon - lon) * 60.0 * lat_cos) ** 2
        d2 += ((self._alt_ft - alt_ft) / 100.0) ** 2
        idx = int(np.argmin(d2))
        return (
            float(self._u[idx]) * (ureg.meter / ureg.second),
            float(self._v[idx]) * (ureg.meter / ureg.second),
        )
