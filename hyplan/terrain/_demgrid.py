"""Lightweight DEM container decoupled from rasterio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DEMGrid:
    """Backend-independent DEM raster.

    All terrain math consumes ``DEMGrid`` instead of live rasterio datasets.
    The ``geotransform`` uses the GDAL convention: ``(origin_x, pixel_width,
    x_skew, origin_y, y_skew, pixel_height)`` where ``pixel_height`` is
    typically negative (north-up).

    Attributes:
        array: 2-D elevation array (rows, cols) in meters.
        geotransform: 6-element affine transform tuple.
        bounds: ``(west, south, east, north)`` in degrees.
        nodata: No-data sentinel value, or ``None``.
    """

    array: np.ndarray
    geotransform: Tuple[float, float, float, float, float, float]
    bounds: Tuple[float, float, float, float]
    nodata: float | None = None

    @property
    def raster_min(self) -> float:
        return float(np.nanmin(self.array))

    @property
    def raster_max(self) -> float:
        return float(np.nanmax(self.array))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape  # type: ignore[no-any-return]
