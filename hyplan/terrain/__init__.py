"""Terrain analysis using Copernicus DEM data.

Downloads, caches, and queries 30-meter Copernicus GLO-30 DEM tiles
from AWS. Provides bulk elevation lookup, DEM tile merging via rasterio,
and a vectorized ray-terrain intersection algorithm for computing
off-nadir ground intersection points.

Assumptions and limitations
---------------------------
**DEM is treated as static.**
  Tiles are downloaded once and cached indefinitely. There is no
  versioning or temporal awareness — the DEM is assumed to represent a
  fixed surface for the duration of a planning session.

**Nearest-neighbor elevation sampling.**
  ``get_elevations`` rounds query coordinates to the nearest pixel center
  and returns that pixel's value. There is no bilinear or bicubic
  interpolation. At 30 m resolution this introduces up to ~15 m of
  horizontal positioning error, which is negligible for flight planning
  but would matter for sub-pixel terrain modeling.

**No nodata masking.**
  Pixels with nodata values (e.g. ocean voids) are returned as-is.
  Callers that operate over coastlines or data gaps should check for
  nodata sentinels (available via ``DEMGrid.nodata``) themselves.

**Out-of-bounds queries clamp to edge pixels.**
  Query points outside the DEM extent return the elevation of the
  nearest edge pixel rather than NaN. A warning is logged.

**Ray-terrain intersection uses fixed-step marching.**
  ``ray_terrain_intersection`` steps along the slant range at a fixed
  interval (default 10 m) and detects the first range step where the
  DEM surface rises above the ray altitude. This is not a root-finding
  algorithm — the intersection point is located to within one step of
  the true crossing. Smaller ``precision`` values improve accuracy at
  the cost of a larger search grid.

**Surface normals use 3x3 central differences.**
  ``surface_normal_at`` estimates slope from the two adjacent pixels in
  each direction, converted to metric gradients using a latitude-dependent
  meters-per-degree approximation. There is no sub-pixel terrain modeling
  or higher-order surface fitting.

**Ellipsoidal altitude only.**
  All altitudes are heights above the WGS84 ellipsoid. The module does
  not apply geoid corrections (EGM96/EGM2008). For most flight-planning
  purposes the ~20 m geoid-ellipsoid separation is small relative to
  typical AGL flight altitudes (hundreds to thousands of meters).

Terrain references
------------------
Data source: Copernicus DEM GLO-30, European Space Agency, distributed
via AWS Open Data (s3://copernicus-dem-30m).
"""

from ._demgrid import DEMGrid  # noqa: F401
from .elevation import (  # noqa: F401
    get_elevations,
    get_elevations_from_grid,
    get_min_max_elevations,
    terrain_aspect_azimuth,
    terrain_elevation_along_track,
)
from .intersection import (  # noqa: F401
    _COS_TILT_MIN,
    _M_PER_DEG_LAT,
    ray_terrain_intersection,
    surface_normal_at,
)
from .io import (  # noqa: F401
    build_tile_index,
    clear_cache,
    clear_localdem_cache,
    download_dem_files,
    generate_demfile,
    get_cache_root,
    load_dem,
    merge_tiles,
)

__all__ = [
    "DEMGrid",
    "get_cache_root",
    "clear_cache",
    "clear_localdem_cache",
    "build_tile_index",
    "download_dem_files",
    "merge_tiles",
    "generate_demfile",
    "load_dem",
    "get_elevations",
    "get_elevations_from_grid",
    "get_min_max_elevations",
    "ray_terrain_intersection",
    "terrain_elevation_along_track",
    "terrain_aspect_azimuth",
    "surface_normal_at",
]
