"""Terrain analysis using Copernicus DEM data.

Downloads, caches, and queries 30-meter Copernicus GLO-30 DEM tiles
from AWS. Provides bulk elevation lookup, DEM tile merging via rasterio,
and a vectorized ray-terrain intersection algorithm for computing
off-nadir ground intersection points.

References
----------
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
