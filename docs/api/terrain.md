# Terrain

Download DEM tiles, merge them into local rasters, and compute where a
sensor's field of view intersects the ground surface.

## Assumptions and limitations

- **DEM is treated as static.** Tiles are downloaded once and cached
  indefinitely. There is no versioning or temporal awareness.

- **Nearest-neighbor elevation sampling.** Query coordinates are rounded
  to the nearest pixel center. There is no bilinear or bicubic
  interpolation. At 30 m resolution this introduces up to ~15 m of
  horizontal positioning error.

- **No nodata masking.** Pixels with nodata values (e.g. ocean voids) are
  returned as-is. Callers operating over coastlines or data gaps should
  check {py:attr}`DEMGrid.nodata <hyplan.terrain.DEMGrid.nodata>`.

- **Out-of-bounds queries clamp to edge pixels** rather than returning NaN.

- **Ray-terrain intersection uses fixed-step marching** along the slant
  range (default 10 m step). The intersection is located to within one
  step of the true crossing — not a root-finding algorithm.

- **Surface normals use 3x3 central differences** with a
  latitude-dependent meters-per-degree approximation. No sub-pixel
  terrain modeling or higher-order surface fitting.

- **Ellipsoidal altitude only.** All altitudes are WGS84 ellipsoid
  heights. No geoid correction (EGM96/EGM2008) is applied.

## API

```{eval-rst}
.. automodule:: hyplan.terrain
   :members:
   :show-inheritance:
```
