#%%
import logging
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPolygon

from hyplan.geometry import (
    wrap_to_180,
    wrap_to_360,
    _validate_polygon,
    calculate_geographic_mean,
    get_utm_crs,
    get_utm_transforms,
    haversine,
    process_linestring,
    minimum_rotated_rectangle,
    rotated_rectangle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#%% Test wrap_to_180
logging.info("Testing wrap_to_180...")
assert wrap_to_180(0.0) == 0.0
assert wrap_to_180(180.0) == 180.0 or wrap_to_180(180.0) == -180.0
assert wrap_to_180(270.0) == -90.0
assert wrap_to_180(-270.0) == 90.0
assert wrap_to_180(360.0) == 0.0

# Array input
result = wrap_to_180([0, 90, 180, 270, 360])
assert len(result) == 5
logging.info(f"wrap_to_180([0, 90, 180, 270, 360]) = {result}")

#%% Test wrap_to_360
logging.info("Testing wrap_to_360...")
assert wrap_to_360(0.0) == 0.0
assert wrap_to_360(360.0) == 0.0
assert wrap_to_360(-90.0) == 270.0
assert wrap_to_360(450.0) == 90.0

# Array input
result = wrap_to_360([-90, 0, 90, 360, 450])
logging.info(f"wrap_to_360([-90, 0, 90, 360, 450]) = {result}")

#%% Test _validate_polygon (Bug #7 regression test)
logging.info("Testing _validate_polygon...")

# None should pass validation
result = _validate_polygon(None)
assert result is None, "None input should return None"
logging.info("None input: passed")

# Valid polygon should return True
valid_poly = Polygon([(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)])
result = _validate_polygon(valid_poly)
assert result is True, f"Valid polygon should return True, got {result}"
logging.info("Valid polygon: returned True")

# Invalid inputs should raise ValueError
try:
    _validate_polygon("not a polygon")
    assert False, "Should have raised ValueError"
except ValueError as e:
    logging.info(f"String input correctly rejected: {e}")

try:
    _validate_polygon(Polygon())  # Empty polygon
    assert False, "Should have raised ValueError"
except ValueError as e:
    logging.info(f"Empty polygon correctly rejected: {e}")

try:
    mp = MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    ])
    _validate_polygon(mp)
    assert False, "Should have raised ValueError"
except ValueError as e:
    logging.info(f"MultiPolygon correctly rejected: {e}")

#%% Test process_linestring (Bug #9 regression test)
logging.info("Testing process_linestring...")

# Two-point linestring (minimal case that triggered dtype issues)
line2 = LineString([(-118.25, 34.05), (-118.20, 34.10)])
lats, lons, azimuths, distances = process_linestring(line2)

assert isinstance(lats, np.ndarray)
assert isinstance(lons, np.ndarray)
assert isinstance(azimuths, np.ndarray)
assert isinstance(distances, np.ndarray)

# All azimuth values should be plain floats in the array
assert azimuths.dtype in (np.float64, np.float32), f"Expected float dtype, got {azimuths.dtype}"
assert azimuths.ndim == 1, f"Expected 1-d array, got ndim={azimuths.ndim}"
logging.info(f"Two-point linestring: azimuths={azimuths}, distances={distances}")

# Multi-point linestring
line_multi = LineString([
    (-118.25, 34.05),
    (-118.20, 34.10),
    (-118.15, 34.08),
    (-118.10, 34.12),
])
lats, lons, azimuths, distances = process_linestring(line_multi)
assert len(lats) == 4
assert len(azimuths) == 4  # One per point (last point gets reverse azimuth)
assert distances[0] == 0.0  # First distance is always 0
assert distances[-1] > 0  # Total distance should be positive
logging.info(f"Multi-point linestring: {len(lats)} points, total distance={distances[-1]:.0f} m")

# Verify all elements are float
for i, az in enumerate(azimuths):
    assert isinstance(float(az), float), f"Azimuth {i} is not convertible to float: {type(az)}"

#%% Test calculate_geographic_mean
logging.info("Testing calculate_geographic_mean...")
point = Point(-118.25, 34.05)
mean_pt = calculate_geographic_mean(point)
logging.info(f"Mean of single point: ({mean_pt.y:.4f}, {mean_pt.x:.4f})")
assert abs(mean_pt.y - 34.05) < 0.01
assert abs(mean_pt.x - (-118.25)) < 0.01

# Multiple geometries
line = LineString([(-118.25, 34.05), (-118.20, 34.10)])
poly = Polygon([(-118.3, 34.0), (-118.2, 34.0), (-118.2, 34.1), (-118.3, 34.1)])
mean_pt = calculate_geographic_mean([line, poly])
logging.info(f"Mean of line + polygon: ({mean_pt.y:.4f}, {mean_pt.x:.4f})")

#%% Test haversine
logging.info("Testing haversine...")
# Distance from LA to SF (approx 559 km)
dist = haversine(34.05, -118.25, 37.77, -122.42)
logging.info(f"LA to SF haversine distance: {dist / 1000:.0f} km")
assert 550_000 < dist < 570_000, f"Expected ~559 km, got {dist / 1000:.0f} km"

# Zero distance
assert haversine(34.0, -118.0, 34.0, -118.0) == 0.0

#%% Test get_utm_crs
logging.info("Testing get_utm_crs...")
crs = get_utm_crs(-118.25, 34.05)
logging.info(f"UTM CRS for LA: {crs}")
assert "UTM" in crs.name or "utm" in str(crs).lower()

#%% Test get_utm_transforms
logging.info("Testing get_utm_transforms...")
poly = Polygon([(-118.3, 34.0), (-118.2, 34.0), (-118.2, 34.1), (-118.3, 34.1)])
to_utm, to_wgs84 = get_utm_transforms(poly)

# Round-trip test
x_utm, y_utm = to_utm(-118.25, 34.05)
lon_back, lat_back = to_wgs84(x_utm, y_utm)
assert abs(lon_back - (-118.25)) < 1e-6, f"Round-trip longitude error: {lon_back}"
assert abs(lat_back - 34.05) < 1e-6, f"Round-trip latitude error: {lat_back}"
logging.info("UTM round-trip transform: passed")

#%% Test minimum_rotated_rectangle
logging.info("Testing minimum_rotated_rectangle...")
poly = Polygon([(-118.5, 34.0), (-118.4, 34.3), (-118.3, 34.2), (-118.4, 34.0)])
mrr = minimum_rotated_rectangle(poly)
assert isinstance(mrr, Polygon), f"Expected Polygon, got {type(mrr)}"
logging.info(f"Minimum rotated rectangle bounds: {mrr.bounds}")

#%% Test rotated_rectangle
logging.info("Testing rotated_rectangle...")
rr = rotated_rectangle(poly, azimuth=45.0)
assert isinstance(rr, Polygon), f"Expected Polygon, got {type(rr)}"
logging.info(f"Rotated rectangle (az=45) bounds: {rr.bounds}")

logging.info("All geometry tests passed.")
# %%
