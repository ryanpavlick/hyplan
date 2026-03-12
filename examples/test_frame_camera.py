#%%
import logging
import numpy as np
from hyplan.units import ureg
from hyplan.frame_camera import FrameCamera

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#%% Create a FrameCamera instance
logging.info("Creating FrameCamera instance...")
camera = FrameCamera(
    name="TestCam",
    sensor_width=ureg.Quantity(36.0, "mm"),
    sensor_height=ureg.Quantity(24.0, "mm"),
    focal_length=ureg.Quantity(50.0, "mm"),
    resolution_x=6000,
    resolution_y=4000,
    frame_rate=ureg.Quantity(1.0, "Hz"),
    f_speed=2.8,
)
logging.info(f"Camera created: {camera.name}")

#%% Test FOV calculations
logging.info("Testing FOV calculations...")
fov_x = camera.fov_x
fov_y = camera.fov_y
logging.info(f"FOV X: {fov_x:.2f} degrees")
logging.info(f"FOV Y: {fov_y:.2f} degrees")

# For a 36mm sensor with 50mm focal length: FOV ~= 2 * atan(18/50) ~= 39.6 degrees
assert 38 < fov_x < 41, f"Expected FOV_X ~39.6 deg, got {fov_x}"
assert 25 < fov_y < 28, f"Expected FOV_Y ~27 deg, got {fov_y}"

#%% Test ground sample distance
logging.info("Testing ground sample distance...")
altitude = ureg.Quantity(5000, "meter")
gsd = camera.ground_sample_distance(altitude)
logging.info(f"GSD at 5000 m: x={gsd['x']:.4f}, y={gsd['y']:.4f}")
assert gsd["x"].magnitude > 0
assert gsd["y"].magnitude > 0

#%% Test altitude for GSD
logging.info("Testing altitude_for_ground_sample_distance...")
target_gsd = ureg.Quantity(1.0, "meter")
required_alt = camera.altitude_for_ground_sample_distance(target_gsd, target_gsd)
logging.info(f"Required altitude for 1m GSD: {required_alt:.1f}")
assert required_alt.magnitude > 0

#%% Test footprint
logging.info("Testing footprint_at...")
footprint = camera.footprint_at(altitude)
logging.info(f"Footprint at 5000 m: width={footprint['width']:.1f}, height={footprint['height']:.1f}")
assert footprint["width"].magnitude > 0
assert footprint["height"].magnitude > 0

#%% Test critical ground speed
logging.info("Testing critical_ground_speed...")
cgs = camera.critical_ground_speed(altitude)
logging.info(f"Critical ground speed at 5000 m: {cgs:.2f}")
assert cgs.magnitude > 0

#%% Test footprint_corners (Bug #4 regression test)
# This test requires a DEM file, so we only test the signature and basic flow
# with a mock scenario. In practice, this would need actual DEM data.
logging.info("Testing footprint_corners signature (Bug #4 regression test)...")

# Verify the static method exists and accepts correct positional args
import inspect
sig = inspect.signature(FrameCamera.footprint_corners)
params = list(sig.parameters.keys())
logging.info(f"footprint_corners parameters: {params}")
assert params == ["lat", "lon", "altitude", "fov_x", "fov_y", "dem_file"], (
    f"Unexpected parameters: {params}"
)

logging.info("All frame camera tests passed.")
# %%
