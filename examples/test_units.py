#%%
import logging
from hyplan.units import ureg, convert_distance, convert_speed, altitude_to_flight_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#%% Test convert_distance
logging.info("Testing convert_distance...")
dist_m = convert_distance(1.0, "nautical_miles", "meters")
logging.info(f"1 NM = {dist_m:.2f} meters")
assert abs(dist_m - 1852.0) < 1.0, f"Expected ~1852 m, got {dist_m}"

dist_km = convert_distance(5280, "feet", "miles")
logging.info(f"5280 feet = {dist_km:.4f} miles")
assert abs(dist_km - 1.0) < 0.01, f"Expected ~1.0 miles, got {dist_km}"

#%% Test convert_speed
logging.info("Testing convert_speed...")
speed_mps = convert_speed(1.0, "knots", "mps")
logging.info(f"1 knot = {speed_mps:.4f} m/s")
assert abs(speed_mps - 0.5144) < 0.01, f"Expected ~0.5144 m/s, got {speed_mps}"

speed_kph = convert_speed(60.0, "mph", "kph")
logging.info(f"60 mph = {speed_kph:.2f} kph")
assert abs(speed_kph - 96.56) < 0.5, f"Expected ~96.56 kph, got {speed_kph}"

#%% Test altitude_to_flight_level with float input
logging.info("Testing altitude_to_flight_level with float...")
fl = altitude_to_flight_level(10000.0)
logging.info(f"10000.0 m -> {fl}")
assert fl.startswith("FL"), f"Expected FL string, got {fl}"

#%% Test altitude_to_flight_level with int input (Bug #2 fix)
logging.info("Testing altitude_to_flight_level with int (Bug #2 regression test)...")
fl_int = altitude_to_flight_level(5000)
logging.info(f"5000 m (int) -> {fl_int}")
assert fl_int.startswith("FL"), f"Expected FL string, got {fl_int}"

# Verify int and float produce the same result
fl_float = altitude_to_flight_level(5000.0)
assert fl_int == fl_float, f"int and float should match: {fl_int} != {fl_float}"
logging.info("int and float inputs produce the same result.")

#%% Test altitude_to_flight_level with pint Quantity
logging.info("Testing altitude_to_flight_level with Quantity...")
fl_qty = altitude_to_flight_level(ureg.Quantity(20000, "feet"))
logging.info(f"20000 feet -> {fl_qty}")
assert fl_qty == "FL200", f"Expected FL200, got {fl_qty}"

#%% Test altitude_to_flight_level with pressure adjustment
logging.info("Testing altitude_to_flight_level with non-standard pressure...")
fl_low_p = altitude_to_flight_level(5000.0, pressure=980.0)
fl_std_p = altitude_to_flight_level(5000.0, pressure=1013.25)
logging.info(f"5000 m at 980 hPa -> {fl_low_p}")
logging.info(f"5000 m at 1013.25 hPa -> {fl_std_p}")
assert fl_low_p != fl_std_p, "Different pressures should produce different FLs"

# Int pressure should also work
fl_int_p = altitude_to_flight_level(5000, pressure=1013)
logging.info(f"5000 m (int) at 1013 hPa (int) -> {fl_int_p}")

#%% Test invalid inputs
logging.info("Testing invalid inputs...")
try:
    altitude_to_flight_level("not a number")
    assert False, "Should have raised ValueError"
except ValueError as e:
    logging.info(f"Correctly rejected string input: {e}")

try:
    altitude_to_flight_level(ureg.Quantity(5, "kilogram"))
    assert False, "Should have raised ValueError"
except ValueError as e:
    logging.info(f"Correctly rejected non-length Quantity: {e}")

logging.info("All units tests passed.")
# %%
