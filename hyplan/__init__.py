"""
HyPlan - Planning software for airborne remote sensing science campaigns.

Core objects are re-exported here for convenience::

    from hyplan import FlightLine, FlightBox, Airport, ureg
    from hyplan import DynamicAviation_B200, AVIRIS3

Specialized modules (clouds, terrain, satellites, glint, sun) should be
imported directly::

    from hyplan.clouds import create_cloud_data_array_with_limit
    from hyplan.terrain import download_dem
"""

try:
    from ._version import version as __version__
except ImportError:
    # Package not installed via setuptools-scm (e.g. editable dev install
    # before first build), fall back to a default.
    __version__ = "0.0.0.dev0"

# --- Core re-exports ---

# Units
from .units import ureg, convert_distance, convert_speed, altitude_to_flight_level

# Flight geometry
from .flight_line import FlightLine
from .flight_box import box_around_center_line, box_around_polygon

# Aircraft
from .aircraft import (
    Aircraft,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_C20A,
    NASA_P3,
    NASA_WB57,
    NASA_B777,
    DynamicAviation_B200,
    DynamicAviation_DH8,
    DynamicAviation_A90,
    C130,
    BAe146,
    Learjet,
    TwinOtter,
)

# Airports
from .airports import (
    Airport,
    initialize_data,
    find_nearest_airport,
    find_nearest_airports,
    airports_within_radius,
)

# Sensors
from .sensors import (
    Sensor,
    LineScanner,
    AVIRISClassic,
    AVIRISNextGen,
    AVIRIS3,
    AVIRIS4,
    HyTES,
    PRISM,
    MASTER,
    GLiHT_VNIR,
    GLiHT_Thermal,
    GLiHT_SIF,
    GCAS_UV_Vis,
    GCAS_VNIR,
    eMAS,
    PICARD,
    SENSOR_REGISTRY,
    create_sensor,
)
from .frame_camera import FrameCamera
from .lvis import LVIS
from .radar import SidelookingRadar

# Dubins path planning
from .dubins_path import Waypoint, DubinsPath

# Swath
from .swath import generate_swath_polygon, calculate_swath_widths

# Flight planning and optimization
from .flight_plan import compute_flight_plan
from .flight_optimizer import build_graph, greedy_optimize

# Plotting
from .plotting import map_flight_lines, plot_flight_plan, plot_altitude_trajectory, terrain_profile_along_track

__all__ = [
    # Units
    "ureg", "convert_distance", "convert_speed", "altitude_to_flight_level",
    # Flight geometry
    "FlightLine", "box_around_center_line", "box_around_polygon",
    # Aircraft
    "Aircraft",
    "NASA_ER2", "NASA_GIII", "NASA_GIV", "NASA_C20A", "NASA_P3",
    "NASA_WB57", "NASA_B777",
    "DynamicAviation_B200", "DynamicAviation_DH8", "DynamicAviation_A90",
    "C130", "BAe146", "Learjet", "TwinOtter",
    # Airports
    "Airport", "initialize_data",
    "find_nearest_airport", "find_nearest_airports", "airports_within_radius",
    # Sensors
    "Sensor", "LineScanner",
    "AVIRISClassic", "AVIRISNextGen", "AVIRIS3", "AVIRIS4",
    "HyTES", "PRISM", "MASTER",
    "GLiHT_VNIR", "GLiHT_Thermal", "GLiHT_SIF",
    "GCAS_UV_Vis", "GCAS_VNIR", "eMAS", "PICARD",
    "SENSOR_REGISTRY", "create_sensor",
    "FrameCamera", "LVIS", "SidelookingRadar",
    # Dubins
    "Waypoint", "DubinsPath",
    # Swath
    "generate_swath_polygon", "calculate_swath_widths",
    # Flight planning
    "compute_flight_plan", "plot_flight_plan", "plot_altitude_trajectory",
    "build_graph", "greedy_optimize",
    # Plotting
    "map_flight_lines", "terrain_profile_along_track",
]
