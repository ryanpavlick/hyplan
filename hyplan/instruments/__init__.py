"""HyPlan instrument models.

Groups all sensor classes under a single subpackage::

    from hyplan.instruments import LVIS, SidelookingRadar, AVIRIS3
    from hyplan.instruments import FrameCamera, LineScanner, Sensor
"""

from ._base import Sensor, ScanningSensor
from .line_scanner import (
    LineScanner,
    AVIRISClassic,
    AVIRISNextGen,
    AVIRIS3,
    AVIRIS5,
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
from .lvis import (
    LVISLens,
    LVIS_LENS_NARROW,
    LVIS_LENS_MEDIUM,
    LVIS_LENS_WIDE,
    LVIS_LENSES,
    LVIS,
)
from .awp import (
    AerosolWindProfiler,
    flag_awp_stable_segments,
    awp_profile_locations_for_flight_line,
    awp_profile_locations_for_plan,
)
from .profilinglidar import ProfilingLidar, HSRL2, HALO, CPL
from .als_lidar import (
    ALSLidar,
    ContiguityError,
    GLIHT_DUAL_VQ_480I,
    LidarMount,
    MultiALSLidarRig,
    RIEGL_VQ_480II,
)
from .dropsondes import (
    AVAPS_NRD41,
    AXCTD,
    AircraftTrackSample,
    DropsondePlan,
    DropsondeRelease,
    DropsondeReleaseSolution,
    DropsondeSystem,
    DropsondeTrajectory,
    FlightPlanTrack,
    PlannedSegment,
    RD94,
    releases_along_flight_line,
    simulate_descent_trajectory,
    simulate_release,
    solve_release_for_target,
    summarize_trajectories,
    terminal_velocity_nrd41,
    terminal_velocity_sippican_axctd,
)
from .radar import (
    RadarExclusionConflict,
    check_lband_radar_exclusions,
    SidelookingRadar,
    UAVSAR_Lband,
    UAVSAR_Pband,
    UAVSAR_Kaband,
)
from .frame_camera import FrameCamera, MultiCameraRig

__all__ = [
    # Base
    "Sensor",
    "ScanningSensor",
    # Line scanners
    "LineScanner",
    "AVIRISClassic", "AVIRISNextGen", "AVIRIS3", "AVIRIS5",
    "HyTES", "PRISM", "MASTER",
    "GLiHT_VNIR", "GLiHT_Thermal", "GLiHT_SIF",
    "GCAS_UV_Vis", "GCAS_VNIR", "eMAS", "PICARD",
    "SENSOR_REGISTRY", "create_sensor",
    # LVIS lidar
    "LVISLens", "LVIS_LENS_NARROW", "LVIS_LENS_MEDIUM", "LVIS_LENS_WIDE", "LVIS_LENSES", "LVIS",
    # Profiling lidars
    "AerosolWindProfiler",
    "flag_awp_stable_segments", "awp_profile_locations_for_flight_line", "awp_profile_locations_for_plan",
    "ProfilingLidar", "HSRL2", "HALO", "CPL",
    # ALS topographic lidar
    "ALSLidar", "ContiguityError", "RIEGL_VQ_480II",
    "LidarMount", "MultiALSLidarRig", "GLIHT_DUAL_VQ_480I",
    # Dropsondes
    "DropsondeSystem", "AVAPS_NRD41", "RD94", "AXCTD",
    "terminal_velocity_nrd41", "terminal_velocity_sippican_axctd",
    "DropsondeRelease", "DropsondeTrajectory", "DropsondePlan",
    "DropsondeReleaseSolution",
    "FlightPlanTrack", "PlannedSegment", "AircraftTrackSample",
    "simulate_descent_trajectory", "simulate_release",
    "releases_along_flight_line", "solve_release_for_target",
    "summarize_trajectories",
    # SAR radar
    "RadarExclusionConflict", "check_lband_radar_exclusions",
    "SidelookingRadar", "UAVSAR_Lband", "UAVSAR_Pband", "UAVSAR_Kaband",
    # Frame cameras
    "FrameCamera", "MultiCameraRig",
]
