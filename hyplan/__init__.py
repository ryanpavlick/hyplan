"""
HyPlan - Planning software for airborne remote sensing science campaigns.

Core objects are re-exported here for convenience::

    from hyplan import FlightLine, FlightBox, Airport, ureg
    from hyplan import KingAirB200, AVIRIS3

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

import logging as _logging


def setup_logging(
    level: int = _logging.INFO,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
) -> None:
    """Attach a StreamHandler to the ``hyplan`` logger.

    Library code uses ``logging.getLogger(__name__)`` everywhere and never
    configures handlers itself. Call this once from a notebook, script, or
    CLI to see hyplan's INFO/WARNING messages. Idempotent — re-calling
    replaces the handler instead of stacking duplicates.
    """
    logger = _logging.getLogger("hyplan")
    for h in list(logger.handlers):
        if getattr(h, "_hyplan_managed", False):
            logger.removeHandler(h)
    handler = _logging.StreamHandler()
    handler.setFormatter(_logging.Formatter(format))
    handler._hyplan_managed = True  # type: ignore[attr-defined]  # custom attr on logging.Handler
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# --- Core re-exports ---

# Exceptions
from .exceptions import (
    HyPlanError,
    HyPlanValueError,
    HyPlanTypeError,
    HyPlanRuntimeError,
)

# Units
from .units import ureg, convert_distance, convert_speed, convert_angle, convert_time, altitude_to_flight_level

# Flight geometry
from .flight_line import FlightLine
from .pattern import Pattern
from .flight_box import box_around_center_line, box_around_polygon, box_around_center_terrain, box_around_polygon_terrain, altitude_msl_for_pixel_size

# Aircraft
from .aircraft import (
    Aircraft,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    NASA_C20A,
    NASA_P3,
    NOAA_WP3D,
    NOAA_GIV,
    NASA_WB57,
    NASA_B777,
    KingAirB200,
    KingAirA90,
    KingAir350,
    NASA_C130,
    NOAA_TwinOtter,
    BAS_TwinOtter,
    FAAM_BAe146,
    SAFIRE_ATR42,
    NERC_DO228,
    AWI_BaslerBT67,
    DLR_HALO,
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
from .instruments import (
    Sensor,
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
    FrameCamera,
    MultiCameraRig,
    LVISLens,
    LVIS_LENS_NARROW,
    LVIS_LENS_MEDIUM,
    LVIS_LENS_WIDE,
    LVIS_LENSES,
    LVIS,
    AerosolWindProfiler,
    flag_awp_stable_segments,
    awp_profile_locations_for_flight_line,
    awp_profile_locations_for_plan,
    ProfilingLidar,
    HSRL2,
    HALO,
    CPL,
    ALSLidar,
    ContiguityError,
    RIEGL_VQ_480II,
    LidarMount,
    MultiALSLidarRig,
    GLIHT_DUAL_VQ_480I,
    DropsondeSystem,
    AVAPS_NRD41,
    RD94,
    AXCTD,
    terminal_velocity_nrd41,
    terminal_velocity_sippican_axctd,
    DropsondeRelease,
    DropsondeTrajectory,
    DropsondePlan,
    DropsondeReleaseSolution,
    FlightPlanTrack,
    PlannedSegment,
    AircraftTrackSample,
    simulate_descent_trajectory,
    simulate_release,
    releases_along_flight_line,
    solve_release_for_target,
    summarize_trajectories,
    RadarExclusionConflict,
    check_lband_radar_exclusions,
    SidelookingRadar,
    UAVSAR_Lband,
    UAVSAR_Pband,
    UAVSAR_Kaband,
)
# Waypoint and Dubins path planning
from .waypoint import Waypoint
from .dubins3d import DubinsPath2D

# Swath
from .swath import generate_swath_polygon, calculate_swath_widths, analyze_swath_gaps_overlaps

# Flight patterns
from .flight_patterns import racetrack, rosette, polygon, sawtooth, spiral, flight_lines_to_waypoint_path, coordinated_line

# Flight planning and optimization
from .planning import compute_flight_plan, compute_isochrone, compute_concentric_isochrones, compute_multi_base_isochrone, compute_multi_refuel_isochrone, compute_refuel_isochrone, evaluate_target_reachability, isochrone_polygon, plot_isochrone

# Wind fields
from .winds import WindField, StillAirField, ConstantWindField, MERRA2WindField, GMAOWindField, GFSWindField, wind_field_from_plan
from .flight_optimizer import build_graph, greedy_optimize

# Plotting
from .plotting import (
    map_flight_lines, plot_flight_plan, plot_altitude_trajectory,
    terrain_profile_along_track,
    plot_airspace_map, plot_isochrone_static,
    plot_oceanic_tracks, plot_vertical_profile,
    plot_conflict_matrix, map_airspace,
)

# Exports
from .exports import (
    to_excel, to_pilot_excel, to_foreflight_csv,
    to_honeywell_fms, to_er2_csv, to_icartt,
    to_kml, to_gpx, to_txt,
)

# Airspace
from .airspace import (
    Airspace, AirspaceConflict, OpenAIPClient,
    check_airspace_conflicts, check_airspace_proximity,
    fetch_and_check, clear_airspace_cache,
    classify_severity, FAATFRClient, NASRAirspaceSource,
    convert_agl_floors, filter_by_schedule,
    summarize_airspaces,
    OceanicTrack, FlightPlanDBClient,
)

# Campaign
from .campaign import Campaign

__all__ = [
    # Logging
    "setup_logging",
    # Exceptions
    "HyPlanError", "HyPlanValueError", "HyPlanTypeError", "HyPlanRuntimeError",
    # Units
    "ureg", "convert_distance", "convert_speed", "convert_angle", "convert_time", "altitude_to_flight_level",
    # Flight geometry
    "FlightLine", "Pattern", "box_around_center_line", "box_around_polygon", "box_around_center_terrain", "box_around_polygon_terrain", "altitude_msl_for_pixel_size",
    # Aircraft
    "Aircraft",
    "NASA_ER2", "NASA_GIII", "NASA_GIV", "NASA_GV", "NASA_C20A", "NASA_P3", "NOAA_WP3D", "NOAA_GIV",
    "NASA_WB57", "NASA_B777",
    "KingAirB200", "KingAirA90", "KingAir350",
    "NASA_C130", "NOAA_TwinOtter",
    "BAS_TwinOtter", "FAAM_BAe146", "SAFIRE_ATR42", "NERC_DO228", "AWI_BaslerBT67", "DLR_HALO",
    # Airports
    "Airport", "initialize_data",
    "find_nearest_airport", "find_nearest_airports", "airports_within_radius",
    # Sensors
    "Sensor", "LineScanner",
    "AVIRISClassic", "AVIRISNextGen", "AVIRIS3", "AVIRIS5",
    "HyTES", "PRISM", "MASTER",
    "GLiHT_VNIR", "GLiHT_Thermal", "GLiHT_SIF",
    "GCAS_UV_Vis", "GCAS_VNIR", "eMAS", "PICARD",
    "SENSOR_REGISTRY", "create_sensor",
    "FrameCamera", "MultiCameraRig",
    "LVISLens", "LVIS_LENS_NARROW", "LVIS_LENS_MEDIUM", "LVIS_LENS_WIDE", "LVIS_LENSES", "LVIS",
    "AerosolWindProfiler", "ProfilingLidar", "HSRL2", "HALO", "CPL",
    "ALSLidar", "ContiguityError", "RIEGL_VQ_480II",
    "LidarMount", "MultiALSLidarRig", "GLIHT_DUAL_VQ_480I",
    "DropsondeSystem", "AVAPS_NRD41", "RD94", "AXCTD",
    "terminal_velocity_nrd41", "terminal_velocity_sippican_axctd",
    "DropsondeRelease", "DropsondeTrajectory", "DropsondePlan",
    "DropsondeReleaseSolution",
    "FlightPlanTrack", "PlannedSegment", "AircraftTrackSample",
    "simulate_descent_trajectory", "simulate_release",
    "releases_along_flight_line", "solve_release_for_target",
    "summarize_trajectories",
    "RadarExclusionConflict", "check_lband_radar_exclusions",
    "SidelookingRadar", "UAVSAR_Lband", "UAVSAR_Pband", "UAVSAR_Kaband",
    # Dubins
    "Waypoint", "DubinsPath2D",
    # Swath
    "generate_swath_polygon", "calculate_swath_widths", "analyze_swath_gaps_overlaps",
    # AWP profiling
    "flag_awp_stable_segments", "awp_profile_locations_for_flight_line", "awp_profile_locations_for_plan",
    # Flight patterns
    "racetrack", "rosette", "polygon", "sawtooth", "spiral", "flight_lines_to_waypoint_path",
    "coordinated_line",
    # Wind
    "WindField", "StillAirField", "ConstantWindField", "MERRA2WindField", "GMAOWindField", "GFSWindField", "wind_field_from_plan",
    # Flight planning
    "compute_flight_plan", "plot_flight_plan", "plot_altitude_trajectory",
    "compute_isochrone", "compute_concentric_isochrones", "compute_multi_base_isochrone", "compute_multi_refuel_isochrone", "compute_refuel_isochrone", "evaluate_target_reachability", "isochrone_polygon", "plot_isochrone",
    "build_graph", "greedy_optimize",
    # Plotting
    "map_flight_lines", "terrain_profile_along_track",
    "plot_airspace_map", "plot_isochrone_static",
    "plot_oceanic_tracks", "plot_vertical_profile",
    "plot_conflict_matrix", "map_airspace",
    # Exports
    "to_excel", "to_pilot_excel", "to_foreflight_csv",
    "to_honeywell_fms", "to_er2_csv", "to_icartt",
    "to_kml", "to_gpx", "to_txt",
    # Airspace
    "Airspace", "AirspaceConflict", "OpenAIPClient",
    "check_airspace_conflicts", "check_airspace_proximity",
    "fetch_and_check", "clear_airspace_cache",
    "classify_severity", "FAATFRClient", "NASRAirspaceSource",
    "convert_agl_floors", "filter_by_schedule",
    "summarize_airspaces",
    "OceanicTrack", "FlightPlanDBClient",
    # Campaign
    "Campaign",
]
