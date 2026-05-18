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
# Aircraft
from .aircraft import (
    DLR_HALO,
    NASA_B777,
    NASA_C20A,
    NASA_C130,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    NASA_P3,
    NASA_WB57,
    NERC_DO228,
    NOAA_GIV,
    NOAA_WP3D,
    SAFIRE_ATR42,
    Aircraft,
    AWI_BaslerBT67,
    BAS_TwinOtter,
    FAAM_BAe146,
    KingAir350,
    KingAirA90,
    KingAirB200,
    NOAA_TwinOtter,
)

# Airports
from .airports import (
    Airport,
    airports_within_radius,
    find_nearest_airport,
    find_nearest_airports,
    initialize_data,
)

# Airspace
from .airspace import (
    Airspace,
    AirspaceConflict,
    FAATFRClient,
    FlightPlanDBClient,
    NASRAirspaceSource,
    OceanicTrack,
    OpenAIPClient,
    check_airspace_conflicts,
    check_airspace_proximity,
    classify_severity,
    clear_airspace_cache,
    convert_agl_floors,
    fetch_and_check,
    filter_by_schedule,
    summarize_airspaces,
)

# Campaign
from .campaign import Campaign
from .dubins3d import DubinsPath2D
from .exceptions import (
    HyPlanError,
    HyPlanRuntimeError,
    HyPlanTypeError,
    HyPlanValueError,
)

# Exports
from .exports import (
    to_er2_csv,
    to_excel,
    to_foreflight_csv,
    to_gpx,
    to_honeywell_fms,
    to_icartt,
    to_kml,
    to_pilot_excel,
    to_txt,
)
from .flight_box import (
    altitude_msl_for_pixel_size,
    box_around_center_line,
    box_around_center_terrain,
    box_around_polygon,
    box_around_polygon_terrain,
)

# Flight geometry
from .flight_line import FlightLine
from .flight_optimizer import build_graph, greedy_optimize

# Flight patterns
from .flight_patterns import (
    coordinated_line,
    flight_lines_to_waypoint_path,
    polygon,
    racetrack,
    rosette,
    sawtooth,
    spiral,
)

# Sensors
from .instruments import (
    AVAPS_NRD41,
    AVIRIS3,
    AVIRIS5,
    AXCTD,
    CPL,
    GCAS_VNIR,
    GLIHT_DUAL_VQ_480I,
    GLIHT_HRAC,
    GLIHT_THERMAL,
    HALO,
    HSRL2,
    LVIS,
    LVIS_LENS_MEDIUM,
    LVIS_LENS_NARROW,
    LVIS_LENS_WIDE,
    LVIS_LENSES,
    MASTER,
    PICARD,
    PRISM,
    RD94,
    RIEGL_VQ_480II,
    SENSOR_REGISTRY,
    AerosolWindProfiler,
    AircraftTrackSample,
    ALSLidar,
    AVIRISClassic,
    AVIRISNextGen,
    ContiguityError,
    DropsondePlan,
    DropsondeRelease,
    DropsondeReleaseSolution,
    DropsondeSystem,
    DropsondeTrajectory,
    FlightPlanTrack,
    FrameCamera,
    GCAS_UV_Vis,
    GLiHT_SIF,
    GLiHT_SWIR,
    GLiHT_VNIR,
    HyTES,
    LidarMount,
    LineScanner,
    LVISLens,
    MultiALSLidarRig,
    MultiCameraRig,
    PlannedSegment,
    ProfilingLidar,
    RadarExclusionConflict,
    Sensor,
    SidelookingRadar,
    UAVSAR_Kaband,
    UAVSAR_Lband,
    UAVSAR_Pband,
    awp_profile_locations_for_flight_line,
    awp_profile_locations_for_plan,
    check_lband_radar_exclusions,
    create_sensor,
    eMAS,
    flag_awp_stable_segments,
    register_sensor,
    releases_along_flight_line,
    simulate_descent_trajectory,
    simulate_release,
    solve_release_for_target,
    summarize_trajectories,
    terminal_velocity_nrd41,
    terminal_velocity_sippican_axctd,
)
from .pattern import Pattern

# Flight planning and optimization
from .planning import (
    compute_concentric_isochrones,
    compute_flight_plan,
    compute_isochrone,
    compute_multi_base_isochrone,
    compute_multi_refuel_isochrone,
    compute_refuel_isochrone,
    evaluate_target_reachability,
    isochrone_polygon,
    plot_isochrone,
)

# Plotting
from .plotting import (
    map_airspace,
    map_flight_lines,
    plot_airspace_map,
    plot_altitude_trajectory,
    plot_conflict_matrix,
    plot_flight_plan,
    plot_isochrone_static,
    plot_oceanic_tracks,
    plot_vertical_profile,
    terrain_profile_along_track,
)

# Swath
from .swath import (
    analyze_swath_gaps_overlaps,
    calculate_swath_widths,
    generate_swath_polygon,
)

# Units
from .units import (
    altitude_to_flight_level,
    convert_angle,
    convert_distance,
    convert_speed,
    convert_time,
    ureg,
)

# Waypoint and Dubins path planning
from .waypoint import Waypoint

# Wind fields
from .winds import (
    ConstantWindField,
    GFSWindField,
    GMAOWindField,
    MERRA2WindField,
    StillAirField,
    WindField,
    wind_field_from_plan,
)

__all__ = [
    "AVAPS_NRD41",
    "AVIRIS3",
    "AVIRIS5",
    "AXCTD",
    "CPL",
    "DLR_HALO",
    "GCAS_VNIR",
    "GLIHT_DUAL_VQ_480I",
    "GLIHT_HRAC",
    "GLIHT_THERMAL",
    "HALO",
    "HSRL2",
    "LVIS",
    "LVIS_LENSES",
    "LVIS_LENS_MEDIUM",
    "LVIS_LENS_NARROW",
    "LVIS_LENS_WIDE",
    "MASTER",
    "NASA_B777",
    "NASA_C20A",
    "NASA_C130",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_P3",
    "NASA_WB57",
    "NERC_DO228",
    "NOAA_GIV",
    "NOAA_WP3D",
    "PICARD",
    "PRISM",
    "RD94",
    "RIEGL_VQ_480II",
    "SAFIRE_ATR42",
    "SENSOR_REGISTRY",
    "ALSLidar",
    "AVIRISClassic",
    "AVIRISNextGen",
    "AWI_BaslerBT67",
    "AerosolWindProfiler",
    # Aircraft
    "Aircraft",
    "AircraftTrackSample",
    # Airports
    "Airport",
    # Airspace
    "Airspace",
    "AirspaceConflict",
    "BAS_TwinOtter",
    # Campaign
    "Campaign",
    "ConstantWindField",
    "ContiguityError",
    "DropsondePlan",
    "DropsondeRelease",
    "DropsondeReleaseSolution",
    "DropsondeSystem",
    "DropsondeTrajectory",
    "DubinsPath2D",
    "FAAM_BAe146",
    "FAATFRClient",
    # Flight geometry
    "FlightLine",
    "FlightPlanDBClient",
    "FlightPlanTrack",
    "FrameCamera",
    "GCAS_UV_Vis",
    "GFSWindField",
    "GLiHT_SIF",
    "GLiHT_SWIR",
    "GLiHT_VNIR",
    "GMAOWindField",
    # Exceptions
    "HyPlanError",
    "HyPlanRuntimeError",
    "HyPlanTypeError",
    "HyPlanValueError",
    "HyTES",
    "KingAir350",
    "KingAirA90",
    "KingAirB200",
    "LVISLens",
    "LidarMount",
    "LineScanner",
    "MERRA2WindField",
    "MultiALSLidarRig",
    "MultiCameraRig",
    "NASRAirspaceSource",
    "NOAA_TwinOtter",
    "OceanicTrack",
    "OpenAIPClient",
    "Pattern",
    "PlannedSegment",
    "ProfilingLidar",
    "RadarExclusionConflict",
    # Sensors
    "Sensor",
    "SidelookingRadar",
    "StillAirField",
    "UAVSAR_Kaband",
    "UAVSAR_Lband",
    "UAVSAR_Pband",
    # Dubins
    "Waypoint",
    # Wind
    "WindField",
    "airports_within_radius",
    "altitude_msl_for_pixel_size",
    "altitude_to_flight_level",
    "analyze_swath_gaps_overlaps",
    "awp_profile_locations_for_flight_line",
    "awp_profile_locations_for_plan",
    "box_around_center_line",
    "box_around_center_terrain",
    "box_around_polygon",
    "box_around_polygon_terrain",
    "build_graph",
    "calculate_swath_widths",
    "check_airspace_conflicts",
    "check_airspace_proximity",
    "check_lband_radar_exclusions",
    "classify_severity",
    "clear_airspace_cache",
    "compute_concentric_isochrones",
    # Flight planning
    "compute_flight_plan",
    "compute_isochrone",
    "compute_multi_base_isochrone",
    "compute_multi_refuel_isochrone",
    "compute_refuel_isochrone",
    "convert_agl_floors",
    "convert_angle",
    "convert_distance",
    "convert_speed",
    "convert_time",
    "coordinated_line",
    "create_sensor",
    "eMAS",
    "evaluate_target_reachability",
    "fetch_and_check",
    "filter_by_schedule",
    "find_nearest_airport",
    "find_nearest_airports",
    # AWP profiling
    "flag_awp_stable_segments",
    "flight_lines_to_waypoint_path",
    # Swath
    "generate_swath_polygon",
    "greedy_optimize",
    "initialize_data",
    "isochrone_polygon",
    "map_airspace",
    # Plotting
    "map_flight_lines",
    "plot_airspace_map",
    "plot_altitude_trajectory",
    "plot_conflict_matrix",
    "plot_flight_plan",
    "plot_isochrone",
    "plot_isochrone_static",
    "plot_oceanic_tracks",
    "plot_vertical_profile",
    "polygon",
    # Flight patterns
    "racetrack",
    "register_sensor",
    "releases_along_flight_line",
    "rosette",
    "sawtooth",
    # Logging
    "setup_logging",
    "simulate_descent_trajectory",
    "simulate_release",
    "solve_release_for_target",
    "spiral",
    "summarize_airspaces",
    "summarize_trajectories",
    "terminal_velocity_nrd41",
    "terminal_velocity_sippican_axctd",
    "terrain_profile_along_track",
    "to_er2_csv",
    # Exports
    "to_excel",
    "to_foreflight_csv",
    "to_gpx",
    "to_honeywell_fms",
    "to_icartt",
    "to_kml",
    "to_pilot_excel",
    "to_txt",
    # Units
    "ureg",
    "wind_field_from_plan",
]
