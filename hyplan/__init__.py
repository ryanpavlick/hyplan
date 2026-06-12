"""
HyPlan - Planning software for airborne remote sensing science campaigns.

Core objects are re-exported here for convenience and loaded lazily on
first access (PEP 562), so ``import hyplan`` stays fast::

    from hyplan import FlightLine, box_around_polygon, Airport, ureg
    from hyplan import KingAirB200, AVIRIS3

Specialized modules (clouds, terrain, satellites, glint, sun) should be
imported directly::

    from hyplan.clouds import create_cloud_data_array_with_limit
    from hyplan.terrain import download_dem_files

Plotting naming convention: ``map_*`` functions return interactive
Folium maps (:func:`hyplan.map_flight_lines`, :func:`hyplan.map_airspace`,
:func:`hyplan.map_isochrone`) while ``plot_*`` functions return static
Matplotlib figures (:func:`hyplan.plot_airspace_map`,
:func:`hyplan.plot_isochrone_static`).
"""

import importlib
import logging as _logging
from typing import TYPE_CHECKING, Any

try:
    from ._version import version as __version__
except ImportError:
    # Package not installed via setuptools-scm (e.g. editable dev install
    # before first build), fall back to a default.
    __version__ = "0.0.0.dev0"

# Exceptions are the only eager re-export: they're dependency-free and
# users need them resolvable in `except` clauses without side effects.
from .exceptions import (
    HyPlanError,
    HyPlanRuntimeError,
    HyPlanTypeError,
    HyPlanValueError,
)


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


# --- Lazy core re-exports (PEP 562) ---

# Owning submodule -> names re-exported at the top level.  This is the
# single source of truth for the lazy loader; keep it in sync with
# ``__all__`` when adding names.
_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".aircraft": (
        "DLR_HALO", "NASA_B777", "NASA_C20A", "NASA_C130", "NASA_ER2",
        "NASA_GIII", "NASA_GIV", "NASA_GV", "NASA_P3", "NASA_WB57",
        "NERC_DO228", "NOAA_GIV", "NOAA_WP3D", "SAFIRE_ATR42", "Aircraft",
        "AWI_BaslerBT67", "BAS_TwinOtter", "FAAM_BAe146", "KingAir350",
        "KingAirA90", "KingAirB200", "NOAA_TwinOtter",
    ),
    ".airports": (
        "Airport", "airports_within_radius", "find_nearest_airport",
        "find_nearest_airports", "initialize_data",
    ),
    ".airspace": (
        "Airspace", "AirspaceConflict", "FAATFRClient", "FlightPlanDBClient",
        "NASRAirspaceSource", "OceanicTrack", "OpenAIPClient",
        "check_airspace_conflicts", "check_airspace_proximity",
        "classify_severity", "clear_airspace_cache", "convert_agl_floors",
        "fetch_and_check", "filter_by_schedule", "summarize_airspaces",
    ),
    ".campaign": ("Campaign",),
    ".dubins3d": ("DubinsPath2D",),
    ".exports": (
        "to_er2_csv", "to_excel", "to_foreflight_csv", "to_gpx",
        "to_honeywell_fms", "to_icartt", "to_kml", "to_pilot_excel",
        "to_trackair", "to_txt",
    ),
    ".flight_box": (
        "altitude_msl_for_pixel_size", "box_around_center_line",
        "box_around_center_terrain", "box_around_polygon",
        "box_around_polygon_terrain",
    ),
    ".flight_line": ("FlightLine",),
    ".flight_optimizer": ("build_graph", "greedy_optimize"),
    ".flight_patterns": (
        "coordinated_line", "flight_lines_to_waypoint_path", "polygon",
        "racetrack", "rosette", "sawtooth", "spiral",
    ),
    ".instruments": (
        "AVAPS_NRD41", "AVIRIS3", "AVIRIS5", "AXCTD", "CPL", "GCAS_VNIR",
        "GLIHT_DUAL_VQ_480I", "GLIHT_HRAC", "GLIHT_THERMAL", "HALO", "HSRL2",
        "LVIS", "LVIS_LENS_MEDIUM", "LVIS_LENS_NARROW", "LVIS_LENS_WIDE",
        "LVIS_LENSES", "MASTER", "PICARD", "PRISM", "RD94", "RIEGL_VQ_480II",
        "SENSOR_REGISTRY", "AerosolWindProfiler", "AircraftTrackSample",
        "ALSLidar", "AVIRISClassic", "AVIRISNextGen", "ContiguityError",
        "DropsondePlan", "DropsondeRelease", "DropsondeReleaseSolution",
        "DropsondeSystem", "DropsondeTrajectory", "FlightPlanTrack",
        "FrameCamera", "GCAS_UV_Vis", "GLiHT_SIF", "GLiHT_SWIR", "GLiHT_VNIR",
        "HyTES", "LidarMount", "LineScanner", "LVISLens", "MultiALSLidarRig",
        "MultiCameraRig", "PlannedSegment", "ProfilingLidar",
        "RadarExclusionConflict", "Sensor", "SidelookingRadar",
        "UAVSAR_Kaband", "UAVSAR_Lband", "UAVSAR_Pband",
        "awp_profile_locations_for_flight_line",
        "awp_profile_locations_for_plan", "check_lband_radar_exclusions",
        "create_sensor", "eMAS", "flag_awp_stable_segments",
        "register_sensor", "releases_along_flight_line",
        "simulate_descent_trajectory", "simulate_release",
        "solve_release_for_target", "summarize_trajectories",
        "terminal_velocity_nrd41", "terminal_velocity_sippican_axctd",
    ),
    ".pattern": ("Pattern",),
    ".planning": (
        "compute_concentric_isochrones", "compute_flight_plan",
        "compute_isochrone", "compute_multi_base_isochrone",
        "compute_multi_refuel_isochrone", "compute_refuel_isochrone",
        "evaluate_target_reachability", "flag_below_min_safe_speed",
        "isochrone_polygon", "plot_isochrone",
    ),
    ".plotting": (
        "map_airspace", "map_flight_lines", "plot_airspace_map",
        "plot_altitude_trajectory", "plot_conflict_matrix",
        "plot_flight_plan", "plot_isochrone_static", "plot_oceanic_tracks",
        "plot_vertical_profile", "terrain_profile_along_track",
    ),
    ".swath": (
        "analyze_swath_gaps_overlaps", "calculate_swath_widths",
        "generate_swath_polygon",
    ),
    ".units": (
        "altitude_to_flight_level", "convert_angle", "convert_distance",
        "convert_speed", "convert_time", "ureg",
    ),
    ".waypoint": ("Waypoint",),
    ".winds": (
        "ConstantWindField", "GFSWindField", "GMAOWindField",
        "MERRA2WindField", "StillAirField", "WindField",
        "wind_field_from_plan",
    ),
}

_LAZY_ATTRS: dict[str, str] = {
    name: module for module, names in _LAZY_MODULES.items() for name in names
}

# Top-level names whose attribute in the owning module differs.
_LAZY_ALIASES: dict[str, tuple[str, str]] = {
    # Interactive Folium isochrone plotter under the map_* convention.
    "map_isochrone": (".planning", "plot_isochrone"),
}

if TYPE_CHECKING:
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
    from .airports import (
        Airport,
        airports_within_radius,
        find_nearest_airport,
        find_nearest_airports,
        initialize_data,
    )
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
    from .campaign import Campaign
    from .dubins3d import DubinsPath2D
    from .exports import (
        to_er2_csv,
        to_excel,
        to_foreflight_csv,
        to_gpx,
        to_honeywell_fms,
        to_icartt,
        to_kml,
        to_pilot_excel,
        to_trackair,
        to_txt,
    )
    from .flight_box import (
        altitude_msl_for_pixel_size,
        box_around_center_line,
        box_around_center_terrain,
        box_around_polygon,
        box_around_polygon_terrain,
    )
    from .flight_line import FlightLine
    from .flight_optimizer import build_graph, greedy_optimize
    from .flight_patterns import (
        coordinated_line,
        flight_lines_to_waypoint_path,
        polygon,
        racetrack,
        rosette,
        sawtooth,
        spiral,
    )
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
    from .planning import (
        compute_concentric_isochrones,
        compute_flight_plan,
        compute_isochrone,
        compute_multi_base_isochrone,
        compute_multi_refuel_isochrone,
        compute_refuel_isochrone,
        evaluate_target_reachability,
        flag_below_min_safe_speed,
        isochrone_polygon,
        plot_isochrone,
    )
    from .planning import plot_isochrone as map_isochrone
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
    from .swath import (
        analyze_swath_gaps_overlaps,
        calculate_swath_widths,
        generate_swath_polygon,
    )
    from .units import (
        altitude_to_flight_level,
        convert_angle,
        convert_distance,
        convert_speed,
        convert_time,
        ureg,
    )
    from .waypoint import Waypoint
    from .winds import (
        ConstantWindField,
        GFSWindField,
        GMAOWindField,
        MERRA2WindField,
        StillAirField,
        WindField,
        wind_field_from_plan,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        attr: Any = getattr(module, name)
    elif name in _LAZY_ALIASES:
        module_name, attr_name = _LAZY_ALIASES[name]
        module = importlib.import_module(module_name, __name__)
        attr = getattr(module, attr_name)
    elif not name.startswith("_"):
        # Plain submodule access (e.g. ``hyplan.plotting`` after a bare
        # ``import hyplan``) — the eager imports used to provide this.
        try:
            attr = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as err:
            if err.name != f"{__name__}.{name}":
                raise
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


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
    "flag_below_min_safe_speed",
    "flight_lines_to_waypoint_path",
    # Swath
    "generate_swath_polygon",
    "greedy_optimize",
    "initialize_data",
    "isochrone_polygon",
    "map_airspace",
    # Plotting
    "map_flight_lines",
    "map_isochrone",
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
    "to_trackair",
    "to_txt",
    # Units
    "ureg",
    "wind_field_from_plan",
]
