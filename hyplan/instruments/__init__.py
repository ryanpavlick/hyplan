"""HyPlan instrument models.

Groups all sensor classes under a single subpackage::

    from hyplan.instruments import LVIS, SidelookingRadar, AVIRIS3
    from hyplan.instruments import FrameCamera, LineScanner, Sensor
"""

from ._base import Sensor, ScanningSensor
from .registry import (
    SENSOR_REGISTRY,
    create_sensor,
    register_sensor,
)
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
    GLiHT_SWIR,
    GLiHT_SIF,
    GCAS_UV_Vis,
    GCAS_VNIR,
    eMAS,
    PICARD,
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
from .frame_camera import FrameCamera, GLIHT_HRAC, GLIHT_THERMAL, MultiCameraRig


# ── Sensor registration ──────────────────────────────────────────────
# Every name HyPlan resolves via `create_sensor` is registered here,
# explicitly, after every instrument module has been imported. Adding a
# new instrument means one new `register_sensor` call here — no edits
# to other modules required.

def _register_defaults() -> None:
    # Line scanners — class-backed (calling the class returns a fresh
    # instance). Aliases match the names accepted by the historic registry.
    for canonical, klass, aliases in (
        ("AVIRISClassic",  AVIRISClassic,  ("AVIRIS Classic",)),
        ("AVIRISNextGen",  AVIRISNextGen,  ("AVIRIS-NG",)),
        ("AVIRIS3",        AVIRIS3,        ("AVIRIS-3",)),
        ("AVIRIS5",        AVIRIS5,        ("AVIRIS-5",)),
        ("HyTES",          HyTES,          ()),
        ("PRISM",          PRISM,          ()),
        ("MASTER",         MASTER,         ()),
        ("GLiHT_VNIR",     GLiHT_VNIR,     ()),
        ("GLiHT_SWIR",     GLiHT_SWIR,     ()),
        ("GLiHT_SIF",      GLiHT_SIF,      ()),
        ("GCAS_UV_Vis",    GCAS_UV_Vis,    ()),
        ("GCAS_VNIR",      GCAS_VNIR,      ()),
        ("eMAS",           eMAS,           ()),
        ("PICARD",         PICARD,         ()),
    ):
        register_sensor(canonical, klass, aliases=aliases)

    # LVIS, AWP, profiling lidars — class-backed.
    register_sensor("LVIS", LVIS)
    register_sensor(
        "AWP", AerosolWindProfiler,
        aliases=("AerosolWindProfiler", "Aerosol Wind Profiler"),
    )
    register_sensor("HSRL-2", HSRL2, aliases=("HSRL2", "HSRL"))
    register_sensor("HALO", HALO, aliases=("High Altitude Lidar Observatory",))
    register_sensor("CPL", CPL, aliases=("Cloud Physics Lidar",))

    # SAR radars — class-backed.
    register_sensor("UAVSAR_Lband", UAVSAR_Lband, aliases=("UAVSAR L-band",))
    register_sensor("UAVSAR_Pband", UAVSAR_Pband, aliases=("UAVSAR P-band",))
    register_sensor("UAVSAR_Kaband", UAVSAR_Kaband, aliases=("GLISTIN-A",))

    # Frame cameras — pre-built singletons; the factory returns the shared
    # instance so `create_sensor(name) is SINGLETON` holds.
    register_sensor(
        "GLIHT_HRAC", lambda: GLIHT_HRAC,
        # G-LiHT HRAC: Phase One iXM-RS100F-RS (2022+) / iXU1000-R (2017),
        # same imaging chain.
        aliases=(
            "G-LiHT HRAC",
            "Phase One iXM-RS100F-RS",
            "Phase One iXU1000-R",
            "Phase One iXU-R 1000",
        ),
    )
    register_sensor(
        "GLIHT_THERMAL", lambda: GLIHT_THERMAL,
        aliases=("G-LiHT Thermal", "Xenics Gobi-640", "Gobi-640"),
    )

    # Dropsondes — pre-built singletons.
    register_sensor(
        "AVAPS_NRD41", lambda: AVAPS_NRD41,
        aliases=("Vaisala NRD41", "NRD41"),
    )
    register_sensor("RD94", lambda: RD94, aliases=("Vaisala RD94",))
    register_sensor(
        "AXCTD", lambda: AXCTD,
        aliases=("Sippican AXCTD", "SIPPICAN_AXCTD"),
    )


_register_defaults()


__all__ = [
    # Base
    "Sensor",
    "ScanningSensor",
    # Registry
    "SENSOR_REGISTRY", "create_sensor", "register_sensor",
    # Line scanners
    "LineScanner",
    "AVIRISClassic", "AVIRISNextGen", "AVIRIS3", "AVIRIS5",
    "HyTES", "PRISM", "MASTER",
    "GLiHT_VNIR", "GLiHT_SWIR", "GLiHT_SIF",
    "GCAS_UV_Vis", "GCAS_VNIR", "eMAS", "PICARD",
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
    "FrameCamera", "MultiCameraRig", "GLIHT_HRAC", "GLIHT_THERMAL",
]
