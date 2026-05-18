"""HyPlan instrument models.

Groups all sensor classes under a single subpackage::

    from hyplan.instruments import LVIS, SidelookingRadar, AVIRIS3
    from hyplan.instruments import FrameCamera, LineScanner, Sensor
"""

from ._base import ScanningSensor, Sensor
from .als_lidar import (
    GLIHT_DUAL_VQ_480I,
    RIEGL_VQ_480II,
    ALSLidar,
    ContiguityError,
    LidarMount,
    MultiALSLidarRig,
)
from .awp import (
    AerosolWindProfiler,
    awp_profile_locations_for_flight_line,
    awp_profile_locations_for_plan,
    flag_awp_stable_segments,
)
from .dropsondes import (
    AVAPS_NRD41,
    AXCTD,
    RD94,
    AircraftTrackSample,
    DropsondePlan,
    DropsondeRelease,
    DropsondeReleaseSolution,
    DropsondeSystem,
    DropsondeTrajectory,
    FlightPlanTrack,
    PlannedSegment,
    releases_along_flight_line,
    simulate_descent_trajectory,
    simulate_release,
    solve_release_for_target,
    summarize_trajectories,
    terminal_velocity_nrd41,
    terminal_velocity_sippican_axctd,
)
from .frame_camera import GLIHT_HRAC, GLIHT_THERMAL, FrameCamera, MultiCameraRig
from .line_scanner import (
    AVIRIS3,
    AVIRIS5,
    GCAS_VNIR,
    MASTER,
    PICARD,
    PRISM,
    AVIRISClassic,
    AVIRISNextGen,
    GCAS_UV_Vis,
    GLiHT_SIF,
    GLiHT_SWIR,
    GLiHT_VNIR,
    HyTES,
    LineScanner,
    eMAS,
)
from .lvis import (
    LVIS,
    LVIS_LENS_MEDIUM,
    LVIS_LENS_NARROW,
    LVIS_LENS_WIDE,
    LVIS_LENSES,
    LVISLens,
)
from .profilinglidar import CPL, HALO, HSRL2, ProfilingLidar
from .radar import (
    RadarExclusionConflict,
    SidelookingRadar,
    UAVSAR_Kaband,
    UAVSAR_Lband,
    UAVSAR_Pband,
    check_lband_radar_exclusions,
)
from .registry import (
    SENSOR_REGISTRY,
    create_sensor,
    register_sensor,
)

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
    "AVAPS_NRD41",
    "AVIRIS3",
    "AVIRIS5",
    "AXCTD",
    "CPL",
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
    "PICARD",
    "PRISM",
    "RD94",
    "RIEGL_VQ_480II",
    # Registry
    "SENSOR_REGISTRY",
    # ALS topographic lidar
    "ALSLidar",
    "AVIRISClassic",
    "AVIRISNextGen",
    # Profiling lidars
    "AerosolWindProfiler",
    "AircraftTrackSample",
    "ContiguityError",
    "DropsondePlan",
    "DropsondeRelease",
    "DropsondeReleaseSolution",
    # Dropsondes
    "DropsondeSystem",
    "DropsondeTrajectory",
    "FlightPlanTrack",
    # Frame cameras
    "FrameCamera",
    "GCAS_UV_Vis",
    "GLiHT_SIF",
    "GLiHT_SWIR",
    "GLiHT_VNIR",
    "HyTES",
    # LVIS lidar
    "LVISLens",
    "LidarMount",
    # Line scanners
    "LineScanner",
    "MultiALSLidarRig",
    "MultiCameraRig",
    "PlannedSegment",
    "ProfilingLidar",
    # SAR radar
    "RadarExclusionConflict",
    "ScanningSensor",
    # Base
    "Sensor",
    "SidelookingRadar",
    "UAVSAR_Kaband",
    "UAVSAR_Lband",
    "UAVSAR_Pband",
    "awp_profile_locations_for_flight_line",
    "awp_profile_locations_for_plan",
    "check_lband_radar_exclusions",
    "create_sensor",
    "eMAS",
    "flag_awp_stable_segments",
    "register_sensor",
    "releases_along_flight_line",
    "simulate_descent_trajectory",
    "simulate_release",
    "solve_release_for_target",
    "summarize_trajectories",
    "terminal_velocity_nrd41",
    "terminal_velocity_sippican_axctd",
]
