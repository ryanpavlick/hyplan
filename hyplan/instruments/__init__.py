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
    # SAR radar
    "RadarExclusionConflict", "check_lband_radar_exclusions",
    "SidelookingRadar", "UAVSAR_Lband", "UAVSAR_Pband", "UAVSAR_Kaband",
    # Frame cameras
    "FrameCamera", "MultiCameraRig",
]
