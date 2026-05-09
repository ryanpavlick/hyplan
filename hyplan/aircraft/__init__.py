"""Aircraft performance model and concrete aircraft definitions.

Core definitions live in :mod:`hyplan.aircraft._base`.  Concrete aircraft
subclasses live in :mod:`hyplan.aircraft._models`.  This package re-exports
both so that ``from hyplan.aircraft import Aircraft, NASA_GV`` continues
to work.
"""

from ._base import (  # noqa: F401
    Aircraft,
    ApproachProfile,
    CasMachSchedule,
    ClimbOutPolicy,
    ClimbPlan,
    TasSchedule,
    SpeedSchedule,
    VerticalProfile,
    TurnModel,
    PhaseBankAngles,
    PerformanceConfidence,
    SourceRecord,
)

from .iwg1 import load_iwg1, split_iwg1_alltracks, trim_ground_taxi  # noqa: F401
from .wind_path import climb_with_wind_field, descend_with_wind_field  # noqa: F401
from ._planned_sortie import (  # noqa: F401
    PlannedSortie,
    load_planned_sortie,
    parse_kml,
    parse_green_card_xlsx,
    parse_green_card_pdf,
)

from ._models import (  # noqa: F401
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    NCAR_GV,
    NASA_C20A,
    NASA_P3,
    NOAA_WP3D,
    NOAA_GIV,
    NASA_WB57,
    NASA_B777,
    KingAirA90,
    KingAirB200,
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

__all__ = [
    "DLR_HALO",
    "NASA_B777",
    "NASA_C20A",
    "NASA_C130",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_P3",
    "NASA_WB57",
    "NCAR_GV",
    "NERC_DO228",
    "NOAA_GIV",
    "NOAA_WP3D",
    "SAFIRE_ATR42",
    "AWI_BaslerBT67",
    "Aircraft",
    "ApproachProfile",
    "BAS_TwinOtter",
    "CasMachSchedule",
    "ClimbOutPolicy",
    "ClimbPlan",
    "FAAM_BAe146",
    "KingAir350",
    "KingAirA90",
    "KingAirB200",
    "NOAA_TwinOtter",
    "PerformanceConfidence",
    "PhaseBankAngles",
    "PlannedSortie",
    "SourceRecord",
    "SpeedSchedule",
    "TasSchedule",
    "TurnModel",
    "VerticalProfile",
    "load_iwg1",
    "load_planned_sortie",
    "parse_green_card_pdf",
    "parse_green_card_xlsx",
    "parse_kml",
    "split_iwg1_alltracks",
    "trim_ground_taxi",
]
