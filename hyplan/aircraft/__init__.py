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
    NASA_WB57,
    NASA_B777,
    KingAirA90,
    KingAirB200,
    C130,
    TwinOtter,
)

__all__ = [
    "Aircraft",
    "ApproachProfile",
    "CasMachSchedule",
    "ClimbOutPolicy",
    "ClimbPlan",
    "TasSchedule",
    "SpeedSchedule",
    "VerticalProfile",
    "TurnModel",
    "PhaseBankAngles",
    "SourceRecord",
    "PerformanceConfidence",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NCAR_GV",
    "NASA_C20A",
    "NASA_P3",
    "NASA_WB57",
    "NASA_B777",
    "KingAirA90",
    "KingAirB200",
    "C130",
    "TwinOtter",
    "load_iwg1",
    "split_iwg1_alltracks",
    "trim_ground_taxi",
    "PlannedSortie",
    "load_planned_sortie",
    "parse_kml",
    "parse_green_card_xlsx",
    "parse_green_card_pdf",
]
