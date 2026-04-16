"""Aircraft performance model and concrete aircraft definitions.

Core definitions live in :mod:`hyplan.aircraft._base`.  Concrete aircraft
subclasses live in :mod:`hyplan.aircraft._models`.  This package re-exports
both so that ``from hyplan.aircraft import Aircraft, NASA_GV`` continues
to work.
"""

from ._base import (  # noqa: F401
    Aircraft,
    CasMachSchedule,
    TasSchedule,
    SpeedSchedule,
    VerticalProfile,
    TurnModel,
    PhaseBankAngles,
    PerformanceConfidence,
    SourceRecord,
)

from ._models import (  # noqa: F401
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_GV,
    NASA_C20A,
    NASA_P3,
    NASA_WB57,
    NASA_B777,
    Dash8,
    KingAirA90,
    KingAirB200,
    C130,
    BAe146,
    Learjet,
    TwinOtter,
)

__all__ = [
    "Aircraft",
    "CasMachSchedule",
    "TasSchedule",
    "VerticalProfile",
    "TurnModel",
    "PhaseBankAngles",
    "SourceRecord",
    "PerformanceConfidence",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_C20A",
    "NASA_P3",
    "NASA_WB57",
    "NASA_B777",
    "Dash8",
    "KingAirA90",
    "KingAirB200",
    "C130",
    "BAe146",
    "Learjet",
    "TwinOtter",
]
