"""Backward-compatible shim — flight planning now lives in :mod:`hyplan.planning`."""

from .planning import (
    compute_flight_plan,
    create_flight_line_record,
    process_flight_phase,
)
from .planning.segments import _direct_segment_record
from .winds.utils import (
    _resolve_track_hold_solution,
    _resolve_wind_factor,
    _resolve_wind_uv,
    _track_hold_solution_from_uv,
    _wind_factor,
    _wind_factor_from_uv,
)

__all__ = [
    "compute_flight_plan",
    "create_flight_line_record",
    "process_flight_phase",
]
