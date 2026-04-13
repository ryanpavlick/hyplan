"""Data structures for ADS-B trajectory fitting results.

These dataclasses carry intermediate and final results through the fitting
pipeline.  They have no external dependencies beyond HyPlan core.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .._base import (
    PerformanceConfidence,
    SourceRecord,
    TasSchedule,
    VerticalProfile,
)


@dataclass
class ScheduleFitMetrics:
    """Quality metrics for a single fitted schedule or profile.

    Attributes:
        r_squared: Coefficient of determination of the piecewise-linear
            fit against the altitude-binned medians (0--1).
        rmse: Root-mean-square error (knots for speed, ft/min for
            vertical rate).
        n_observations: Total trajectory points contributing to the fit.
        altitude_coverage_pct: Fraction of the altitude range that
            contained data (0--1).
        n_breakpoints: Number of vertices in the fitted piecewise-linear
            schedule.
    """

    r_squared: float
    rmse: float
    n_observations: int
    altitude_coverage_pct: float
    n_breakpoints: int


@dataclass
class FlightPhaseData:
    """Altitude-binned statistics for a single flight phase.

    Stores the aggregated data used as input to the schedule fitter.

    Attributes:
        phase: One of ``"climb"``, ``"cruise"``, ``"descent"``.
        altitude_bins_ft: Bin centre altitudes (feet).
        median_tas_kt: Median true airspeed per bin (knots).
        median_vs_fpm: Median absolute vertical speed per bin (ft/min).
        count_per_bin: Number of observations per bin.
        altitude_range_ft: ``(min, max)`` observed altitude in feet.
    """

    phase: str
    altitude_bins_ft: List[float]
    median_tas_kt: List[float]
    median_vs_fpm: List[float]
    count_per_bin: List[int]
    altitude_range_ft: Tuple[float, float]


@dataclass
class FitResult:
    """Complete result of fitting speed schedules and vertical profiles.

    Carries the fitted objects, quality metrics, and provenance metadata
    needed to construct an :class:`~hyplan.aircraft.Aircraft` instance.
    """

    # Fitted schedules
    climb_schedule: TasSchedule
    cruise_schedule: TasSchedule
    descent_schedule: TasSchedule

    # Fitted vertical profiles
    climb_profile: VerticalProfile
    descent_profile: VerticalProfile

    # Inferred parameters
    service_ceiling_ft: float
    approach_speed_kt: float

    # Per-schedule fit quality
    # Keys: "climb_speed", "cruise_speed", "descent_speed",
    #        "climb_vertical", "descent_vertical"
    metrics: Dict[str, ScheduleFitMetrics]

    # Provenance
    icao24: Optional[str] = None
    callsign: Optional[str] = None
    aircraft_type_code: Optional[str] = None
    n_flights: int = 1
    flight_ids: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None
    wind_source: str = "unknown"

    # Phase data (for diagnostics / plotting)
    phase_data: List[FlightPhaseData] = field(default_factory=list)

    def overall_confidence(self) -> PerformanceConfidence:
        """Derive :class:`PerformanceConfidence` from fit metrics.

        Confidence is a weighted combination of R-squared (50 %),
        altitude coverage (25 %), and observation count (25 %,
        saturating at 500 points).
        """

        def _phase_conf(
            speed_key: str, vert_key: Optional[str] = None
        ) -> float:
            sm = self.metrics.get(speed_key)
            if sm is None:
                return 0.2
            conf = 0.5 * max(0.0, sm.r_squared)
            conf += 0.25 * sm.altitude_coverage_pct
            conf += 0.25 * min(1.0, sm.n_observations / 500.0)
            if vert_key and vert_key in self.metrics:
                vm = self.metrics[vert_key]
                vert_conf = (
                    0.5 * max(0.0, vm.r_squared)
                    + 0.25 * vm.altitude_coverage_pct
                )
                conf = 0.6 * conf + 0.4 * vert_conf
            return round(min(1.0, conf), 2)

        return PerformanceConfidence(
            climb=_phase_conf("climb_speed", "climb_vertical"),
            cruise=_phase_conf("cruise_speed"),
            descent=_phase_conf("descent_speed", "descent_vertical"),
            turns=0.3,
        )

    def source_records(self) -> List[SourceRecord]:
        """Build :class:`SourceRecord` list for the fitted model."""
        pc = self.overall_confidence()
        avg_conf = (pc.climb + pc.cruise + pc.descent) / 3.0
        return [
            SourceRecord(
                source_type="adsb",
                reference=(
                    f"ADS-B fitting: {self.n_flights} flight(s), "
                    f"icao24={self.icao24 or 'unknown'}, "
                    f"wind={self.wind_source}"
                ),
                notes=(
                    f"callsign={self.callsign or 'unknown'}, "
                    f"flights={self.flight_ids}"
                ),
                confidence=round(avg_conf, 2),
            )
        ]
