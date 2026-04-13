"""End-to-end ADS-B trajectory fitting pipeline.

Chains ingestion, phase labeling, wind correction, and schedule fitting
into a single call that produces an :class:`~hyplan.aircraft.Aircraft`.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import pandas as pd

from .._base import Aircraft, TurnModel
from ...exceptions import HyPlanValueError
from ...units import ureg
from .airdata import reconstruct_airdata, resolve_wind_field
from .fitting import fit_schedules
from .io import load_flights
from .phases import label_phases

if TYPE_CHECKING:
    from ...winds import WindField

logger = logging.getLogger(__name__)


def fit_aircraft_from_adsb(
    source: Union[str, Path, object],
    *,
    # Identity
    aircraft_type: Optional[str] = None,
    tail_number: Optional[str] = None,
    operator: str = "Unknown",
    engine_type: Literal["jet", "turboprop", "piston"] = "jet",
    # Flight selection
    icao24: Optional[Union[str, List[str]]] = None,
    callsign: Optional[Union[str, List[str]]] = None,
    start: Optional[datetime.datetime] = None,
    stop: Optional[datetime.datetime] = None,
    # Wind correction
    wind_source: Union[str, WindField, None] = "still_air",
    # Phase labeling
    phase_backend: str = "heuristic",
    # Fitting
    altitude_bin_ft: float = 2000.0,
    max_schedule_points: int = 6,
    service_ceiling_ft: Optional[float] = None,
    # Pipeline control
    aggregate: bool = True,
) -> Aircraft:
    """Fit an Aircraft model from ADS-B surveillance data.

    End-to-end pipeline: ingest -> clean -> phase label -> wind correct
    -> fit schedules -> build Aircraft.

    Args:
        source: Path to ADS-B data file (``.parquet``, ``.csv``,
            ``.pkl``) or a pre-loaded ``traffic.core.Traffic`` object.
        aircraft_type: Aircraft type string.  If *None*, inferred from
            the first flight's icao24 via traffic's aircraft database.
        tail_number: Tail number.  If *None*, uses the icao24 address.
        operator: Operating organisation string.
        engine_type: Propulsion category for schedule compatibility.
        icao24: Filter flights by ICAO 24-bit address(es).
        callsign: Filter flights by callsign(s).
        start: Start of time window (UTC).
        stop: End of time window (UTC).
        wind_source: Wind field specification — ``"still_air"``
            (default), ``"merra2"``, ``"gfs"``, ``"gmao"``, a
            :class:`~hyplan.winds.WindField` instance, or *None*.
        phase_backend: Phase labeling method (``"heuristic"``).
        altitude_bin_ft: Altitude bin width for fitting.
        max_schedule_points: Maximum breakpoints per schedule.
        service_ceiling_ft: Override service ceiling.
        aggregate: If *True* and multiple flights are loaded, aggregate
            their data before fitting.  If *False*, uses only the first
            flight.

    Returns:
        :class:`Aircraft` with fitted speed schedules, vertical profiles,
        and ADS-B provenance metadata.

    Raises:
        HyPlanValueError: If no valid flights remain after filtering.
    """
    # --- Stage 1: Ingest ---
    flights = load_flights(
        source,
        icao24=icao24,
        callsign=callsign,
        start=start,
        stop=stop,
    )
    if not flights:
        raise HyPlanValueError("No valid flights found after filtering.")

    # --- Stage 2: Phase labeling ---
    phased_dfs = [label_phases(f, backend=phase_backend) for f in flights]

    # --- Stage 3: Wind correction ---
    wind_field = resolve_wind_field(wind_source, phased_dfs)
    airdata_dfs = [reconstruct_airdata(df, wind_field) for df in phased_dfs]

    # --- Stage 4: Aggregate or select ---
    if aggregate and len(airdata_dfs) > 1:
        combined_df = pd.concat(airdata_dfs, ignore_index=True)
    else:
        combined_df = airdata_dfs[0]

    # --- Stage 5: Fit schedules ---
    fit = fit_schedules(
        combined_df,
        altitude_bin_ft=altitude_bin_ft,
        max_schedule_points=max_schedule_points,
        service_ceiling_ft=service_ceiling_ft,
    )

    # --- Enrich metadata ---
    fit.n_flights = len(flights)
    fit.flight_ids = [_flight_id(f) for f in flights]
    fit.icao24 = (
        icao24
        if isinstance(icao24, str)
        else getattr(flights[0], "icao24", None)
    )
    fit.callsign = (
        callsign
        if isinstance(callsign, str)
        else getattr(flights[0], "callsign", None)
    )
    fit.wind_source = (
        wind_source
        if isinstance(wind_source, str)
        else type(wind_field).__name__
    )

    # Time range from data
    all_ts = pd.concat([df["timestamp"] for df in airdata_dfs])
    fit.time_range = (all_ts.min().to_pydatetime(), all_ts.max().to_pydatetime())

    # --- Resolve identity ---
    resolved_type = aircraft_type or _infer_aircraft_type(flights[0]) or "Unknown"
    resolved_tail = tail_number or fit.icao24 or "Unknown"

    # --- Stage 6: Build Aircraft ---
    confidence = fit.overall_confidence()
    sources = fit.source_records()

    return Aircraft(
        aircraft_type=resolved_type,
        tail_number=resolved_tail,
        operator=operator,
        service_ceiling=fit.service_ceiling_ft * ureg.feet,
        approach_speed=fit.approach_speed_kt * ureg.knot,
        climb_schedule=fit.climb_schedule,
        cruise_schedule=fit.cruise_schedule,
        descent_schedule=fit.descent_schedule,
        climb_profile=fit.climb_profile,
        descent_profile=fit.descent_profile,
        turn_model=TurnModel(),
        engine_type=engine_type,
        confidence=confidence,
        sources=sources,
    )


def _flight_id(flight) -> str:
    """Build a compact string identifier for a flight."""
    icao = getattr(flight, "icao24", "unknown")
    cs = getattr(flight, "callsign", "")
    try:
        start = flight.data["timestamp"].min().isoformat()[:16]
    except Exception:
        start = "?"
    return f"{icao}_{cs}_{start}"


def _infer_aircraft_type(flight) -> Optional[str]:
    """Try to look up aircraft type from traffic's database."""
    try:
        from traffic.data import aircraft as ac_db

        icao24 = getattr(flight, "icao24", None)
        if icao24:
            info = ac_db.get(icao24)
            if info is not None:
                return getattr(info, "typecode", None) or getattr(
                    info, "model", None
                )
    except Exception:
        pass
    return None
