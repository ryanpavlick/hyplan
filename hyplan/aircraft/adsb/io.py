"""ADS-B trajectory ingestion via the ``traffic`` library.

This is the only module in the ADS-B pipeline that imports ``traffic``
directly.  All downstream stages operate on pandas DataFrames.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import List, Optional, Union

from ...exceptions import HyPlanRuntimeError

logger = logging.getLogger(__name__)


def _require_traffic():
    """Import and return the traffic module, raising a clear error if missing."""
    try:
        import traffic  # noqa: F811
        import traffic.core  # noqa: F811

        return traffic
    except ImportError:
        raise HyPlanRuntimeError(
            "The 'traffic' library is required for ADS-B ingestion. "
            "Install it with:  pip install hyplan[adsb]"
        )


def load_flights(
    source: Union[str, Path, object],
    *,
    icao24: Optional[Union[str, List[str]]] = None,
    callsign: Optional[Union[str, List[str]]] = None,
    start: Optional[datetime.datetime] = None,
    stop: Optional[datetime.datetime] = None,
    resample: str = "5s",
    filter_strategy: Optional[str] = "default",
    min_duration_minutes: float = 10.0,
    min_altitude_ft: float = 1000.0,
    max_altitude_ft: float = 60000.0,
) -> list:
    """Load and clean ADS-B flights from file or Traffic object.

    Args:
        source: Path to a ``.parquet``, ``.csv``, or ``.pkl`` file, or a
            pre-loaded ``traffic.core.Traffic`` object.
        icao24: Filter by ICAO 24-bit address(es).
        callsign: Filter by callsign(s).
        start: Keep only data after this UTC time.
        stop: Keep only data before this UTC time.
        resample: Resample interval (pandas frequency string).
            ``"5s"`` gives smooth trajectories without excessive data.
        filter_strategy: Smoothing strategy passed to
            ``Flight.filter()``.  ``"default"`` uses traffic's built-in
            EKF.  ``None`` skips filtering.
        min_duration_minutes: Drop flights shorter than this.
        min_altitude_ft: Drop trajectory points below this altitude.
        max_altitude_ft: Drop trajectory points above this altitude.

    Returns:
        List of cleaned ``traffic.core.Flight`` objects.
    """
    traffic = _require_traffic()
    from traffic.core import Traffic

    # --- Load source ---
    if isinstance(source, Traffic):
        data = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"ADS-B data file not found: {path}")
        data = Traffic.from_file(str(path))
        logger.info("Loaded Traffic from %s (%d flights)", path, len(data))
    else:
        raise TypeError(
            f"source must be a file path or Traffic object, got {type(source)}"
        )

    # --- Filter by icao24 ---
    if icao24 is not None:
        if isinstance(icao24, str):
            icao24 = [icao24]
        data = data.query(f"icao24 in {icao24}")
        if data is None or len(data) == 0:
            logger.warning("No flights match icao24=%s", icao24)
            return []

    # --- Filter by callsign ---
    if callsign is not None:
        if isinstance(callsign, str):
            callsign = [callsign]
        data = data.query(f"callsign in {callsign}")
        if data is None or len(data) == 0:
            logger.warning("No flights match callsign=%s", callsign)
            return []

    # --- Filter by time ---
    if start is not None:
        data = data.query(f"timestamp >= '{start.isoformat()}'")
    if stop is not None:
        data = data.query(f"timestamp <= '{stop.isoformat()}'")
    if data is None or len(data) == 0:
        return []

    # --- Process individual flights ---
    flights = []
    for flight in data:
        flight = _clean_flight(
            flight,
            resample=resample,
            filter_strategy=filter_strategy,
            min_altitude_ft=min_altitude_ft,
            max_altitude_ft=max_altitude_ft,
            min_duration_minutes=min_duration_minutes,
        )
        if flight is not None:
            flights.append(flight)

    logger.info(
        "Loaded %d flights (%d after cleaning)",
        len(data),
        len(flights),
    )
    return flights


def _clean_flight(
    flight,
    *,
    resample: str,
    filter_strategy: Optional[str],
    min_altitude_ft: float,
    max_altitude_ft: float,
    min_duration_minutes: float,
):
    """Apply quality filters to a single Flight. Returns None if dropped."""
    df = flight.data

    # Drop rows with missing essential columns
    essential = ["latitude", "longitude", "altitude", "groundspeed", "track"]
    available = [c for c in essential if c in df.columns]
    df = df.dropna(subset=available)

    # Altitude filter
    if "altitude" in df.columns:
        df = df[(df["altitude"] >= min_altitude_ft) & (df["altitude"] <= max_altitude_ft)]

    if len(df) < 2:
        return None

    # Rebuild flight from filtered DataFrame
    from traffic.core import Flight

    flight = Flight(df)

    # Duration check
    duration = flight.duration
    if hasattr(duration, "total_seconds"):
        dur_min = duration.total_seconds() / 60.0
    else:
        dur_min = float(duration) / 60.0
    if dur_min < min_duration_minutes:
        return None

    # Resample
    if resample:
        resampled = flight.resample(resample)
        if resampled is not None:
            flight = resampled

    # EKF filter
    if filter_strategy is not None:
        try:
            filtered = flight.filter()
            if filtered is not None:
                flight = filtered
        except Exception:
            logger.debug(
                "Flight.filter() failed for %s, using unfiltered data",
                getattr(flight, "callsign", "?"),
            )

    # Ensure vertical_rate column exists
    if "vertical_rate" not in flight.data.columns:
        _add_vertical_rate(flight)

    return flight


def _add_vertical_rate(flight) -> None:
    """Compute vertical_rate from altitude differences if missing."""
    df = flight.data
    if "altitude" not in df.columns or "timestamp" not in df.columns:
        df["vertical_rate"] = 0.0
        return

    dt_sec = df["timestamp"].diff().dt.total_seconds()
    dalt = df["altitude"].diff()
    # ft/min from ft/sec
    df["vertical_rate"] = (dalt / dt_sec * 60.0).fillna(0.0)
