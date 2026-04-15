"""Airspace data retrieval, conflict detection, and proximity analysis.

Checks flight lines for airspace conflicts (horizontal intersection plus
vertical altitude overlap) with entry/exit point extraction and near-miss
proximity warnings.

Data sources
------------
**US airspace (no API key required):**

- **FAA ArcGIS** — Special Use Airspace (R, P, MOA, W, A), SFRAs (DC,
  Grand Canyon, etc.), and Class B/C/D/E via :class:`NASRAirspaceSource`.
  Schedule data (``TIMESOFUSE``) and floor reference (MSL/SFC) are
  preserved for filtering.
- **FAA GeoServer + tfrapi** — Active TFRs with geometry and metadata
  via :class:`FAATFRClient`.  Supports date-based filtering
  (``effective_only``).

**International airspace:**

- **OpenAIP** (https://www.openaip.net) — Global airspace via
  :class:`OpenAIPClient`.  Requires an API key set via the
  ``OPENAIP_API_KEY`` environment variable or passed directly.

**Oceanic tracks (no API key required):**

- **FlightPlanDB** — NAT and PACOT tracks via
  :class:`FlightPlanDBClient`.

Key functions
-------------
- :func:`check_airspace_conflicts` — conflict detection with entry/exit
  extraction.
- :func:`check_airspace_proximity` — near-miss proximity warnings.
- :func:`fetch_and_check` — one-call convenience: auto-selects FAA
  sources for US queries, OpenAIP for international.
- :func:`filter_by_schedule` — filter MOA/restricted areas by active
  schedule (time-of-day + day-of-week).
- :func:`convert_agl_floors` — convert SFC-referenced floors to MSL
  using DEM terrain data.
- :func:`classify_severity` — maps airspace type to HARD / ADVISORY /
  INFO / NEAR_MISS.
- :func:`summarize_airspaces` — formatted text table of airspace
  properties for display.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

import requests  # type: ignore[import-untyped]
from shapely.geometry import Polygon, MultiPolygon, box as box_geom, shape
from shapely.geometry.base import BaseGeometry
from shapely import STRtree

if TYPE_CHECKING:
    from datetime import datetime
    from shapely.geometry import LineString

from .exceptions import HyPlanRuntimeError, HyPlanValueError
from .terrain import get_cache_root
from .units import ureg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Airspace:
    """A single airspace volume with lateral boundary and vertical limits.

    Attributes:
        name: Human-readable airspace name (e.g. "R-2508 Complex").
        airspace_class: ICAO classification or SUA type.  Values include
            ``"A"`` through ``"G"`` for controlled airspace, plus
            ``"RESTRICTED"``, ``"PROHIBITED"``, ``"DANGER"``, ``"TMA"``,
            ``"CTR"``, ``"FIR"``, ``"SFRA"``, ``"TFR"``, etc.
        airspace_type: Numeric type code (0=other, 1=restricted,
            2=danger, 3=prohibited, 4=CTR, 31=TFR, 33=CLASS_B,
            34=CLASS_C, 35=CLASS_D, 36=CLASS_E, 37=SFRA, …).
        floor_ft: Lower altitude limit in feet MSL.  0 means surface.
        ceiling_ft: Upper altitude limit in feet MSL.
        geometry: Shapely polygon of the lateral boundary.
        country: ISO 3166-1 alpha-2 country code (e.g. ``"US"``).
        source: Data source identifier (``"openaip"``, ``"faa_nasr"``,
            ``"faa_tfr"``).
        ceiling_unlimited: True if the airspace has no defined ceiling.
        effective_start: ISO date string for TFR start, or None.
        effective_end: ISO date string for TFR end, or None.
        floor_reference: ``"MSL"`` or ``"SFC"`` (AGL).  Use
            :func:`convert_agl_floors` to convert SFC to MSL.
        schedule: Active-hours text from NASR (e.g.
            ``"0700 - 1800, MON - FRI"``).  Use :func:`filter_by_schedule`
            to filter inactive airspaces.
        gmt_offset: Hours from UTC for the airspace location.
        dst_code: ``0`` = no DST, ``1`` = uses DST.
    """

    name: str
    airspace_class: str
    airspace_type: int
    floor_ft: float
    ceiling_ft: float
    geometry: Union[Polygon, MultiPolygon]
    country: str = ""
    source: str = "openaip"
    ceiling_unlimited: bool = False
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    floor_reference: str = "MSL"  # "MSL", "SFC" (AGL), or "STD"
    schedule: Optional[str] = None  # e.g. "0700 - 1800, MON - FRI"
    gmt_offset: Optional[float] = None  # hours from UTC
    dst_code: Optional[int] = None  # 0=no DST, 1=uses DST


@dataclass
class AirspaceConflict:
    """A detected conflict between a flight line and an airspace volume.

    Attributes:
        airspace: The conflicting airspace.
        flight_line_index: Index of the flight line in the input list.
        horizontal_intersection: Shapely geometry of the lateral overlap.
        vertical_overlap_ft: ``(overlap_floor, overlap_ceiling)`` in feet
            MSL describing the altitude band where the flight line and
            airspace overlap.
        severity: ``"HARD"``, ``"ADVISORY"``, ``"INFO"``, or
            ``"NEAR_MISS"`` (from :func:`classify_severity` or
            :func:`check_airspace_proximity`).
        entry_point: ``(lon, lat)`` where the flight line enters the
            airspace, or None.
        exit_point: ``(lon, lat)`` where the flight line exits, or None.
        distance_to_boundary_m: For near-miss conflicts, the closest
            approach distance in meters.  None for actual conflicts.
    """

    airspace: Airspace
    flight_line_index: int
    horizontal_intersection: BaseGeometry
    vertical_overlap_ft: Tuple[float, float]
    severity: str = ""
    entry_point: Optional[Tuple[float, float]] = None  # (lon, lat)
    exit_point: Optional[Tuple[float, float]] = None  # (lon, lat)
    distance_to_boundary_m: Optional[float] = None  # near-miss distance


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def summarize_airspaces(airspaces: List["Airspace"], header: str = "") -> str:
    """Return a formatted summary table of airspaces.

    Args:
        airspaces: List of Airspace objects.
        header: Optional header line (e.g. ``"SUA zones"``).

    Returns:
        Multi-line string suitable for ``print()``.
    """
    lines = []
    if header:
        lines.append(f"{header}: {len(airspaces)}\n")
    for a in airspaces:
        ceil_str = "UNLTD" if a.ceiling_unlimited else f"{a.ceiling_ft:>8,.0f}"
        lines.append(
            f"  {a.name:40s}  {a.airspace_class:12s}  "
            f"{a.floor_ft:>8,.0f} \u2013 {ceil_str:>8s} ft  "
            f"[{a.floor_reference}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conflict detection (pure geometry, no network)
# ---------------------------------------------------------------------------


def check_airspace_conflicts(
    flight_lines,
    airspaces: List[Airspace],
) -> List[AirspaceConflict]:
    """Check flight lines for airspace conflicts.

    For each flight line the function tests:

    1. **Horizontal** — does the line geometry intersect the airspace polygon?
    2. **Vertical** — does the flight altitude overlap the airspace
       ``[floor_ft, ceiling_ft]`` range?

    A conflict is reported only when *both* checks are true.

    Args:
        flight_lines: Iterable of objects with ``.geometry`` (Shapely
            LineString) and ``.altitude_msl`` (pint Quantity) attributes —
            typically :class:`~hyplan.flight_line.FlightLine` instances.
        airspaces: List of :class:`Airspace` objects to check against.

    Returns:
        List of :class:`AirspaceConflict` for every flight-line / airspace
        pair that conflicts.
    """
    if not airspaces or not flight_lines:
        return []

    # Build spatial index over airspace polygons
    as_geoms = [a.geometry for a in airspaces]
    tree = STRtree(as_geoms)

    conflicts: List[AirspaceConflict] = []

    for fl_idx, fl in enumerate(flight_lines):
        fl_geom = fl.geometry
        fl_alt_ft = fl.altitude_msl.m_as(ureg.foot)

        # Query spatial index for candidate airspaces
        candidate_indices = tree.query(fl_geom, predicate="intersects")

        for as_idx in candidate_indices:
            airspace = airspaces[as_idx]

            # Vertical overlap check
            overlap_floor = max(fl_alt_ft, airspace.floor_ft)
            overlap_ceil = min(fl_alt_ft, airspace.ceiling_ft)
            if overlap_floor > overlap_ceil:
                continue

            # Compute the actual intersection geometry
            intersection = fl_geom.intersection(airspace.geometry)
            if intersection.is_empty:
                continue

            entry, exit_ = _extract_entry_exit(intersection)

            conflicts.append(
                AirspaceConflict(
                    airspace=airspace,
                    flight_line_index=fl_idx,
                    horizontal_intersection=intersection,
                    vertical_overlap_ft=(overlap_floor, overlap_ceil),
                    severity=classify_severity(airspace.airspace_type),
                    entry_point=entry,
                    exit_point=exit_,
                )
            )

    return conflicts


def _extract_entry_exit(
    intersection: BaseGeometry,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Extract entry and exit (lon, lat) from an intersection geometry.

    For a LineString, the first coordinate is the entry point and the
    last is the exit point.  For MultiLineString, uses the first coord
    of the first part and the last coord of the last part.  For Point
    geometries, entry and exit are the same.
    """
    try:
        if intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            if coords:
                return coords[0], coords[-1]
        elif intersection.geom_type == "MultiLineString":
            parts = list(intersection.geoms)
            if parts:
                first_coords = list(parts[0].coords)
                last_coords = list(parts[-1].coords)
                return first_coords[0], last_coords[-1]
        elif intersection.geom_type == "Point":
            pt = (intersection.x, intersection.y)
            return pt, pt
        elif intersection.geom_type == "GeometryCollection":
            # Extract first and last non-empty sub-geometry
            geoms = [g for g in intersection.geoms if not g.is_empty]
            if geoms:
                first_entry, _ = _extract_entry_exit(geoms[0])
                _, last_exit = _extract_entry_exit(geoms[-1])
                return first_entry, last_exit
    except (AttributeError, IndexError, StopIteration):
        pass
    return None, None


def check_airspace_proximity(
    flight_lines,
    airspaces: List[Airspace],
    buffer_m: float = 1000.0,
) -> List[AirspaceConflict]:
    """Check for near-miss proximity to airspace without penetration.

    Returns conflicts where the flight line does *not* intersect the
    airspace polygon but passes within *buffer_m* meters of its boundary.
    The ``distance_to_boundary_m`` field is populated with the closest
    approach distance.

    Args:
        flight_lines: Iterable of objects with ``.geometry`` and
            ``.altitude_msl`` attributes.
        airspaces: List of :class:`Airspace` objects.
        buffer_m: Proximity threshold in meters.

    Returns:
        List of :class:`AirspaceConflict` with ``severity="NEAR_MISS"``
        for each flight-line / airspace pair within the buffer.
    """
    import math

    if not airspaces or not flight_lines:
        return []

    # Approximate buffer in degrees (at mid-latitudes)
    buf_deg = buffer_m / 111_000.0

    # Build buffered geometries for spatial indexing
    buffered_geoms = [a.geometry.buffer(buf_deg) for a in airspaces]
    tree = STRtree(buffered_geoms)

    near_misses: List[AirspaceConflict] = []

    for fl_idx, fl in enumerate(flight_lines):
        fl_geom = fl.geometry
        fl_alt_ft = fl.altitude_msl.m_as(ureg.foot)

        candidate_indices = tree.query(fl_geom, predicate="intersects")

        for as_idx in candidate_indices:
            airspace = airspaces[as_idx]

            # Skip if actually inside (that's a conflict, not a near-miss)
            if fl_geom.intersects(airspace.geometry):
                continue

            # Vertical overlap check
            overlap_floor = max(fl_alt_ft, airspace.floor_ft)
            overlap_ceil = min(fl_alt_ft, airspace.ceiling_ft)
            if overlap_floor > overlap_ceil:
                continue

            # Compute distance in degrees, convert to meters
            dist_deg = fl_geom.distance(airspace.geometry)
            # Approximate conversion using mid-latitude
            mid_lat = (fl_geom.centroid.y + airspace.geometry.centroid.y) / 2
            dist_m = dist_deg * 111_000 * math.cos(math.radians(mid_lat))

            if dist_m <= buffer_m:
                near_misses.append(
                    AirspaceConflict(
                        airspace=airspace,
                        flight_line_index=fl_idx,
                        horizontal_intersection=fl_geom,  # no actual intersection
                        vertical_overlap_ft=(overlap_floor, overlap_ceil),
                        severity="NEAR_MISS",
                        distance_to_boundary_m=dist_m,
                    )
                )

    return near_misses


def convert_agl_floors(
    airspaces: List[Airspace],
    dem_file: str,
) -> List[Airspace]:
    """Convert AGL (SFC-referenced) floors to MSL using terrain elevations.

    For each airspace whose ``floor_reference`` is ``"SFC"``, computes
    the maximum terrain elevation within its boundary and adds the AGL
    floor to it.  This is a conservative estimate — the worst-case MSL
    floor across the airspace polygon.

    Args:
        airspaces: List of airspaces (modified in place and returned).
        dem_file: Path to a DEM GeoTIFF file.

    Returns:
        The same list with AGL floors converted to MSL.
    """
    import numpy as np
    from .terrain import get_elevations

    for a in airspaces:
        if a.floor_reference != "SFC" or a.floor_ft == 0.0:
            continue

        # Sample terrain along the airspace boundary
        boundary = a.geometry.exterior if hasattr(a.geometry, "exterior") else (
            a.geometry.geoms[0].exterior if a.geometry.geom_type == "MultiPolygon"
            else None
        )
        if boundary is None:
            continue

        coords = list(boundary.coords)
        lats = np.array([c[1] for c in coords])
        lons = np.array([c[0] for c in coords])

        try:
            elevations_m = get_elevations(lats, lons, dem_file)
            max_elev_ft = float(np.max(elevations_m)) * 3.28084
            a.floor_ft = a.floor_ft + max_elev_ft
            a.floor_reference = "MSL"
            logger.debug(
                "Converted %s floor: +%.0f ft AGL → %.0f ft MSL (max terrain %.0f ft)",
                a.name, a.floor_ft - max_elev_ft, a.floor_ft, max_elev_ft,
            )
        except Exception as exc:
            logger.warning("AGL conversion failed for %s: %s", a.name, exc)

    return airspaces


def filter_by_schedule(
    airspaces: List[Airspace],
    at_datetime: Optional["datetime"] = None,
) -> List[Airspace]:
    """Filter airspaces by their active schedule.

    Parses the ``schedule`` field (e.g. ``"0700 - 1800, MON - FRI"``)
    and returns only those airspaces active at the given time.  Airspaces
    without schedule data are always included (conservative).

    Args:
        airspaces: List of airspaces to filter.
        at_datetime: UTC datetime to check against.  Defaults to now.

    Returns:
        Filtered list of airspaces that are active at the given time.
    """
    import re
    from datetime import datetime as _dt, timezone as _tz, timedelta

    if at_datetime is None:
        at_datetime = _dt.now(_tz.utc)

    result = []
    for a in airspaces:
        if not a.schedule:
            result.append(a)
            continue

        if _is_schedule_active(a.schedule, a.gmt_offset, a.dst_code, at_datetime):
            result.append(a)
        else:
            logger.debug("Filtered inactive airspace: %s (schedule: %s)", a.name, a.schedule)

    return result


_DAY_MAP = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _is_schedule_active(
    schedule: str,
    gmt_offset: Optional[float],
    dst_code: Optional[int],
    at_utc: "datetime",
) -> bool:
    """Check if a schedule string is active at a given UTC time.

    Handles patterns like:
    - ``"0700 - 1800, MON - FRI"``
    - ``"INTERMITTENT, 0600 - 2200, DAILY"``
    - ``"CONTINUOUS"`` / ``"H24"``
    - ``"OTHER TIMES BY NOTAM"``

    Returns True if definitely or possibly active (conservative).
    """
    import re
    from datetime import timedelta

    sched_upper = schedule.upper().strip()

    # Always-active keywords
    if any(kw in sched_upper for kw in ("CONTINUOUS", "H24", "24HR", "BY NOTAM")):
        return True

    # Convert UTC to local time for the airspace
    offset_hours = gmt_offset if gmt_offset is not None else 0
    local_dt = at_utc + timedelta(hours=offset_hours)
    local_hour = local_dt.hour * 100 + local_dt.minute
    local_weekday = local_dt.weekday()  # 0=Monday

    # Parse time range: "0700 - 1800" or "0600-2200"
    time_match = re.search(r'(\d{4})\s*-\s*(\d{4})', sched_upper)
    if time_match:
        start_time = int(time_match.group(1))
        end_time = int(time_match.group(2))
        if not (start_time <= local_hour <= end_time):
            return False

    # Parse day range: "MON - FRI", "DAILY", "MON-SAT"
    if "DAILY" in sched_upper:
        return True  # already passed time check

    day_match = re.search(r'(MON|TUE|WED|THU|FRI|SAT|SUN)\s*-\s*(MON|TUE|WED|THU|FRI|SAT|SUN)', sched_upper)
    if day_match:
        start_day = _DAY_MAP.get(day_match.group(1), 0)
        end_day = _DAY_MAP.get(day_match.group(2), 6)
        if start_day <= end_day:
            if not (start_day <= local_weekday <= end_day):
                return False
        else:
            # Wraps around (e.g., FRI-MON)
            if not (local_weekday >= start_day or local_weekday <= end_day):
                return False
        return True

    # If we extracted a time range but no day range, assume it passed
    if time_match:
        return True

    # Unparseable schedule — be conservative, assume active
    return True


# ---------------------------------------------------------------------------
# OpenAIP API client with caching
# ---------------------------------------------------------------------------

# OpenAIP airspace type codes (subset)
_OPENAIP_TYPE_NAMES = {
    0: "OTHER",
    1: "RESTRICTED",
    2: "DANGER",
    3: "PROHIBITED",
    4: "CTR",
    5: "TMZ",
    6: "RMZ",
    7: "TMA",
    8: "TRA",
    9: "TSA",
    10: "FIR",
    11: "UIR",
    12: "ADIZ",
    13: "ATZ",
    14: "MATZ",
    15: "AIRWAY",
    16: "MTR",
    17: "ALERT_AREA",
    18: "WARNING_AREA",
    19: "PROTECTED",
    20: "HTZ",
    21: "GLIDING_SECTOR",
    22: "TRP",
    23: "TIZ",
    24: "TIA",
    25: "MTA",
    26: "CTA",
    27: "ACC_SECTOR",
    28: "AERIAL_SPORTING_RECREATIONAL",
    29: "OVERFLIGHT_RESTRICTION",
    30: "MRT",
    31: "TFR",
    32: "GCA",
    33: "CLASS_B",
    34: "CLASS_C",
    35: "CLASS_D",
    36: "CLASS_E",
    37: "SFRA",
    38: "MODE_C",
}

# Reverse lookup: name string → OpenAIP type code
_TYPE_NAME_TO_CODE = {v: k for k, v in _OPENAIP_TYPE_NAMES.items()}

# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

# Maps OpenAIP type codes to conflict severity.
_SEVERITY_MAP = {
    3: "HARD",       # PROHIBITED
    1: "HARD",       # RESTRICTED
    37: "HARD",      # SFRA
    2: "ADVISORY",   # DANGER
    4: "ADVISORY",   # CTR
    7: "ADVISORY",   # TMA
    6: "ADVISORY",   # RMZ
    13: "ADVISORY",  # ATZ
    14: "ADVISORY",  # MATZ
    31: "ADVISORY",  # TFR
    33: "ADVISORY",  # CLASS_B
    34: "ADVISORY",  # CLASS_C
    35: "ADVISORY",  # CLASS_D
    36: "INFO",      # CLASS_E
    38: "INFO",      # MODE_C
}


def classify_severity(airspace_type: int) -> str:
    """Return conflict severity for an airspace type code.

    Returns ``"HARD"``, ``"ADVISORY"``, or ``"INFO"``.
    """
    return _SEVERITY_MAP.get(airspace_type, "INFO")


# ---------------------------------------------------------------------------
# Type filtering
# ---------------------------------------------------------------------------


def _resolve_type_filter(
    type_filter: "int | str | list[int | str] | None",
) -> Optional[set]:
    """Resolve a type filter specification to a set of integer type codes.

    Accepts a single type code (int), a type name (str), a list of
    either, or ``None`` (no filtering).

    Raises:
        HyPlanValueError: If a string name is not recognised.
    """
    if type_filter is None:
        return None

    if isinstance(type_filter, (int, str)):
        type_filter = [type_filter]

    codes: set = set()
    for item in type_filter:
        if isinstance(item, int):
            codes.add(item)
        elif isinstance(item, str):
            name = item.upper()
            if name not in _TYPE_NAME_TO_CODE:
                raise HyPlanValueError(
                    f"Unknown airspace type name: {item!r}. "
                    f"Valid names: {sorted(_TYPE_NAME_TO_CODE.keys())}"
                )
            codes.add(_TYPE_NAME_TO_CODE[name])
        else:
            raise HyPlanValueError(
                f"type_filter items must be int or str, got {type(item)}"
            )
    return codes


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _get_airspace_cache_dir() -> str:
    """Return the airspace cache subdirectory under HyPlan's cache root."""
    cache_dir = os.path.join(get_cache_root(), "airspace_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def clear_airspace_cache() -> None:
    """Remove all cached airspace data.

    This forces the next :meth:`OpenAIPClient.fetch_airspaces` call to
    re-query the API instead of using stale local data.
    """
    import shutil

    cache_dir = os.path.join(get_cache_root(), "airspace_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Airspace cache cleared: %s", cache_dir)
    else:
        logger.info("Airspace cache directory does not exist: %s", cache_dir)


def _cache_key(
    bounds: Tuple[float, float, float, float],
    country: "str | list | None",
) -> str:
    """Compute a deterministic cache filename from query parameters."""
    # Round bounds to 1 decimal degree so nearby queries share cache
    rounded = tuple(round(b, 1) for b in bounds)
    if isinstance(country, list):
        country_str = ",".join(sorted(str(c) for c in country if c))
        if not country_str:
            country_str = "all"
    else:
        country_str = country or "all"
    raw = f"{rounded}_{country_str}"
    return hashlib.md5(raw.encode()).hexdigest() + ".json"


def _is_cache_stale(cache_path: str, max_age_hours: float) -> bool:
    """Check if a cached file is older than *max_age_hours*."""
    if not os.path.exists(cache_path):
        return True
    age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
    return age_hours > max_age_hours


def _parse_airspace_item(item: dict) -> Optional[Airspace]:
    """Parse a single OpenAIP airspace JSON object into an Airspace."""
    try:
        geojson = item.get("geometry")
        if geojson is None:
            return None
        geom = shape(geojson)
        if not isinstance(geom, (Polygon, MultiPolygon)):
            return None

        name = item.get("name", "Unknown")
        icao_class_raw = item.get("icaoClass", "")
        as_type = int(item.get("type", 0))
        country = item.get("country", "")

        # Altitude limits — stored in feet MSL
        # OpenAIP unit codes: 0=meters, 1=feet, 6=flight level (FL)
        # referenceDatum: 0=GND, 1=MSL, 2=STD
        def _parse_alt_ft(alt_obj: dict, default: float) -> float:
            if not alt_obj:
                return default
            value = alt_obj.get("value", default)
            unit = alt_obj.get("unit", 0)
            if unit == 0:  # meters → feet
                value = value * 3.28084
            elif unit in (2, 6):  # flight level → feet
                value = value * 100
            # unit == 1 (feet) → use as-is
            return float(value)

        floor_ft = _parse_alt_ft(item.get("lowerLimit", {}), 0.0)
        ceiling_ft = _parse_alt_ft(item.get("upperLimit", {}), 60000.0)
        ceiling_unlimited = (
            not item.get("upperLimit")
            or ceiling_ft >= 60000.0
        )

        # Determine airspace_class from icaoClass or type code
        # icaoClass may be a letter string ("A"-"G") or an integer
        # (0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=UNCLASSIFIED, 8=SUA)
        _ICAO_CLASS_INT_MAP = {0: "A", 1: "B", 2: "C", 3: "D",
                               4: "E", 5: "F", 6: "G"}
        if isinstance(icao_class_raw, int):
            airspace_class = _ICAO_CLASS_INT_MAP.get(icao_class_raw, "")
        else:
            airspace_class = str(icao_class_raw)

        if not airspace_class or airspace_class not in "ABCDEFG":
            airspace_class = _OPENAIP_TYPE_NAMES.get(as_type, "OTHER")

        return Airspace(
            name=name,
            airspace_class=airspace_class,
            airspace_type=as_type,
            floor_ft=floor_ft,
            ceiling_ft=ceiling_ft,
            geometry=geom,
            country=country,
            source="openaip",
            ceiling_unlimited=ceiling_unlimited,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Skipping unparseable airspace item: %s", exc)
        return None


def parse_airspace_items(items: List[dict]) -> List[Airspace]:
    """Parse a list of raw OpenAIP JSON items into Airspace objects.

    Items that cannot be parsed (missing geometry, non-polygon geometry,
    etc.) are silently skipped.
    """
    results = []
    for item in items:
        parsed = _parse_airspace_item(item)
        if parsed is not None:
            results.append(parsed)
    return results


class OpenAIPClient:
    """Fetch and cache airspace data from the OpenAIP API.

    Args:
        api_key: OpenAIP API key.  If *None*, reads from the
            ``OPENAIP_API_KEY`` environment variable.

    Raises:
        HyPlanValueError: If no API key is provided or found.
    """

    BASE_URL = "https://api.core.openaip.net/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAIP_API_KEY", "")
        if not self.api_key:
            raise HyPlanValueError(
                "OpenAIP API key required. Set the OPENAIP_API_KEY "
                "environment variable or pass api_key= to OpenAIPClient."
            )

    def fetch_airspaces(
        self,
        bounds: Tuple[float, float, float, float],
        country: "str | list[str] | None" = None,
        max_age_hours: float = 24.0,
        type_filter: "int | str | list[int | str] | None" = None,
    ) -> List[Airspace]:
        """Fetch airspaces within a bounding box.

        Uses a local JSON cache; stale entries are re-fetched.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            country: Optional ISO 2-letter country code(s).  A single
                string or a list of strings for multi-country queries.
            max_age_hours: Maximum cache age before re-fetching.
            type_filter: Filter results to specific airspace types.
                Accepts an int type code, a type name string (e.g.
                ``"RESTRICTED"``), a list of either, or ``None`` for all.

        Returns:
            List of :class:`Airspace` objects.

        Raises:
            HyPlanRuntimeError: On network or API errors.
        """
        airspaces, _ = self.fetch_airspaces_raw(bounds, country, max_age_hours)
        codes = _resolve_type_filter(type_filter)
        if codes is not None:
            airspaces = [a for a in airspaces if a.airspace_type in codes]
        return airspaces

    def fetch_airspaces_raw(
        self,
        bounds: Tuple[float, float, float, float],
        country: "str | list[str] | None" = None,
        max_age_hours: float = 24.0,
    ) -> Tuple[List[Airspace], List[dict]]:
        """Fetch airspaces and return both parsed objects and raw JSON items.

        Same as :meth:`fetch_airspaces` but also returns the raw API
        response items, which can be persisted for later re-parsing.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            country: Optional ISO 2-letter country code(s).

        Returns:
            ``(airspaces, raw_items)`` tuple.
        """
        # Normalise country to a list (or [None] for "all countries")
        if country is None:
            countries = [None]
        elif isinstance(country, str):
            countries = [country]  # type: ignore[list-item]
        else:
            countries = list(country)  # type: ignore[arg-type]

        cache_dir = _get_airspace_cache_dir()
        cache_key_str = _cache_key(bounds, countries)
        cache_file = os.path.join(cache_dir, cache_key_str)

        if not _is_cache_stale(cache_file, max_age_hours):
            logger.info("Using cached airspace data: %s", cache_file)
            with open(cache_file) as f:
                items = json.load(f)
            return parse_airspace_items(items), items

        min_lon, min_lat, max_lon, max_lat = bounds

        seen_ids: set = set()
        items: List[dict] = []  # type: ignore[no-redef]
        for c in countries:
            page_items = self._fetch_all_pages(bounds, c)
            for it in page_items:
                item_id = it.get("_id") or it.get("id")
                if item_id is None:
                    item_id = hashlib.sha1(
                        json.dumps(it, sort_keys=True, default=str).encode()
                    ).hexdigest()
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    items.append(it)

        # Cache the raw JSON. The cache is a pure performance optimization:
        # `items` is already fully populated above, and the function returns
        # the parsed result unconditionally below — a write failure (disk
        # full, read-only mount, permission error) just means the next call
        # will re-fetch instead of hitting the cache. We log the warning
        # rather than promoting to an exception, which would discard a
        # successful fetch.
        try:
            with open(cache_file, "w") as f:
                json.dump(items, f)
            logger.info("Cached %d airspace items to %s", len(items), cache_file)
        except OSError as exc:
            logger.warning("Failed to write airspace cache: %s", exc)

        return parse_airspace_items(items), items

    def _fetch_all_pages(
        self,
        bounds: Tuple[float, float, float, float],
        country: Optional[str],
    ) -> List[dict]:
        """Fetch all pages of airspace results from the API."""
        min_lon, min_lat, max_lon, max_lat = bounds
        all_items: List[dict] = []
        page = 1
        limit = 100

        while True:
            params = {
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "page": page,
                "limit": limit,
            }
            if country:
                params["country"] = country.upper()

            headers = {"x-openaip-api-key": self.api_key}

            try:
                resp = requests.get(
                    f"{self.BASE_URL}/airspaces",
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise HyPlanRuntimeError(
                    f"OpenAIP API request failed: {exc}"
                ) from exc

            data = resp.json()
            items = data.get("items", data if isinstance(data, list) else [])
            all_items.extend(items)

            total_pages = data.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1

        logger.info("Fetched %d airspace items from OpenAIP", len(all_items))
        return all_items

    @staticmethod
    def _parse_items(items: List[dict]) -> List[Airspace]:
        """Parse a list of raw JSON items into Airspace objects."""
        return parse_airspace_items(items)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def _bounds_within_us(bounds: Tuple[float, float, float, float]) -> bool:
    """Return True if the bounding box falls within US airspace.

    Uses a generous box covering CONUS, Alaska, Hawaii, and territories.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    # US territory: lon ∈ [-180, -60], lat ∈ [17, 72]
    return min_lon >= -180 and max_lon <= -60 and min_lat >= 17 and max_lat <= 72


def fetch_and_check(
    flight_lines,
    api_key: Optional[str] = None,
    buffer_m: float = 1000.0,
    country: "str | list[str] | None" = None,
    max_age_hours: float = 24.0,
    type_filter: "int | str | list[int | str] | None" = None,
    use_faa: bool = True,
) -> List[AirspaceConflict]:
    """Fetch nearby airspaces and check flight lines for conflicts.

    This is a one-call convenience function that:

    1. Computes a bounding box from the flight lines (with buffer).
    2. Fetches airspaces (US: FAA SUA + SFRA + TFRs; international:
       OpenAIP).
    3. Runs :func:`check_airspace_conflicts`.

    For US queries (when *use_faa* is True and bounds fall within US
    territory), airspace data is sourced from the FAA ArcGIS portal
    and GeoServer — **no API key required**.  For international queries,
    an OpenAIP API key is needed.

    Args:
        flight_lines: Iterable of objects with ``.geometry`` and
            ``.altitude_msl`` attributes.
        api_key: OpenAIP API key (or set ``OPENAIP_API_KEY`` env var).
            Only used for international queries.
        buffer_m: Buffer in meters added to the bounding box.
        country: Optional ISO 2-letter country code(s).
        max_age_hours: Maximum cache age before re-fetching.
        type_filter: Filter to specific airspace types (see
            :meth:`OpenAIPClient.fetch_airspaces`).
        use_faa: If True (default), use FAA sources for US queries
            instead of OpenAIP.

    Returns:
        List of :class:`AirspaceConflict` objects.
    """
    fl_list = list(flight_lines)
    if not fl_list:
        return []

    # Compute bounding box from all flight line geometries
    all_lons = []
    all_lats = []
    for fl in fl_list:
        coords = list(fl.geometry.coords)
        for lon, lat in coords:
            all_lons.append(lon)
            all_lats.append(lat)

    # Buffer in approximate degrees (1° ≈ 111 km)
    buf_deg = buffer_m / 111_000.0
    bounds = (
        min(all_lons) - buf_deg,
        min(all_lats) - buf_deg,
        max(all_lons) + buf_deg,
        max(all_lats) + buf_deg,
    )

    if use_faa and _bounds_within_us(bounds):
        nasr = NASRAirspaceSource()
        airspaces = nasr.fetch_airspaces(bounds)
        airspaces.extend(nasr.fetch_sfras(bounds))
        airspaces.extend(nasr.fetch_class_airspace(bounds))
        try:
            tfr_client = FAATFRClient()
            airspaces.extend(tfr_client.fetch_tfrs(bounds, effective_only=True))
        except HyPlanRuntimeError as exc:
            logger.warning("TFR fetch failed, continuing without TFRs: %s", exc)

        # Apply type_filter on the FAA path
        type_codes = _resolve_type_filter(type_filter)
        if type_codes is not None:
            airspaces = [a for a in airspaces if a.airspace_type in type_codes]
    else:
        client = OpenAIPClient(api_key=api_key)
        airspaces = client.fetch_airspaces(
            bounds=bounds, country=country, max_age_hours=max_age_hours,
            type_filter=type_filter,
        )

    return check_airspace_conflicts(fl_list, airspaces)


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def _circle_to_polygon(
    center_lat: float,
    center_lon: float,
    radius_nm: float,
    n_points: int = 64,
) -> Polygon:
    """Convert a center + radius (nautical miles) to a Shapely polygon.

    Uses a simple equirectangular approximation suitable for airspace
    circles (typically < 30 NM).
    """
    import math

    radius_deg_lat = radius_nm / 60.0
    radius_deg_lon = radius_deg_lat / math.cos(math.radians(center_lat))

    angles = [2 * math.pi * i / n_points for i in range(n_points)]
    coords = [
        (
            center_lon + radius_deg_lon * math.cos(a),
            center_lat + radius_deg_lat * math.sin(a),
        )
        for a in angles
    ]
    coords.append(coords[0])
    return Polygon(coords)


# ---------------------------------------------------------------------------
# FAA TFR (Temporary Flight Restrictions) client
# ---------------------------------------------------------------------------


class FAATFRClient:
    """Fetch active Temporary Flight Restrictions from the FAA.

    Queries the FAA GeoServer WFS endpoint for TFR polygons and the
    ``tfrapi`` JSON endpoint for metadata, then merges them by NOTAM ID.
    **No API key required.**

    TFRs are cached locally with a short TTL (default 1 hour) since
    they change frequently.

    Args:
        cache_ttl_hours: Cache time-to-live in hours.
    """

    TFR_WFS_URL = "https://tfr.faa.gov/geoserver/TFR/ows"
    TFR_LIST_URL = "https://tfr.faa.gov/tfrapi/getTfrList"

    def __init__(self, cache_ttl_hours: float = 1.0):
        self._cache_ttl = cache_ttl_hours

    def fetch_tfrs(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        effective_only: bool = False,
    ) -> List[Airspace]:
        """Fetch active TFRs, optionally filtered to a bounding box.

        Args:
            bounds: Optional ``(min_lon, min_lat, max_lon, max_lat)``.
                If provided, only TFRs whose geometry intersects the
                bounding box are returned.
            effective_only: If True, filter out TFRs whose description
                text indicates a start date in the future.  Best-effort
                parsing — TFRs without parseable dates are kept.

        Returns:
            List of :class:`Airspace` objects with ``source="faa_tfr"``.
        """
        cache_dir = os.path.join(_get_airspace_cache_dir(), "tfr")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "tfr_list.json")

        if not _is_cache_stale(cache_file, self._cache_ttl):
            logger.info("Using cached TFR data: %s", cache_file)
            with open(cache_file) as f:
                cached = json.load(f)
            airspaces = [self._dict_to_airspace(d) for d in cached]
            airspaces = [a for a in airspaces if a is not None]
        else:
            # Fetch TFR polygons from GeoServer WFS
            wfs_params = {
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typeName": "TFR:V_TFR_LOC",
                "maxFeatures": "300",
                "outputFormat": "application/json",
                "srsname": "EPSG:4326",
            }
            try:
                wfs_resp = requests.get(
                    self.TFR_WFS_URL, params=wfs_params, timeout=60,
                )
                wfs_resp.raise_for_status()
            except requests.RequestException as exc:
                raise HyPlanRuntimeError(
                    f"FAA TFR WFS request failed: {exc}"
                ) from exc

            wfs_data = wfs_resp.json()
            features = wfs_data.get("features", [])

            # Fetch metadata from tfrapi for enrichment
            meta_map: dict = {}
            try:
                meta_resp = requests.get(self.TFR_LIST_URL, timeout=30)
                meta_resp.raise_for_status()
                for item in meta_resp.json():
                    nid = item.get("notam_id", "")
                    meta_map[nid] = item
            except (requests.RequestException, ValueError) as exc:
                logger.warning("TFR metadata fetch failed, using WFS only: %s", exc)

            airspaces = []
            cache_dicts = []
            for feat in features:
                parsed = self._parse_wfs_feature(feat, meta_map)
                if parsed is not None:
                    airspaces.append(parsed)
                    cache_dicts.append(self._airspace_to_dict(parsed))

            logger.info("Fetched %d active TFRs from FAA GeoServer", len(airspaces))

            try:
                with open(cache_file, "w") as f:
                    json.dump(cache_dicts, f)
            except OSError as exc:
                logger.warning("Failed to write TFR cache: %s", exc)

        if bounds is not None:
            bbox = box_geom(*bounds)
            airspaces = [a for a in airspaces if a.geometry.intersects(bbox)]  # type: ignore[union-attr]

        if effective_only:
            airspaces = self._filter_effective(airspaces)  # type: ignore[assignment, arg-type]

        return airspaces  # type: ignore[return-value]

    @staticmethod
    def _parse_date_from_description(description: str) -> Optional[str]:
        """Try to extract a start date from a tfrapi description string.

        Descriptions look like:
        - "Cape Canaveral, FL, Thursday, April 16, 2026 Local"
        - "11NM NW HOT SPRINGS, SD, Sunday, April 12, 2026 through ..."
        - "Miami, FL" (no date)

        Returns an ISO date string or None.
        """
        import re
        from datetime import datetime as _dt

        # Match patterns like "April 16, 2026" or "January 1, 2026"
        m = re.search(
            r'(\w+ \d{1,2},\s*\d{4})', description,
        )
        if m:
            try:
                return _dt.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
        return None

    @staticmethod
    def _filter_effective(airspaces: List[Airspace]) -> List[Airspace]:
        """Remove TFRs whose start date is in the future."""
        from datetime import date as _date

        today = _date.today().isoformat()
        result = []
        for a in airspaces:
            start = a.effective_start
            if start is None or start <= today:
                result.append(a)
            else:
                logger.debug("Filtering future TFR: %s (starts %s)", a.name, start)
        return result

    @staticmethod
    def _extract_notam_id(notam_key: str) -> str:
        """Extract the base NOTAM ID from a WFS NOTAM_KEY.

        WFS keys look like ``"6/1222-1-FDC-F"``; the tfrapi uses
        ``"6/1222"``.  Strip the suffix after the first ``-``.
        """
        return notam_key.split("-")[0] if notam_key else ""

    @staticmethod
    def _parse_wfs_feature(
        feature: dict,
        meta_map: dict,
    ) -> Optional[Airspace]:
        """Convert a WFS GeoJSON feature + tfrapi metadata to an Airspace."""
        try:
            geom_data = feature.get("geometry")
            if not geom_data:
                return None
            geom = shape(geom_data)
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None

            props = feature.get("properties", {})
            notam_key = props.get("NOTAM_KEY", "")
            notam_id = FAATFRClient._extract_notam_id(notam_key)
            title = props.get("TITLE", "")
            state = props.get("STATE", "")

            # Enrich with tfrapi metadata if available
            meta = meta_map.get(notam_id, {})
            tfr_type = meta.get("type", "")
            description = meta.get("description", "")

            name = title or description or f"TFR {notam_id}"
            if tfr_type:
                name = f"TFR ({tfr_type}): {name}"

            # Try to extract effective date from description text
            effective_start = FAATFRClient._parse_date_from_description(
                description or title
            )

            return Airspace(
                name=name,
                airspace_class="TFR",
                airspace_type=31,
                floor_ft=0.0,
                ceiling_ft=60000.0,
                geometry=geom,
                country="US",
                source="faa_tfr",
                ceiling_unlimited=True,
                effective_start=effective_start,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping unparseable WFS TFR feature: %s", exc)
            return None

    @staticmethod
    def _airspace_to_dict(a: Airspace) -> dict:
        """Serialise an Airspace to a JSON-compatible dict for caching."""
        from shapely.geometry import mapping
        return {
            "name": a.name,
            "airspace_class": a.airspace_class,
            "airspace_type": a.airspace_type,
            "floor_ft": a.floor_ft,
            "ceiling_ft": a.ceiling_ft,
            "geometry": mapping(a.geometry),
            "country": a.country,
            "source": a.source,
            "ceiling_unlimited": a.ceiling_unlimited,
            "effective_start": a.effective_start,
            "effective_end": a.effective_end,
        }

    @staticmethod
    def _dict_to_airspace(d: dict) -> Optional[Airspace]:
        """Deserialise a cached dict back to an Airspace."""
        try:
            geom = shape(d["geometry"])
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None
            return Airspace(
                name=d["name"],
                airspace_class=d["airspace_class"],
                airspace_type=d["airspace_type"],
                floor_ft=d["floor_ft"],
                ceiling_ft=d["ceiling_ft"],
                geometry=geom,
                country=d.get("country", "US"),
                source=d.get("source", "faa_tfr"),
                ceiling_unlimited=d.get("ceiling_unlimited", False),
                effective_start=d.get("effective_start"),
                effective_end=d.get("effective_end"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping unparseable cached TFR: %s", exc)
            return None


# ---------------------------------------------------------------------------
# FAA NASR/SUA data source
# ---------------------------------------------------------------------------

# Map NASR SUA type codes to OpenAIP type codes
_NASR_TYPE_MAP = {
    "R": 1,    # RESTRICTED
    "P": 3,    # PROHIBITED
    "W": 18,   # WARNING_AREA
    "A": 17,   # ALERT_AREA
    "MOA": 0,  # OTHER (Military Operations Area)
}

# Map FAA class airspace LOCAL_TYPE to internal type codes
_CLASS_AIRSPACE_TYPE_MAP = {
    "CLASS_B": 33,
    "CLASS_C": 34,
    "CLASS_D": 35,
    "CLASS_E": 36,
    "MODE C": 38,
}


class NASRAirspaceSource:
    """Fetch US airspace data from the FAA ArcGIS open data portal.

    Uses the `AIS Data Delivery Service
    <https://adds-faa.opendata.arcgis.com/>`_ to query Special Use
    Airspace (SUA) and Special Flight Rules Areas (SFRAs) as GeoJSON.
    **No API key required.**

    Cached locally with a 28-day TTL matching the NASR subscription cycle.

    Args:
        cache_ttl_days: Cache time-to-live in days.
        base_url: Override the SUA ArcGIS Feature Server URL.
    """

    # ArcGIS Feature Server endpoint for Special Use Airspace
    DEFAULT_BASE_URL = (
        "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ"
        "/arcgis/rest/services/Special_Use_Airspace/FeatureServer/0/query"
    )

    # ArcGIS Feature Server endpoint for SFRA / Special Air Traffic Rules
    SFRA_BASE_URL = (
        "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ"
        "/arcgis/rest/services/Airspace/FeatureServer/0/query"
    )

    # ArcGIS Feature Server endpoint for Class B/C/D/E airspace
    CLASS_AIRSPACE_URL = (
        "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ"
        "/arcgis/rest/services/Class_Airspace/FeatureServer/0/query"
    )

    def __init__(
        self,
        cache_ttl_days: float = 28.0,
        base_url: Optional[str] = None,
    ):
        self._cache_ttl_hours = cache_ttl_days * 24.0
        self._base_url = base_url or self.DEFAULT_BASE_URL

    def fetch_airspaces(
        self,
        bounds: Tuple[float, float, float, float],
        sua_types: Optional[List[str]] = None,
    ) -> List[Airspace]:
        """Fetch NASR airspace data within bounds.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            sua_types: Optional filter for SUA types (e.g.
                ``["R", "P", "MOA"]``).

        Returns:
            List of :class:`Airspace` objects with ``source="faa_nasr"``.
        """
        cache_dir = os.path.join(_get_airspace_cache_dir(), "nasr")
        os.makedirs(cache_dir, exist_ok=True)
        rounded = tuple(round(b, 1) for b in bounds)
        cache_file = os.path.join(
            cache_dir,
            hashlib.md5(str(rounded).encode()).hexdigest() + ".json",
        )

        if not _is_cache_stale(cache_file, self._cache_ttl_hours):
            logger.info("Using cached NASR data: %s", cache_file)
            with open(cache_file) as f:
                features = json.load(f)
        else:
            features = self._fetch_arcgis(bounds)
            try:
                with open(cache_file, "w") as f:
                    json.dump(features, f)
            except OSError as exc:
                logger.warning("Failed to write NASR cache: %s", exc)

        airspaces = []
        for feat in features:
            parsed = self._feature_to_airspace(feat)
            if parsed is not None:
                airspaces.append(parsed)

        if sua_types is not None:
            upper = {t.upper() for t in sua_types}
            airspaces = [
                a for a in airspaces
                if a.name.split("-")[0].strip().upper() in upper
                or a.airspace_class.upper() in upper
            ]

        return airspaces

    def _fetch_arcgis(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> List[dict]:
        """Query the ArcGIS Feature Server for airspace features."""
        min_lon, min_lat, max_lon, max_lat = bounds

        params = {
            "where": "1=1",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "f": "geojson",
        }

        try:
            resp = requests.get(self._base_url, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise HyPlanRuntimeError(
                f"NASR ArcGIS query failed: {exc}"
            ) from exc

        data = resp.json()
        return data.get("features", [])  # type: ignore[no-any-return]

    @staticmethod
    def _feature_to_airspace(feature: dict) -> Optional[Airspace]:
        """Convert a GeoJSON feature to an Airspace."""
        try:
            geom_data = feature.get("geometry")
            if not geom_data or "type" not in geom_data:
                return None
            geom = shape(geom_data)
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None

            props = feature.get("properties", {})
            name = props.get("NAME", props.get("name", "Unknown"))
            sua_type = props.get("TYPE_CODE", props.get("type_code", ""))

            # Parse altitudes (NASR uses various field names)
            floor_ft = 0.0
            ceiling_ft = 60000.0
            ceiling_unlimited = False
            for key in ("LOWER_VAL", "lower_val", "FLOOR"):
                if key in props and props[key] is not None:
                    try:
                        floor_ft = float(props[key])
                    except (ValueError, TypeError):
                        pass
                    break
            for key in ("UPPER_VAL", "upper_val", "CEILING"):
                if key in props and props[key] is not None:
                    try:
                        ceiling_ft = float(props[key])
                    except (ValueError, TypeError):
                        pass
                    break

            # Handle NASR sentinel values and unlimited ceilings
            upper_code = props.get("UPPER_CODE", "")
            if upper_code == "UNLTD" or ceiling_ft < 0:
                ceiling_ft = 60000.0
                ceiling_unlimited = True
            # Handle flight-level ceilings (e.g. 180 = FL180 = 18,000 ft)
            upper_uom = props.get("UPPER_UOM", "")
            if upper_uom == "FL" or (not upper_uom and 0 < ceiling_ft <= 600):
                ceiling_ft *= 100.0
            lower_uom = props.get("LOWER_UOM", "")
            if lower_uom == "FL":
                floor_ft *= 100.0

            as_type = _NASR_TYPE_MAP.get(sua_type, 0)
            as_class = _OPENAIP_TYPE_NAMES.get(as_type, sua_type or "OTHER")

            # Floor reference (SFC = AGL, MSL = MSL)
            lower_code = props.get("LOWER_CODE", props.get("lower_code", "MSL"))
            floor_ref = "SFC" if lower_code in ("SFC", "GND", "AGL") else "MSL"

            # Schedule data
            schedule = props.get("TIMESOFUSE") or None
            gmt_offset = None
            dst_code = None
            raw_offset = props.get("GMTOFFSET")
            if raw_offset is not None:
                try:
                    gmt_offset = float(raw_offset)
                except (ValueError, TypeError):
                    pass
            raw_dst = props.get("DST_CODE")
            if raw_dst is not None:
                try:
                    dst_code = int(raw_dst)
                except (ValueError, TypeError):
                    pass

            return Airspace(
                name=name,
                airspace_class=as_class,
                airspace_type=as_type,
                floor_ft=floor_ft,
                ceiling_ft=ceiling_ft,
                geometry=geom,
                country="US",
                source="faa_nasr",
                ceiling_unlimited=ceiling_unlimited,
                floor_reference=floor_ref,
                schedule=schedule,
                gmt_offset=gmt_offset,
                dst_code=dst_code,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping unparseable NASR feature: %s", exc)
            return None

    def fetch_sfras(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> List[Airspace]:
        """Fetch Special Flight Rules Areas (SFRAs) within bounds.

        Queries the FAA Airspace ArcGIS layer for features with
        ``TYPE_CODE='SATA'`` (Special Air Traffic Rules), which includes
        DC SFRA, Grand Canyon, NYC Hudson River SFRA, and others.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.

        Returns:
            List of :class:`Airspace` objects with ``source="faa_nasr"``
            and ``airspace_class="SFRA"``.
        """
        cache_dir = os.path.join(_get_airspace_cache_dir(), "nasr")
        os.makedirs(cache_dir, exist_ok=True)
        rounded = tuple(round(b, 1) for b in bounds)
        cache_file = os.path.join(
            cache_dir,
            "sfra_" + hashlib.md5(str(rounded).encode()).hexdigest() + ".json",
        )

        if not _is_cache_stale(cache_file, self._cache_ttl_hours):
            logger.info("Using cached SFRA data: %s", cache_file)
            with open(cache_file) as f:
                features = json.load(f)
        else:
            features = self._fetch_sfra_arcgis(bounds)
            try:
                with open(cache_file, "w") as f:
                    json.dump(features, f)
            except OSError as exc:
                logger.warning("Failed to write SFRA cache: %s", exc)

        airspaces = []
        for feat in features:
            parsed = self._sfra_feature_to_airspace(feat)
            if parsed is not None:
                airspaces.append(parsed)
        return airspaces

    def _fetch_sfra_arcgis(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> List[dict]:
        """Query the ArcGIS Airspace layer for SFRA features."""
        min_lon, min_lat, max_lon, max_lat = bounds

        params = {
            "where": "TYPE_CODE='SATA'",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "f": "geojson",
        }

        try:
            resp = requests.get(
                self.SFRA_BASE_URL, params=params, timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise HyPlanRuntimeError(
                f"NASR SFRA ArcGIS query failed: {exc}"
            ) from exc

        data = resp.json()
        return data.get("features", [])  # type: ignore[no-any-return]

    @staticmethod
    def _sfra_feature_to_airspace(feature: dict) -> Optional[Airspace]:
        """Convert an Airspace-layer GeoJSON feature to an Airspace."""
        try:
            geom_data = feature.get("geometry")
            if not geom_data or "type" not in geom_data:
                return None
            geom = shape(geom_data)
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None

            props = feature.get("properties", {})
            name = props.get("NAME_TXT", props.get("name", "Unknown SFRA"))

            # Parse altitudes
            floor_ft = 0.0
            ceiling_ft = 60000.0
            ceiling_unlimited = False

            lower_val = props.get("DISTVERTLOWER_VAL")
            if lower_val is not None:
                try:
                    floor_ft = float(lower_val)
                    # Convert if unit is FL (flight level)
                    lower_uom = props.get("DISTVERTLOWER_UOM", "")
                    if lower_uom == "FL":
                        floor_ft *= 100.0
                except (ValueError, TypeError):
                    pass

            upper_val = props.get("DISTVERTUPPER_VAL")
            if upper_val is not None:
                try:
                    ceiling_ft = float(upper_val)
                    upper_uom = props.get("DISTVERTUPPER_UOM", "")
                    if upper_uom == "FL":
                        ceiling_ft *= 100.0
                except (ValueError, TypeError):
                    pass
            else:
                ceiling_unlimited = True

            return Airspace(
                name=name,
                airspace_class="SFRA",
                airspace_type=37,
                floor_ft=floor_ft,
                ceiling_ft=ceiling_ft,
                geometry=geom,
                country="US",
                source="faa_nasr",
                ceiling_unlimited=ceiling_unlimited,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping unparseable SFRA feature: %s", exc)
            return None

    def fetch_class_airspace(
        self,
        bounds: Tuple[float, float, float, float],
        classes: Optional[List[str]] = None,
    ) -> List[Airspace]:
        """Fetch FAA Class B/C/D/E airspace within bounds.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            classes: Optional filter, e.g. ``["B", "C", "D"]``.
                Defaults to ``["B", "C", "D"]`` (E is omitted by default
                as it covers very large areas and is rarely restrictive).

        Returns:
            List of :class:`Airspace` objects with ``source="faa_nasr"``.
        """
        if classes is None:
            classes = ["B", "C", "D"]

        cache_dir = os.path.join(_get_airspace_cache_dir(), "nasr")
        os.makedirs(cache_dir, exist_ok=True)
        rounded = tuple(round(b, 1) for b in bounds)
        cls_key = "_".join(sorted(c.upper() for c in classes))
        cache_file = os.path.join(
            cache_dir,
            f"class_{cls_key}_"
            + hashlib.md5(str(rounded).encode()).hexdigest() + ".json",
        )

        if not _is_cache_stale(cache_file, self._cache_ttl_hours):
            logger.info("Using cached class airspace data: %s", cache_file)
            with open(cache_file) as f:
                features = json.load(f)
        else:
            features = self._fetch_class_arcgis(bounds, classes)
            try:
                with open(cache_file, "w") as f:
                    json.dump(features, f)
            except OSError as exc:
                logger.warning("Failed to write class airspace cache: %s", exc)

        airspaces = []
        for feat in features:
            parsed = self._class_feature_to_airspace(feat)
            if parsed is not None:
                airspaces.append(parsed)
        return airspaces

    def _fetch_class_arcgis(
        self,
        bounds: Tuple[float, float, float, float],
        classes: List[str],
    ) -> List[dict]:
        """Query the ArcGIS Class_Airspace layer."""
        min_lon, min_lat, max_lon, max_lat = bounds

        # Build WHERE clause for the requested classes
        class_values = ",".join(f"'{c.upper()}'" for c in classes)
        where = f"CLASS IN ({class_values})"

        params = {
            "where": where,
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "f": "geojson",
        }

        try:
            resp = requests.get(
                self.CLASS_AIRSPACE_URL, params=params, timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise HyPlanRuntimeError(
                f"FAA class airspace ArcGIS query failed: {exc}"
            ) from exc

        data = resp.json()
        return data.get("features", [])  # type: ignore[no-any-return]

    @staticmethod
    def _class_feature_to_airspace(feature: dict) -> Optional[Airspace]:
        """Convert a Class_Airspace GeoJSON feature to an Airspace."""
        try:
            geom_data = feature.get("geometry")
            if not geom_data or "type" not in geom_data:
                return None
            geom = shape(geom_data)
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None

            props = feature.get("properties", {})
            name = props.get("NAME", "Unknown")
            local_type = props.get("LOCAL_TYPE", "")
            airspace_class = props.get("CLASS", local_type)

            # Parse altitudes
            floor_ft = 0.0
            ceiling_ft = 60000.0

            lower_val = props.get("LOWER_VAL")
            if lower_val is not None:
                try:
                    floor_ft = float(lower_val)
                    if props.get("LOWER_UOM") == "FL":
                        floor_ft *= 100.0
                except (ValueError, TypeError):
                    pass

            upper_val = props.get("UPPER_VAL")
            if upper_val is not None:
                try:
                    ceiling_ft = float(upper_val)
                    if props.get("UPPER_UOM") == "FL":
                        ceiling_ft *= 100.0
                except (ValueError, TypeError):
                    pass

            as_type = _CLASS_AIRSPACE_TYPE_MAP.get(local_type, 0)

            return Airspace(
                name=name,
                airspace_class=airspace_class or "OTHER",
                airspace_type=as_type,
                floor_ft=floor_ft,
                ceiling_ft=ceiling_ft,
                geometry=geom,
                country="US",
                source="faa_nasr",
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping unparseable class airspace feature: %s", exc)
            return None


# ---------------------------------------------------------------------------
# FlightPlanDB oceanic tracks (NAT / PACOT)
# ---------------------------------------------------------------------------


@dataclass
class OceanicTrack:
    """A single oceanic track (NAT or PACOT).

    Attributes:
        ident: Track identifier (e.g. "A", "Z", 1, 2).
        system: ``"NAT"`` or ``"PACOT"``.
        valid_from: ISO 8601 start of validity window.
        valid_to: ISO 8601 end of validity window.
        waypoints: List of ``(lon, lat, ident)`` tuples.
        east_levels: Permitted eastbound flight levels, or None.
        west_levels: Permitted westbound flight levels, or None.
        geometry: Shapely LineString of the track.
    """

    ident: "str | int"
    system: str
    valid_from: str
    valid_to: str
    waypoints: List[Tuple[float, float, str]]
    east_levels: Optional[List[str]] = None
    west_levels: Optional[List[str]] = None
    geometry: Optional["LineString"] = None


class FlightPlanDBClient:
    """Fetch NAT and PACOT oceanic tracks from FlightPlanDatabase.com.

    **No API key required** (100 requests/day unauthenticated).
    An optional key raises the limit to 2,500/day.

    Args:
        api_key: Optional FlightPlanDB API key.
        cache_ttl_hours: Cache time-to-live in hours (default 6).
    """

    BASE_URL = "https://api.flightplandatabase.com"
    NAT_URL = f"{BASE_URL}/nav/NATS"
    PACOT_URL = f"{BASE_URL}/nav/PACOTS"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_hours: float = 6.0,
    ):
        self._api_key = api_key
        self._cache_ttl = cache_ttl_hours

    def fetch_nats(self) -> List[OceanicTrack]:
        """Fetch current North Atlantic Tracks."""
        return self._fetch_tracks(self.NAT_URL, "NAT")

    def fetch_pacots(self) -> List[OceanicTrack]:
        """Fetch current Pacific Organized Tracks."""
        return self._fetch_tracks(self.PACOT_URL, "PACOT")

    def _fetch_tracks(self, url: str, system: str) -> List[OceanicTrack]:
        """Fetch and parse tracks from a FlightPlanDB endpoint."""
        from shapely.geometry import LineString

        cache_dir = os.path.join(_get_airspace_cache_dir(), "oceanic")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{system.lower()}.json")

        if not _is_cache_stale(cache_file, self._cache_ttl):
            logger.info("Using cached %s tracks: %s", system, cache_file)
            with open(cache_file) as f:
                raw = json.load(f)
        else:
            headers = {"User-Agent": "hyplan/1.0"}
            if self._api_key:
                import base64
                creds = base64.b64encode(f"{self._api_key}:".encode()).decode()
                headers["Authorization"] = f"Basic {creds}"

            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise HyPlanRuntimeError(
                    f"FlightPlanDB {system} request failed: {exc}"
                ) from exc

            raw = resp.json()
            try:
                with open(cache_file, "w") as f:
                    json.dump(raw, f)
            except OSError as exc:
                logger.warning("Failed to write %s cache: %s", system, exc)

        tracks = []
        for item in raw:
            try:
                route = item.get("route", {})
                nodes = route.get("nodes", [])
                waypoints = [
                    (n["lon"], n["lat"], n.get("ident", ""))
                    for n in nodes
                    if "lon" in n and "lat" in n
                ]
                geom = LineString([(w[0], w[1]) for w in waypoints]) if len(waypoints) >= 2 else None

                tracks.append(OceanicTrack(
                    ident=item.get("ident", ""),
                    system=system,
                    valid_from=item.get("validFrom", ""),
                    valid_to=item.get("validTo", ""),
                    waypoints=waypoints,
                    east_levels=route.get("eastLevels"),
                    west_levels=route.get("westLevels"),
                    geometry=geom,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping unparseable %s track: %s", system, exc)

        logger.info("Fetched %d %s tracks", len(tracks), system)
        return tracks
