"""Airspace data retrieval and conflict detection.

Fetches structured airspace data from the OpenAIP API, caches it locally,
and checks flight lines for airspace conflicts (horizontal intersection
plus vertical altitude overlap).

Requires an OpenAIP API key, set via the ``OPENAIP_API_KEY`` environment
variable or passed directly to :class:`OpenAIPClient`.

Data source
-----------
OpenAIP (https://www.openaip.net). Airspace boundaries and vertical
limits retrieved via the OpenAIP REST API. An API key is required;
see https://www.openaip.net/users/clients for registration.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import requests
from shapely.geometry import LineString, Polygon, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely import STRtree

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
            ``"CTR"``, ``"FIR"``, etc.
        airspace_type: OpenAIP numeric type code (0=other, 1=restricted,
            2=danger, 3=prohibited, 4=CTR, 5=TMZ, …).
        floor_ft: Lower altitude limit in feet MSL.  0 means surface.
        ceiling_ft: Upper altitude limit in feet MSL.
        geometry: Shapely polygon of the lateral boundary.
        country: ISO 3166-1 alpha-2 country code (e.g. ``"US"``).
        source: Data source identifier.
    """

    name: str
    airspace_class: str
    airspace_type: int
    floor_ft: float
    ceiling_ft: float
    geometry: Union[Polygon, MultiPolygon]
    country: str = ""
    source: str = "openaip"


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
    """

    airspace: Airspace
    flight_line_index: int
    horizontal_intersection: BaseGeometry
    vertical_overlap_ft: Tuple[float, float]


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
        fl_alt_ft = fl.altitude_msl.to(ureg.foot).magnitude

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

            conflicts.append(
                AirspaceConflict(
                    airspace=airspace,
                    flight_line_index=fl_idx,
                    horizontal_intersection=intersection,
                    vertical_overlap_ft=(overlap_floor, overlap_ceil),
                )
            )

    return conflicts


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
}


def _tile_range(lo: float, hi: float, step: float) -> List[float]:
    """Return evenly spaced points covering [lo, hi] with at most *step* gap."""
    import math

    n = max(1, math.ceil((hi - lo) / step)) + 1
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


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


def _cache_key(bounds: Tuple[float, float, float, float], country: Optional[str]) -> str:
    """Compute a deterministic cache filename from query parameters."""
    # Round bounds to 1 decimal degree so nearby queries share cache
    rounded = tuple(round(b, 1) for b in bounds)
    raw = f"{rounded}_{country or 'all'}"
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
        country: Optional[str] = None,
        max_age_hours: float = 24.0,
    ) -> List[Airspace]:
        """Fetch airspaces within a bounding box.

        Uses a local JSON cache; stale entries are re-fetched.

        Args:
            bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            country: Optional ISO 2-letter country code filter.
            max_age_hours: Maximum cache age before re-fetching.

        Returns:
            List of :class:`Airspace` objects.

        Raises:
            HyPlanRuntimeError: On network or API errors.
        """
        airspaces, _ = self.fetch_airspaces_raw(bounds, country, max_age_hours)
        return airspaces

    def fetch_airspaces_raw(
        self,
        bounds: Tuple[float, float, float, float],
        country: Optional[str] = None,
        max_age_hours: float = 24.0,
    ) -> Tuple[List[Airspace], List[dict]]:
        """Fetch airspaces and return both parsed objects and raw JSON items.

        Same as :meth:`fetch_airspaces` but also returns the raw API
        response items, which can be persisted for later re-parsing.

        Returns:
            ``(airspaces, raw_items)`` tuple.
        """
        cache_dir = _get_airspace_cache_dir()
        cache_file = os.path.join(cache_dir, _cache_key(bounds, country))

        if not _is_cache_stale(cache_file, max_age_hours):
            logger.info("Using cached airspace data: %s", cache_file)
            with open(cache_file) as f:
                items = json.load(f)
            return parse_airspace_items(items), items

        # The OpenAIP API searches by center + radius but appears to match
        # against airspace centroids rather than polygon boundaries.  This
        # means large airspaces whose centroid is far from the query point
        # can be missed even with a large ``dist``.  To work around this we
        # tile the bounding box with a dense grid of overlapping queries.
        min_lon, min_lat, max_lon, max_lat = bounds

        tile_step = 0.1  # degrees (~11 km) — dense enough to catch most airspaces
        query_dist = 30  # km "dist" value passed to the API

        lat_points = _tile_range(min_lat, max_lat, tile_step)
        lon_points = _tile_range(min_lon, max_lon, tile_step)

        seen_ids: set = set()
        items: List[dict] = []
        for lat in lat_points:
            for lon in lon_points:
                page_items = self._fetch_all_pages(lat, lon, query_dist, country)
                for it in page_items:
                    item_id = it.get("_id") or it.get("id") or id(it)
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        items.append(it)

        # Cache the raw JSON
        try:
            with open(cache_file, "w") as f:
                json.dump(items, f)
            logger.info("Cached %d airspace items to %s", len(items), cache_file)
        except OSError as exc:
            logger.warning("Failed to write airspace cache: %s", exc)

        return parse_airspace_items(items), items

    def _fetch_all_pages(
        self,
        lat: float,
        lon: float,
        radius_nm: float,
        country: Optional[str],
    ) -> List[dict]:
        """Fetch all pages of airspace results from the API."""
        all_items: List[dict] = []
        page = 1
        limit = 100

        while True:
            params = {
                "pos": f"{lat},{lon}",
                "dist": int(radius_nm),
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


def fetch_and_check(
    flight_lines,
    api_key: Optional[str] = None,
    buffer_m: float = 1000.0,
    country: Optional[str] = None,
    max_age_hours: float = 24.0,
) -> List[AirspaceConflict]:
    """Fetch nearby airspaces and check flight lines for conflicts.

    This is a one-call convenience function that:

    1. Computes a bounding box from the flight lines (with buffer).
    2. Fetches airspaces from OpenAIP within that bounding box.
    3. Runs :func:`check_airspace_conflicts`.

    Args:
        flight_lines: Iterable of objects with ``.geometry`` and
            ``.altitude_msl`` attributes.
        api_key: OpenAIP API key (or set ``OPENAIP_API_KEY`` env var).
        buffer_m: Buffer in meters added to the bounding box.
        country: Optional ISO 2-letter country code filter.
        max_age_hours: Maximum cache age before re-fetching.

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

    client = OpenAIPClient(api_key=api_key)
    airspaces = client.fetch_airspaces(
        bounds=bounds, country=country, max_age_hours=max_age_hours
    )

    return check_airspace_conflicts(fl_list, airspaces)
