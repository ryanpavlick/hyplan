"""Campaign management with folder-based persistence.

A :class:`Campaign` defines a geographic domain for a flight campaign,
fetches and caches reference data (airspace, airports, terrain), and
manages a master library of flight lines organized into groups.

The entire campaign is stored as a plain folder::

    my_campaign/
    ├── campaign.json
    ├── domain.geojson
    ├── airspaces.json
    ├── flight_lines/
    │   ├── all_lines.geojson
    │   └── groups.json
    └── waypoints.geojson

Each file uses its natural format (GeoJSON for geometry, JSON for
structured data) and can be opened independently in QGIS, Python, etc.
"""

import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from shapely.geometry import box, mapping, shape, Polygon

from .airspace import (
    Airspace,
    AirspaceConflict,
    OpenAIPClient,
    check_airspace_conflicts,
    parse_airspace_items,
)
from .exceptions import HyPlanRuntimeError, HyPlanValueError
from .flight_line import FlightLine
from .pattern import Pattern
from .units import ureg
from .waypoint import Waypoint

logger = logging.getLogger(__name__)


class Campaign:
    """A flight campaign with geographic domain, reference data, and flight lines.

    Args:
        name: Human-readable campaign name.
        bounds: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            Mutually exclusive with *polygon*.
        polygon: Shapely Polygon defining the domain boundary.
            Mutually exclusive with *bounds*.
        country: Optional ISO 2-letter country code to filter API results.

    Raises:
        HyPlanValueError: If neither or both of bounds/polygon are given,
            or if coordinates are out of range.
    """

    def __init__(
        self,
        name: str,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        polygon: Optional[Polygon] = None,
        country: Optional[str] = None,
    ):
        if bounds is not None and polygon is not None:
            raise HyPlanValueError(
                "Provide either bounds or polygon, not both."
            )
        if bounds is None and polygon is None:
            raise HyPlanValueError(
                "Either bounds or polygon must be provided."
            )

        if bounds is not None:
            min_lon, min_lat, max_lon, max_lat = bounds
            if min_lon >= max_lon:
                raise HyPlanValueError(
                    f"min_lon ({min_lon}) must be less than max_lon ({max_lon})"
                )
            if min_lat >= max_lat:
                raise HyPlanValueError(
                    f"min_lat ({min_lat}) must be less than max_lat ({max_lat})"
                )
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                raise HyPlanValueError("Longitude must be between -180 and 180")
            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                raise HyPlanValueError("Latitude must be between -90 and 90")
            self._polygon = box(min_lon, min_lat, max_lon, max_lat)
            self._bounds = bounds
        else:
            self._polygon = polygon
            self._bounds = polygon.bounds  # type: ignore[union-attr]

        self._name = name
        self._country = country

        # Airspace data
        self._airspaces: Optional[List[Airspace]] = None
        self._raw_airspace_items: Optional[List[dict]] = None
        self._fetch_timestamp: Optional[str] = None

        # Flight lines and groups
        self._flight_lines: Dict[str, FlightLine] = {}  # id -> FlightLine
        self._groups: List[dict] = []
        self._line_counter: int = 0
        self._group_counter: int = 0

        # Patterns (rosette, racetrack, polygon, sawtooth, spiral)
        self._patterns: Dict[str, Pattern] = {}
        self._pattern_counter: int = 0

        # Revision metadata
        import uuid
        self._campaign_id: str = str(uuid.uuid4())
        self._revision: int = 0
        self._updated_at: str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._bounds

    @property
    def country(self) -> Optional[str]:
        return self._country

    @property
    def airspaces(self) -> Optional[List[Airspace]]:
        return self._airspaces

    @property
    def is_fetched(self) -> bool:
        return self._airspaces is not None

    @property
    def flight_lines(self) -> List[FlightLine]:
        return list(self._flight_lines.values())

    @property
    def flight_line_ids(self) -> List[str]:
        return list(self._flight_lines.keys())

    @property
    def groups(self) -> List[dict]:
        return list(self._groups)

    @property
    def patterns(self) -> List[Pattern]:
        """All patterns attached to this campaign, in insertion order."""
        return list(self._patterns.values())

    @property
    def pattern_ids(self) -> List[str]:
        return list(self._patterns.keys())

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def updated_at(self) -> str:
        return self._updated_at

    # ------------------------------------------------------------------
    # Airspace
    # ------------------------------------------------------------------

    def fetch_airspaces(
        self,
        api_key: Optional[str] = None,
        force: bool = False,
    ) -> "Campaign":
        """Fetch airspaces from OpenAIP for this domain.

        If data is already loaded (from a previous fetch or from
        :meth:`load`), this is a no-op unless *force=True*.

        Returns self for chaining.
        """
        if self._airspaces is not None and not force:
            logger.info("Airspace data already loaded; skipping fetch.")
            return self

        client = OpenAIPClient(api_key=api_key)
        airspaces, raw_items = client.fetch_airspaces_raw(
            self._bounds, country=self._country
        )
        self._airspaces = airspaces
        self._raw_airspace_items = raw_items
        self._fetch_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        logger.info("Fetched %d airspaces for campaign '%s'", len(airspaces), self._name)
        return self

    def check_conflicts(
        self,
        flight_lines=None,
    ) -> List[AirspaceConflict]:
        """Check flight lines against this campaign's airspaces.

        Args:
            flight_lines: Lines to check.  If *None*, checks this
                campaign's own flight lines.

        Raises:
            HyPlanRuntimeError: If airspace data has not been fetched.
        """
        if self._airspaces is None:
            raise HyPlanRuntimeError(
                "No airspace data loaded. Call fetch_airspaces() or load() first."
            )
        if flight_lines is None:
            flight_lines = self.flight_lines
        return check_airspace_conflicts(flight_lines, self._airspaces)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Flight lines and groups
    # ------------------------------------------------------------------

    def add_flight_lines(
        self,
        lines: List[FlightLine],
        group_name: Optional[str] = None,
        group_type: str = "manual",
        generation_params: Optional[dict] = None,
    ) -> str:
        """Add flight lines to the campaign and create a group.

        Args:
            lines: Flight lines to add.
            group_name: Human-readable group name.  Defaults to
                ``"group_NNN"``.
            group_type: Group type label (``"flight_box"``,
                ``"pattern"``, ``"single_line"``, ``"manual"``).
            generation_params: Optional dict recording how the lines
                were generated (method, parameters) for reproducibility.

        Returns:
            The group ID string.
        """
        self._group_counter += 1
        group_id = f"group_{self._group_counter:03d}"

        line_ids = []
        for fl in lines:
            self._line_counter += 1
            line_id = f"line_{self._line_counter:03d}"
            self._flight_lines[line_id] = fl
            line_ids.append(line_id)

        group = {
            "id": group_id,
            "name": group_name or group_id,
            "type": group_type,
            "line_ids": line_ids,
        }
        if generation_params is not None:
            group["generation"] = generation_params  # type: ignore[assignment]

        self._groups.append(group)
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(
            "Added group '%s' with %d lines to campaign '%s'",
            group["name"], len(line_ids), self._name,
        )
        return group_id

    def remove_group(self, group_id: str) -> None:
        """Remove a group and its flight lines from the campaign."""
        group = None
        for g in self._groups:
            if g["id"] == group_id:
                group = g
                break
        if group is None:
            raise HyPlanValueError(f"Group '{group_id}' not found.")

        for line_id in group["line_ids"]:
            self._flight_lines.pop(line_id, None)
        self._groups.remove(group)
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("Removed group '%s' from campaign '%s'", group_id, self._name)

    def remove_flight_line(self, line_id: str) -> None:
        """Remove a flight line by ID and update group memberships.

        Removes *line_id* from the campaign-wide flight-line registry and
        from any groups that reference it.  Groups left with no remaining
        lines are removed.

        Args:
            line_id: Stable campaign line ID, e.g. ``"line_001"``.

        Raises:
            HyPlanValueError: If *line_id* is not present in the campaign.
        """
        if line_id not in self._flight_lines:
            raise HyPlanValueError(f"Flight line '{line_id}' not found.")

        self._flight_lines.pop(line_id)

        empty_group_ids = []
        for group in self._groups:
            if line_id in group["line_ids"]:
                group["line_ids"] = [lid for lid in group["line_ids"] if lid != line_id]
                if not group["line_ids"]:
                    empty_group_ids.append(group["id"])

        if empty_group_ids:
            self._groups = [g for g in self._groups if g["id"] not in empty_group_ids]

        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(
            "Removed flight line '%s' from campaign '%s'%s",
            line_id, self._name,
            f" and removed {len(empty_group_ids)} empty group(s)" if empty_group_ids else "",
        )

    def replace_flight_line(self, line_id: str, line: FlightLine) -> None:
        """Replace an existing flight line in place, preserving its ID.

        Args:
            line_id: Stable campaign line ID to replace.
            line: New :class:`~hyplan.flight_line.FlightLine` object.

        Raises:
            HyPlanValueError: If *line_id* is not present in the campaign.
            HyPlanValueError: If *line* is not a ``FlightLine`` instance.
        """
        if line_id not in self._flight_lines:
            raise HyPlanValueError(f"Flight line '{line_id}' not found.")
        if not isinstance(line, FlightLine):
            raise HyPlanValueError("line must be a FlightLine instance.")

        self._flight_lines[line_id] = line
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("Replaced flight line '%s' in campaign '%s'", line_id, self._name)

    def flight_lines_to_geojson(self) -> dict:
        """Return all campaign flight lines as a GeoJSON FeatureCollection.

        Each feature includes the stable ``line_id`` in both the feature
        ``id`` field and in ``properties.line_id``. Includes free-standing
        lines as well as flight lines that belong to line-based patterns;
        pattern lines carry ``pattern_id`` and ``pattern_kind`` in their
        properties.
        """
        features = []
        for line_id, fl in self._flight_lines.items():
            feature = fl.to_geojson()
            feature["id"] = line_id
            feature["properties"]["line_id"] = line_id
            features.append(feature)
        for pattern in self._patterns.values():
            if not pattern.is_line_based:
                continue
            for line_id, fl in pattern.lines.items():
                feature = fl.to_geojson()
                feature["id"] = line_id
                feature["properties"]["line_id"] = line_id
                feature["properties"]["pattern_id"] = pattern.pattern_id
                feature["properties"]["pattern_kind"] = pattern.kind
                features.append(feature)
        return {
            "type": "FeatureCollection",
            "features": features,
        }

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def add_pattern(self, pattern: Pattern) -> str:
        """Add a pattern to the campaign.

        Assigns a stable ``pattern_id`` and rekeys any contained flight
        lines with campaign-global ``line_id`` values.  The input Pattern
        is mutated in place so the caller holds the live reference.

        Args:
            pattern: A :class:`Pattern` returned by a generator.

        Returns:
            The assigned pattern_id.
        """
        if not isinstance(pattern, Pattern):
            raise HyPlanValueError("pattern must be a Pattern instance.")

        self._pattern_counter += 1
        pattern_id = f"pattern_{self._pattern_counter:03d}"
        pattern.pattern_id = pattern_id

        if pattern.is_line_based:
            rekeyed: Dict[str, FlightLine] = {}
            for fl in pattern.lines.values():
                self._line_counter += 1
                line_id = f"line_{self._line_counter:03d}"
                rekeyed[line_id] = fl
            pattern.lines = rekeyed

        self._patterns[pattern_id] = pattern
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(
            "Added %s pattern '%s' to campaign '%s' (%d %s)",
            pattern.kind, pattern_id, self._name,
            len(pattern.lines) if pattern.is_line_based else len(pattern.waypoints),
            "lines" if pattern.is_line_based else "waypoints",
        )
        return pattern_id

    def remove_pattern(self, pattern_id: str) -> None:
        """Remove a pattern and all of its contained flight lines or waypoints."""
        if pattern_id not in self._patterns:
            raise HyPlanValueError(f"Pattern '{pattern_id}' not found.")
        del self._patterns[pattern_id]
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("Removed pattern '%s' from campaign '%s'", pattern_id, self._name)

    def replace_pattern(self, pattern_id: str, new_pattern: Pattern) -> None:
        """Replace a pattern in place, preserving its ``pattern_id``.

        Contained flight lines receive fresh campaign-global ``line_id``
        values — callers that want to preserve individual line IDs should
        use :meth:`Pattern.replace_line` instead.
        """
        if pattern_id not in self._patterns:
            raise HyPlanValueError(f"Pattern '{pattern_id}' not found.")
        if not isinstance(new_pattern, Pattern):
            raise HyPlanValueError("new_pattern must be a Pattern instance.")

        new_pattern.pattern_id = pattern_id
        if new_pattern.is_line_based:
            rekeyed: Dict[str, FlightLine] = {}
            for fl in new_pattern.lines.values():
                self._line_counter += 1
                line_id = f"line_{self._line_counter:03d}"
                rekeyed[line_id] = fl
            new_pattern.lines = rekeyed

        self._patterns[pattern_id] = new_pattern
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("Replaced pattern '%s' in campaign '%s'", pattern_id, self._name)

    def get_pattern(self, pattern_id: str) -> Pattern:
        """Return the Pattern with the given id (raises if not found)."""
        if pattern_id not in self._patterns:
            raise HyPlanValueError(f"Pattern '{pattern_id}' not found.")
        return self._patterns[pattern_id]

    def patterns_to_geojson(self) -> dict:
        """Return waypoint-based patterns as one GeoJSON FeatureCollection for
        display.

        Only **waypoint-based** patterns (polygon, sawtooth, spiral) contribute
        features here; their geometry (Point waypoints + connecting track) is
        not available elsewhere. Line-based patterns (rosette, racetrack) are
        intentionally omitted because their legs are already exposed through
        :meth:`flight_lines_to_geojson`; emitting them again would cause
        duplicate rendering in map clients.
        """
        features = []
        for pattern in self._patterns.values():
            if pattern.is_line_based:
                continue
            features.extend(pattern.to_geojson().get("features", []))
        return {"type": "FeatureCollection", "features": features}

    # ------------------------------------------------------------------
    # Line lookup across free-standing + pattern lines
    # ------------------------------------------------------------------

    def get_line(self, line_id: str) -> FlightLine:
        """Return the FlightLine with *line_id* from free-standing lines
        or any line-based pattern.  Raises if not found."""
        if line_id in self._flight_lines:
            return self._flight_lines[line_id]
        for pattern in self._patterns.values():
            if pattern.is_line_based and line_id in pattern.lines:
                return pattern.lines[line_id]
        raise HyPlanValueError(f"Flight line '{line_id}' not found.")

    def find_pattern_for_line(self, line_id: str) -> Optional[Pattern]:
        """Return the Pattern owning *line_id*, or None if the line is
        free-standing or does not exist."""
        for pattern in self._patterns.values():
            if pattern.is_line_based and line_id in pattern.lines:
                return pattern
        return None

    def all_flight_lines(self) -> List[FlightLine]:
        """Return all FlightLines in the campaign: free-standing plus
        pattern-owned lines, in a stable order."""
        out = list(self._flight_lines.values())
        for pattern in self._patterns.values():
            if pattern.is_line_based:
                out.extend(pattern.lines.values())
        return out

    def all_flight_lines_dict(self) -> Dict[str, FlightLine]:
        """Return ``{line_id: FlightLine}`` for every flight line in the
        campaign — free-standing or pattern-owned."""
        out: Dict[str, FlightLine] = dict(self._flight_lines)
        for pattern in self._patterns.values():
            if pattern.is_line_based:
                out.update(pattern.lines)
        return out

    def replace_line_anywhere(self, line_id: str, line: FlightLine) -> None:
        """Replace a flight line by id whether free-standing or pattern-owned.

        Routes to :meth:`replace_flight_line` for free-standing lines or
        :meth:`Pattern.replace_line` on the owning pattern.
        """
        if line_id in self._flight_lines:
            self.replace_flight_line(line_id, line)
            return
        pattern = self.find_pattern_for_line(line_id)
        if pattern is None:
            raise HyPlanValueError(f"Flight line '{line_id}' not found.")
        pattern.replace_line(line_id, line)
        self._revision += 1
        self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def remove_line_anywhere(self, line_id: str) -> None:
        """Remove a flight line by id whether free-standing or pattern-owned.

        Free-standing removal uses :meth:`remove_flight_line` (deletes the
        line and trims empty groups).  Pattern-owned removal drops the
        leg from the pattern and, if the pattern has no legs left,
        removes the pattern entirely.
        """
        if line_id in self._flight_lines:
            self.remove_flight_line(line_id)
            return
        pattern = self.find_pattern_for_line(line_id)
        if pattern is None:
            raise HyPlanValueError(f"Flight line '{line_id}' not found.")
        del pattern.lines[line_id]
        if not pattern.lines:
            self.remove_pattern(pattern.pattern_id)
        else:
            self._revision += 1
            self._updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Persistence — save
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the campaign to a folder.

        Creates the directory structure and writes all files.
        Existing files are overwritten.

        Args:
            path: Path to the campaign folder.
        """
        os.makedirs(path, exist_ok=True)

        # campaign.json
        meta = {
            "version": 1,
            "name": self._name,
            "country": self._country,
            "campaign_id": self._campaign_id,
            "revision": self._revision,
            "updated_at": self._updated_at,
            "fetched_at": self._fetch_timestamp,
            "line_counter": self._line_counter,
            "group_counter": self._group_counter,
            "pattern_counter": self._pattern_counter,
        }
        _write_json(os.path.join(path, "campaign.json"), meta)

        # domain.geojson
        domain_geojson = {
            "type": "Feature",
            "geometry": mapping(self._polygon),
            "properties": {
                "name": self._name,
                "bounds": list(self._bounds),
            },
        }
        _write_json(os.path.join(path, "domain.geojson"), domain_geojson)

        # airspaces.json (raw items for re-parsing)
        if self._raw_airspace_items is not None:
            _write_json(os.path.join(path, "airspaces.json"), {
                "source": "openaip",
                "fetched_at": self._fetch_timestamp,
                "count": len(self._raw_airspace_items),
                "items": self._raw_airspace_items,
            })

        # flight_lines/ — only free-standing lines (pattern-owned lines
        # are persisted inside their patterns).
        if self._flight_lines:
            fl_dir = os.path.join(path, "flight_lines")
            os.makedirs(fl_dir, exist_ok=True)

            # all_lines.geojson — FeatureCollection with IDs
            free_standing = {
                "type": "FeatureCollection",
                "features": [
                    {**fl.to_geojson(), "id": lid,
                     "properties": {**fl.to_geojson()["properties"], "line_id": lid}}
                    for lid, fl in self._flight_lines.items()
                ],
            }
            _write_json(os.path.join(fl_dir, "all_lines.geojson"), free_standing)

            # groups.json
            _write_json(os.path.join(fl_dir, "groups.json"), {
                "groups": self._groups,
            })

        # patterns/all_patterns.json
        if self._patterns:
            patterns_dir = os.path.join(path, "patterns")
            os.makedirs(patterns_dir, exist_ok=True)
            _write_json(os.path.join(patterns_dir, "all_patterns.json"), {
                "patterns": [p.to_dict() for p in self._patterns.values()],
            })

        logger.info("Saved campaign '%s' to %s", self._name, path)

    # ------------------------------------------------------------------
    # Persistence — load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "Campaign":
        """Load a campaign from a previously saved folder.

        No API calls are made.  Raw airspace items are re-parsed.

        Args:
            path: Path to the campaign folder.

        Returns:
            A fully populated Campaign instance.
        """
        # campaign.json
        meta = _read_json(os.path.join(path, "campaign.json"))
        version = meta.get("version", 0)
        if version != 1:
            raise HyPlanValueError(
                f"Unsupported campaign version: {version}"
            )

        # domain.geojson
        domain_data = _read_json(os.path.join(path, "domain.geojson"))
        polygon = shape(domain_data["geometry"])

        campaign = cls(
            name=meta["name"],
            polygon=polygon,
            country=meta.get("country"),
        )
        campaign._fetch_timestamp = meta.get("fetched_at")
        campaign._line_counter = meta.get("line_counter", 0)
        campaign._group_counter = meta.get("group_counter", 0)
        campaign._pattern_counter = meta.get("pattern_counter", 0)
        if "campaign_id" in meta:
            campaign._campaign_id = meta["campaign_id"]
        campaign._revision = meta.get("revision", 0)
        campaign._updated_at = meta.get("updated_at", campaign._updated_at)

        # airspaces.json (optional)
        airspace_file = os.path.join(path, "airspaces.json")
        if os.path.exists(airspace_file):
            airspace_data = _read_json(airspace_file)
            raw_items = airspace_data.get("items", [])
            campaign._raw_airspace_items = raw_items
            campaign._airspaces = parse_airspace_items(raw_items)
            campaign._fetch_timestamp = airspace_data.get(
                "fetched_at", campaign._fetch_timestamp
            )
            logger.info("Loaded %d airspaces from %s", len(campaign._airspaces), airspace_file)

        # flight_lines/
        fl_file = os.path.join(path, "flight_lines", "all_lines.geojson")
        if os.path.exists(fl_file):
            fc = _read_json(fl_file)
            for feature in fc.get("features", []):
                line_id = feature.get("id") or feature["properties"].get("line_id")
                fl = FlightLine.from_geojson(feature)
                campaign._flight_lines[line_id] = fl
            logger.info("Loaded %d flight lines from %s", len(campaign._flight_lines), fl_file)

        groups_file = os.path.join(path, "flight_lines", "groups.json")
        if os.path.exists(groups_file):
            groups_data = _read_json(groups_file)
            campaign._groups = groups_data.get("groups", [])

        # patterns/
        patterns_file = os.path.join(path, "patterns", "all_patterns.json")
        if os.path.exists(patterns_file):
            patterns_data = _read_json(patterns_file)
            for entry in patterns_data.get("patterns", []):
                pattern = Pattern.from_dict(entry)
                campaign._patterns[pattern.pattern_id] = pattern
            logger.info(
                "Loaded %d patterns from %s",
                len(campaign._patterns), patterns_file,
            )

        return campaign

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the campaign."""
        lines = [f"Campaign: {self._name}"]
        lines.append(f"  Domain: {self._bounds}")
        if self._country:
            lines.append(f"  Country: {self._country}")
        if self._airspaces is not None:
            lines.append(f"  Airspaces: {len(self._airspaces)}")
            if self._fetch_timestamp:
                lines.append(f"  Fetched: {self._fetch_timestamp}")
        else:
            lines.append("  Airspaces: not fetched")
        lines.append(f"  Flight lines (free-standing): {len(self._flight_lines)}")
        lines.append(f"  Groups: {len(self._groups)}")
        for g in self._groups:
            lines.append(
                f"    {g['id']}: {g['name']} ({g['type']}, "
                f"{len(g['line_ids'])} lines)"
            )
        lines.append(f"  Patterns: {len(self._patterns)}")
        for p in self._patterns.values():
            elt_count = len(p.lines) if p.is_line_based else len(p.waypoints)
            elt_label = "lines" if p.is_line_based else "waypoints"
            lines.append(
                f"    {p.pattern_id}: {p.name} ({p.kind}, {elt_count} {elt_label})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Campaign(name={self._name!r}, bounds={self._bounds}, "
            f"airspaces={len(self._airspaces) if self._airspaces else 0}, "
            f"lines={len(self._flight_lines)}, groups={len(self._groups)}, "
            f"patterns={len(self._patterns)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(filepath: str, data: dict) -> None:
    """Write a dict to a JSON file with nice formatting."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _read_json(filepath: str) -> dict:
    """Read a JSON file and return the parsed dict."""
    with open(filepath) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _flight_line_from_geojson(feature: dict) -> FlightLine:
    """Reconstruct a FlightLine from a GeoJSON Feature dict."""
    coords = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})

    lon1, lat1 = coords[0]
    lon2, lat2 = coords[-1]
    alt_m = props.get("altitude_msl")

    if lat1 == lat2 and lon1 == lon2:
        raise HyPlanValueError(
            f"Degenerate flight line in GeoJSON: start == end ({lat1}, {lon1})"
        )

    import pymap3d.vincenty
    _, az12 = pymap3d.vincenty.vdist(lat1, lon1, lat2, lon2)
    _, az21 = pymap3d.vincenty.vdist(lat2, lon2, lat1, lon1)

    alt = ureg.Quantity(alt_m, "meter") if alt_m is not None else None

    wp1 = Waypoint(
        latitude=lat1, longitude=lon1,
        heading=float(az12) % 360,
        altitude_msl=alt,
    )
    wp2 = Waypoint(
        latitude=lat2, longitude=lon2,
        heading=(float(az21) + 180.0) % 360,
        altitude_msl=alt,
    )
    return FlightLine(
        waypoint1=wp1,
        waypoint2=wp2,
        site_name=props.get("site_name"),
        site_description=props.get("site_description"),
        investigator=props.get("investigator"),
    )
