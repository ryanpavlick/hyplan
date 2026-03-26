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
from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import box, mapping, shape, LineString, Polygon

from .airspace import (
    Airspace,
    AirspaceConflict,
    OpenAIPClient,
    check_airspace_conflicts,
    parse_airspace_items,
)
from .exceptions import HyPlanRuntimeError, HyPlanValueError
from .flight_line import FlightLine
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
            self._bounds = polygon.bounds  # (minx, miny, maxx, maxy)

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
        return check_airspace_conflicts(flight_lines, self._airspaces)

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
            group["generation"] = generation_params

        self._groups.append(group)
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
        logger.info("Removed group '%s' from campaign '%s'", group_id, self._name)

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
            "fetched_at": self._fetch_timestamp,
            "line_counter": self._line_counter,
            "group_counter": self._group_counter,
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

        # flight_lines/
        if self._flight_lines:
            fl_dir = os.path.join(path, "flight_lines")
            os.makedirs(fl_dir, exist_ok=True)

            # all_lines.geojson — FeatureCollection with IDs
            features = []
            for line_id, fl in self._flight_lines.items():
                feature = fl.to_geojson()
                feature["id"] = line_id
                feature["properties"]["line_id"] = line_id
                features.append(feature)

            _write_json(os.path.join(fl_dir, "all_lines.geojson"), {
                "type": "FeatureCollection",
                "features": features,
            })

            # groups.json
            _write_json(os.path.join(fl_dir, "groups.json"), {
                "groups": self._groups,
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
                fl = _flight_line_from_geojson(feature)
                campaign._flight_lines[line_id] = fl
            logger.info("Loaded %d flight lines from %s", len(campaign._flight_lines), fl_file)

        groups_file = os.path.join(path, "flight_lines", "groups.json")
        if os.path.exists(groups_file):
            groups_data = _read_json(groups_file)
            campaign._groups = groups_data.get("groups", [])

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
        lines.append(f"  Flight lines: {len(self._flight_lines)}")
        lines.append(f"  Groups: {len(self._groups)}")
        for g in self._groups:
            lines.append(
                f"    {g['id']}: {g['name']} ({g['type']}, "
                f"{len(g['line_ids'])} lines)"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Campaign(name={self._name!r}, bounds={self._bounds}, "
            f"airspaces={len(self._airspaces) if self._airspaces else 0}, "
            f"lines={len(self._flight_lines)}, groups={len(self._groups)})"
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
        return json.load(f)


def _flight_line_from_geojson(feature: dict) -> FlightLine:
    """Reconstruct a FlightLine from a GeoJSON Feature dict."""
    coords = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})

    lon1, lat1 = coords[0]
    lon2, lat2 = coords[-1]
    alt_m = props.get("altitude_msl")

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
