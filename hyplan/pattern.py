"""Flight pattern objects.

A :class:`Pattern` is a first-class campaign entity that bundles the output
of a pattern generator (``rosette``, ``racetrack``, ``polygon``,
``sawtooth``, ``spiral``) with the parameters used to generate it.  Patterns
hold either :class:`~hyplan.flight_line.FlightLine` objects (line-based
patterns) or :class:`~hyplan.waypoint.Waypoint` objects (continuous
patterns).  ``compute_flight_plan`` accepts ``Pattern`` in its
``flight_sequence`` and expands it inline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List

from .flight_line import FlightLine
from .waypoint import Waypoint
from .units import ureg
from .exceptions import HyPlanValueError


LINE_BASED_KINDS = frozenset({"rosette", "racetrack"})
WAYPOINT_BASED_KINDS = frozenset({"polygon", "sawtooth", "spiral", "glint_arc"})
PATTERN_KINDS = LINE_BASED_KINDS | WAYPOINT_BASED_KINDS


@dataclass
class Pattern:
    """A named, parameterized pattern within a campaign.

    Attributes:
        pattern_id: Stable identifier assigned by the owning Campaign.
            Empty string for patterns not yet added to a campaign.
        kind: Generator kind ("rosette", "racetrack", "polygon",
            "sawtooth", "spiral").
        name: Human-readable name.
        params: Generator parameters as a plain-JSON-compatible dict
            (lengths/altitudes in meters).  Sufficient to regenerate.
        lines: Ordered mapping of line_id -> FlightLine for line-based
            patterns.  Empty for waypoint-based patterns.
        waypoints: Ordered list of Waypoints for continuous patterns.
            Empty for line-based patterns.
    """

    kind: str
    name: str
    params: dict
    pattern_id: str = ""
    lines: Dict[str, FlightLine] = field(default_factory=dict)
    waypoints: List[Waypoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in PATTERN_KINDS:
            raise HyPlanValueError(
                f"Unknown pattern kind: {self.kind!r}. "
                f"Expected one of {sorted(PATTERN_KINDS)}."
            )
        if self.is_line_based and self.waypoints:
            raise HyPlanValueError(
                f"Line-based pattern '{self.kind}' cannot carry waypoints."
            )
        if self.is_waypoint_based and self.lines:
            raise HyPlanValueError(
                f"Waypoint-based pattern '{self.kind}' cannot carry flight lines."
            )

    @property
    def is_line_based(self) -> bool:
        return self.kind in LINE_BASED_KINDS

    @property
    def is_waypoint_based(self) -> bool:
        return self.kind in WAYPOINT_BASED_KINDS

    @property
    def line_ids(self) -> List[str]:
        return list(self.lines.keys())

    def elements(self):
        """Return the ordered flight lines or waypoints for this pattern."""
        if self.is_line_based:
            return list(self.lines.values())
        return list(self.waypoints)

    def replace_line(self, line_id: str, line: FlightLine) -> None:
        """Replace a line in place, preserving its ID and pattern membership."""
        if not self.is_line_based:
            raise HyPlanValueError(
                f"Cannot replace line on waypoint-based pattern '{self.kind}'."
            )
        if line_id not in self.lines:
            raise HyPlanValueError(
                f"Line '{line_id}' is not part of pattern '{self.pattern_id}'."
            )
        if not isinstance(line, FlightLine):
            raise HyPlanValueError("line must be a FlightLine instance.")
        self.lines[line_id] = line

    def to_dict(self) -> dict:
        """Serialize the pattern to a plain JSON-compatible dict."""
        out: dict = {
            "pattern_id": self.pattern_id,
            "kind": self.kind,
            "name": self.name,
            "params": self.params,
        }
        if self.is_line_based:
            out["lines"] = [
                {"line_id": lid, **fl.to_geojson()}
                for lid, fl in self.lines.items()
            ]
        else:
            out["waypoints"] = [_waypoint_to_dict(wp) for wp in self.waypoints]
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Pattern":
        """Reconstruct a Pattern from a dict produced by :meth:`to_dict`."""
        kind = data["kind"]
        pattern = cls(
            pattern_id=data.get("pattern_id", ""),
            kind=kind,
            name=data.get("name", kind),
            params=dict(data.get("params", {})),
        )
        if kind in LINE_BASED_KINDS:
            for entry in data.get("lines", []):
                line_id = entry["line_id"]
                feature = {
                    "type": "Feature",
                    "geometry": entry["geometry"],
                    "properties": entry.get("properties", {}),
                }
                pattern.lines[line_id] = FlightLine.from_geojson(feature)
        else:
            for wd in data.get("waypoints", []):
                pattern.waypoints.append(_waypoint_from_dict(wd))
        return pattern

    def to_geojson(self) -> dict:
        """Return a GeoJSON FeatureCollection of this pattern's elements.

        Line-based patterns yield one LineString feature per leg (with
        ``line_id`` and ``pattern_id`` in properties).  Waypoint-based
        patterns yield one Point feature per waypoint plus one LineString
        for the connecting track.
        """
        features: list = []
        if self.is_line_based:
            for line_id, fl in self.lines.items():
                feat = fl.to_geojson()
                feat["id"] = line_id
                feat.setdefault("properties", {})
                feat["properties"]["line_id"] = line_id
                feat["properties"]["pattern_id"] = self.pattern_id
                feat["properties"]["pattern_kind"] = self.kind
                features.append(feat)
        else:
            coords = []
            for i, wp in enumerate(self.waypoints):
                coords.append([wp.longitude, wp.latitude])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [wp.longitude, wp.latitude],
                    },
                    "properties": {
                        "name": wp.name,
                        "altitude_msl": (
                            wp.altitude_msl.m_as(ureg.meter)
                            if wp.altitude_msl is not None else None
                        ),
                        "heading": wp.heading,
                        "index": i,
                        "pattern_id": self.pattern_id,
                        "pattern_kind": self.kind,
                    },
                })
            if len(coords) >= 2:
                features.insert(0, {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "pattern_id": self.pattern_id,
                        "pattern_kind": self.kind,
                        "name": self.name,
                    },
                })
        return {"type": "FeatureCollection", "features": features}

    def regenerate(self, **overrides) -> "Pattern":
        """Return a new Pattern by re-invoking the generator with params.

        Any keyword overrides are merged into :attr:`params` for the
        regeneration call.  The returned Pattern is not yet added to a
        campaign; use :meth:`Campaign.replace_pattern` to swap it in.
        """
        from . import flight_patterns  # lazy import to avoid cycles

        generator = getattr(flight_patterns, self.kind)
        merged = copy.deepcopy(self.params)
        merged.update(overrides)
        new_pattern = _invoke_generator(generator, self.kind, merged)
        new_pattern.name = self.name
        return new_pattern


def _waypoint_to_dict(wp: Waypoint) -> dict:
    return {
        "latitude": wp.latitude,
        "longitude": wp.longitude,
        "heading": wp.heading,
        "altitude_msl_m": (
            wp.altitude_msl.m_as(ureg.meter) if wp.altitude_msl is not None else None
        ),
        "name": wp.name,
        "segment_type": wp.segment_type,
    }


def _waypoint_from_dict(d: dict) -> Waypoint:
    alt = d.get("altitude_msl_m")
    return Waypoint(
        latitude=d["latitude"],
        longitude=d["longitude"],
        heading=d["heading"],
        altitude_msl=(alt * ureg.meter if alt is not None else None),
        name=d.get("name"),
        segment_type=d.get("segment_type"),
    )


def _invoke_generator(generator, kind: str, params: dict) -> "Pattern":
    """Re-invoke a generator from a stored params dict (meters/degrees only)."""
    center = (params["center_lat"], params["center_lon"])
    heading = params.get("heading", 0.0)
    if kind == "rosette":
        return generator(
            center=center,
            heading=heading,
            altitude=params["altitude_msl_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_lines=params.get("n_lines", 3),
            angles=params.get("angles"),
        )
    if kind == "racetrack":
        offset = params.get("offset_m", 0)
        if isinstance(offset, list):
            offset_q = [o * ureg.meter for o in offset]
        else:
            offset_q = offset * ureg.meter
        altitudes = params.get("altitudes_m")
        stack_altitudes = params.get("stack_altitudes_m")
        return generator(
            center=center,
            heading=heading,
            altitude=params["altitude_msl_m"] * ureg.meter,
            leg_length=params["leg_length_m"] * ureg.meter,
            n_legs=params.get("n_legs", 1),
            offset=offset_q,
            altitudes=[a * ureg.meter for a in altitudes] if altitudes else None,
            stack_altitudes=(
                [a * ureg.meter for a in stack_altitudes] if stack_altitudes else None
            ),
        )
    if kind == "polygon":
        return generator(
            center=center,
            heading=heading,
            altitude=params["altitude_msl_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_sides=int(params.get("n_sides", 4)),
            aspect_ratio=float(params.get("aspect_ratio", 1.0)),
            closed=bool(params.get("closed", True)),
        )
    if kind == "sawtooth":
        return generator(
            center=center,
            heading=heading,
            altitude_min=params["altitude_min_m"] * ureg.meter,
            altitude_max=params["altitude_max_m"] * ureg.meter,
            leg_length=params["leg_length_m"] * ureg.meter,
            n_cycles=int(params.get("n_cycles", 1)),
        )
    if kind == "spiral":
        return generator(
            center=center,
            heading=heading,
            altitude_start=params["altitude_start_m"] * ureg.meter,
            altitude_end=params["altitude_end_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_turns=float(params.get("n_turns", 3.0)),
            direction=str(params.get("direction", "right")),
            points_per_turn=int(params.get("points_per_turn", 36)),
        )
    if kind == "glint_arc":
        import datetime as _dt
        obs_raw = params["observation_datetime"]
        if isinstance(obs_raw, str):
            obs_dt = _dt.datetime.fromisoformat(obs_raw.replace("Z", "+00:00"))
        else:
            obs_dt = obs_raw
        cl_m = params.get("collection_length_m")
        return generator(
            center=center,
            observation_datetime=obs_dt,
            altitude=params["altitude_msl_m"] * ureg.meter,
            speed=params["speed_mps"] * (ureg.meter / ureg.second),
            bank_angle=params.get("bank_angle"),
            bank_direction=str(params.get("bank_direction", "right")),
            collection_length=(cl_m * ureg.meter if cl_m is not None else None),
            densify_m=float(params.get("densify_m", 200.0)),
        )
    raise HyPlanValueError(f"Unknown pattern kind: {kind}")


__all__ = [
    "Pattern",
    "LINE_BASED_KINDS",
    "WAYPOINT_BASED_KINDS",
    "PATTERN_KINDS",
]
