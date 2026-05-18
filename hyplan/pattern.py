"""Flight pattern objects.

A :class:`Pattern` is a first-class campaign entity that bundles the output
of a pattern generator (``rosette``, ``racetrack``, ``polygon``,
``sawtooth``, ``spiral``) with the parameters used to generate it.  Patterns
hold either :class:`~hyplan.flight_line.FlightLine` objects (line-based
patterns) or :class:`~hyplan.waypoint.Waypoint` objects (continuous
patterns).  ``compute_flight_plan`` accepts ``Pattern`` in its
``flight_sequence`` and expands it inline.

Atomicity in the flight-line optimizer
--------------------------------------
A :class:`Pattern` is an **atomic visit item** in
:func:`~hyplan.flight_optimizer.greedy_optimize`. The optimizer may reorder
whole patterns relative to free-standing :class:`~hyplan.flight_line.FlightLine`
items, but **may not split a pattern apart**: every element of a pattern is
visited consecutively, in pattern definition order, before the optimizer
moves on to any other item. Pattern traversal direction is fixed (entry → exit)
in this release; reversal is not supported.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pymap3d
from pint import Quantity

from .exceptions import HyPlanValueError
from .flight_line import FlightLine
from .geometry import wrap_to_180, wrap_to_360
from .units import ureg
from .waypoint import Waypoint

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
    params: dict[str, Any]
    pattern_id: str = ""
    lines: dict[str, FlightLine] = field(default_factory=dict)
    waypoints: list[Waypoint] = field(default_factory=list)

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
    def line_ids(self) -> list[str]:
        return list(self.lines.keys())

    def elements(self) -> list[FlightLine] | list[Waypoint]:
        """Return the ordered flight lines or waypoints for this pattern."""
        if self.is_line_based:
            return list(self.lines.values())
        return list(self.waypoints)

    @property
    def entry_waypoint(self) -> Waypoint:
        """Waypoint where this pattern's traversal begins.

        For line-based patterns, this is the start of the first leg
        (``self.lines[first_line_id].waypoint1``). For waypoint-based
        patterns, this is the first element of ``self.waypoints``.

        Used by the flight-line optimizer to compute transit-in cost
        when scheduling this pattern in a sortie.

        Raises:
            HyPlanValueError: If the pattern has no elements (empty
                ``lines`` and empty ``waypoints``).
        """
        if self.is_line_based:
            if not self.lines:
                raise HyPlanValueError(
                    f"Pattern '{self.pattern_id or self.name}' has no flight lines; "
                    "entry_waypoint is undefined."
                )
            first_line = next(iter(self.lines.values()))
            return first_line.waypoint1
        if not self.waypoints:
            raise HyPlanValueError(
                f"Pattern '{self.pattern_id or self.name}' has no waypoints; "
                "entry_waypoint is undefined."
            )
        return self.waypoints[0]

    @property
    def exit_waypoint(self) -> Waypoint:
        """Waypoint where this pattern's traversal ends.

        For line-based patterns, this is the end of the last leg
        (``self.lines[last_line_id].waypoint2``). For waypoint-based
        patterns, this is the last element of ``self.waypoints``.

        Used by the flight-line optimizer to compute transit-out cost
        when scheduling this pattern in a sortie.

        Raises:
            HyPlanValueError: If the pattern has no elements (empty
                ``lines`` and empty ``waypoints``).
        """
        if self.is_line_based:
            if not self.lines:
                raise HyPlanValueError(
                    f"Pattern '{self.pattern_id or self.name}' has no flight lines; "
                    "exit_waypoint is undefined."
                )
            last_line = next(reversed(self.lines.values()))
            return last_line.waypoint2
        if not self.waypoints:
            raise HyPlanValueError(
                f"Pattern '{self.pattern_id or self.name}' has no waypoints; "
                "exit_waypoint is undefined."
            )
        return self.waypoints[-1]

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pattern to a plain JSON-compatible dict."""
        out: dict[str, Any] = {
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
    def from_dict(cls, data: dict[str, Any]) -> Pattern:
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

    def to_geojson(self) -> dict[str, Any]:
        """Return a GeoJSON FeatureCollection of this pattern's elements.

        Line-based patterns yield one LineString feature per leg (with
        ``line_id`` and ``pattern_id`` in properties).  Waypoint-based
        patterns yield one Point feature per waypoint plus one LineString
        for the connecting track.
        """
        features: list[dict[str, Any]] = []
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

    def regenerate(self, **overrides: Any) -> Pattern:
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

    # ----------------------------------------------------------------
    # Whole-pattern movement (functional — return new Pattern)
    # ----------------------------------------------------------------

    def translate(
        self,
        offset_north: Quantity | float,
        offset_east: Quantity | float,
    ) -> Pattern:
        """Return a new Pattern shifted by the given N/E offsets.

        Every contained :class:`FlightLine` and :class:`Waypoint` is
        moved by the same geodetic N/E offset (delegating to each
        element's existing ``offset_north_east`` method).  The pattern's
        stored ``center_lat`` / ``center_lon`` params are updated to
        match so a subsequent :meth:`regenerate` produces the same
        geometry.

        Args:
            offset_north: Northward distance (negative for south).
                ``float`` is interpreted as metres; pass a pint Quantity
                for other units.
            offset_east: Eastward distance (negative for west); same
                conventions as ``offset_north``.

        Returns:
            A new Pattern at the translated position.  The original
            is unchanged.
        """
        n_m = _length_m(offset_north)
        e_m = _length_m(offset_east)

        new_lines = {
            lid: fl.offset_north_east(offset_north, offset_east)
            for lid, fl in self.lines.items()
        }
        new_waypoints = [
            wp.offset_north_east(offset_north, offset_east)
            for wp in self.waypoints
        ]
        new_params = _translated_params(self.params, n_m, e_m)

        return Pattern(
            kind=self.kind,
            name=self.name,
            params=new_params,
            pattern_id=self.pattern_id,
            lines=new_lines,
            waypoints=new_waypoints,
        )

    def move_to(self, latitude: float, longitude: float) -> Pattern:
        """Return a new Pattern re-anchored at the given centre.

        For built-in generator patterns (those whose ``params`` carry
        ``center_lat`` / ``center_lon``), the new pattern is produced
        by :meth:`regenerate` at the new centre — exact, regardless
        of displacement size.  For ad-hoc patterns, computes the
        geodetic N/E delta from the pattern centroid to
        ``(latitude, longitude)`` and delegates to :meth:`translate`.

        Args:
            latitude: New centre latitude in decimal degrees.
            longitude: New centre longitude in decimal degrees.

        Returns:
            A new Pattern centred on ``(latitude, longitude)``.
        """
        if "center_lat" in self.params and "center_lon" in self.params:
            return self.regenerate(
                center_lat=float(latitude),
                center_lon=float(longitude),
            )
        cur_lat, cur_lon = self._current_centre()
        n_m, e_m, _ = pymap3d.geodetic2ned(
            latitude, longitude, 0, cur_lat, cur_lon, 0,
        )
        return self.translate(float(n_m) * ureg.meter, float(e_m) * ureg.meter)

    def rotate(
        self,
        angle_deg: float,
        around: tuple[float, float] | None = None,
    ) -> Pattern:
        """Return a new Pattern rotated by ``angle_deg`` (compass CW).

        Each element is rotated about ``around`` (the pattern's centre
        by default).  Headings on every contained :class:`Waypoint`
        and :class:`FlightLine` are shifted by the same angle.
        ``params["heading"]`` is also shifted if present, so a
        subsequent :meth:`regenerate` produces matching geometry.

        Args:
            angle_deg: Clockwise rotation in degrees.  ``rotate(360)``
                is the identity; ``rotate(-90)`` rotates 90° CCW.
            around: ``(latitude, longitude)`` pivot.  ``None``
                (default) rotates about the pattern's current centre.

        Returns:
            A new rotated Pattern.
        """
        pivot_lat, pivot_lon = around if around is not None else self._current_centre()
        theta = math.radians(float(angle_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def _rotate_lat_lon(lat: float, lon: float) -> tuple[float, float]:
            e_m, n_m, _ = pymap3d.geodetic2enu(lat, lon, 0, pivot_lat, pivot_lon, 0)
            # CW rotation about the +Up axis (looking down):
            #   e' =  e·cos(θ) + n·sin(θ)
            #   n' = -e·sin(θ) + n·cos(θ)
            e_new = float(e_m) * cos_t + float(n_m) * sin_t
            n_new = -float(e_m) * sin_t + float(n_m) * cos_t
            new_lat, new_lon, _ = pymap3d.enu2geodetic(
                e_new, n_new, 0, pivot_lat, pivot_lon, 0,
            )
            return (
                round(float(new_lat), 6),
                round(float(wrap_to_180(float(new_lon))), 6),
            )

        def _rotate_waypoint(wp: Waypoint) -> Waypoint:
            new_lat, new_lon = _rotate_lat_lon(wp.latitude, wp.longitude)
            return Waypoint(
                latitude=new_lat,
                longitude=new_lon,
                heading=float(wrap_to_360(float(wp.heading) + float(angle_deg))),
                altitude_msl=wp.altitude_msl,
                name=wp.name,
                speed=wp.speed,
                delay=wp.delay,
                segment_type=wp.segment_type,
            )

        def _rotate_flight_line(fl: FlightLine) -> FlightLine:
            lat1, lon1 = _rotate_lat_lon(fl.waypoint1.latitude, fl.waypoint1.longitude)
            lat2, lon2 = _rotate_lat_lon(fl.waypoint2.latitude, fl.waypoint2.longitude)
            return FlightLine.from_endpoints(
                lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2,
                altitude_msl=fl.altitude_msl,
                site_name=fl.site_name,
                site_description=fl.site_description,
                investigator=fl.investigator,
            )

        new_lines = {lid: _rotate_flight_line(fl) for lid, fl in self.lines.items()}
        new_waypoints = [_rotate_waypoint(wp) for wp in self.waypoints]

        new_params = copy.deepcopy(self.params)
        if "heading" in new_params:
            new_params["heading"] = float(
                wrap_to_360(float(new_params["heading"]) + float(angle_deg))
            )
        if "center_lat" in new_params and "center_lon" in new_params:
            new_centre_lat, new_centre_lon = _rotate_lat_lon(
                float(new_params["center_lat"]),
                float(new_params["center_lon"]),
            )
            new_params["center_lat"] = new_centre_lat
            new_params["center_lon"] = new_centre_lon

        return Pattern(
            kind=self.kind,
            name=self.name,
            params=new_params,
            pattern_id=self.pattern_id,
            lines=new_lines,
            waypoints=new_waypoints,
        )

    @classmethod
    def from_relative(
        cls,
        anchor: Waypoint | tuple[float, float],
        *,
        bearing: float,
        distance: Quantity | float,
        generator: Callable[..., Pattern],
        **generator_kwargs: Any,
    ) -> Pattern:
        """Build a pattern centred at a geodesic offset from an anchor.

        Combines :meth:`Waypoint.relative_to` with a pattern
        generator: computes the offset point from ``anchor`` along
        ``bearing`` for ``distance``, then calls ``generator`` with
        that point as the ``center`` keyword.  All other kwargs are
        forwarded to the generator unchanged.

        Args:
            anchor: A :class:`Waypoint` or ``(latitude, longitude)``
                tuple.
            bearing: Initial true bearing from ``anchor`` (compass deg).
            distance: Geodesic distance.  ``float`` interpreted as
                nautical miles (matches :meth:`Waypoint.relative_to`).
            generator: A pattern generator from
                :mod:`hyplan.flight_patterns` (e.g. ``racetrack``,
                ``rosette``, ``polygon``, ``sawtooth``, ``spiral``).
            **generator_kwargs: Forwarded to ``generator`` (e.g.
                ``heading``, ``altitude``, ``leg_length``, ``n_legs``).

        Returns:
            The Pattern produced by ``generator(center=offset, ...)``.

        Example:
            >>> from hyplan.units import ureg
            >>> from hyplan.flight_patterns import racetrack
            >>> from hyplan.waypoint import Waypoint
            >>> from hyplan.pattern import Pattern
            >>> edw = Waypoint(34.92, -117.87, heading=0, name="EDW")
            >>> pattern = Pattern.from_relative(
            ...     edw,
            ...     bearing=90,
            ...     distance=200,                       # 200 nmi east
            ...     generator=racetrack,
            ...     heading=0,
            ...     altitude=35_000 * ureg.foot,
            ...     leg_length=10 * ureg.nautical_mile,
            ...     n_legs=5,
            ... )
        """
        offset_wp = Waypoint.relative_to(
            anchor, bearing=bearing, distance=distance,
        )
        return generator(
            center=(offset_wp.latitude, offset_wp.longitude),
            **generator_kwargs,
        )

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------

    def _current_centre(self) -> tuple[float, float]:
        """Best-effort lat/lon centre for this pattern.

        Uses the stored ``params["center_lat"]`` / ``["center_lon"]``
        if present (always true for built-in generator patterns);
        otherwise returns the arithmetic mean of element coordinates.
        """
        if "center_lat" in self.params and "center_lon" in self.params:
            return (
                float(self.params["center_lat"]),
                float(self.params["center_lon"]),
            )
        lats: list[float] = []
        lons: list[float] = []
        for fl in self.lines.values():
            lats.extend((fl.waypoint1.latitude, fl.waypoint2.latitude))
            lons.extend((fl.waypoint1.longitude, fl.waypoint2.longitude))
        for wp in self.waypoints:
            lats.append(wp.latitude)
            lons.append(wp.longitude)
        if not lats:
            raise HyPlanValueError(
                f"Pattern '{self.pattern_id or self.name}' has no elements; "
                "centre is undefined."
            )
        return (sum(lats) / len(lats), sum(lons) / len(lons))


def _waypoint_to_dict(wp: Waypoint) -> dict[str, Any]:
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


def _waypoint_from_dict(d: dict[str, Any]) -> Waypoint:
    alt = d.get("altitude_msl_m")
    return Waypoint(
        latitude=d["latitude"],
        longitude=d["longitude"],
        heading=d["heading"],
        altitude_msl=(alt * ureg.meter if alt is not None else None),
        name=d.get("name"),
        segment_type=d.get("segment_type"),
    )


def _length_m(value: Quantity | float) -> float:
    """Convert a length argument (Quantity or float-meters) to plain metres."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.m_as(ureg.meter))


def _translated_params(
    params: dict[str, Any], n_m: float, e_m: float,
) -> dict[str, Any]:
    """Return a copy of ``params`` with ``center_lat``/``center_lon``
    shifted by the given geodetic N/E offset (metres).  Other entries
    are deep-copied unchanged.  Patterns whose params dict does not
    carry an explicit centre are returned as-is.
    """
    new_params = copy.deepcopy(params)
    if "center_lat" in new_params and "center_lon" in new_params:
        cur_lat = float(new_params["center_lat"])
        cur_lon = float(new_params["center_lon"])
        new_lat, new_lon, _ = pymap3d.ned2geodetic(
            n_m, e_m, 0, cur_lat, cur_lon, 0,
        )
        new_params["center_lat"] = float(new_lat)
        new_params["center_lon"] = float(wrap_to_180(float(new_lon)))
    return new_params


def _invoke_generator(generator: Any, kind: str, params: dict[str, Any]) -> Pattern:
    """Re-invoke a generator from a stored params dict (meters/degrees only)."""
    center = (params["center_lat"], params["center_lon"])
    heading = params.get("heading", 0.0)

    if kind == "rosette":
        return cast("Pattern", generator(
            center=center,
            heading=heading,
            altitude=params["altitude_msl_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_lines=params.get("n_lines", 3),
            angles=params.get("angles"),
        ))
    if kind == "racetrack":
        offset = params.get("offset_m", 0)
        if isinstance(offset, list):
            offset_q = [o * ureg.meter for o in offset]
        else:
            offset_q = offset * ureg.meter
        altitudes = params.get("altitudes_m")
        stack_altitudes = params.get("stack_altitudes_m")
        return cast("Pattern", generator(
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
        ))
    if kind == "polygon":
        return cast("Pattern", generator(
            center=center,
            heading=heading,
            altitude=params["altitude_msl_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_sides=int(params.get("n_sides", 4)),
            aspect_ratio=float(params.get("aspect_ratio", 1.0)),
            closed=bool(params.get("closed", True)),
        ))
    if kind == "sawtooth":
        return cast("Pattern", generator(
            center=center,
            heading=heading,
            altitude_min=params["altitude_min_m"] * ureg.meter,
            altitude_max=params["altitude_max_m"] * ureg.meter,
            leg_length=params["leg_length_m"] * ureg.meter,
            n_cycles=int(params.get("n_cycles", 1)),
        ))
    if kind == "spiral":
        return cast("Pattern", generator(
            center=center,
            heading=heading,
            altitude_start=params["altitude_start_m"] * ureg.meter,
            altitude_end=params["altitude_end_m"] * ureg.meter,
            radius=params["radius_m"] * ureg.meter,
            n_turns=float(params.get("n_turns", 3.0)),
            direction=str(params.get("direction", "right")),
            points_per_turn=int(params.get("points_per_turn", 36)),
        ))
    if kind == "glint_arc":
        import datetime as _dt
        obs_raw = params["observation_datetime"]
        if isinstance(obs_raw, str):
            obs_dt = _dt.datetime.fromisoformat(obs_raw.replace("Z", "+00:00"))
        else:
            obs_dt = obs_raw
        cl_m = params.get("collection_length_m")
        return cast("Pattern", generator(
            center=center,
            observation_datetime=obs_dt,
            altitude=params["altitude_msl_m"] * ureg.meter,
            speed=params["speed_mps"] * (ureg.meter / ureg.second),
            bank_angle=params.get("bank_angle"),
            bank_direction=str(params.get("bank_direction", "right")),
            collection_length=(cl_m * ureg.meter if cl_m is not None else None),
            densify_m=float(params.get("densify_m", 200.0)),
        ))
    raise HyPlanValueError(f"Unknown pattern kind: {kind}")


__all__ = [
    "LINE_BASED_KINDS",
    "PATTERN_KINDS",
    "WAYPOINT_BASED_KINDS",
    "Pattern",
]
