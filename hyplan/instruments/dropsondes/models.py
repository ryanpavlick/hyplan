"""Frozen domain objects: ``DropsondeRelease`` and ``DropsondeTrajectory``.

A :class:`DropsondeRelease` is the planned event — lat/lon/altitude/time
plus provenance back to the source ``FlightLine`` / ``Pattern`` / plan
segment, the sensor, the host aircraft, and tri-state QC flags.  It is
immutable; the underlying mutable :class:`Waypoint` is defensively
copied on construction and on extraction so the frozen-record promise
is not a paper shield.

A :class:`DropsondeTrajectory` wraps a single descent integration —
the per-step GeoDataFrame returned by the kernel — plus a few derived
splash diagnostics and identity back to the release that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable

import geopandas as gpd
from pint import Quantity
from shapely.geometry import Point

from ...waypoint import Waypoint
from .sensor import AVAPS_NRD41, DropsondeSystem

if TYPE_CHECKING:
    from ...aircraft._base import Aircraft
    from ...flight_line import FlightLine
    from ...pattern import Pattern

__all__ = ["DropsondeRelease", "DropsondeTrajectory"]


def _copy_waypoint(wp: Waypoint) -> Waypoint:
    """Defensive copy of a :class:`Waypoint`.

    Waypoint is mutable in the rest of HyPlan; we copy on ingress to
    :class:`DropsondeRelease` and on egress so the frozen release
    behaves as an immutable record.
    """
    alt = wp.altitude_msl
    return Waypoint(
        latitude=wp.latitude,
        longitude=wp.longitude,
        heading=wp.heading,
        altitude_msl=alt,
        name=wp.name,
        speed=wp.speed,
        delay=wp.delay,
        segment_type=wp.segment_type,
    )


@dataclass(frozen=True, eq=False)
class DropsondeRelease:
    """A planned single-sonde release event.

    All gating QC flags are tri-state (``True``/``False``/``None``);
    :attr:`qc_release_ok` aggregates them.  ``qc_splash_in_target_polygon``
    is **not** a gate — it is a post-simulation diagnostic populated by
    :meth:`DropsondePlan.simulate`.

    Equality/hashing is identity-based (``eq=False``): two releases with
    structurally identical fields are still distinct objects.  Use
    :attr:`release_id` as the explicit join key when set-like semantics
    are needed.
    """

    waypoint: Waypoint
    sensor: DropsondeSystem = AVAPS_NRD41
    aircraft: "Aircraft | None" = None
    release_time: datetime.datetime | None = None
    aircraft_velocity_mps: tuple[float, float] | None = None

    # Provenance
    source: "FlightLine | Pattern | Hashable | None" = None
    source_id: Hashable | None = None
    source_pattern_id: str | None = None
    source_segment_type: str | None = None

    # Gating QC (tri-state)
    qc_min_alt_ok: bool | None = None
    qc_aircraft_envelope_ok: bool | None = None
    qc_segment_allowed: bool | None = None

    # Post-sim diagnostic (NOT part of the gate)
    qc_splash_in_target_polygon: bool | None = None

    release_id: int = 0

    @property
    def qc_release_ok(self) -> bool | None:
        """Tri-state aggregate over the gating flags only.

        Rule (locked):

        - any known gate is ``False`` → ``False``
        - all gates are ``True`` → ``True``
        - otherwise (at least one ``None``, none ``False``) → ``None``

        ``qc_splash_in_target_polygon`` is intentionally not part of
        this aggregate.
        """
        gates = (
            self.qc_min_alt_ok,
            self.qc_aircraft_envelope_ok,
            self.qc_segment_allowed,
        )
        if any(g is False for g in gates):
            return False
        if all(g is True for g in gates):
            return True
        return None

    def to_waypoint(self) -> Waypoint:
        """Return a defensive copy of the release waypoint."""
        return _copy_waypoint(self.waypoint)

    @classmethod
    def from_waypoint(
        cls,
        wp: Waypoint,
        *,
        sensor: DropsondeSystem = AVAPS_NRD41,
        release_time: datetime.datetime | None = None,
        release_id: int = 0,
        **kwargs: Any,
    ) -> "DropsondeRelease":
        """Build a release from a :class:`Waypoint` (defensively copied)."""
        return cls(
            waypoint=_copy_waypoint(wp),
            sensor=sensor,
            release_time=release_time,
            release_id=release_id,
            **kwargs,
        )

    def to_record(self) -> dict[str, Any]:
        """One-row dict for manifest export.

        Columns are stable (used by :meth:`DropsondePlan.to_manifest_gdf`).
        """
        wp = self.waypoint
        alt_m = float(wp.altitude_msl.m_as("meter")) if wp.altitude_msl is not None else float("nan")
        alt_ft = alt_m / 0.3048 if alt_m == alt_m else float("nan")
        return {
            "release_id": self.release_id,
            "source_id": self.source_id,
            "source_pattern_id": self.source_pattern_id,
            "source_segment_type": self.source_segment_type,
            "release_lat": float(wp.latitude),
            "release_lon": float(wp.longitude),
            "release_altitude_msl_m": alt_m,
            "release_altitude_msl_ft": alt_ft,
            "release_time_utc": self.release_time,
            "qc_min_alt_ok": self.qc_min_alt_ok,
            "qc_aircraft_envelope_ok": self.qc_aircraft_envelope_ok,
            "qc_segment_allowed": self.qc_segment_allowed,
            "qc_release_ok": self.qc_release_ok,
            "qc_splash_in_target_polygon": self.qc_splash_in_target_polygon,
            "geometry": Point(float(wp.longitude), float(wp.latitude)),
        }

    def with_qc(
        self,
        *,
        qc_min_alt_ok: bool | None | object = dataclasses.MISSING,
        qc_aircraft_envelope_ok: bool | None | object = dataclasses.MISSING,
        qc_segment_allowed: bool | None | object = dataclasses.MISSING,
        qc_splash_in_target_polygon: bool | None | object = dataclasses.MISSING,
    ) -> "DropsondeRelease":
        """Return a copy of self with one or more QC flags replaced."""
        kwargs: dict[str, Any] = {}
        if qc_min_alt_ok is not dataclasses.MISSING:
            kwargs["qc_min_alt_ok"] = qc_min_alt_ok
        if qc_aircraft_envelope_ok is not dataclasses.MISSING:
            kwargs["qc_aircraft_envelope_ok"] = qc_aircraft_envelope_ok
        if qc_segment_allowed is not dataclasses.MISSING:
            kwargs["qc_segment_allowed"] = qc_segment_allowed
        if qc_splash_in_target_polygon is not dataclasses.MISSING:
            kwargs["qc_splash_in_target_polygon"] = qc_splash_in_target_polygon
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True, eq=False)
class DropsondeTrajectory:
    """A single simulated descent — the kernel output plus splash diagnostics."""

    release: DropsondeRelease
    track: gpd.GeoDataFrame
    splash_waypoint: Waypoint
    time_to_surface: Quantity
    drift_distance: Quantity
    drift_bearing_deg: float
    ensemble_member: int = 0
    qc_terminated_at_ground: bool = True
    qc_max_steps_exceeded: bool = False
    qc_dem_gap_count: int = 0

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return the per-step trajectory GeoDataFrame."""
        return self.track

    def summary_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release.release_id,
            "ensemble_member": self.ensemble_member,
            "time_to_surface_s": float(self.time_to_surface.m_as("second")),
            "splash_lat": float(self.splash_waypoint.latitude),
            "splash_lon": float(self.splash_waypoint.longitude),
            "drift_distance_m": float(self.drift_distance.m_as("meter")),
            "drift_bearing_deg": float(self.drift_bearing_deg),
            "qc_terminated_at_ground": bool(self.qc_terminated_at_ground),
            "qc_max_steps_exceeded": bool(self.qc_max_steps_exceeded),
            "qc_dem_gap_count": int(self.qc_dem_gap_count),
        }
