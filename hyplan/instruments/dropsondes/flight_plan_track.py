"""Typed adapter over the ``compute_flight_plan`` GeoDataFrame.

The computed-plan GeoDataFrame ships timing in minutes, altitudes in
feet, and a column-by-column schema spread across several optional
fields.  :class:`FlightPlanTrack` validates that schema once at the
boundary, normalises units, and exposes a clean object-typed view of
the segments — including cumulative ``start_elapsed_s`` /
``end_elapsed_s`` per segment so consumers (release planning, inverse
targeting) share one authoritative timeline.

The canonical trajectory sampler :meth:`FlightPlanTrack.sample_at_elapsed`
returns an :class:`AircraftTrackSample` describing where the aircraft
is at a given elapsed time after takeoff.  This is the single source
of "where is the aircraft when" — release planning, pattern timing,
and targeting all go through it instead of each duplicating the
interpolation logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from collections.abc import Hashable

import geopandas as gpd
import numpy as np
import pandas as pd
import pymap3d.vincenty
from pint import Quantity
from shapely.geometry import LineString

from ...exceptions import HyPlanTypeError, HyPlanValueError
from ...geometry import process_linestring, wrap_to_180
from ...units import ureg

__all__ = ["AircraftTrackSample", "FlightPlanTrack", "PlannedSegment"]


@dataclass(frozen=True)
class PlannedSegment:
    """One row of a computed flight plan, normalised to SI units."""

    index: Hashable
    geometry: LineString
    segment_type: str
    start_altitude_m: float
    end_altitude_m: float
    duration_s: float
    start_elapsed_s: float
    end_elapsed_s: float
    planned_track_deg: float | None
    wind_corrected_heading_deg: float | None
    groundspeed_mps: float | None
    segment_name: str | None
    _length_m: float = 0.0


@dataclass(frozen=True)
class AircraftTrackSample:
    """Aircraft state at a specific elapsed time after takeoff."""

    elapsed_s: float
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float | None
    groundspeed_mps: float | None
    segment_index: Hashable
    segment_type: str
    segment_name: str | None


def _line_length_m(geom: LineString) -> float:
    _, _, _, cumulative = process_linestring(geom)
    if len(cumulative) == 0:
        return 0.0
    return float(cumulative[-1])


def _opt_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _opt_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    s = str(value)
    return s if s else None


class FlightPlanTrack:
    """Typed view of a computed flight plan, with a trajectory sampler."""

    def __init__(self, segments: list[PlannedSegment]) -> None:
        self.segments: list[PlannedSegment] = list(segments)

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_compute_flight_plan(cls, plan: gpd.GeoDataFrame) -> FlightPlanTrack:
        """Validate + normalise a ``compute_flight_plan`` GeoDataFrame."""
        if not isinstance(plan, gpd.GeoDataFrame):
            raise HyPlanTypeError("plan must be a GeoDataFrame")
        if "geometry" not in plan.columns:
            raise HyPlanValueError("plan must have a 'geometry' column")
        required = ("segment_type", "start_altitude", "end_altitude", "time_to_segment")
        missing = [c for c in required if c not in plan.columns]
        if missing:
            raise HyPlanValueError(
                f"plan is missing required columns: {missing}"
            )

        segments: list[PlannedSegment] = []
        elapsed_s = 0.0
        for idx, row in plan.iterrows():
            geom = row.get("geometry")
            seg_type = str(row.get("segment_type", ""))
            t_min = row.get("time_to_segment")
            duration_s = float(t_min) * 60.0 if pd.notna(t_min) else 0.0
            alt0 = row.get("start_altitude")
            alt1 = row.get("end_altitude")
            if pd.isna(alt0) or pd.isna(alt1):
                # Skip segments with missing altitudes (cannot release).
                # We still advance elapsed_s so the timeline stays
                # consistent.
                elapsed_s += duration_s
                continue
            alt0_m = float(alt0) * 0.3048
            alt1_m = float(alt1) * 0.3048

            track_deg = _opt_float(row.get("planned_track"))
            if track_deg is None:
                track_deg = _opt_float(row.get("start_heading"))
            wch_deg = _opt_float(row.get("wind_corrected_heading"))

            gs_kt = row.get("groundspeed_kts")
            gs_mps: float | None
            if pd.notna(gs_kt) and float(gs_kt) > 0:
                gs_mps = float(gs_kt) * 0.5144444
            elif isinstance(geom, LineString) and duration_s > 0:
                # Fall back to segment length / duration.
                gs_mps = _line_length_m(geom) / duration_s
            else:
                gs_mps = None

            length_m = _line_length_m(geom) if isinstance(geom, LineString) else 0.0
            seg_name = _opt_str(row.get("segment_name", row.get("site_name")))

            segments.append(
                PlannedSegment(
                    index=idx,
                    geometry=geom,
                    segment_type=seg_type,
                    start_altitude_m=alt0_m,
                    end_altitude_m=alt1_m,
                    duration_s=duration_s,
                    start_elapsed_s=elapsed_s,
                    end_elapsed_s=elapsed_s + duration_s,
                    planned_track_deg=track_deg,
                    wind_corrected_heading_deg=wch_deg,
                    groundspeed_mps=gs_mps,
                    segment_name=seg_name,
                    _length_m=length_m,
                )
            )
            elapsed_s += duration_s

        return cls(segments)

    # ------------------------------------------------------------------ ops
    def filter(self, segment_types: tuple[str, ...]) -> FlightPlanTrack:
        """Return a track with only the given segment types, preserving cumulative timing."""
        wanted = set(segment_types)
        return FlightPlanTrack(
            [s for s in self.segments if s.segment_type in wanted]
        )

    def total_duration_s(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.end_elapsed_s for s in self.segments)

    def elapsed_range(self) -> tuple[float, float]:
        if not self.segments:
            return (0.0, 0.0)
        return (
            min(s.start_elapsed_s for s in self.segments),
            max(s.end_elapsed_s for s in self.segments),
        )

    # --------------------------------------------------------------- sample
    def sample_at_elapsed(self, elapsed: Quantity | float) -> AircraftTrackSample:
        """Aircraft state at the given elapsed time after takeoff."""
        if isinstance(elapsed, Quantity):
            t_s = float(elapsed.m_as("second"))
        else:
            t_s = float(elapsed)

        seg = self._segment_at_elapsed(t_s)
        # Clamp to segment bounds for safety; callers shouldn't ask for
        # times outside the range, but we tolerate it.
        frac = 0.0
        if seg.duration_s > 0:
            frac = max(0.0, min(1.0, (t_s - seg.start_elapsed_s) / seg.duration_s))

        if not isinstance(seg.geometry, LineString) or seg._length_m <= 0:
            # Degenerate segment (no geometry); fall back to the first
            # coordinate of the geometry if one exists, else (0,0).
            if isinstance(seg.geometry, LineString) and len(seg.geometry.coords) > 0:
                lon, lat = seg.geometry.coords[0]
            else:
                lon, lat = 0.0, 0.0
        else:
            d_m = frac * seg._length_m
            lats, lons, azimuths, cumulative = process_linestring(seg.geometry)
            seg_idx = int(np.searchsorted(cumulative, d_m, side="right") - 1)
            seg_idx = max(0, min(seg_idx, len(cumulative) - 2))
            seg_start = float(cumulative[seg_idx])
            remain = d_m - seg_start
            lat0 = float(lats[seg_idx])
            lon0 = float(lons[seg_idx])
            az = float(azimuths[seg_idx])
            lat, lon = pymap3d.vincenty.vreckon(lat0, lon0, remain, az)
            lon = float(wrap_to_180(lon))

        alt_m = seg.start_altitude_m + frac * (seg.end_altitude_m - seg.start_altitude_m)
        return AircraftTrackSample(
            elapsed_s=t_s,
            latitude=float(lat),
            longitude=float(lon),
            altitude_m=float(alt_m),
            heading_deg=seg.planned_track_deg,
            groundspeed_mps=seg.groundspeed_mps,
            segment_index=seg.index,
            segment_type=seg.segment_type,
            segment_name=seg.segment_name,
        )

    def iter_samples(
        self,
        *,
        step: Quantity | float = 1 * ureg.second,
        segment_types: tuple[str, ...] | None = None,
    ) -> Iterator[AircraftTrackSample]:
        """Iterate samples at uniform elapsed-time spacing.

        Samples that fall inside a segment whose type is **not** in
        ``segment_types`` (when provided) are skipped.  This is what
        the inverse-targeting solver uses to enumerate feasible
        release times.
        """
        step_s = (
            float(step.m_as("second")) if isinstance(step, Quantity) else float(step)
        )
        if step_s <= 0:
            raise HyPlanValueError("step must be positive")
        if not self.segments:
            return
        total = self.total_duration_s()
        n = int(np.floor(total / step_s)) + 1
        wanted = set(segment_types) if segment_types is not None else None
        for k in range(n):
            t_s = k * step_s
            sample = self.sample_at_elapsed(t_s)
            if wanted is not None and sample.segment_type not in wanted:
                continue
            yield sample

    # ------------------------------------------------------------- internal
    def _segment_at_elapsed(self, t_s: float) -> PlannedSegment:
        if not self.segments:
            raise HyPlanValueError("FlightPlanTrack is empty")
        # Clamp into range.
        first = self.segments[0]
        last = self.segments[-1]
        if t_s <= first.start_elapsed_s:
            return first
        if t_s >= last.end_elapsed_s:
            return last
        # Linear scan is fine; plans are tens of segments.
        for seg in self.segments:
            if seg.start_elapsed_s <= t_s <= seg.end_elapsed_s:
                return seg
        return last  # unreachable in practice
