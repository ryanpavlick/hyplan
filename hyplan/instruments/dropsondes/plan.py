"""``DropsondePlan`` — the collection of releases + trajectories for a flight.

The public workflow:

    plan = DropsondePlan.from_flight_plan(
        compute_flight_plan(...),
        sensor=AVAPS_NRD41,
        aircraft=aircraft,
        takeoff_time=t0,
        spacing=20 * ureg.nautical_mile,
        target_polygon=poly,
    )
    plan2 = plan.simulate(wind_field=wind)
    df = plan2.to_manifest_gdf()
    plan2.plot()

:class:`DropsondePlan` is frozen and `simulate()` is functional — the
original pre-sim plan is left intact and the simulated plan is a new
object.  Releases in the simulated plan carry
``qc_splash_in_target_polygon`` populated; the corresponding
trajectories reference those post-sim release objects (see the
trajectory-release identity invariant in the design doc).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable

import geopandas as gpd
import matplotlib.axes
import numpy as np
import pandas as pd
import shapely.geometry
from pint import Quantity
from shapely.geometry import Point

from ...exceptions import HyPlanValueError
from ...units import ureg
from ...winds.base import WindField
from .flight_plan_track import FlightPlanTrack
from .models import DropsondeRelease, DropsondeTrajectory
from .planning import releases_along_flight_line, releases_along_segment
from .sensor import AVAPS_NRD41, DropsondeSystem, _as_quantity
from .simulate import simulate_release

if TYPE_CHECKING:
    from ...aircraft._base import Aircraft
    from ...pattern import Pattern

__all__ = ["DropsondePlan", "summarize_trajectories"]


@dataclass(frozen=True, eq=False)
class DropsondePlan:
    """A collection of planned dropsonde releases (and optionally their simulations).

    Frozen; ``simulate()`` is functional and returns a new plan.
    Equality/hashing are identity-based (``eq=False``) — embedded
    ``Waypoint`` / ``Aircraft`` / ``FlightPlanTrack`` / ``GeoDataFrame``
    objects make structural equality either expensive or wrong.
    """

    releases: tuple[DropsondeRelease, ...] = ()
    trajectories: tuple[DropsondeTrajectory, ...] = ()
    flight_track: FlightPlanTrack | None = None
    target_polygon: shapely.geometry.Polygon | None = None

    # -------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------

    @classmethod
    def from_flight_plan(
        cls,
        plan: gpd.GeoDataFrame | FlightPlanTrack,
        *,
        sensor: DropsondeSystem = AVAPS_NRD41,
        aircraft: Aircraft | None = None,
        takeoff_time: _dt.datetime,
        spacing: Quantity | None = None,
        spacing_time: Quantity | None = None,
        segment_types: tuple[str, ...] = ("flight_line", "transit"),
        min_release_altitude: Quantity | None = None,
        surface_elevation_msl: Quantity | None = None,
        target_polygon: shapely.geometry.Polygon | None = None,
        dem_file: str | None = None,
        terrain_aware: bool = False,
    ) -> DropsondePlan:
        """Build a plan from a ``compute_flight_plan`` GeoDataFrame.

        Set ``flight_track`` on the returned plan so the inverse
        targeting solver can iterate the aircraft trajectory later.
        """
        if isinstance(plan, FlightPlanTrack):
            track = plan
        else:
            track = FlightPlanTrack.from_compute_flight_plan(plan)

        filtered = track.filter(segment_types)
        releases: list[DropsondeRelease] = []
        next_id = 0
        for seg in filtered.segments:
            seg_releases = releases_along_segment(
                seg,
                sensor=sensor,
                aircraft=aircraft,
                takeoff_time=takeoff_time,
                spacing=spacing,
                spacing_time=spacing_time,
                min_release_altitude=min_release_altitude,
                surface_elevation_msl=surface_elevation_msl,
                first_release_id=next_id,
            )
            # When DEM is supplied, refine the AGL gate per release.
            if terrain_aware and dem_file is not None and seg_releases:
                seg_releases = _refine_agl_qc_with_dem(
                    seg_releases,
                    dem_file=dem_file,
                    min_release_altitude_m=float(
                        _as_quantity(
                            min_release_altitude or sensor.min_release_altitude,
                            "meter", "min_release_altitude",
                        ).magnitude,
                    ),
                )
            releases.extend(seg_releases)
            next_id += len(seg_releases)

        return cls(
            releases=tuple(releases),
            trajectories=(),
            flight_track=track,
            target_polygon=target_polygon,
        )

    @classmethod
    def from_pattern(
        cls,
        pattern: Pattern,
        *,
        sensor: DropsondeSystem = AVAPS_NRD41,
        aircraft: Aircraft | None = None,
        takeoff_time: _dt.datetime | None = None,
        spacing: Quantity | None = None,
        spacing_time: Quantity | None = None,
        start_elapsed: Quantity = 0 * ureg.second,
    ) -> DropsondePlan:
        """Build a plan from a line-based :class:`Pattern`.

        Spacing resets per line; line-to-line transit is NOT modelled in
        this release.  For authoritative multi-line timing use
        ``from_flight_plan(compute_flight_plan(...))``.

        Emits a ``UserWarning`` when the pattern has more than one line
        and ``takeoff_time`` is supplied, because the first release on
        every line carries the same timestamp (``takeoff_time +
        start_elapsed``) — physically impossible, just a hazard of
        geometry-only timing.

        For ``is_waypoint_based`` patterns returns an empty plan and
        emits a ``UserWarning``.
        """
        if not pattern.is_line_based:
            warnings.warn(
                "DropsondePlan.from_pattern only supports line-based patterns; "
                f"got kind={pattern.kind!r} (is_waypoint_based). Returning empty plan.",
                UserWarning,
                stacklevel=2,
            )
            return cls()

        releases: list[DropsondeRelease] = []
        next_id = 0
        for line_id, line in pattern.lines.items():
            line_releases = releases_along_flight_line(
                line,
                sensor=sensor,
                aircraft=aircraft,
                takeoff_time=takeoff_time,
                start_elapsed=start_elapsed,
                spacing=spacing,
                spacing_time=spacing_time,
                first_release_id=next_id,
            )
            # Stamp pattern provenance.
            stamped = [
                dataclasses.replace(
                    r,
                    source_pattern_id=str(pattern.pattern_id) if pattern.pattern_id else None,
                    source_id=line_id,
                )
                for r in line_releases
            ]
            releases.extend(stamped)
            next_id += len(line_releases)

        # Multi-line warning: detect duplicate timestamps across lines.
        if takeoff_time is not None and len(pattern.lines) > 1:
            seen: dict[_dt.datetime, int] = {}
            for r in releases:
                if r.release_time is None:
                    continue
                seen[r.release_time] = seen.get(r.release_time, 0) + 1
            n_dupes = sum(1 for c in seen.values() if c > 1)
            if n_dupes > 0:
                warnings.warn(
                    "from_pattern produced multiple releases sharing the same "
                    "release_time because v1.10 does not model line-to-line "
                    "transit; use DropsondePlan.from_flight_plan() for "
                    "authoritative timing.",
                    UserWarning,
                    stacklevel=2,
                )

        return cls(releases=tuple(releases))

    # -------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------

    def simulate(
        self,
        *,
        wind_field: WindField,
        n_ensemble: int = 0,
        wind_perturbation_sigma: Quantity = 1.5 * ureg.meter / ureg.second,
        fall_rate_perturbation_pct: float = 5.0,
        dem_file: str | None = None,
        terrain_aware: bool = False,
        surface_elevation_msl: Quantity | None = None,
        dt: Quantity = 1 * ureg.second,
        rng_seed: int | None = None,
    ) -> DropsondePlan:
        """Forward-simulate all non-skipped releases. Returns a new plan."""
        if n_ensemble < 0:
            raise HyPlanValueError("n_ensemble must be non-negative")

        # Pre-flight: if any release lacks release_time and wind is time-
        # dependent, raise before doing any per-release work.
        time_dep = getattr(wind_field, "is_time_dependent", True)
        if time_dep:
            for r in self.releases:
                if r.qc_release_ok is False:
                    continue
                if r.release_time is None:
                    raise HyPlanValueError(
                        f"release_id={r.release_id} has release_time=None; "
                        "time-dependent wind fields require a timestamp on "
                        "every release."
                    )

        sigma_uv = float(
            _as_quantity(
                wind_perturbation_sigma, "meter / second", "wind_perturbation_sigma",
            ).magnitude
        )
        sigma_w_pct = float(fall_rate_perturbation_pct)
        n_members = max(1, n_ensemble)
        use_perturb = n_ensemble > 0
        rng = np.random.default_rng(rng_seed)

        post_releases: list[DropsondeRelease] = []
        trajectories: list[DropsondeTrajectory] = []
        skipped = 0

        for release in self.releases:
            if release.qc_release_ok is False:
                # Skipped — keep the release in the manifest unchanged.
                post_releases.append(release)
                skipped += 1
                continue

            # Run ensemble (or single deterministic) members.
            member_splashes: list[tuple[float, float]] = []
            member_trajs: list[DropsondeTrajectory] = []
            for m_idx in range(n_members):
                if use_perturb:
                    u_bias = float(rng.normal(0.0, sigma_uv))
                    v_bias = float(rng.normal(0.0, sigma_uv))
                    fall_scale = 1.0 + float(rng.normal(0.0, sigma_w_pct / 100.0))
                else:
                    u_bias = 0.0
                    v_bias = 0.0
                    fall_scale = 1.0

                traj = simulate_release(
                    release,
                    wind_field=wind_field,
                    dem_file=dem_file,
                    terrain_aware=terrain_aware,
                    surface_elevation_msl=surface_elevation_msl,
                    dt=dt,
                    u_bias_mps=u_bias,
                    v_bias_mps=v_bias,
                    fall_rate_scale=fall_scale,
                    ensemble_member=m_idx,
                )
                member_trajs.append(traj)
                member_splashes.append(
                    (
                        float(traj.splash_waypoint.latitude),
                        float(traj.splash_waypoint.longitude),
                    )
                )

            # Polygon QC against the ensemble-mean splash.
            qc_poly: bool | None
            if self.target_polygon is None:
                qc_poly = None
            else:
                mean_lat = float(np.mean([p[0] for p in member_splashes]))
                mean_lon = float(np.mean([p[1] for p in member_splashes]))
                qc_poly = bool(
                    self.target_polygon.contains(Point(mean_lon, mean_lat))
                )

            post_release = release.with_qc(qc_splash_in_target_polygon=qc_poly)
            post_releases.append(post_release)

            # Rebind each trajectory's `release` to the post-sim release.
            for tr in member_trajs:
                trajectories.append(
                    dataclasses.replace(tr, release=post_release),
                )

        if skipped > 0:
            warnings.warn(
                f"Skipped descent simulation for {skipped} release(s) with "
                f"qc_release_ok=False (gating QC failed).",
                UserWarning,
                stacklevel=2,
            )

        return DropsondePlan(
            releases=tuple(post_releases),
            trajectories=tuple(trajectories),
            flight_track=self.flight_track,
            target_polygon=self.target_polygon,
        )

    # -------------------------------------------------------------------
    # Exports
    # -------------------------------------------------------------------

    def to_manifest_gdf(self) -> gpd.GeoDataFrame:
        """Manifest GeoDataFrame — one row per planned release."""
        if not self.releases:
            return _empty_manifest_gdf()
        records = [r.to_record() for r in self.releases]
        df = pd.DataFrame.from_records(records)
        for col in (
            "qc_min_alt_ok", "qc_aircraft_envelope_ok",
            "qc_segment_allowed", "qc_release_ok",
            "qc_splash_in_target_polygon",
        ):
            df[col] = df[col].astype("boolean")
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def trajectories_gdf(self) -> gpd.GeoDataFrame:
        """Concatenated per-step trajectory GeoDataFrame for all releases × members."""
        if not self.trajectories:
            return _empty_trajectory_gdf()
        frames = [t.track for t in self.trajectories]
        out = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    def summary(self) -> pd.DataFrame:
        """Per-release splash diagnostics."""
        return summarize_trajectories(
            self.trajectories, target_polygon=self.target_polygon,
        )

    def plot(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        show_traces: bool = True,
        show_ellipses: bool = True,
        sigma_scale: float = 2.0,
    ) -> matplotlib.axes.Axes:
        """Simple map overlay: track, releases, traces, splash means."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 5))

        # Releases
        manifest = self.to_manifest_gdf()
        if not manifest.empty:
            ax.scatter(
                manifest["release_lon"], manifest["release_lat"],
                color="C0", marker="o", s=30, zorder=4, label="Release",
            )

        # Trajectories (faint per-step traces)
        if show_traces and self.trajectories:
            traj_gdf = self.trajectories_gdf()
            for (_rid, _m_idx), group in traj_gdf.groupby(
                ["release_id", "ensemble_member"],
            ):
                ax.plot(
                    group["longitude"], group["latitude"],
                    color="gray", lw=0.5, alpha=0.4,
                )

        # Splash means and ellipses
        summary = self.summary()
        if not summary.empty:
            ax.scatter(
                summary["splash_lon_mean"], summary["splash_lat_mean"],
                color="C3", marker="x", s=40, zorder=5, label="Splash (mean)",
            )
            if show_ellipses:
                added = False
                for _, row in summary.iterrows():
                    if row["n_ensemble"] < 2:
                        continue
                    center_lat = float(row["splash_lat_mean"])
                    center_lon = float(row["splash_lon_mean"])
                    m_per_deg_lat = 111_320.0
                    m_per_deg_lon = m_per_deg_lat * float(
                        np.cos(np.radians(center_lat))
                    )
                    a_m = sigma_scale * float(row["splash_ellipse_semi_major_m"])
                    b_m = sigma_scale * float(row["splash_ellipse_semi_minor_m"])
                    width_deg = 2 * a_m / m_per_deg_lon
                    height_deg = 2 * b_m / m_per_deg_lat
                    bearing = float(row["splash_ellipse_bearing_deg"])
                    angle = (90.0 - bearing) % 360.0
                    kw: dict[str, Any] = dict(
                        xy=(center_lon, center_lat),
                        width=width_deg,
                        height=height_deg,
                        angle=angle,
                        fill=True,
                        facecolor="C3",
                        alpha=0.18,
                        edgecolor="C3",
                        lw=1.2,
                    )
                    if not added:
                        kw["label"] = f"Splash {sigma_scale:.0f}-σ ellipse"
                        added = True
                    ax.add_patch(Ellipse(**kw))

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"Dropsonde release plan ({len(self.releases)} releases)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MANIFEST_COLUMNS = [
    "release_id", "source_id", "source_pattern_id", "source_segment_type",
    "release_lat", "release_lon", "release_altitude_msl_m",
    "release_altitude_msl_ft", "release_time_utc",
    "qc_min_alt_ok", "qc_aircraft_envelope_ok", "qc_segment_allowed",
    "qc_release_ok", "qc_splash_in_target_polygon", "geometry",
]


_TRAJECTORY_COLUMNS = [
    "release_id", "ensemble_member", "step", "time_utc",
    "latitude", "longitude", "altitude_msl_m", "altitude_agl_m",
    "u_wind_mps", "v_wind_mps", "w_fall_mps", "geometry",
]


def _empty_manifest_gdf() -> gpd.GeoDataFrame:
    df = pd.DataFrame(columns=_MANIFEST_COLUMNS)
    for col in (
        "qc_min_alt_ok", "qc_aircraft_envelope_ok",
        "qc_segment_allowed", "qc_release_ok",
        "qc_splash_in_target_polygon",
    ):
        df[col] = df[col].astype("boolean")
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _empty_trajectory_gdf() -> gpd.GeoDataFrame:
    df = pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _refine_agl_qc_with_dem(
    releases: list[DropsondeRelease],
    *,
    dem_file: str,
    min_release_altitude_m: float,
) -> list[DropsondeRelease]:
    """Set ``qc_min_alt_ok`` per release using a DEM lookup at lat/lon."""
    from ...terrain import get_elevations
    lats = np.asarray([r.waypoint.latitude for r in releases], dtype=float)
    lons = np.asarray([r.waypoint.longitude for r in releases], dtype=float)
    elevs = get_elevations(lats, lons, dem_file)
    out: list[DropsondeRelease] = []
    for r, e in zip(releases, elevs):
        if r.waypoint.altitude_msl is None or np.isnan(e):
            qc_min: bool | None = None
        else:
            agl = float(r.waypoint.altitude_msl.m_as("meter")) - float(e)
            qc_min = bool(agl >= min_release_altitude_m)
        out.append(r.with_qc(qc_min_alt_ok=qc_min))
    return out


def summarize_trajectories(
    trajectories: Iterable[DropsondeTrajectory],
    *,
    target_polygon: shapely.geometry.Polygon | None = None,
) -> pd.DataFrame:
    """Per-release splash diagnostics + (when supplied) polygon-hit fraction."""
    cols = [
        "release_id", "n_ensemble",
        "time_to_surface_s_mean", "time_to_surface_s_std",
        "splash_lat_mean", "splash_lon_mean",
        "splash_lat_std", "splash_lon_std",
        "drift_distance_m_mean", "drift_distance_m_std",
        "drift_bearing_deg_mean",
        "splash_ellipse_semi_major_m", "splash_ellipse_semi_minor_m",
        "splash_ellipse_bearing_deg",
        "n_members_in_polygon", "fraction_in_polygon",
        "qc_terminated_at_ground", "qc_max_steps_exceeded", "qc_dem_gap_count",
    ]
    by_rid: dict[int, list[DropsondeTrajectory]] = defaultdict(list)
    for t in trajectories:
        by_rid[t.release.release_id].append(t)
    if not by_rid:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for rid, group in by_rid.items():
        lats = np.asarray([t.splash_waypoint.latitude for t in group], dtype=float)
        lons = np.asarray([t.splash_waypoint.longitude for t in group], dtype=float)
        fts = np.asarray(
            [float(t.time_to_surface.m_as("second")) for t in group], dtype=float,
        )
        drifts = np.asarray(
            [float(t.drift_distance.m_as("meter")) for t in group], dtype=float,
        )
        bearings = np.asarray(
            [float(t.drift_bearing_deg) for t in group], dtype=float,
        )

        # Splash ellipse: PCA of (delta_lat * deg_to_m_lat, delta_lon * deg_to_m_lon).
        lat_mean = float(np.mean(lats))
        lon_mean = float(np.mean(lons))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = m_per_deg_lat * float(np.cos(np.radians(lat_mean)))
        if len(group) >= 2:
            dx = (lons - lon_mean) * m_per_deg_lon
            dy = (lats - lat_mean) * m_per_deg_lat
            cov = np.cov(np.vstack([dx, dy]))
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            semi_major = float(np.sqrt(max(eigvals[0], 0.0)))
            semi_minor = float(np.sqrt(max(eigvals[1], 0.0)))
            vx, vy = eigvecs[:, 0]
            bearing_deg = (np.degrees(np.arctan2(vx, vy))) % 360.0
        else:
            semi_major = 0.0
            semi_minor = 0.0
            bearing_deg = 0.0

        if target_polygon is not None:
            hits = sum(
                1 for la, lo in zip(lats, lons)
                if target_polygon.contains(Point(float(lo), float(la)))
            )
            frac_in = hits / len(group)
        else:
            hits = 0
            frac_in = float("nan")

        rows.append(
            {
                "release_id": int(rid),
                "n_ensemble": int(len(group)),
                "time_to_surface_s_mean": float(np.mean(fts)),
                "time_to_surface_s_std": float(np.std(fts)) if len(fts) > 1 else 0.0,
                "splash_lat_mean": lat_mean,
                "splash_lon_mean": lon_mean,
                "splash_lat_std": float(np.std(lats)) if len(lats) > 1 else 0.0,
                "splash_lon_std": float(np.std(lons)) if len(lons) > 1 else 0.0,
                "drift_distance_m_mean": float(np.mean(drifts)),
                "drift_distance_m_std": float(np.std(drifts)) if len(drifts) > 1 else 0.0,
                "drift_bearing_deg_mean": float(np.mean(bearings)),
                "splash_ellipse_semi_major_m": semi_major,
                "splash_ellipse_semi_minor_m": semi_minor,
                "splash_ellipse_bearing_deg": float(bearing_deg),
                "n_members_in_polygon": int(hits),
                "fraction_in_polygon": float(frac_in),
                "qc_terminated_at_ground": all(t.qc_terminated_at_ground for t in group),
                "qc_max_steps_exceeded": any(t.qc_max_steps_exceeded for t in group),
                "qc_dem_gap_count": int(sum(t.qc_dem_gap_count for t in group)),
            }
        )
    return pd.DataFrame(rows, columns=cols)
