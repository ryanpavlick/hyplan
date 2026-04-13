"""Schedule and vertical-profile fitting from phase-labeled ADS-B data.

Bins trajectory observations by altitude, computes robust statistics per
bin, simplifies the resulting polylines, and produces
:class:`~hyplan.aircraft._base.TasSchedule` and
:class:`~hyplan.aircraft._base.VerticalProfile` objects ready for use in
an :class:`~hyplan.aircraft.Aircraft`.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from .._base import TasSchedule, VerticalProfile
from ...units import ureg
from .models import FitResult, FlightPhaseData, ScheduleFitMetrics

logger = logging.getLogger(__name__)


def fit_schedules(
    airdata_df,
    *,
    altitude_bin_ft: float = 2000.0,
    min_points_per_bin: int = 5,
    max_schedule_points: int = 6,
    outlier_sigma: float = 2.5,
    service_ceiling_ft: Optional[float] = None,
) -> FitResult:
    """Fit speed schedules and vertical profiles from wind-corrected data.

    Args:
        airdata_df: DataFrame from :func:`reconstruct_airdata` with
            columns ``altitude``, ``tas_kt``, ``vertical_rate``,
            ``phase``.
        altitude_bin_ft: Altitude bin width for aggregation.
        min_points_per_bin: Minimum observations per bin to include.
        max_schedule_points: Maximum breakpoints in fitted schedules.
        outlier_sigma: Remove points beyond this many standard deviations
            from the bin median before computing the final value.
        service_ceiling_ft: Override service ceiling.  If *None*,
            inferred as max observed altitude rounded up to the nearest
            1000 ft.

    Returns:
        :class:`FitResult` with fitted schedules, profiles, and metrics.
    """
    import pandas as pd

    # --- Split by phase ---
    climb_df = airdata_df[airdata_df["phase"] == "climb"]
    cruise_df = airdata_df[airdata_df["phase"] == "cruise"]
    descent_df = airdata_df[airdata_df["phase"] == "descent"]

    # --- Infer service ceiling ---
    all_alts = airdata_df["altitude"].dropna()
    if service_ceiling_ft is None:
        max_alt = float(all_alts.max()) if len(all_alts) > 0 else 10000.0
        service_ceiling_ft = math.ceil(max_alt / 1000.0) * 1000.0

    # --- Fit each phase ---
    climb_spd, climb_vs, climb_metrics, climb_pd = _fit_phase(
        climb_df, "climb",
        altitude_bin_ft=altitude_bin_ft,
        min_points_per_bin=min_points_per_bin,
        max_schedule_points=max_schedule_points,
        outlier_sigma=outlier_sigma,
    )
    cruise_spd, _, cruise_metrics, cruise_pd = _fit_phase(
        cruise_df, "cruise",
        altitude_bin_ft=altitude_bin_ft,
        min_points_per_bin=min_points_per_bin,
        max_schedule_points=max_schedule_points,
        outlier_sigma=outlier_sigma,
    )
    descent_spd, descent_vs, descent_metrics, descent_pd = _fit_phase(
        descent_df, "descent",
        altitude_bin_ft=altitude_bin_ft,
        min_points_per_bin=min_points_per_bin,
        max_schedule_points=max_schedule_points,
        outlier_sigma=outlier_sigma,
    )

    # --- Fallbacks for missing phases ---
    if climb_spd is None:
        climb_spd = cruise_spd or _default_speed_schedule()
    if cruise_spd is None:
        cruise_spd = climb_spd or _default_speed_schedule()
    if descent_spd is None:
        descent_spd = _reduce_schedule(cruise_spd, reduction_kt=30.0)
    if climb_vs is None:
        climb_vs = _default_vertical_profile(1500.0)
    if descent_vs is None:
        descent_vs = _default_vertical_profile(1000.0)

    # --- Approach speed ---
    approach_speed_kt = _estimate_approach_speed(descent_df)

    # --- Assemble metrics dict ---
    metrics = {}
    if climb_metrics:
        metrics["climb_speed"] = climb_metrics[0]
        if len(climb_metrics) > 1:
            metrics["climb_vertical"] = climb_metrics[1]
    if cruise_metrics:
        metrics["cruise_speed"] = cruise_metrics[0]
    if descent_metrics:
        metrics["descent_speed"] = descent_metrics[0]
        if len(descent_metrics) > 1:
            metrics["descent_vertical"] = descent_metrics[1]

    phase_data = [
        p for p in [climb_pd, cruise_pd, descent_pd] if p is not None
    ]

    return FitResult(
        climb_schedule=climb_spd,
        cruise_schedule=cruise_spd,
        descent_schedule=descent_spd,
        climb_profile=climb_vs,
        descent_profile=descent_vs,
        service_ceiling_ft=service_ceiling_ft,
        approach_speed_kt=approach_speed_kt,
        metrics=metrics,
        phase_data=phase_data,
    )


# ------------------------------------------------------------------
# Per-phase fitting
# ------------------------------------------------------------------


def _fit_phase(
    phase_df,
    phase_name: str,
    *,
    altitude_bin_ft: float,
    min_points_per_bin: int,
    max_schedule_points: int,
    outlier_sigma: float,
) -> Tuple[
    Optional[TasSchedule],
    Optional[VerticalProfile],
    List[ScheduleFitMetrics],
    Optional[FlightPhaseData],
]:
    """Fit speed schedule and vertical profile for one phase.

    Returns ``(speed_schedule, vertical_profile, metrics_list, phase_data)``
    where any element may be *None* if the phase has insufficient data.
    """
    if phase_df is None or len(phase_df) == 0:
        return None, None, [], None

    alt = phase_df["altitude"].values.astype(float)
    tas = phase_df["tas_kt"].values.astype(float)
    vs = phase_df["vertical_rate"].values.astype(float)

    # Remove NaNs
    valid = np.isfinite(alt) & np.isfinite(tas) & np.isfinite(vs)
    alt, tas, vs = alt[valid], tas[valid], vs[valid]

    if len(alt) < min_points_per_bin:
        return None, None, [], None

    # --- Altitude binning ---
    alt_min, alt_max = float(alt.min()), float(alt.max())
    bin_edges = np.arange(
        alt_min, alt_max + altitude_bin_ft, altitude_bin_ft
    )
    if len(bin_edges) < 2:
        bin_edges = np.array([alt_min, alt_min + altitude_bin_ft])

    bin_indices = np.digitize(alt, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    bin_centers = []
    bin_tas_medians = []
    bin_vs_medians = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        if mask.sum() < min_points_per_bin:
            continue

        bin_alt = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        bin_t = tas[mask]
        bin_v = np.abs(vs[mask])

        # Outlier rejection
        bin_t = _reject_outliers(bin_t, outlier_sigma)
        bin_v = _reject_outliers(bin_v, outlier_sigma)

        if len(bin_t) == 0:
            continue

        bin_centers.append(bin_alt)
        bin_tas_medians.append(float(np.median(bin_t)))
        bin_vs_medians.append(float(np.median(bin_v)))
        bin_counts.append(int(mask.sum()))

    if len(bin_centers) == 0:
        return None, None, [], None

    centers = np.array(bin_centers)
    tas_meds = np.array(bin_tas_medians)
    vs_meds = np.array(bin_vs_medians)

    # --- RDP simplification ---
    spd_pts = np.column_stack([centers, tas_meds])
    spd_simplified = _rdp_simplify(spd_pts, max_schedule_points)

    # --- Build TasSchedule ---
    schedule = TasSchedule(
        points=[
            (float(a) * ureg.feet, float(s) * ureg.knot)
            for a, s in spd_simplified
        ]
    )

    # --- Build VerticalProfile (for climb/descent only) ---
    profile = None
    if phase_name in ("climb", "descent"):
        vs_pts = np.column_stack([centers, vs_meds])
        vs_simplified = _rdp_simplify(vs_pts, max_schedule_points)
        profile = VerticalProfile(
            points=[
                (float(a) * ureg.feet, float(r) * ureg.feet / ureg.minute)
                for a, r in vs_simplified
            ]
        )

    # --- Compute metrics ---
    metrics_list = []

    spd_metric = _compute_metrics(
        centers, tas_meds, spd_simplified, sum(bin_counts),
        alt_min, alt_max, altitude_bin_ft,
    )
    metrics_list.append(spd_metric)

    if profile is not None:
        vs_metric = _compute_metrics(
            centers, vs_meds, vs_simplified, sum(bin_counts),
            alt_min, alt_max, altitude_bin_ft,
        )
        metrics_list.append(vs_metric)

    phase_data = FlightPhaseData(
        phase=phase_name,
        altitude_bins_ft=list(bin_centers),
        median_tas_kt=list(bin_tas_medians),
        median_vs_fpm=list(bin_vs_medians if phase_name != "cruise" else []),
        count_per_bin=bin_counts,
        altitude_range_ft=(alt_min, alt_max),
    )

    return schedule, profile, metrics_list, phase_data


# ------------------------------------------------------------------
# RDP simplification
# ------------------------------------------------------------------


def _rdp_simplify(points: np.ndarray, max_points: int) -> np.ndarray:
    """Simplify a polyline to at most *max_points* vertices using RDP.

    Normalizes both axes to [0, 1] so altitude and speed/rate contribute
    equally to the perpendicular distance metric.  Uses binary search on
    epsilon to find the tolerance that yields at most *max_points*.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` values.
        max_points: Maximum number of output vertices.

    Returns:
        ``(M, 2)`` array with ``M <= max_points``, preserving endpoints.
    """
    if len(points) <= max_points:
        return points

    # Normalize to [0, 1]
    mins = points.min(axis=0)
    ranges = points.max(axis=0) - mins
    ranges[ranges == 0] = 1.0
    normalized = (points - mins) / ranges

    lo, hi = 0.0, 1.0
    best = points
    for _ in range(50):
        eps = (lo + hi) / 2.0
        simplified = _rdp_core(normalized, eps)
        if len(simplified) <= max_points:
            # Denormalize and save
            best = simplified * ranges + mins
            hi = eps
        else:
            lo = eps

    return best


def _rdp_core(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points

    # Find point with maximum perpendicular distance
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        # Degenerate: all points at same location
        dists = np.linalg.norm(points - start, axis=1)
    else:
        # Perpendicular distance from each point to start-end line
        cross = np.abs(
            (points[:, 1] - start[1]) * line_vec[0]
            - (points[:, 0] - start[0]) * line_vec[1]
        )
        dists = cross / line_len

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = _rdp_core(points[: max_idx + 1], epsilon)
        right = _rdp_core(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([points[0], points[-1]])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _reject_outliers(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Remove values beyond *sigma* standard deviations from the median."""
    if len(arr) < 3:
        return arr
    med = np.median(arr)
    std = np.std(arr)
    if std < 1e-9:
        return arr
    mask = np.abs(arr - med) <= sigma * std
    return arr[mask]


def _compute_metrics(
    bin_centers: np.ndarray,
    bin_values: np.ndarray,
    simplified: np.ndarray,
    n_observations: int,
    alt_min: float,
    alt_max: float,
    altitude_bin_ft: float,
) -> ScheduleFitMetrics:
    """Compute fit quality metrics for a piecewise-linear schedule."""
    # Interpolate the simplified polyline at the bin centers
    predicted = np.interp(bin_centers, simplified[:, 0], simplified[:, 1])
    residuals = bin_values - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((bin_values - np.mean(bin_values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    alt_range = alt_max - alt_min
    n_possible_bins = max(1, int(alt_range / altitude_bin_ft))
    coverage = min(1.0, len(bin_centers) / n_possible_bins)

    return ScheduleFitMetrics(
        r_squared=max(0.0, r_squared),
        rmse=rmse,
        n_observations=n_observations,
        altitude_coverage_pct=coverage,
        n_breakpoints=len(simplified),
    )


def _estimate_approach_speed(descent_df, ground_altitude_ft: float = 3000.0) -> float:
    """Estimate approach speed from low-altitude descent observations."""
    if descent_df is None or len(descent_df) == 0:
        return 130.0  # conservative default

    low = descent_df[descent_df["altitude"] < ground_altitude_ft]
    if len(low) == 0:
        # Fall back to lowest-altitude observations
        low = descent_df.nsmallest(20, "altitude")

    if "tas_kt" in low.columns and len(low) > 0:
        return float(low["tas_kt"].median())
    if "groundspeed" in low.columns and len(low) > 0:
        return float(low["groundspeed"].median())
    return 130.0


def _default_speed_schedule() -> TasSchedule:
    """Fallback single-point speed schedule."""
    return TasSchedule(points=[(0 * ureg.feet, 250 * ureg.knot)])


def _default_vertical_profile(rate_fpm: float) -> VerticalProfile:
    """Fallback single-point vertical profile."""
    return VerticalProfile(
        points=[(0 * ureg.feet, rate_fpm * ureg.feet / ureg.minute)]
    )


def _reduce_schedule(
    schedule: TasSchedule, reduction_kt: float
) -> TasSchedule:
    """Build a descent schedule by reducing TAS at all breakpoints."""
    return TasSchedule(
        points=[
            (alt, max(50.0, spd.m_as(ureg.knot) - reduction_kt) * ureg.knot)
            for alt, spd in schedule.points
        ]
    )
