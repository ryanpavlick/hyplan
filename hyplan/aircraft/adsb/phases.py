"""Flight phase labeling for ADS-B trajectories.

Assigns a phase label (climb, cruise, descent, ground, level_off) to
each observation in a trajectory.  v1 ships a heuristic backend; the
architecture supports plugging in an LSTM model later.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PhaseLabel = Literal["climb", "cruise", "descent", "ground", "level_off"]


def label_phases(
    flight,
    *,
    backend: str = "heuristic",
    climb_vs_threshold_fpm: float = 300.0,
    descent_vs_threshold_fpm: float = -300.0,
    level_band_ft: float = 200.0,
    min_phase_seconds: float = 60.0,
    ground_altitude_ft: float = 2000.0,
) -> pd.DataFrame:
    """Assign flight phase labels to each observation in a trajectory.

    Args:
        flight: A ``traffic.core.Flight`` object (or anything with a
            ``.data`` attribute containing a DataFrame with columns
            ``timestamp``, ``altitude``, ``vertical_rate``).
        backend: Labeling strategy.  ``"heuristic"`` (default) uses
            vertical-rate thresholds.  ``"lstm"`` is reserved for future
            ML-based classification.
        climb_vs_threshold_fpm: Smoothed vertical speed (ft/min) above
            which a point is labeled climb.
        descent_vs_threshold_fpm: Smoothed vertical speed (ft/min) below
            which a point is labeled descent (should be negative).
        level_band_ft: Maximum altitude variation within a cruise run.
            Cruise segments with larger drift are reclassified as
            ``level_off``.
        min_phase_seconds: Minimum phase duration.  Phases shorter than
            this are merged into the adjacent longer phase.
        ground_altitude_ft: Points below this altitude (feet) are
            labeled ``"ground"``.

    Returns:
        Copy of ``flight.data`` with an added ``"phase"`` column.
    """
    if backend == "lstm":
        raise NotImplementedError(
            "LSTM phase labeling is not yet available. "
            "Use backend='heuristic'."
        )
    if backend != "heuristic":
        raise ValueError(f"Unknown phase labeling backend: {backend!r}")

    df = flight.data.copy()
    phases = _label_heuristic(
        altitude_ft=df["altitude"].values,
        vertical_rate_fpm=df["vertical_rate"].values,
        timestamps=df["timestamp"].values,
        climb_threshold=climb_vs_threshold_fpm,
        descent_threshold=descent_vs_threshold_fpm,
        level_band_ft=level_band_ft,
        min_phase_seconds=min_phase_seconds,
        ground_altitude_ft=ground_altitude_ft,
    )
    df["phase"] = phases
    logger.debug(
        "Labeled %d points: %s",
        len(df),
        {p: int((phases == p).sum()) for p in np.unique(phases)},
    )
    return df


# ------------------------------------------------------------------
# Heuristic implementation
# ------------------------------------------------------------------


def _label_heuristic(
    altitude_ft: np.ndarray,
    vertical_rate_fpm: np.ndarray,
    timestamps: np.ndarray,
    climb_threshold: float,
    descent_threshold: float,
    level_band_ft: float,
    min_phase_seconds: float,
    ground_altitude_ft: float,
) -> np.ndarray:
    """Pure-numpy heuristic phase labeler."""
    n = len(altitude_ft)
    if n == 0:
        return np.array([], dtype=object)  # type: ignore[no-any-return]

    # Step 1: smooth vertical rate with rolling median (window=5)
    vs_smooth = _rolling_median(vertical_rate_fpm, window=5)

    # Step 2: initial labeling by smoothed VS
    phases = np.full(n, "cruise", dtype=object)
    phases[vs_smooth > climb_threshold] = "climb"
    phases[vs_smooth < descent_threshold] = "descent"

    # Step 3: override — low altitude → ground
    phases[altitude_ft < ground_altitude_ft] = "ground"

    # Step 4: cruise refinement — check altitude stability
    phases = _refine_cruise(phases, altitude_ft, level_band_ft)

    # Step 5: merge short phases
    phases = _merge_short_phases(phases, timestamps, min_phase_seconds)

    return phases


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling median over *arr* with edge-clamped padding."""
    n = len(arr)
    if n <= window:
        return np.full(n, np.median(arr))  # type: ignore[no-any-return]
    half = window // 2
    padded = np.pad(arr, half, mode="edge")
    # Use stride tricks for a vectorized sliding window
    shape = (n, window)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.median(windows, axis=1)  # type: ignore[no-any-return]


def _refine_cruise(
    phases: np.ndarray,
    altitude_ft: np.ndarray,
    level_band_ft: float,
) -> np.ndarray:
    """Reclassify cruise runs with altitude drift as level_off."""
    runs = _find_runs(phases)
    for start, end, label in runs:
        if label != "cruise":
            continue
        alt_range = altitude_ft[start:end].max() - altitude_ft[start:end].min()
        if alt_range > level_band_ft:
            phases[start:end] = "level_off"
    return phases


def _merge_short_phases(
    phases: np.ndarray,
    timestamps: np.ndarray,
    min_seconds: float,
) -> np.ndarray:
    """Merge phases shorter than *min_seconds* into their longer neighbor."""
    if len(phases) < 2:
        return phases

    # Convert timestamps to seconds for duration calculation
    ts_sec = _timestamps_to_seconds(timestamps)

    changed = True
    while changed:
        changed = False
        runs = _find_runs(phases)
        if len(runs) <= 1:
            break
        for i, (start, end, label) in enumerate(runs):
            duration = ts_sec[end - 1] - ts_sec[start]
            if duration >= min_seconds:
                continue
            # Find longest neighbor
            if i > 0 and i < len(runs) - 1:
                prev_dur = ts_sec[runs[i - 1][1] - 1] - ts_sec[runs[i - 1][0]]
                next_dur = ts_sec[runs[i + 1][1] - 1] - ts_sec[runs[i + 1][0]]
                merge_label = (
                    runs[i - 1][2] if prev_dur >= next_dur else runs[i + 1][2]
                )
            elif i > 0:
                merge_label = runs[i - 1][2]
            else:
                merge_label = runs[i + 1][2]
            phases[start:end] = merge_label
            changed = True
            break  # restart scan after modification

    return phases


def _find_runs(arr: np.ndarray) -> list:
    """Find consecutive runs in *arr*.

    Returns list of ``(start_idx, end_idx, value)`` tuples where
    ``end_idx`` is exclusive.
    """
    if len(arr) == 0:
        return []
    changes = np.where(arr[:-1] != arr[1:])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(arr)]])
    return [(int(s), int(e), arr[s]) for s, e in zip(starts, ends)]


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert a numpy array of timestamps to float seconds from epoch."""
    if np.issubdtype(timestamps.dtype, np.datetime64):
        epoch = np.datetime64(0, "s")
        return (timestamps - epoch) / np.timedelta64(1, "s")  # type: ignore[no-any-return]
    # Already numeric (e.g. from pandas .values on a numeric column)
    return timestamps.astype(float)  # type: ignore[no-any-return]
