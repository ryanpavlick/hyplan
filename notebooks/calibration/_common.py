"""Shared helpers for the per-aircraft calibration notebook builders.

Each ``_build_notebook.py`` under ``notebooks/calibration/<aircraft>/``
emits a notebook that follows the same recipe:

1. Load IWG1 / ICARTT files, trim ground taxi, filter sortie length
   and peak altitude, phase-label by vertical-rate threshold.
2. Active-VS per-altitude-bin medians for climb / descent profiles.
3. Per-phase TAS schedules (climb / cruise / descent).
4. Bank-angle p90 in turns.
5. Paste-ready constructor block for ``hyplan/aircraft/_models.py``.

The aircraft-specific knobs (active-VS threshold, target altitudes,
rotation TAS, brochure ceiling, hold bands for the ER-2) stay in the
per-aircraft builder; everything else lives here.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phase labeling
# ---------------------------------------------------------------------------

def label_phases(
    df: pd.DataFrame,
    climb_fpm: float = 300.0,
    descent_fpm: float = -300.0,
) -> pd.DataFrame:
    """Tag each fix as climb / cruise / descent / unlabeled.

    ``vertical_rate`` is expected to be in fpm.  The default thresholds
    separate sustained vertical motion from autopilot ±100 ft cruise
    oscillation across every aircraft we've calibrated.
    """
    out = df.copy()
    vs = out["vertical_rate"].to_numpy()
    phase = np.full(len(out), "unlabeled", dtype=object)
    phase[vs > climb_fpm] = "climb"
    phase[vs < descent_fpm] = "descent"
    phase[(vs >= descent_fpm) & (vs <= climb_fpm)] = "cruise"
    out["phase"] = phase
    return out


# ---------------------------------------------------------------------------
# Sortie filtering
# ---------------------------------------------------------------------------

def apply_sortie_filters(
    a: pd.DataFrame,
    *,
    min_dur_min: float,
    max_dur_min: float,
    min_peak_alt_ft: float,
    max_peak_alt_ft: float = float("inf"),
):
    """Apply the duration / peak-altitude filter loop.

    Returns ``(kept_df_or_None, skip_reason_or_None)``.  When the
    sortie passes, ``kept_df`` is the input frame with NaN altitudes /
    vertical rates dropped; ``skip_reason`` is ``None``.
    """
    if a.empty:
        return None, "no airborne fixes"
    a = a.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    if a.empty:
        return None, "no valid altitude"
    dur_min = (a["timestamp"].iloc[-1] - a["timestamp"].iloc[0]).total_seconds() / 60.0
    peak = float(a["altitude"].max())
    if dur_min < min_dur_min:
        return None, f"too short ({dur_min:.0f} min)"
    if dur_min > max_dur_min:
        return None, f"too long ({dur_min:.0f} min)"
    if peak < min_peak_alt_ft:
        return None, f"low peak alt ({peak:.0f} ft)"
    if peak > max_peak_alt_ft:
        return None, f"high peak alt ({peak:.0f} ft)"
    return a, None


# ---------------------------------------------------------------------------
# Per-altitude-bin aggregations
# ---------------------------------------------------------------------------

def _exclude_hold_bands(df: pd.DataFrame, bands: Sequence[tuple[float, float]]) -> pd.DataFrame:
    if not bands:
        return df
    keep = pd.Series(True, index=df.index)
    for lo, hi in bands:
        keep &= ~((df["altitude"] >= lo) & (df["altitude"] < hi))
    return df[keep]


def per_bin(
    sorties: dict,
    phase: str,
    sign: int,
    active_thr_fpm: float,
    *,
    bin_ft: int = 5000,
    hold_bands_ft: Sequence[tuple[float, float]] | None = None,
    n_min: int = 30,
    extra_cols: Iterable[str] = ("tas_kt",),
) -> pd.DataFrame:
    """Active-VS per-altitude-bin medians for one phase.

    ``sign`` is +1 for climb, -1 for descent.  ``hold_bands_ft`` (ER-2
    only) excludes weight-management hold altitudes from the bin
    medians so they don't contaminate the active-climb shape.
    """
    cols = ["altitude", "vertical_rate", *extra_cols]
    rows = []
    for a in sorties.values():
        sub = a[a["phase"] == phase]
        sub = sub[(sub["vertical_rate"] * sign) >= active_thr_fpm]
        sub = _exclude_hold_bands(sub, hold_bands_ft or [])
        rows.append(sub[[c for c in cols if c in sub.columns]].copy())
    df = pd.concat(rows).dropna(subset=["altitude"])
    df["alt_bin_ft"] = (df["altitude"] // bin_ft * bin_ft).astype(int)
    agg = {
        "n": ("vertical_rate", "count"),
        "vs_med": ("vertical_rate", "median"),
        "vs_p25": ("vertical_rate", lambda x: x.quantile(0.25)),
        "vs_p75": ("vertical_rate", lambda x: x.quantile(0.75)),
    }
    if "tas_kt" in df.columns:
        agg["tas_med"] = ("tas_kt", "median")
    if "mach" in df.columns:
        agg["mach_med"] = ("mach", "median")
    g = df.groupby("alt_bin_ft").agg(**agg).round(1).reset_index()
    return g[g["n"] >= n_min]


def tas_per_bin(
    sorties: dict,
    phases: Sequence[str],
    *,
    bin_ft: int = 5000,
    n_min: int = 200,
) -> pd.DataFrame:
    """Per-altitude-bin median TAS for a given phase set."""
    rows = []
    for a in sorties.values():
        sub = a[a["phase"].isin(phases)]
        cols = ["altitude", "tas_kt"]
        if "mach" in sub.columns:
            cols.append("mach")
        rows.append(sub[cols])
    df = pd.concat(rows).dropna(subset=["tas_kt"])
    df["alt_bin_ft"] = (df["altitude"] // bin_ft * bin_ft).astype(int)
    agg = {"n": ("tas_kt", "count"), "tas_med": ("tas_kt", "median")}
    if "mach" in df.columns:
        agg["mach_med"] = ("mach", "median")
    g = df.groupby("alt_bin_ft").agg(**agg).round(1).reset_index()
    return g[g["n"] >= n_min]


# ---------------------------------------------------------------------------
# Schedule / profile construction
# ---------------------------------------------------------------------------

def schedule_pts(
    bins: pd.DataFrame,
    target_alts: Sequence[int],
    *,
    n_min: int = 200,
) -> list[tuple[int, int]]:
    """Pick the bin nearest each target altitude (rounded TAS).

    Bins below ``n_min`` samples are dropped as too thin to trust.
    """
    out = []
    bins = bins[bins["n"] >= n_min]
    if bins.empty:
        return out
    for target in target_alts:
        row = bins.iloc[(bins["alt_bin_ft"] - target).abs().argsort().iloc[0]]
        out.append((int(target), int(round(float(row["tas_med"])))))
    return out


def evaluate_profile(points: Sequence[tuple[float, float]], alts: np.ndarray) -> np.ndarray:
    """Linear interpolation through ``(alt_ft, vs_fpm)`` breakpoints."""
    pts = sorted(points, key=lambda p: p[0])
    pa = np.array([p[0] for p in pts])
    pv = np.array([p[1] for p in pts])
    return np.interp(alts, pa, pv)


# ---------------------------------------------------------------------------
# Standard summary table
# ---------------------------------------------------------------------------

def summary_table(
    sorties: dict,
    skipped: list[tuple[str, str]],
    *,
    source_label: str | None = None,
    print_it: bool = True,
) -> pd.DataFrame:
    """Standardized post-§1 summary: counts + reason breakdown.

    Returns a one-row DataFrame; also prints a formatted view by
    default.  Each builder calls this in place of the loose
    "loaded N / skipped M / reason counter" output.
    """
    raw_files = len(sorties) + len(skipped)
    valid = len(sorties)
    excluded = len(skipped)
    if sorties:
        ts = pd.concat([a["timestamp"] for a in sorties.values()])
        date_lo = ts.min().date().isoformat()
        date_hi = ts.max().date().isoformat()
    else:
        date_lo = date_hi = "-"

    reasons = Counter(r.split(" (")[0] for _, r in skipped)
    reason_str = ", ".join(f"{n} {k}" for k, n in reasons.most_common()) or "-"

    df = pd.DataFrame([{
        "source": source_label or "(multiple)",
        "raw_files": raw_files,
        "valid_sorties": valid,
        "excluded": excluded,
        "exclusion_reasons": reason_str,
        "date_range": f"{date_lo} → {date_hi}",
    }])

    if print_it:
        print(f"source:            {df.at[0, 'source']}")
        print(f"raw files:         {raw_files}")
        print(f"valid sorties:     {valid}")
        print(f"excluded:          {excluded}")
        if reasons:
            print("excluded by reason:")
            for k, n in reasons.most_common():
                print(f"  {n:4d}  {k}")
        print(f"date range:        {date_lo} → {date_hi}")
    return df
