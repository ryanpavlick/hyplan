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
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared numerical helpers (deduped across per-aircraft loaders)
# ---------------------------------------------------------------------------

#: Conversion factors used everywhere in the calibration pipeline.
M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501
DEG_LAT_TO_M = 111_320.0  # m per degree latitude (sphere approx)


def smooth_diff(
    t_s: np.ndarray, x: np.ndarray, half_s: float = 90.0
) -> np.ndarray:
    """Centered finite difference of ``x(t)`` over a 2*half_s window.

    Default ``half_s=90`` matches the 180-second window used for vertical
    rate estimation across every aircraft loader.  For groundspeed
    estimation from lat/lon, callers typically use ``half_s=5`` to avoid
    over-smoothing.
    """
    lo = np.searchsorted(t_s, t_s - half_s, side="left")
    hi = np.searchsorted(t_s, t_s + half_s, side="right") - 1
    lo = np.clip(lo, 0, len(t_s) - 1)
    hi = np.clip(hi, 0, len(t_s) - 1)
    dt = t_s[hi] - t_s[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dt > 0, (x[hi] - x[lo]) / dt, np.nan)


def vertical_rate_fpm(t_s: np.ndarray, alt_ft: np.ndarray) -> np.ndarray:
    """Vertical rate in fpm via 180-second centered finite difference.

    The standard recipe across all calibration scripts.  Use this so
    every aircraft sees the same smoothing — phase labels (climb /
    cruise / descent thresholds) are calibrated against this window.
    """
    return smooth_diff(t_s, alt_ft, half_s=90.0) * 60.0


def wind_triangle_tas_kt(
    t_s: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    half_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct TAS and groundspeed in kt from position + wind.

    Used for archives that don't ship native TAS (HIAPER DC3, SAFIRE
    ATR-42 EUFAR, BAS Twin Otter ArcticCyclones / IGP, NERC DO-228).

    ``u_wind`` / ``v_wind`` are the eastward / northward wind vector
    components in m/s.  Groundspeed is computed from finite-difference
    of position; TAS is groundspeed minus the wind vector
    (``TAS_vec = GS_vec - wind_vec``).

    Returns ``(tas_kt, groundspeed_kt)``.
    """
    cos_lat = np.cos(np.deg2rad(lat))
    gs_e = smooth_diff(t_s, lon, half_s=half_s) * DEG_LAT_TO_M * cos_lat
    gs_n = smooth_diff(t_s, lat, half_s=half_s) * DEG_LAT_TO_M
    tas_e = gs_e - u_wind
    tas_n = gs_n - v_wind
    tas_ms = np.hypot(tas_e, tas_n)
    gs_ms = np.hypot(gs_e, gs_n)
    return tas_ms * M_PER_S_TO_KT, gs_ms * M_PER_S_TO_KT


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
    manifest_path: Path | str | None = None,
) -> pd.DataFrame:
    """Standardized post-§1 summary: counts + reason breakdown.

    Returns a one-row DataFrame; also prints a formatted view by
    default.  Each builder calls this in place of the loose
    "loaded N / skipped M / reason counter" output.

    If ``manifest_path`` is given, also writes a per-sortie CSV listing
    every input file with its kept/dropped status, exclusion reason,
    duration, peak altitude, and date.  The manifest is the
    auditable provenance trail for a calibration run; reviewers can
    answer "did sortie X contribute?" without re-running the script.
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

    if manifest_path is not None:
        rows = []
        for key, frame in sorties.items():
            t0 = frame["timestamp"].iloc[0]
            t1 = frame["timestamp"].iloc[-1]
            rows.append({
                "key": key,
                "status": "kept",
                "reason": "",
                "date": t0.date().isoformat(),
                "duration_min": round((t1 - t0).total_seconds() / 60.0, 1),
                "peak_alt_ft": int(round(float(frame["altitude"].max()))),
                "n_fixes": int(len(frame)),
            })
        for key, reason in skipped:
            rows.append({
                "key": key, "status": "dropped", "reason": reason,
                "date": "", "duration_min": "", "peak_alt_ft": "", "n_fixes": "",
            })
        manifest_df = pd.DataFrame(rows).sort_values(["status", "key"])
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(path, index=False)
        if print_it:
            print(f"manifest:          {path}")
    return df
