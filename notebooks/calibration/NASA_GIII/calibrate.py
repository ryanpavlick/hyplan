"""NASA G-III (NASA 520, LaRC) calibration from IWG1 sorties.

Source: NASA ASP archive at ``asp-archive.arc.nasa.gov/N520NA``
(per-sortie IWG1 files prefixed ``n520_*.txt``) plus the local
``n520NA_g3_alltracks.csv`` delivery split into per-sortie files
(prefix ``g3ih_*.txt``).  Both sources land in
``data/NASA_GIII/`` and are loaded together by glob.

Per-phase TAS schedule targets differ — climb anchored at SL with a
rotation TAS, cruise restricted to FL250-FL400 (the typical operating
band; below FL250 the cruise label is dominated by brief level-offs
during step climbs), descent anchored at SL approach TAS.

Run from repo root::

    python -m notebooks.calibration.NASA_GIII.calibrate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (  # noqa: E402
    apply_sortie_filters, label_phases, per_bin, tas_per_bin,
    schedule_pts, summary_table,
)

from hyplan.aircraft import load_iwg1, trim_ground_taxi  # noqa: E402


# ---- glob -----------------------------------------------------------
GIII_GLOB = "data/NASA_GIII/*_*.txt"

# ---- calibration parameters ----------------------------------------
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 25000  # below this is local pattern / test, not a cruise sortie
MAX_PEAK_ALT_FT = 50000  # above the certified ceiling -> mislabeled tail
TARGET_ALTS_FT = (0, 10000, 20000, 30000, 40000)

# Per-phase schedule targets (used by main(); the notebook contract
# uses TARGET_ALTS_FT for all three phases).
CLIMB_TARGET_ALTS_FT = (10000, 20000, 30000, 40000)
CRUISE_TARGET_ALTS_FT = (25000, 30000, 35000, 40000)
DESCENT_TARGET_ALTS_FT = (10000, 20000, 30000, 40000)

# SL anchors for climb (rotation) and descent (final approach) — the
# climb-phase SL bin is contaminated by takeoff-roll fixes still
# accelerating, so anchor at typical jet rotation TAS instead.
ROTATION_TAS_KT = 150
APPROACH_ANCHOR_TAS_KT = 180

# Vertical-profile anchors (G-III certified ceiling + residual rate).
CEILING_FT = 45000
CEILING_VS_FPM = 500.0


def load_sorties() -> dict[str, pd.DataFrame]:
    paths = sorted(Path(".").glob(GIII_GLOB))
    print(f"  scanning {len(paths)} IWG1 files")

    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        try:
            raw = load_iwg1(p)
            a = trim_ground_taxi(raw)
        except Exception as e:
            skipped.append((p.stem, f"load fail: {e}"))
            continue
        kept, reason = apply_sortie_filters(
            a,
            min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((p.stem, reason or "?"))
            continue
        sorties[p.stem] = label_phases(kept)

    summary_table(
        sorties, skipped,
        source_label="NASA G-III (NASA 520): ASP archive + n520NA_g3_alltracks delivery",
        print_it=False,
        manifest_path="data/NASA_GIII/calibration_manifest.csv",
    )
    print(f"  loaded {len(sorties)} sorties, skipped {len(skipped)}")
    return sorties


def main() -> None:
    print("NASA G-III calibration (NASA 520 / LaRC)")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties loaded")

    climb_bins = per_bin(sorties, "climb", +1, ACTIVE_VS_THR_FPM,
                         n_min=30, extra_cols=("tas_kt",))
    desc_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM,
                        n_min=30, extra_cols=("tas_kt",))
    cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, ("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, ("descent",), n_min=200)

    print("\nCLIMB VS bins (active VS >= 1500 fpm):")
    print(climb_bins.to_string(index=False))
    print("\nDESCENT VS bins:")
    print(desc_bins.to_string(index=False))
    print("\nCRUISE TAS bins:")
    print(cruise_tas.to_string(index=False))

    # Per-phase TAS schedules with custom anchors.
    climb_pts = [(0, ROTATION_TAS_KT)] + schedule_pts(
        climb_tas, CLIMB_TARGET_ALTS_FT, n_min=200,
    )
    cruise_pts = schedule_pts(cruise_tas, CRUISE_TARGET_ALTS_FT, n_min=200)
    descent_pts = [(0, APPROACH_ANCHOR_TAS_KT)] + schedule_pts(
        desc_tas, DESCENT_TARGET_ALTS_FT, n_min=200,
    )
    print(f"\nClimb TAS schedule:   {climb_pts}")
    print(f"Cruise TAS schedule:  {cruise_pts}")
    print(f"Descent TAS schedule: {descent_pts}")

    # Vertical profiles: per-bin medians + ceiling residual anchor.
    climb_profile_pts: list[tuple[int, float]] = []
    for _, r in climb_bins.iterrows():
        if 0 <= r["alt_bin_ft"] < CEILING_FT:
            climb_profile_pts.append((int(r["alt_bin_ft"]), float(r["vs_med"])))
    climb_profile_pts.append((CEILING_FT, CEILING_VS_FPM))

    descent_profile_pts: list[tuple[int, float]] = []
    for _, r in desc_bins.iterrows():
        if 0 <= r["alt_bin_ft"] < CEILING_FT:
            descent_profile_pts.append(
                (int(r["alt_bin_ft"]), abs(float(r["vs_med"]))),
            )
    descent_profile_pts = sorted(descent_profile_pts)

    # Approach speed: median TAS at last fix above ground floor where
    # VS < -300 fpm and AGL between 200 ft and 1500 ft.
    approach_tas: list[float] = []
    for a in sorties.values():
        floor = a["altitude"].min()
        sub = a[(a["altitude"] - floor < 1500)
                & (a["altitude"] - floor > 200)
                & (a["vertical_rate"] < -300)]
        if not sub.empty and sub["tas_kt"].notna().any():
            approach_tas.append(float(sub["tas_kt"].median()))
    ap_s = pd.Series(approach_tas).dropna()
    approach_kt = float(ap_s.median()) if len(ap_s) else float("nan")

    # Service ceiling: p99 of per-sortie peak altitudes.
    peaks = pd.Series([float(a["altitude"].max()) for a in sorties.values()])
    ceiling = float(peaks.quantile(0.99))

    # Bank: p90 |roll| during turn fixes (>5° gate).
    rolls: list[pd.Series] = []
    for a in sorties.values():
        if "roll_deg" in a.columns:
            r = a["roll_deg"].abs()
            rolls.append(r[r > 5.0])
    all_banks = pd.concat(rolls).dropna() if rolls else pd.Series(dtype=float)
    roll_p90 = float(all_banks.quantile(0.90)) if len(all_banks) else float("nan")

    print(f"\nApproach TAS median:        {approach_kt:.0f} kt (n={len(ap_s)})")
    print(f"Service ceiling (op-p99):   {ceiling:.0f} ft")
    if len(all_banks):
        print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}° "
              f"(n={len(all_banks):,} fixes)")

    from notebooks.calibration._common import apply_calibration_to_profile

    print()
    print("=" * 70)
    path = apply_calibration_to_profile(
        "nasa_giii",
        service_ceiling_ft=int(round(ceiling / 1000) * 1000),
        approach_speed_kt=int(round(approach_kt)),
        climb_pts=climb_pts,
        cruise_pts=cruise_pts,
        descent_pts=descent_pts,
        climb_profile_pts=climb_profile_pts,
        descent_profile_pts=descent_profile_pts,
        max_bank_deg=(max(30.0, round(roll_p90)) if len(all_banks) else 30.0),
    )
    print(f"Wrote calibrated profile to {path}")
    print(f"  fit n_sorties={len(sorties)}")


if __name__ == "__main__":
    main()
