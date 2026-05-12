"""NASA WB-57 (NASA 926 + 927, JSC) calibration from IWG1 + ICARTT sorties.

High-altitude reconnaissance twin-jet operating from NASA Johnson
Space Center.  Typical cruise FL500-FL620; brochure ceiling FL650.

Sources (both land in ``data/NASA_WB57/``):

* IWG1 per-sortie .txt files split from local in-house
  ``n92[67]NA_alltracks.csv`` deliveries (prefix ``n926_*.txt`` /
  ``n927_*.txt``).
* ICARTT MMS-1HZ .ICT files from the ACCLIP 2022 campaign archive at
  NASA LaRC ASDC, fetched via ``_fetch_acclip.py``
  (``n926_*_acclip-mms.ICT``).  Both formats produce DataFrames with
  the same canonical schema, so the calibration pipeline treats them
  uniformly after load + ``trim_ground_taxi``.

Per-phase TAS schedule targets — climb anchored at SL with rotation
TAS, cruise restricted to FL500-FL620 (the typical operating band;
below FL500 cruise-labeled bins are mostly transient level-offs),
descent anchored at SL approach TAS.

Run from repo root::

    python -m notebooks.calibration.NASA_WB57.calibrate
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
from hyplan.aircraft.icartt import load_icartt  # noqa: E402


# ---- globs ----------------------------------------------------------
WB57_IWG1_GLOB = "data/NASA_WB57/n92[67]_*.txt"
WB57_ICT_GLOB = "data/NASA_WB57/n926_*_acclip-mms.ICT"

# ---- calibration parameters ----------------------------------------
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 35000
MAX_PEAK_ALT_FT = 70000
TARGET_ALTS_FT = (0, 10000, 20000, 30000, 40000, 50000, 60000)

# Per-phase schedule targets.
CLIMB_TARGET_ALTS_FT = (10000, 20000, 30000, 40000, 50000, 60000)
CRUISE_TARGET_ALTS_FT = (50000, 55000, 60000, 62000)
DESCENT_TARGET_ALTS_FT = (10000, 20000, 30000, 40000, 50000, 60000)

ROTATION_TAS_KT = 150
APPROACH_ANCHOR_TAS_KT = 140

CEILING_FT = 65000  # brochure ceiling
CEILING_VS_FPM = 500.0


def load_sorties() -> dict[str, pd.DataFrame]:
    iwg1_paths = sorted(Path(".").glob(WB57_IWG1_GLOB))
    ict_paths = sorted(Path(".").glob(WB57_ICT_GLOB))
    print(f"  scanning {len(iwg1_paths)} IWG1 files (n926 + n927) "
          f"+ {len(ict_paths)} ACCLIP MMS-1HZ ICARTT files")

    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    # Dispatch on suffix: .txt → IWG1, .ICT/.ict → ICARTT.  Both
    # produce canonical-schema DataFrames consumable by
    # trim_ground_taxi / apply_sortie_filters / label_phases.  ACCLIP
    # MMS-1HZ files ship TAS but not groundspeed; backfill GS from TAS
    # so trim_ground_taxi's >25 kt airborne gate fires.  (Wind effect
    # on WB-57 cruise is ≲10% of TAS at FL550+; the gate's job is
    # "definitely airborne or not" not precise GS, so this proxy is
    # fine for the trim step.)
    for p in iwg1_paths + ict_paths:
        try:
            if p.suffix.lower() == ".ict":
                raw = load_icartt(p)
                if "groundspeed" not in raw.columns or raw["groundspeed"].isna().all():
                    raw = raw.copy()
                    raw["groundspeed"] = raw["tas_kt"]
            else:
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
        source_label=(
            "NASA WB-57 (NASA 926 + 927): n92[67]NA_alltracks IWG1 delivery "
            "+ ACCLIP 2022 MMS-1HZ ICARTT (LaRC ASDC)"
        ),
        print_it=False,
        manifest_path="data/NASA_WB57/calibration_manifest.csv",
    )
    print(f"  loaded {len(sorties)} sorties, skipped {len(skipped)}")
    return sorties


def main() -> None:
    print("NASA WB-57 calibration (NASA 926 + 927 / JSC)")
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

    # Approach speed: tighter window than G-III (50-500 ft AGL,
    # VS < -200 fpm) to isolate final approach not pattern speed.
    approach_tas: list[float] = []
    for a in sorties.values():
        floor = a["altitude"].min()
        sub = a[(a["altitude"] - floor < 500)
                & (a["altitude"] - floor > 50)
                & (a["vertical_rate"] < -200)]
        if not sub.empty and sub["tas_kt"].notna().any():
            approach_tas.append(float(sub["tas_kt"].median()))
    ap_s = pd.Series(approach_tas).dropna()
    approach_kt = float(ap_s.median()) if len(ap_s) else float("nan")

    peaks = pd.Series([float(a["altitude"].max()) for a in sorties.values()])
    ceiling = float(peaks.quantile(0.99))

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
        "nasa_wb57",
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
