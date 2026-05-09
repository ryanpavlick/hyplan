"""NASA P-3 Orion (NASA 426, LaRC) calibration from IWG1 sorties.

Lockheed P-3 Orion four-engine turboprop, NASA tail N426NA, operated
from NASA Wallops / LaRC for atmospheric chemistry and remote-sensing
missions (ARCTAS, DISCOVER-AQ, FIREX-AQ, etc.).  Distinct from the
NOAA WP-3D 'Hurricane Hunter' (different operator, different mission
profile, separate calibration class ``NOAA_WP3D``).

Source: per-sortie IWG1 files under ``data/NASA_P3/``:
  * ``p3_*.txt`` — local NASA-supplied delivery, split into per-sortie
    files via ``hyplan.aircraft.split_iwg1_alltracks``.
  * ``n426_*.txt`` — NASA ASP archive (``asp-archive.arc.nasa.gov/N426NA``).

Per-phase TAS schedule targets — climb anchored at SL with rotation
TAS, cruise restricted to FL150-FL280 (typical operating band),
descent anchored at SL approach TAS.

Run from repo root::

    python -m notebooks.calibration.NASA_P3.calibrate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (  # noqa: E402
    apply_sortie_filters, label_phases, per_bin, tas_per_bin,
    schedule_pts, summary_table,
)

from hyplan.aircraft import load_iwg1, trim_ground_taxi  # noqa: E402


# Match both p3_*.txt (local delivery) and n426_*.txt (ASP archive).
P3_GLOB = "data/NASA_P3/*_*.txt"

ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 900
MIN_PEAK_ALT_FT = 10000
MAX_PEAK_ALT_FT = 32000  # P-3 brochure ceiling; sorties peaking above
                         # are mislabeled tails (jets under same N number)
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000, 25000)

CLIMB_TARGET_ALTS_FT = (5000, 10000, 15000, 20000, 25000)
CRUISE_TARGET_ALTS_FT = (15000, 20000, 25000, 28000)
DESCENT_TARGET_ALTS_FT = (5000, 10000, 15000, 20000, 25000)

ROTATION_TAS_KT = 110
APPROACH_ANCHOR_TAS_KT = 130

CEILING_FT = 30000
CEILING_VS_FPM = 500.0


def load_sorties() -> dict[str, pd.DataFrame]:
    paths = sorted(Path(".").glob(P3_GLOB))
    print(f"  scanning {len(paths)} IWG1 files (p3 + n426)")

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
        source_label="NASA P-3 (NASA 426): local delivery + ASP archive",
        print_it=False,
        manifest_path="data/NASA_P3/calibration_manifest.csv",
    )
    print(f"  loaded {len(sorties)} sorties, skipped {len(skipped)}")
    return sorties


def main() -> None:
    print("NASA P-3 calibration (NASA 426 / LaRC)")
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

    print()
    print("=" * 70)
    print("PASTE-READY NASA_P3() PERFORMANCE BLOCK")
    print("=" * 70)
    print(f"# Calibrated against {len(sorties)} IWG1 sorties from NASA 426")
    print("# (local delivery + NASA ASP archive at asp-archive.arc.nasa.gov/N426NA).")
    print(f"service_ceiling={int(round(ceiling/1000)*1000)} * ureg.feet,")
    print(f"approach_speed={int(round(approach_kt))} * ureg.knot,")
    print(f"climb_schedule=TasSchedule(points={climb_pts!r}),")
    print(f"cruise_schedule=TasSchedule(points={cruise_pts!r}),")
    print(f"descent_schedule=TasSchedule(points={descent_pts!r}),")
    print(f"climb_profile=VerticalProfile(points={climb_profile_pts!r}),")
    print(f"descent_profile=VerticalProfile(points={descent_profile_pts!r}),")
    if len(all_banks):
        print(f"max_bank_deg={max(30.0, round(roll_p90))},")
    else:
        print("max_bank_deg=30.0,  # AFM default; no roll data")


if __name__ == "__main__":
    main()
