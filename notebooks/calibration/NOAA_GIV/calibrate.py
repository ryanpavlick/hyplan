"""NOAA G-IV-SP "Gonzo" (N49RF) calibration from NOAA AOML HRD data.

NOAA's Gulfstream IV-SP, tail N49RF callsign "Gonzo", is the
hurricane synoptic-surveillance jet operated by NOAA AOC.  Cruise
FL420-FL450 dropping sondes around developing tropical cyclones.
Distinct mission profile and operator from NASA's brochure-only
``NASA_GIV``.

Data: HRD AOML public flight-level archive at
``data/HRD/G-IV-SP_N49RF/<YYYY>/<storm>/<YYYYMMDD>N<#>.1sec.txt``,
1-second fixed-width text (same format as NOAA P-3 H/I files).
See ``notebooks/calibration/_hrd_loader.py:load_p3_1sec`` for the
loader (handles both P-3 and G-IV variants of the format).

Run from repo root::

    python -m notebooks.calibration.NOAA_GIV.calibrate
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
    apply_sortie_filters, label_phases, per_bin, schedule_pts,
    summary_table, tas_per_bin,
)
from _hrd_loader import load_p3_1sec  # noqa: E402


GIV_GLOB = "data/HRD/G-IV-SP_N49RF/*/*/*N*.1sec.txt"

# G-IV envelope: cruise M0.80 (~470 KTAS), MMO M0.88, certified ceiling 45000 ft.
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 720
MIN_PEAK_ALT_FT = 15000  # synoptic surveillance routinely FL420+
MAX_PEAK_ALT_FT = 50000
TARGET_ALTS_FT = (0, 10000, 20000, 30000, 40000, 45000)


def load_sorties() -> dict[str, pd.DataFrame]:
    paths = sorted(Path(".").glob(GIV_GLOB))
    print(f"  scanning {len(paths)} G-IV-SP ARWO files")
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        key = f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"
        df = load_p3_1sec(p)
        if df is None or df.empty:
            skipped.append((key, "load failed"))
            continue
        kept, reason = apply_sortie_filters(
            df, min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((key, reason or "?"))
            continue
        sorties[key] = label_phases(kept)
    summary_table(sorties, skipped, source_label="NOAA HRD AOML G-IV-SP ARWO",
                   manifest_path="data/HRD/G-IV-SP_N49RF/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("NOAA G-IV-SP 'Gonzo' calibration (HRD AOML 2021-2025)")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties loaded")

    climb_bins = per_bin(sorties, "climb", +1, ACTIVE_VS_THR_FPM, n_min=30)
    desc_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM, n_min=30)
    cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, ("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, ("descent",), n_min=200)

    print("\nCLIMB VS bins (active VS >= 1500 fpm):")
    print(climb_bins.to_string(index=False))
    print("\nDESCENT VS bins:")
    print(desc_bins.to_string(index=False))
    print("\nCRUISE TAS bins:")
    print(cruise_tas.to_string(index=False))

    cs = schedule_pts(cruise_tas, TARGET_ALTS_FT, n_min=200)
    klms = schedule_pts(climb_tas, TARGET_ALTS_FT, n_min=200)
    ds = schedule_pts(desc_tas, TARGET_ALTS_FT, n_min=200)
    print(f"\nCruise TAS schedule: {cs}")
    print(f"Climb  TAS schedule: {klms}")
    print(f"Desc.  TAS schedule: {ds}")

    final_rows = []
    for a in sorties.values():
        if a.empty:
            continue
        t_end = a["timestamp"].iloc[-1]
        sub = a[a["timestamp"] >= t_end - pd.Timedelta(seconds=60)]
        sub = sub[sub["vertical_rate"] < -200]
        final_rows.append(sub[["tas_kt"]])
    final = pd.concat(final_rows).dropna() if final_rows else pd.DataFrame()
    approach_kt = float(final["tas_kt"].median()) if len(final) else float("nan")

    peaks = [float(a["altitude"].max()) for a in sorties.values()]
    ceiling = float(np.percentile(peaks, 99))

    rolls = []
    for a in sorties.values():
        if "roll_deg" in a.columns:
            r = a["roll_deg"].abs()
            rolls.append(r[r > 5.0])
    roll_p90 = float(pd.concat(rolls).quantile(0.90)) if rolls else float("nan")

    print(f"\nApproach TAS median: {approach_kt:.0f} kt")
    print(f"Service ceiling (op-p99): {ceiling:.0f} ft")
    print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}°")

    print()
    print("=" * 70)
    print("PASTE-READY NOAA_GIV() PERFORMANCE BLOCK")
    print("=" * 70)

    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {len(sorties)} HRD AOML G-IV-SP ARWO sorties")
    print("# (NOAA hurricane synoptic-surveillance, 2021-2025).")
    print(f"service_ceiling={int(round(ceiling/100)*100)} * ureg.feet,")
    print(f"approach_speed={int(round(approach_kt))} * ureg.knot,")
    print(f"climb_schedule=TasSchedule(points={klms!r}),")
    print(f"cruise_schedule=TasSchedule(points={cs!r}),")
    print(f"descent_schedule=TasSchedule(points={ds!r}),")
    print(f"climb_profile=VerticalProfile(points={_vs_pts(climb_bins)!r}),")
    print(f"descent_profile=VerticalProfile(points={_vs_pts(desc_bins)!r}),")
    print(f"max_bank_deg={max(30.0, round(roll_p90))},")


if __name__ == "__main__":
    main()
