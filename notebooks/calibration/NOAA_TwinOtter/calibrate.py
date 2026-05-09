"""Twin Otter calibration combining FIREX-AQ N48RF + NOAA CSL N46RF.

The existing TwinOtter() class is calibrated against 17 N48RF FIREX-AQ
ICARTT sorties.  ``data/TwinOtter/NOAA_CSL/`` curates additional CSL
campaigns with N46RF (and possibly N48RF), now under
``data/NOAA_TwinOtter/NOAA_CSL/``:

* topdown-2014   — TOPDOWN (?)
* uwfps-2017     — Utah Winter Fine Particulate Study
* calfide-2022   — California Fire Dynamics Experiment
* cupids-aeromma-2023 — AEROMMA NYC megacity
* ammbec-2024    — AMMBEC NYC follow-on
* usos-2024      — Urban-rural Sources / sinks of Ozone Spring

This script loads everything, applies per-file unit detection (some
PIs label TAS / WindSpd as m/s when the values are actually in knots),
runs the standard per-aircraft calibration recipe, and prints a
paste-ready ``NOAA_TwinOtter()`` constructor block.

Run from repo root::

    python -m notebooks.calibration.NOAA_TwinOtter.calibrate
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (
    apply_sortie_filters, label_phases, per_bin, tas_per_bin, schedule_pts,
)

from hyplan.aircraft.icartt import load_icartt
from hyplan.aircraft.iwg1 import trim_ground_taxi


# ─── data sources ────────────────────────────────────────────────────
FIREXAQ_GLOB = "data/NOAA_TwinOtter/FIREXAQ_TwinOtter_N48_FLIGHTDATA/*.ict"
NOAA_CSL_GLOB = "data/NOAA_TwinOtter/NOAA_CSL/*/raw/*.ict"

# Twin Otter physical envelope: max TAS ~85 m/s = 165 kt.  If raw TAS
# values exceed 100 (would mean ~360 kt after the m/s→kt conversion),
# the loader's m/s assumption is wrong and the values are already kt.
TAS_KT_THRESHOLD = 100.0

# Calibration parameters (mirrors the existing _build_notebook.py)
ACTIVE_VS_THR_FPM = 500.0    # Twin Otter is a slow climber
MIN_DUR_MIN = 30
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 4000
MAX_PEAK_ALT_FT = 25000
TARGET_ALTS_FT = (0, 5000, 10000, 12000)


def _load_with_unit_check(path: Path) -> pd.DataFrame:
    """Load an ICARTT file with TAS / wind unit verification.

    The icartt loader's ``_convert_to_canonical`` multiplies by
    1.94 if the unit string contains ``m/s``.  Some campaigns label
    TAS/WindSpd as m/s while the underlying values are already in
    kt — in those cases we undo the spurious conversion.
    """
    df = load_icartt(path)
    # Sanity threshold per aircraft class.  Twin Otter can't fly faster
    # than ~85 m/s air-mass-relative; if loaded TAS exceeds ~190 kt
    # (which would be 98 m/s native) something's wrong.
    if "tas_kt" in df.columns and df["tas_kt"].notna().any():
        if df["tas_kt"].quantile(0.95) > 190:
            df["tas_kt"] = df["tas_kt"] / 1.9438444924406046
    if "wind_speed_kt" in df.columns and df["wind_speed_kt"].notna().any():
        if df["wind_speed_kt"].quantile(0.95) > 100:  # 100 kt is rare; >100 likely m/s mislabel
            df["wind_speed_kt"] = df["wind_speed_kt"] / 1.9438444924406046
    return df


def load_sorties() -> dict[str, pd.DataFrame]:
    paths: list[Path] = (
        sorted(Path(".").glob(FIREXAQ_GLOB))
        + sorted(Path(".").glob(NOAA_CSL_GLOB))
    )
    print(f"  scanning {len(paths)} ICARTT files")

    sorties: dict[str, pd.DataFrame] = {}
    skipped = []
    for p in paths:
        try:
            df = _load_with_unit_check(p)
        except Exception as e:
            skipped.append((p.stem, f"load fail: {e}"))
            continue
        # Trim taxi if groundspeed available; else altitude-based
        if df["groundspeed"].notna().sum() > 100:
            try:
                a = trim_ground_taxi(df)
            except Exception:
                a = df
        else:
            alt = df["altitude"]
            if alt.dropna().empty:
                skipped.append((p.stem, "no alt"))
                continue
            airborne = (alt - alt.min()) > 200
            if not airborne.any():
                skipped.append((p.stem, "no airborne"))
                continue
            i0 = int(airborne.values.argmax())
            i1 = int(len(airborne) - 1 - airborne.values[::-1].argmax())
            a = df.iloc[i0:i1+1].reset_index(drop=True)

        kept, reason = apply_sortie_filters(
            a,
            min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((p.stem, reason or "?"))
            continue
        sorties[p.stem] = label_phases(kept)
    print(f"  loaded {len(sorties)} sorties, skipped {len(skipped)}")
    if skipped[:5]:
        for n, r in skipped[:5]:
            print(f"    skip {n}: {r}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped)-5} more")
    return sorties


def main():
    print("Twin Otter combined calibration (FIREX-AQ + NOAA CSL archive)")
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

    print()
    print("CLIMB VS bins (active VS >= 500 fpm):")
    print(climb_bins.to_string(index=False))
    print()
    print("DESCENT VS bins:")
    print(desc_bins.to_string(index=False))
    print()
    print("CRUISE TAS bins:")
    print(cruise_tas.to_string(index=False))

    cs = schedule_pts(cruise_tas, TARGET_ALTS_FT, n_min=200)
    klms = schedule_pts(climb_tas, TARGET_ALTS_FT, n_min=200)
    ds = schedule_pts(desc_tas, TARGET_ALTS_FT, n_min=200)
    print()
    print(f"Cruise TAS schedule: {cs}")
    print(f"Climb  TAS schedule: {klms}")
    print(f"Desc.  TAS schedule: {ds}")

    # Approach speed: median TAS in last < 3000 ft AGL on descent
    final_rows = []
    for a in sorties.values():
        sub = a[(a["altitude"] < 3000) & (a["vertical_rate"] < -200)]
        if "tas_kt" in sub.columns:
            final_rows.append(sub[["tas_kt"]])
    final = pd.concat(final_rows).dropna() if final_rows else pd.DataFrame()
    approach_kt = float(final["tas_kt"].median()) if len(final) else float("nan")

    # Service ceiling: p99 of per-sortie peak altitudes
    peaks = [float(a["altitude"].max()) for a in sorties.values()]
    ceiling = float(np.percentile(peaks, 99))

    # Bank angle p90
    rolls = []
    for a in sorties.values():
        if "roll_deg" in a.columns:
            r = a["roll_deg"].abs()
            rolls.append(r[r > 5.0])
    roll_p90 = float(pd.concat(rolls).quantile(0.90)) if rolls else float("nan")

    print()
    print(f"Approach TAS median (final < 3000 ft, descent): {approach_kt:.0f} kt")
    print(f"Service ceiling (op-p99): {ceiling:.0f} ft")
    print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}°")

    print()
    print("=" * 70)
    print("PASTE-READY NOAA_TwinOtter() PERFORMANCE BLOCK")
    print("=" * 70)
    _print_block(cs, klms, ds, climb_bins, desc_bins,
                 approach_kt=approach_kt, ceiling=ceiling, roll_p90=roll_p90,
                 n_sorties=len(sorties))


def _print_block(cs, klms, ds, climb_bins, desc_bins, *,
                 approach_kt, ceiling, roll_p90, n_sorties):
    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {n_sorties} ICARTT sorties: FIREX-AQ N48RF +")
    print(f"# NOAA CSL N46RF (TopDown, UWFPS, CalFiDE, AEROMMA, AMMBEC, USOS).")
    print(f"# Per-file unit detection handles inconsistent m/s vs kt labeling.")
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
