"""KingAirB200 calibration from multi-campaign ICARTT files.

Beechcraft King Air B-200 (NASA Wallops / LaRC tails N801NA, N529NA,
N526NA, etc.).  Calibrated from six ICARTT campaign archives under
``data/KingAirB200/``: ACT-AMERICA housekeeping, three DISCOVER-AQ
deployments (California, Colorado, Texas), KORUS-AQ NAV, and LMOS
NAV.

Active-VS threshold is 1000 fpm — lower than the 1500 fpm used for
jets, since B-200 climb rates drop below 1500 fpm above FL150 and a
1500 fpm gate would lose the upper-altitude bins.

Run from repo root::

    python -m notebooks.calibration.KingAirB200.calibrate
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

from hyplan.aircraft.icartt import load_icartt  # noqa: E402
from hyplan.aircraft.iwg1 import trim_ground_taxi  # noqa: E402


# ---- campaign dirs --------------------------------------------------
CAMPAIGN_GLOBS = [
    "data/KingAirB200/ACTAMERICA_B200_Hskping/*.ict",
    "data/KingAirB200/DISCOVERAQ_California_B200_APPLANIX/*.ict",
    "data/KingAirB200/DISCOVERAQ_Colorado_B200_APPLANIX/*.ict",
    "data/KingAirB200/DISCOVERAQ_Texas_B200_APPLANIX/*.ict",
    "data/KingAirB200/KORUSAQ_B200_NAV/*.ict",
    "data/KingAirB200/LMOS_UC12_NAV/*.ict",
]

# ---- calibration parameters ----------------------------------------
ACTIVE_VS_THR_FPM = 1000.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 8000   # boundary-layer campaigns peak around FL080
MAX_PEAK_ALT_FT = 35000  # B-200 brochure ceiling
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000, 25000)

CLIMB_TARGET_ALTS_FT = (5000, 10000, 15000, 20000, 25000)
CRUISE_TARGET_ALTS_FT = (10000, 15000, 20000, 25000, 28000)
DESCENT_TARGET_ALTS_FT = (5000, 10000, 15000, 20000, 25000)

ROTATION_TAS_KT = 110
APPROACH_ANCHOR_TAS_KT = 130

CEILING_FT = 35000
CEILING_VS_FPM = 500.0


def _trim_or_altitude_fallback(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Trim ground taxi via groundspeed if available, else altitude."""
    if "groundspeed" in raw.columns and raw["groundspeed"].notna().sum() > 100:
        return trim_ground_taxi(raw)
    alt = raw["altitude"]
    if alt.dropna().empty:
        return None
    airborne = (alt - alt.min()) > 200
    if not airborne.any():
        return None
    first = int(airborne.values.argmax())
    last = int(len(airborne) - 1 - airborne.values[::-1].argmax())
    return raw.iloc[first:last + 1].reset_index(drop=True)


def load_sorties() -> dict[str, pd.DataFrame]:
    paths: list[Path] = []
    for g in CAMPAIGN_GLOBS:
        paths.extend(sorted(Path(".").glob(g)))
    print(f"  scanning {len(paths)} ICARTT files across "
          f"{len(CAMPAIGN_GLOBS)} campaigns")

    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        try:
            raw = load_icartt(p)
        except Exception as e:
            skipped.append((p.stem, f"load failed ({type(e).__name__})"))
            continue
        a = _trim_or_altitude_fallback(raw)
        if a is None or a.empty:
            skipped.append((p.stem, "no airborne fixes"))
            continue
        kept, reason = apply_sortie_filters(
            a,
            min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((p.stem, reason or "?"))
            continue
        # Use parent dir as campaign tag in the key to avoid collisions.
        key = f"{p.parent.name}/{p.stem}"
        sorties[key] = label_phases(kept)

    summary_table(
        sorties, skipped,
        source_label="KingAirB200: ACT-AMERICA + DISCOVER-AQ + KORUS-AQ + LMOS ICARTT",
        print_it=False,
        manifest_path="data/KingAirB200/calibration_manifest.csv",
    )
    print(f"  loaded {len(sorties)} sorties, skipped {len(skipped)}")
    return sorties


def main() -> None:
    print("KingAirB200 calibration (multi-campaign ICARTT)")
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

    print("\nCLIMB VS bins (active VS >= 1000 fpm):")
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

    from notebooks.calibration._common import apply_calibration_to_profile

    print()
    print("=" * 70)
    path = apply_calibration_to_profile(
        "king_air_b200",
        service_ceiling_ft=int(round(ceiling / 1000) * 1000),
        approach_speed_kt=int(round(approach_kt)),
        climb_pts=climb_pts,
        cruise_pts=cruise_pts,
        descent_pts=descent_pts,
        climb_profile_pts=climb_profile_pts,
        descent_profile_pts=descent_profile_pts,
        max_bank_deg=30.0,
    )
    print(f"Wrote calibrated profile to {path}")
    print(f"  fit n_sorties={len(sorties)}")


if __name__ == "__main__":
    main()
