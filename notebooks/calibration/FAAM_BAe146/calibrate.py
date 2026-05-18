"""FAAM BAe-146 calibration from CEDA core_processed 1 Hz NetCDFs.

The FAAM Core Data Product (G-LUXE, BAe-146-301) ships every flight as
``core_faam_<YYYYMMDD>_v###_r#_<flight>_1hz.nc`` with GPS-aided INS nav
variables (LAT_GIN, ALT_GIN, ROLL_GIN, PTCH_GIN, VELD_GIN), RVSM air
data (TAS_RVSM, IAS_RVSM, PALT_RVS), and a weight-on-wheels flag
(WOW_IND).  This is much cleaner than the IWG1/ICARTT inputs we've
calibrated against — every variable has units, every sortie carries
WOW for a perfect on-ground filter.

Files at ``data/FAAM/<YYYY>/<flight_dir>/core_faam_*_1hz.nc``.

Run from repo root::

    python -m notebooks.calibration.FAAM_BAe146.calibrate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (
    apply_sortie_filters,
    label_phases,
    per_bin,
    schedule_pts,
    summary_table,
    tas_per_bin,
)


FAAM_GLOB = "data/FAAM/*/*/core_faam_*_1hz.nc"

# BAe-146-301 envelope: cruise ~360 KTAS, MMO M0.7, ceiling 35000 ft.
# Active-VS gate at 1500 fpm (typical for jets/turboprops).
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 8000
MAX_PEAK_ALT_FT = 35000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000, 25000, 30000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501


def _load_one(path: Path) -> pd.DataFrame | None:
    """Load one FAAM 1 Hz core NetCDF, return canonical DataFrame.

    Filters to airborne fixes (WOW_IND == 0).  Altitude is GIN
    geometric where available, otherwise RVSM pressure altitude.
    """
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception:
        return None

    df = pd.DataFrame({"timestamp": ds["Time"].values})
    # Altitude: GIN geometric preferred, fall back to PALT_RVS pressure alt.
    alt_m = ds["ALT_GIN"].values if "ALT_GIN" in ds.variables else None
    if alt_m is None or np.isnan(alt_m).all():
        alt_m = ds["PALT_RVS"].values
    df["altitude"] = alt_m * M_TO_FT  # ft
    # TAS in m/s → kt
    df["tas_kt"] = ds["TAS_RVSM"].values * M_PER_S_TO_KT
    if "GSPD_GIN" in ds.variables:
        df["groundspeed"] = ds["GSPD_GIN"].values * M_PER_S_TO_KT
    # Roll, pitch, heading
    if "ROLL_GIN" in ds.variables:
        df["roll_deg"] = ds["ROLL_GIN"].values
    if "PTCH_GIN" in ds.variables:
        df["pitch_deg"] = ds["PTCH_GIN"].values
    if "HDG_GIN" in ds.variables:
        df["heading_deg"] = ds["HDG_GIN"].values
    # Vertical rate from VELD_GIN (NED down velocity in m/s) → fpm.
    # VELD positive = descent, so flip sign.
    if "VELD_GIN" in ds.variables:
        veld = ds["VELD_GIN"].values
        df["vertical_rate"] = -veld * M_TO_FT * 60.0  # fpm
    else:
        df["vertical_rate"] = np.nan
    # Radar altimeter (AGL, m → ft) — the cleanest gate for final approach.
    if "HGT_RADR" in ds.variables:
        df["radar_alt_ft"] = ds["HGT_RADR"].values * M_TO_FT
    # Weight-on-wheels — keep airborne fixes only.
    if "WOW_IND" in ds.variables:
        wow = ds["WOW_IND"].values
        df = df[wow == 0]
    ds.close()
    df = df.dropna(subset=["altitude", "vertical_rate"])
    if df.empty:
        return None
    # Re-derive vertical_rate via 180s centered diff so the gate is
    # consistent with what _common.label_phases expects (the raw VELD is
    # noisy; the smoothed version matches the IWG1 calibrations).
    t = df["timestamp"].to_numpy()
    t_s = (t - t[0]) / np.timedelta64(1, "s")
    alt_ft = df["altitude"].to_numpy(dtype=float)
    half = 90.0
    lo = np.searchsorted(t_s, t_s - half, side="left")
    hi = np.searchsorted(t_s, t_s + half, side="right") - 1
    lo = np.clip(lo, 0, len(t_s) - 1)
    hi = np.clip(hi, 0, len(t_s) - 1)
    dt = t_s[hi] - t_s[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        vs = np.where(dt > 0, (alt_ft[hi] - alt_ft[lo]) / dt * 60.0, np.nan)
    df["vertical_rate"] = vs
    return df.reset_index(drop=True)


def load_sorties() -> dict[str, pd.DataFrame]:
    paths = sorted(Path(".").glob(FAAM_GLOB))
    print(f"  scanning {len(paths)} FAAM 1 Hz files")
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        key = f"{p.parent.parent.name}/{p.parent.name}"
        df = _load_one(p)
        if df is None or df.empty:
            skipped.append((key, "empty after WOW filter"))
            continue
        kept, reason = apply_sortie_filters(
            df,
            min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((key, reason or "?"))
            continue
        sorties[key] = label_phases(kept)
    summary_table(sorties, skipped, source_label="CEDA FAAM core_processed 1Hz",
                   manifest_path="data/FAAM/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("FAAM BAe-146 calibration (CEDA G-LUXE 2017–2024 ASMM-tagged)")
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

    # Approach speed: median TAS in the final 60 s of each sortie's
    # airborne segment.  WOW filtering during load means the last
    # airborne fix sits just before touchdown; the final 60 s captures
    # the stabilized landing approach.  An MSL/AGL filter doesn't work
    # because FAAM flies low-altitude science surveys at cruise speed.
    final_rows = []
    for a in sorties.values():
        if "tas_kt" not in a.columns or a.empty:
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

    from notebooks.calibration._common import apply_calibration_to_profile

    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]

    print()
    print("=" * 70)
    path = apply_calibration_to_profile(
        "faam_bae146",
        service_ceiling_ft=int(round(ceiling / 100) * 100),
        approach_speed_kt=int(round(approach_kt)),
        climb_pts=klms,
        cruise_pts=cs,
        descent_pts=ds,
        climb_profile_pts=_vs_pts(climb_bins),
        descent_profile_pts=_vs_pts(desc_bins),
        max_bank_deg=max(30.0, round(roll_p90)),
    )
    print(f"Wrote calibrated profile to {path}")
    print(f"  fit n_sorties={len(sorties)}")


if __name__ == "__main__":
    main()
