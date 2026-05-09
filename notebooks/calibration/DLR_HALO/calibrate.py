"""DLR HALO (Gulfstream G550) calibration from BAHAMAS HALO-AC3 data.

HALO is DLR's high-altitude long-range research aircraft, a Gulfstream
G550 (D-ADLR).  Same airframe family as NCAR_GV (HIAPER, N677F) but
operated independently by DLR Flugexperimente.

The HALO-AC3 campaign (March-April 2022, Arctic) shipped 18 research
flights as 10 Hz BAHAMAS NetCDF files.  BAHAMAS is HALO's basic
sensor system, with native TAS, IRS-derived groundspeed/heading/
attitude, and per-fix wind reconstruction — much cleaner than IWG1.

Files at ``data/HALO/halo-ac3-bahamas/raw/HALO-AC3_HALO_BAHAMAS_*.nc``.

Run from repo root::

    python -m notebooks.calibration.DLR_HALO.calibrate
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
from _common import (  # noqa: E402
    apply_sortie_filters,
    label_phases,
    per_bin,
    schedule_pts,
    summary_table,
    tas_per_bin,
)


HALO_GLOB = "data/HALO/halo-ac3-bahamas/raw/HALO-AC3_HALO_BAHAMAS_*.nc"

# G550 envelope: cruise M0.80 (~470 KTAS), MMO M0.885, ceiling 51000 ft.
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 720
MIN_PEAK_ALT_FT = 15000
MAX_PEAK_ALT_FT = 55000
TARGET_ALTS_FT = (0, 10000, 20000, 30000, 40000, 45000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501


def _load_one(path: Path) -> pd.DataFrame | None:
    """Load HALO BAHAMAS 10 Hz NetCDF, downsample to 1 Hz."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None

    t = ds["TIME"].values  # seconds since 1970-01-01 UTC
    df = pd.DataFrame({
        "time_s": t,
        "altitude_m": ds["H"].values,                  # geometric MSL
        "tas_ms": ds["TAS"].values,                     # native TAS
        "vv_ms": ds["IRS_VV"].values,                   # vertical velocity, +up
        "roll_deg": ds["IRS_PHI"].values,
        "pitch_deg": ds["IRS_THE"].values,
        "heading_deg": ds["IRS_HDG"].values,
        "groundspeed_ms": ds["IRS_GS"].values,
        "lat": ds["IRS_LAT"].values,
        "lon": ds["IRS_LON"].values,
    })
    ds.close()

    # Downsample to 1 Hz (10:1 mean) — keeps n>30 per bin while the
    # high-rate noise on roll/VV doesn't pollute the climb gate.
    df["time_int"] = df["time_s"].astype(int)
    df = df.groupby("time_int").mean().reset_index()
    df["timestamp"] = pd.to_datetime(df["time_int"], unit="s", origin="unix")
    df = df.drop(columns=["time_int", "time_s"])

    df["altitude"] = df["altitude_m"] * M_TO_FT
    df["tas_kt"] = df["tas_ms"] * M_PER_S_TO_KT
    df["groundspeed"] = df["groundspeed_ms"] * M_PER_S_TO_KT
    df["vertical_rate"] = df["vv_ms"] * M_TO_FT * 60.0  # m/s up → fpm
    df = df.drop(columns=["altitude_m", "tas_ms", "vv_ms", "groundspeed_ms"])
    df = df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)

    # Drop ground fixes via low-altitude + low-TAS heuristic (no WOW
    # in BAHAMAS).  Below 100 ft MSL with TAS < 50 kt = on the ground.
    if not df.empty:
        airborne = ~((df["altitude"] < 100) & (df["tas_kt"] < 50))
        df = df[airborne].reset_index(drop=True)
    return df if not df.empty else None


def load_sorties() -> dict[str, pd.DataFrame]:
    paths = sorted(Path(".").glob(HALO_GLOB))
    print(f"  scanning {len(paths)} HALO BAHAMAS files")
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        key = p.stem
        df = _load_one(p)
        if df is None:
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
    summary_table(sorties, skipped, source_label="DLR HALO-AC3 BAHAMAS",
                   manifest_path="data/HALO/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("DLR HALO (G550) calibration (HALO-AC3 March-April 2022)")
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

    print()
    print("=" * 70)
    print("PASTE-READY DLR_HALO() PERFORMANCE BLOCK")
    print("=" * 70)

    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {len(sorties)} HALO-AC3 BAHAMAS sorties")
    print(f"# (DLR, March-April 2022, Arctic).  Single-campaign dataset;")
    print(f"# confidence 0.7 reflects narrower envelope sample.")
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
