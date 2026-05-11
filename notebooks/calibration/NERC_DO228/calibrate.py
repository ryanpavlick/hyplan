"""NERC ARSF Dornier Do228-101 (D-CALM) calibration.

Two CEDA archives:

* **active-core** (ACTIVE 2005-2006 Aerosol Coupling in the Earth
  System): ``active-package_arsf-dornier_*.nc``.  TIME-indexed NetCDF
  with LATITUDE, LONGITUDE, ALTITUDE, U/V wind, plus aerosol products.
  No native TAS / roll / pitch — TAS reconstructed via wind triangle
  from position derivatives + wind components.

* **eyjafjallajokull-arsf** (Eyjafjallajökull volcanic ash 2010
  ARSF flights): ``arsf_uk_<YYYYMMDD>_r<n>_1Hz.csv``.  Plain CSV with
  ``sec, lon, lat, alt, u, v, w, air temp, RH, pressure, ...``.
  Same wind-triangle approach.

NetCDF file inventory (44 NetCDFs total):
  - active-core:           27 sorties, every file has nav + winds
  - eyjafjallajokull-arsf: 17 NetCDFs are payload (PCASP cloud-phy);
                          the **CSVs** (17 across 5 sortie dates) are
                          where the nav lives.

Run from repo root::

    python -m notebooks.calibration.NERC_DO228.calibrate
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
    vertical_rate_fpm,
    wind_triangle_tas_kt,
)


ACTIVE_GLOB = "data/DO228/NERC_DCALM/active-core/active-package_arsf-dornier_*.nc"
EYJAF_GLOB = "data/DO228/NERC_DCALM/eyjafjallajokull-arsf/arsf_uk_*_1Hz.csv"

# Do228-101 envelope: cruise ~165 KTAS, ceiling 22000 ft (NERC operational).
ACTIVE_VS_THR_FPM = 700.0
MIN_DUR_MIN = 30
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 3000
MAX_PEAK_ALT_FT = 28000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501
DEG_LAT_TO_M = 111_320.0




def _build_df(t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray,
              alt_m: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray,
              start_time: pd.Timestamp) -> pd.DataFrame:
    """Common assembly: compute groundspeed, TAS via wind triangle, VS."""
    tas_kt, gs_kt = wind_triangle_tas_kt(t_s, lat, lon, u_wind, v_wind)
    alt_ft = alt_m * M_TO_FT
    df = pd.DataFrame({
        "timestamp": start_time + pd.to_timedelta(t_s, unit="s"),
        "altitude": alt_ft,
        "tas_kt": tas_kt,
        "groundspeed": gs_kt,
        "vertical_rate": vertical_rate_fpm(t_s, alt_ft),
    })
    return df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)


def _load_active(path: Path) -> pd.DataFrame | None:
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None
    secs = ds["SECONDS"].values
    lat = ds["LATITUDE"].values
    lon = ds["LONGITUDE"].values
    alt_m = ds["ALTITUDE"].values
    u = ds["U"].values
    v = ds["V"].values
    ds.close()
    # Sortie date from filename: active-package_arsf-dornier_<YYYYMMDDHHMMSS>_<...>.nc
    stem = path.stem
    parts = stem.split("_")
    date_str = next((p for p in parts if p.isdigit() and len(p) >= 8), None)
    start = pd.Timestamp(date_str[:8]) if date_str else pd.Timestamp("2005-01-01")
    df = _build_df(secs.astype(float), lat, lon, alt_m, u, v, start)
    return df if not df.empty else None


def _load_eyjaf_csv(path: Path) -> pd.DataFrame | None:
    try:
        df_raw = pd.read_csv(path, na_values=["NaN", "nan"])
    except Exception:
        return None
    df_raw.columns = [c.strip() for c in df_raw.columns]
    cmap = {c: c.lower().split("(")[0].strip() for c in df_raw.columns}
    df_raw = df_raw.rename(columns=cmap)
    needed = {"sec past midnight", "lon", "lat", "alt", "u", "v"}
    if not needed.issubset(df_raw.columns):
        return None
    df_raw = df_raw.dropna(subset=["sec past midnight", "lon", "lat", "alt"])
    if df_raw.empty:
        return None
    secs = df_raw["sec past midnight"].to_numpy(dtype=float)
    # Sortie date from filename: arsf_uk_<YYYYMMDD>_r<n>_1Hz[_raw].csv
    parts = path.stem.split("_")
    date_str = next((p for p in parts if p.isdigit() and len(p) == 8), None)
    start = pd.Timestamp(date_str) if date_str else pd.Timestamp("2010-01-01")
    df = _build_df(secs, df_raw["lat"].to_numpy(dtype=float),
                   df_raw["lon"].to_numpy(dtype=float),
                   df_raw["alt"].to_numpy(dtype=float),
                   df_raw["u"].to_numpy(dtype=float),
                   df_raw["v"].to_numpy(dtype=float), start)
    return df if not df.empty else None


def load_sorties() -> dict[str, pd.DataFrame]:
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []

    for p in sorted(Path(".").glob(ACTIVE_GLOB)):
        key = f"active/{p.stem}"
        df = _load_active(p)
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

    # Eyjaf CSVs: prefer the highest-revision processed file per date
    # (skip "_raw" variants, prefer r2 > r1 > r0).
    eyjaf_files = sorted(Path(".").glob(EYJAF_GLOB))
    eyjaf_files = [p for p in eyjaf_files if "_raw" not in p.stem]
    by_date: dict[str, Path] = {}
    for p in eyjaf_files:
        parts = p.stem.split("_")
        date = next((q for q in parts if q.isdigit() and len(q) == 8), None)
        if date and (date not in by_date or p.stem > by_date[date].stem):
            by_date[date] = p
    for _date, p in sorted(by_date.items()):
        key = f"eyjaf/{p.stem}"
        df = _load_eyjaf_csv(p)
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

    summary_table(sorties, skipped, source_label="ACTIVE + Eyjafjallajokull",
                   manifest_path="data/DO228/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("NERC DO228 D-CALM calibration (ACTIVE 2005-2006 + Eyjafjallajökull 2010)")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties loaded")

    climb_bins = per_bin(sorties, "climb", +1, ACTIVE_VS_THR_FPM, n_min=30)
    desc_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM, n_min=30)
    cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, ("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, ("descent",), n_min=200)

    print("\nCLIMB VS bins (active VS >= 700 fpm):")
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

    print(f"\nApproach TAS median: {approach_kt:.0f} kt")
    print(f"Service ceiling (op-p99): {ceiling:.0f} ft")
    print("Bank angle: not available; using AFM default 30°")

    print()
    print("=" * 70)
    print("PASTE-READY NERC_DO228() PERFORMANCE BLOCK")
    print("=" * 70)

    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {len(sorties)} sorties (ACTIVE + Eyjafjallajökull).")
    print("# TAS reconstructed via wind triangle (no native TAS in either archive).")
    print(f"service_ceiling={int(round(ceiling/100)*100)} * ureg.feet,")
    print(f"approach_speed={int(round(approach_kt))} * ureg.knot,")
    print(f"climb_schedule=TasSchedule(points={klms!r}),")
    print(f"cruise_schedule=TasSchedule(points={cs!r}),")
    print(f"descent_schedule=TasSchedule(points={ds!r}),")
    print(f"climb_profile=VerticalProfile(points={_vs_pts(climb_bins)!r}),")
    print(f"descent_profile=VerticalProfile(points={_vs_pts(desc_bins)!r}),")


if __name__ == "__main__":
    main()
