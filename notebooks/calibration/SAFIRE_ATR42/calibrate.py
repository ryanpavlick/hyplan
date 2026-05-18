"""SAFIRE ATR-42 calibration combining CEDA EUFAR + AERIS EUREC4A.

Two source archives, two NetCDF schemas:

* **CEDA EUFAR** (28 flights across 7 EUFAR Transnational Access projects:
  geomad, i-wake2, icare-qad, micwa, olacta2, tetrad, walitemp).  Files
  use SAFIRE prefix-based variable names (``alt_aipov_1``,
  ``pos_lat_aipov_1``, ``att_roul_aipov_1``, ``ven_e_paipov_1``).  No
  TAS or groundspeed shipped — we derive groundspeed from position
  finite differences and TAS via wind triangle (GS minus wind vector).
  This mirrors the HIAPER calibration's wind-triangle TAS reconstruction.

* **AERIS EUREC4A** (19 flights from the EUREC4A 2020 campaign).
  L2 1 Hz files use uppercase semantic names (``ALTITUDE``, ``TAS``,
  ``GS``, ``ROLL``, ``PITCH``, ``HEADING``).  Native TAS, used directly.

Run from repo root::

    python -m notebooks.calibration.SAFIRE_ATR42.calibrate
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
    smooth_diff,
    vertical_rate_fpm
)


CEDA_GLOB = "data/ATR42/ceda-eufar/*/*/*.nc"
AERIS_GLOB = "data/ATR42/eurec4a-aeris/EUREC4A_*.nc"

# ATR-42-300/320 envelope: cruise ~280 KTAS, MMO M0.55, ceiling 25000 ft.
ACTIVE_VS_THR_FPM = 1000.0  # ATR-42 is a turboprop, lower active gate
MIN_DUR_MIN = 30
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 5000
MAX_PEAK_ALT_FT = 28000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501
DEG_LAT_TO_M = 111_320.0  # m per degree latitude






def _load_ceda(path: Path) -> pd.DataFrame | None:
    """Load CEDA EUFAR SAFIRE ATR-42 NetCDF, derive GS+TAS via wind triangle."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None

    # Pick variable names — the 2016 "core" files use *_aipov_1 (AIRINS),
    # the 2010 "as-core" files use *_ins_deg_1 / *_pinsdat_1.
    def pick(*names):
        for n in names:
            if n in ds.variables:
                return ds[n].values
        return None

    lat = pick("pos_lat_aipov_1", "pos_lat_gps_1")
    lon = pick("pos_lon_aipov_1", "pos_lon_gps_1")
    alt_m = pick("alt_aipov_1", "alt_alti_gps_1", "alt_baro_m_1")
    roll = pick("att_roul_aipov_1", "att_roul_ins_deg_1")
    pitch = pick("att_tang_aipov_1", "att_tang_ins_deg_1")
    hdg = pick("att_capgeo_aipov_1", "att_cap_ins_deg_1")
    u_wind = pick("ven_e_paipov_1", "ven_e_pinsdat_1")  # m/s
    v_wind = pick("ven_n_paipov_1", "ven_n_pinsdat_1")

    if lat is None or lon is None or alt_m is None:
        ds.close()
        return None

    t_raw = ds["time"].values  # seconds since epoch
    ds.close()

    # Reference time
    t_s = (t_raw - t_raw[0]).astype(float)
    timestamp = pd.to_datetime(t_raw, unit="s", origin="unix")

    df = pd.DataFrame({"timestamp": timestamp})
    df["altitude"] = alt_m * M_TO_FT
    if roll is not None:
        df["roll_deg"] = roll
    if pitch is not None:
        df["pitch_deg"] = pitch
    if hdg is not None:
        df["heading_deg"] = hdg

    # Groundspeed from position derivatives.  Use 5-second window for
    # smoothing (1 Hz data is noisy at single-step diff).
    cos_lat = np.cos(np.deg2rad(lat))
    gs_e = smooth_diff(t_s, lon, half_s=5.0) * DEG_LAT_TO_M * cos_lat  # m/s
    gs_n = smooth_diff(t_s, lat, half_s=5.0) * DEG_LAT_TO_M
    df["groundspeed"] = np.hypot(gs_e, gs_n) * M_PER_S_TO_KT

    # TAS via wind triangle: TAS_vec = GS_vec - wind_vec.  Wind components
    # are reported as the wind direction-from convention (eastward wind
    # = +u, northward = +v, both blowing TO that direction).
    if u_wind is not None and v_wind is not None:
        tas_e = gs_e - u_wind
        tas_n = gs_n - v_wind
        df["tas_kt"] = np.hypot(tas_e, tas_n) * M_PER_S_TO_KT
    else:
        df["tas_kt"] = df["groundspeed"]  # still-air assumption fallback

    # Vertical rate from altitude finite difference
    df["vertical_rate"] = vertical_rate_fpm(t_s, df["altitude"].to_numpy(dtype=float))

    # Drop fixes with no altitude (most aren't ground; SAFIRE files don't ship WOW)
    df = df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    return df if not df.empty else None


def _load_aeris(path: Path) -> pd.DataFrame | None:
    """Load AERIS EUREC4A L2 ATR-42 NetCDF (native TAS, simpler schema)."""
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception:
        return None

    t = ds["time"].values
    df = pd.DataFrame({"timestamp": pd.to_datetime(t)})
    t_raw = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    df["altitude"] = ds["ALTITUDE"].values * M_TO_FT
    df["tas_kt"] = ds["TAS"].values * M_PER_S_TO_KT
    if "GS" in ds.variables:
        df["groundspeed"] = ds["GS"].values * M_PER_S_TO_KT
    if "ROLL" in ds.variables:
        df["roll_deg"] = ds["ROLL"].values
    if "PITCH" in ds.variables:
        df["pitch_deg"] = ds["PITCH"].values
    if "HEADING" in ds.variables:
        df["heading_deg"] = ds["HEADING"].values
    ds.close()

    df["vertical_rate"] = vertical_rate_fpm(t_raw, df["altitude"].to_numpy(dtype=float))
    df = df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    return df if not df.empty else None


def load_sorties() -> dict[str, pd.DataFrame]:
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []

    for p in sorted(Path(".").glob(CEDA_GLOB)):
        key = f"ceda/{p.parent.parent.name}/{p.parent.name}"
        df = _load_ceda(p)
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

    for p in sorted(Path(".").glob(AERIS_GLOB)):
        key = f"aeris/{p.stem}"
        df = _load_aeris(p)
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

    summary_table(sorties, skipped, source_label="CEDA EUFAR + AERIS EUREC4A",
                   manifest_path="data/ATR42/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("SAFIRE ATR-42 calibration (CEDA EUFAR + AERIS EUREC4A)")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties loaded")

    climb_bins = per_bin(sorties, "climb", +1, ACTIVE_VS_THR_FPM, n_min=30)
    desc_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM, n_min=30)
    cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, ("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, ("descent",), n_min=200)

    print("\nCLIMB VS bins (active VS >= 1000 fpm):")
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

    # Approach speed: last 60 s of each sortie's airborne segment with VS<-200
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
        "safire_atr42",
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
