"""BAS Twin Otter calibration from CEDA MASIN data.

The British Antarctic Survey operates DHC-6-300 Twin Otters with the
MASIN (Met Airborne Science Instrumentation) payload.  Same airframe
class as NASA / NOAA Twin Otters but the polar/Antarctic operating
profile (sustained low altitude survey, cold-soak, high winds) is
distinct enough to warrant its own calibration.

Two CEDA archives, two file schemas:

* **ofcap** (Orographic Flows and Clouds in Sub-Antarctic Falklands,
  2010-2011, 23 flights).  ``bas-core_masin_*_1hz.nc`` with native
  TAS, ROLL_JAVAD/PTCH_JAVAD, VELZ_JAVAD vertical velocity — full nav.
* **arcticcyclones** (Arctic Summer-time Cyclones, 2022, 15 flights).
  ``core_masin_*_asc-qc.nc`` ships QC scientific subset only — no
  TAS, no roll/pitch.  Reconstruct groundspeed from lat/lon, VS from
  gps_alt, TAS via wind-triangle from 50 Hz wind components.

Run from repo root::

    python -m notebooks.calibration.BAS_TwinOtter.calibrate
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
    smooth_diff,
    summary_table,
    tas_per_bin,
    wind_triangle_tas_kt,
)


# Two file flavors across five CEDA archives:
#  * 1 Hz native-nav: ofcap, accacia (bas-core_masin_*_1hz.nc),
#    orchestra (core_masin_*_1hz.nc).  TAS shipped natively.
#  * High-rate QC: arcticcyclones (asc-qc), igp (igp-qc).  Lacks TAS;
#    derive via wind triangle from 50 Hz lat/lon/wind.
ONEHZ_GLOB = "data/BAS_TwinOtter/*/*/*core_masin_*_1hz.nc"
QC_GLOB = "data/BAS_TwinOtter/*/*/core_masin_*-qc.nc"

# DHC-6-300 envelope: cruise ~150 KTAS, ceiling 25000 ft.  Active-VS
# threshold matches the existing NOAA_TwinOtter calibration.
ACTIVE_VS_THR_FPM = 500.0
MIN_DUR_MIN = 30
MAX_DUR_MIN = 600
MIN_PEAK_ALT_FT = 1000  # polar surveys often stay low
MAX_PEAK_ALT_FT = 28000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501
DEG_LAT_TO_M = 111_320.0




def _load_ofcap(path: Path) -> pd.DataFrame | None:
    """Load OFCAP 1 Hz file — native TAS / roll / pitch."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None
    t_raw = ds["Time"].values
    units_str = ds["Time"].attrs["units"]
    origin_str = units_str.split("since", 1)[1].strip().replace(" UTC", "").strip()
    origin = pd.Timestamp(origin_str)
    if origin.tz is not None:
        origin = origin.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame({"timestamp": pd.to_datetime(t_raw, unit="s", origin=origin)})
    # Variable name suffix varies: _JAVAD (OFCAP 2010-2011) vs _OXTS
    # (ACCACIA 2013, ORCHESTRA 2017-2018).  Pick whichever is present.
    suffix = "_OXTS" if "ALT_OXTS" in ds.variables else "_JAVAD"
    df["altitude"] = ds[f"ALT{suffix}"].values * M_TO_FT
    df["tas_kt"] = ds["TAS"].values * M_PER_S_TO_KT
    # ROLL / PTCH / HDG / VELZ often ship as all-NaN; only attach if they
    # carry data.
    for src, dst in [(f"ROLL{suffix}", "roll_deg"),
                     (f"PTCH{suffix}", "pitch_deg"),
                     (f"HDG{suffix}", "heading_deg")]:
        if src in ds.variables and not np.isnan(ds[src].values).all():
            df[dst] = ds[src].values
    velz = f"VELZ{suffix}"
    if velz in ds.variables and not np.isnan(ds[velz].values).all():
        df["vertical_rate"] = ds[velz].values * M_TO_FT * 60.0
    else:
        # Derive VS from altitude finite difference (180s centered window).
        t_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        df["vertical_rate"] = smooth_diff(t_s, df["altitude"].to_numpy(dtype=float),
                                            half_s=90.0) * 60.0
    if "HGT_RADR1" in ds.variables:
        df["radar_alt_ft"] = ds["HGT_RADR1"].values * M_TO_FT
    ds.close()
    df = df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    return df if not df.empty else None


def _load_arcticcyclones(path: Path) -> pd.DataFrame | None:
    """Load ArcticCyclones asc-qc — reconstruct TAS / VS from 50 Hz nav."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None
    # 50 Hz nav: lat, lon, gps_alt, heading, u/v wind components.
    if "posixtime_50hz" not in ds.variables:
        ds.close()
        return None
    t = ds["posixtime_50hz"].values
    lat = ds["lat_50hz"].values
    lon = ds["lon_50hz"].values
    alt_m = ds["gps_alt_50hz"].values
    u_wind = ds["u_50hz"].values  # eastward wind (m/s)
    v_wind = ds["v_50hz"].values  # northward wind (m/s)
    hdg = ds["heading_inu_50hz"].values
    ds.close()

    t_s = (t - t[0]).astype(float)

    # TAS / groundspeed via wind triangle from 50 Hz position + wind.
    tas_kt, gs_kt = wind_triangle_tas_kt(t_s, lat, lon, u_wind, v_wind, half_s=2.0)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(t, unit="s", origin="unix"),
        "altitude": alt_m * M_TO_FT,
        "tas_kt": tas_kt,
        "groundspeed": gs_kt,
        "heading_deg": hdg,
    })
    # Resample to 1 Hz mean (50:1)
    df["t_int"] = (t).astype(int)
    df = df.groupby("t_int").mean(numeric_only=True)
    df["timestamp"] = pd.to_datetime(df.index, unit="s", origin="unix")
    df = df.reset_index(drop=True)
    # Vertical rate from altitude
    t_smooth = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    df["vertical_rate"] = smooth_diff(t_smooth, df["altitude"].to_numpy(dtype=float), half_s=90.0) * 60.0
    df = df.dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    return df if not df.empty else None


def load_sorties() -> dict[str, pd.DataFrame]:
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in sorted(Path(".").glob(ONEHZ_GLOB)):
        key = f"{p.parent.parent.name}/{p.parent.name}"
        df = _load_ofcap(p)
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
    for p in sorted(Path(".").glob(QC_GLOB)):
        key = f"{p.parent.parent.name}/{p.parent.name}"
        df = _load_arcticcyclones(p)
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
    summary_table(sorties, skipped, source_label="OFCAP + ACCACIA + ARCTICCYCLONES + IGP + ORCHESTRA",
                   manifest_path="data/BAS_TwinOtter/calibration_manifest.csv")
    return sorties


def main() -> None:
    print("BAS Twin Otter calibration (5 CEDA archives 2010-2022)")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties loaded")

    climb_bins = per_bin(sorties, "climb", +1, ACTIVE_VS_THR_FPM, n_min=30)
    desc_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM, n_min=30)
    cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, ("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, ("descent",), n_min=200)

    print("\nCLIMB VS bins (active VS >= 500 fpm):")
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
    if rolls:
        roll_p90 = float(pd.concat(rolls).quantile(0.90))
    else:
        roll_p90 = float("nan")

    print(f"\nApproach TAS median: {approach_kt:.0f} kt")
    print(f"Service ceiling (op-p99): {ceiling:.0f} ft")
    if rolls:
        print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}°")
    else:
        print("Bank angle: not available in ArcticCyclones; using AFM default 30°")

    print()
    print("=" * 70)
    print("PASTE-READY BAS_TwinOtter() PERFORMANCE BLOCK")
    print("=" * 70)

    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {len(sorties)} sorties (OFCAP 2010-11 + ArcticCyclones 2022).")
    print("# OFCAP files ship native TAS/roll/VS; ArcticCyclones asc-qc reconstructs")
    print("# TAS via wind triangle and VS from gps_alt finite difference.")
    print(f"service_ceiling={int(round(ceiling/100)*100)} * ureg.feet,")
    print(f"approach_speed={int(round(approach_kt))} * ureg.knot,")
    print(f"climb_schedule=TasSchedule(points={klms!r}),")
    print(f"cruise_schedule=TasSchedule(points={cs!r}),")
    print(f"descent_schedule=TasSchedule(points={ds!r}),")
    print(f"climb_profile=VerticalProfile(points={_vs_pts(climb_bins)!r}),")
    print(f"descent_profile=VerticalProfile(points={_vs_pts(desc_bins)!r}),")
    print(f"max_bank_deg={max(30.0, round(roll_p90)) if not np.isnan(roll_p90) else 30.0},")


if __name__ == "__main__":
    main()
