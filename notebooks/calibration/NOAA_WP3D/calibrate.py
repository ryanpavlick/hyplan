"""NOAA WP-3D Orion calibration combining NOAA CSL chemistry + HRD hurricane data.

NOAA WP-3D ('Hurricane Hunter' / atmospheric chemistry P-3 Orion,
tails N42RF / N43RF) flies a different mission profile than NASA's
``NASA_P3`` (N426, the LaRC tail).  Same Lockheed P-3 Orion airframe,
different operators / outfits / cruise altitudes.

**Two source archives:**

1. **NOAA CSL chemistry ICARTT** — per-sortie data split across three
   ICARTT files merged on ``AOCTimewave`` UTC-seconds-past-midnight:

   * ``AircraftMet_NP3_*.ict``  — AmbTemp, WindSpd, WindDir, StaticPrs
   * ``AircraftPos_NP3_*.ict``  — GpsLat, GpsLon, GpsAlt, PAlt, RAlt
   * ``AircraftMis_NP3_*.ict``  — TrueAirSpd, GndSpd, Heading, Pitch, Roll

   Locations under ``data/WP3D/NOAA_CSL/``:

   * arcpac-2008    — 36 ICT (12 sortie-dates × 3 products)
   * calnex-2010    — 81 ICT (27 sorties × 3)
   * senex-2013     — 60 ICT (20 sorties × 3)
   * songnex-2015   — 57 ICT (19 sorties × 3)

2. **NOAA HRD hurricane field-program 1-sec text** — ``YYYYMMDDH<#>.1sec.txt``
   (N42RF "Kermit") and ``YYYYMMDDI<#>.1sec.txt`` (N43RF "Miss Piggy").
   Single fixed-width text file per sortie with TIME, Lat, Lon, Head,
   GnSpd, TAS, GeoAl, etc. native nav columns.  Loader at
   ``notebooks/calibration/_hrd_loader.py``.

   Locations under ``data/HRD/P-3_N42RF/<YYYY>/<storm>/`` and
   ``data/HRD/P-3_N43RF/<YYYY>/<storm>/`` (downloaded by
   ``notebooks.calibration._hrd_fetch``).

Run from repo root::

    python -m notebooks.calibration.NOAA_WP3D.calibrate
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


WP3D_GLOB = "data/WP3D/NOAA_CSL/*/raw/Aircraft*_NP3_*.ict"

# WP-3D Orion physical envelope: max TAS ~150 m/s (~290 kt at altitude);
# values exceeding ~250 m/s after the loader's m/s→kt conversion would
# imply a kt-mislabel.
TAS_KT_MAX_REASONABLE = 380.0  # max TAS ever, with margin

# Calibration parameters (matches existing NASA_P3 recipe)
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 720
MIN_PEAK_ALT_FT = 8000
MAX_PEAK_ALT_FT = 35000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000, 25000)


def _load_with_unit_check(path: Path) -> pd.DataFrame:
    df = load_icartt(path)
    if "tas_kt" in df.columns and df["tas_kt"].notna().any():
        if df["tas_kt"].quantile(0.95) > TAS_KT_MAX_REASONABLE:
            df["tas_kt"] = df["tas_kt"] / 1.9438444924406046
    if "wind_speed_kt" in df.columns and df["wind_speed_kt"].notna().any():
        if df["wind_speed_kt"].quantile(0.95) > 200:  # 200 kt + winds rare
            df["wind_speed_kt"] = df["wind_speed_kt"] / 1.9438444924406046
    if "groundspeed" in df.columns and df["groundspeed"].notna().any():
        if df["groundspeed"].quantile(0.95) > TAS_KT_MAX_REASONABLE:
            df["groundspeed"] = df["groundspeed"] / 1.9438444924406046
    return df


def _merge_sortie(met: Path, pos: Path, mis: Path) -> pd.DataFrame | None:
    """Load and merge a sortie's three ICARTT files.

    Returns a DataFrame with columns from all three, or None on
    inconsistent timestamps.
    """
    df_met = _load_with_unit_check(met)
    df_pos = _load_with_unit_check(pos)
    df_mis = _load_with_unit_check(mis)
    if df_met.empty or df_pos.empty or df_mis.empty:
        return None
    # Use timestamp (computed from AOCTimewave by load_icartt) as the
    # join key.  All three files share the same UTC time base.
    # Drop fully-empty backfilled columns from base before joining so
    # populated columns from Mis/Met can replace them without overlap.
    def _strip_empty(df):
        return df.loc[:, df.notna().any()]
    base = _strip_empty(df_pos).set_index("timestamp")
    out = base.copy()
    # Position file already has lat/lon/altitude (PAlt → altitude).
    # Pull Mis (TAS, GndSpd, Heading, Pitch, Roll) and Met (Wind*) too.
    for src in (df_mis, df_met):
        src = _strip_empty(src).set_index("timestamp")
        # Only add columns we don't already have populated.
        keep = [c for c in src.columns if c not in out.columns]
        if keep:
            out = out.join(src[keep], how="left")
    out = out.reset_index()
    return out


def _group_sortie_files(paths: list[Path]) -> dict[str, dict[str, Path]]:
    """Group files by (campaign, sortie_date_revision) into Met/Pos/Mis."""
    sorties: dict[str, dict[str, Path]] = {}
    for p in paths:
        # Filename: Aircraft<Met|Pos|Mis>_NP3_<date>_R<rev>.ict
        # Some have _L1/_L2 suffix; treat each (date, rev, segment) as a sortie.
        name = p.stem
        if name.startswith("AircraftMet_"):
            kind = "met"
            key = name.removeprefix("AircraftMet_")
        elif name.startswith("AircraftPos_"):
            kind = "pos"
            key = name.removeprefix("AircraftPos_")
        elif name.startswith("AircraftMis_"):
            kind = "mis"
            key = name.removeprefix("AircraftMis_")
        else:
            continue
        # Include the parent-campaign in the key so re-flown dates across
        # campaigns don't collide.
        key = f"{p.parent.parent.name}/{key}"
        sorties.setdefault(key, {})[kind] = p
    return sorties


HRD_P3_GLOB = "data/HRD/P-3_N4{2,3}RF/*/*/*.1sec.txt"


def _load_hrd_sorties() -> dict[str, pd.DataFrame]:
    """Load NOAA HRD hurricane-program 1-sec P-3 files (H + I tails)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _hrd_loader import load_p3_1sec  # noqa: E402
    out: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for tail in ("P-3_N42RF", "P-3_N43RF"):
        paths = sorted(Path(".").glob(f"data/HRD/{tail}/*/*/*.1sec.txt"))
        for p in paths:
            key = f"hrd/{tail}/{p.parent.parent.name}/{p.parent.name}/{p.stem}"
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
            out[key] = label_phases(kept)
    print(f"  HRD: {len(out)} sorties, {len(skipped)} skipped")
    return out


def load_sorties() -> dict[str, pd.DataFrame]:
    # Phase 1: NOAA CSL chemistry (existing path)
    paths = sorted(Path(".").glob(WP3D_GLOB))
    print(f"  scanning {len(paths)} CSL ICARTT files")
    grouped = _group_sortie_files(paths)
    complete = {k: v for k, v in grouped.items()
                if {"met", "pos", "mis"} <= set(v.keys())}
    print(f"  CSL: {len(complete)} complete sorties (all 3 files); "
          f"{len(grouped) - len(complete)} partial dropped")

    sorties: dict[str, pd.DataFrame] = {}
    skipped = []
    for key, files in complete.items():
        try:
            df = _merge_sortie(files["met"], files["pos"], files["mis"])
        except Exception as e:
            skipped.append((key, f"merge fail: {e}"))
            continue
        if df is None:
            skipped.append((key, "no merge"))
            continue
        if "vertical_rate" not in df.columns or df["vertical_rate"].isna().all():
            t_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
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
        kept, reason = apply_sortie_filters(
            df,
            min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((key, reason or "?"))
            continue
        sorties[f"csl/{key}"] = label_phases(kept)

    # Phase 2: HRD hurricane-program H + I files
    sorties.update(_load_hrd_sorties())

    print(f"  TOTAL: {len(sorties)} sorties (CSL + HRD), skipped {len(skipped)}")
    # Provenance: write a calibration manifest for the CSL portion.
    # (HRD load_sorties tracks its own skipped list internally; this
    # manifest covers the CSL phase + the merged kept set.)
    from _common import summary_table
    summary_table(
        sorties, skipped,
        source_label="NOAA WP-3D: NOAA CSL ICARTT + HRD AOML hurricane",
        print_it=False,
        manifest_path="data/WP3D/calibration_manifest.csv",
    )
    return sorties


def main():
    print("NOAA WP-3D calibration (ARCPAC + CalNex + SENEX + SONGNEX)")
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

    cs = schedule_pts(cruise_tas, TARGET_ALTS_FT, n_min=200)
    klms = schedule_pts(climb_tas, TARGET_ALTS_FT, n_min=200)
    ds = schedule_pts(desc_tas, TARGET_ALTS_FT, n_min=200)
    print(f"\nCruise TAS schedule: {cs}")
    print(f"Climb  TAS schedule: {klms}")
    print(f"Desc.  TAS schedule: {ds}")

    final_rows = []
    for a in sorties.values():
        sub = a[(a["altitude"] < 3000) & (a["vertical_rate"] < -200)]
        if "tas_kt" in sub.columns:
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
    print("PASTE-READY NOAA_WP3D() PERFORMANCE BLOCK")
    print("=" * 70)
    def _vs_pts(bins):
        return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
                for _, r in bins.iterrows()]
    print(f"# Calibrated against {len(sorties)} merged ICARTT sorties from")
    print("# NOAA CSL archive: ARCPAC 2008, CalNex 2010, SENEX 2013, SONGNEX 2015")
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
