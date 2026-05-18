"""AWI Polar 5 / Polar 6 Basler BT-67 calibration from PANGAEA data.

Two public PANGAEA wind/temperature products are used:

* **ACLOUD 2017 1 Hz** — per-flight ``.asc.gz`` files for Polar 5 and
  Polar 6, DOI ``10.1594/PANGAEA.902849``.
* **HALO-AC3 2022 wind/temperature** — per-flight PANGAEA text files for
  Polar 5 and Polar 6, DOI ``10.1594/PANGAEA.968911``.

Both products ship native TAS, groundspeed, pitch, roll, pressure,
temperature, and U/V/W wind components.  Vertical rate is derived from
1 Hz altitude.  The public files store altitude at whole-metre precision,
so the calibration uses a central-difference smoother before binning.

Run from repo root::

    python -m notebooks.calibration.AWI_BaslerBT67.calibrate
"""
from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

ROOT = Path("data/BT67/AWI_Polar")
ACLOUD_MANIFEST = ROOT / "acloud-2017-1hz" / "manifest.csv"
HALO_AC3_MANIFEST = ROOT / "halo-ac3-2022-wind-temp" / "manifest.csv"

# Basler BT-67 polar operations: usually low / mid troposphere, with sparse
# public-data coverage above FL150 even though the aircraft envelope is higher.
ACTIVE_VS_THR_FPM = 500.0
MIN_DUR_MIN = 30
MAX_DUR_MIN = 720
MIN_PEAK_ALT_FT = 3000
MAX_PEAK_ALT_FT = 28000
TARGET_ALTS_FT = (0, 5000, 10000, 15000, 20000)

M_PER_S_TO_KT = 1.9438444924406046
M_TO_FT = 3.28083989501


def _smooth_diff(t_s: np.ndarray, x: np.ndarray, half_s: float = 30.0) -> np.ndarray:
    """Central-difference derivative over a +/- ``half_s`` window."""
    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    lo = np.searchsorted(t_s, t_s - half_s, side="left")
    hi = np.searchsorted(t_s, t_s + half_s, side="right") - 1
    lo = np.clip(lo, 0, len(t_s) - 1)
    hi = np.clip(hi, 0, len(t_s) - 1)
    dt = t_s[hi] - t_s[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dt > 0, (x[hi] - x[lo]) / dt, np.nan)


def _date_from_acloud_name(path: Path) -> pd.Timestamp:
    match = re.search(r"Flight_(20\d{6})_", path.name)
    return pd.Timestamp(match.group(1)) if match else pd.Timestamp("2017-01-01")


def _date_from_halo_flight(flight: str) -> pd.Timestamp:
    # Flight identifiers look like P5_232_HALO_2022_2203170201.
    match = re.search(r"_22(\d{2})(\d{2})(\d{2})\d{2}$", flight)
    if match:
        return pd.Timestamp(f"2022-{match.group(1)}-{match.group(2)}")
    return pd.Timestamp("2022-01-01")


def _read_acloud(path: Path) -> pd.DataFrame:
    """Read ACLOUD whitespace-delimited 1 Hz ``.asc.gz`` file."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="!",
        names=[
            "utc",
            "altitude_m",
            "lon",
            "lat",
            "pressure_hpa",
            "groundspeed_ms",
            "pitch_deg",
            "roll_deg",
            "rh_pct",
            "temp_c",
            "u_ms",
            "v_ms",
            "tas_ms",
        ],
        compression="gzip",
    )


def _read_halo_ac3(path: Path) -> pd.DataFrame:
    """Read PANGAEA HALO-AC3 tab-delimited child dataset."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Time sec"))
    df = pd.read_csv(path, sep="\t", skiprows=header_idx)
    column_map = {}
    for col in df.columns:
        key = col.lower()
        if key.startswith("time sec"):
            column_map[col] = "utc"
        elif key.startswith("altitude"):
            column_map[col] = "altitude_m"
        elif key.startswith("longitude"):
            column_map[col] = "lon"
        elif key.startswith("latitude"):
            column_map[col] = "lat"
        elif key.startswith("ground speed"):
            column_map[col] = "groundspeed_ms"
        elif key.startswith("pitch"):
            column_map[col] = "pitch_deg"
        elif key.startswith("roll"):
            column_map[col] = "roll_deg"
        elif key.startswith("u "):
            column_map[col] = "u_ms"
        elif key.startswith("v "):
            column_map[col] = "v_ms"
        elif key.startswith("w "):
            column_map[col] = "w_ms"
        elif key.startswith("tas"):
            column_map[col] = "tas_ms"
        elif key.startswith("pppp"):
            column_map[col] = "pressure_hpa"
        elif key.startswith("ttt"):
            column_map[col] = "temp_c"
    return df.rename(columns=column_map)


def _standardize(
    df: pd.DataFrame,
    *,
    sortie_key: str,
    aircraft: str,
    source: str,
    date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Convert one source file into the shared calibration schema."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    needed = ["utc", "altitude_m", "lat", "lon", "groundspeed_ms", "tas_ms"]
    df = df.dropna(subset=needed).sort_values("utc").drop_duplicates("utc")
    if df.empty:
        return None

    out = pd.DataFrame({
        "timestamp": date + pd.to_timedelta(df["utc"], unit="s"),
        "altitude": df["altitude_m"] * M_TO_FT,
        "tas_kt": df["tas_ms"] * M_PER_S_TO_KT,
        "groundspeed": df["groundspeed_ms"] * M_PER_S_TO_KT,
        "vertical_rate": _smooth_diff(
            df["utc"].to_numpy(),
            (df["altitude_m"] * M_TO_FT).to_numpy(),
            half_s=30.0,
        ) * 60.0,
        "lat": df["lat"],
        "lon": df["lon"],
        "source": source,
        "aircraft": aircraft,
        "sortie_key": sortie_key,
    })
    if "roll_deg" in df.columns:
        out["roll_deg"] = df["roll_deg"]
    if "pitch_deg" in df.columns:
        out["pitch_deg"] = df["pitch_deg"]

    airborne = (
        (out["tas_kt"] > 45)
        & (out["groundspeed"] > 35)
        & out["lat"].between(-90, 90)
        & out["lon"].between(-180, 180)
        & out["altitude"].between(-1000, 40000)
    )
    out = out[airborne].dropna(subset=["altitude", "vertical_rate"]).reset_index(drop=True)
    return out if not out.empty else None


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    for row in csv.DictReader(path.open()):
        if str(row.get("status", "")).startswith(("downloaded", "exists")):
            rows.append(row)
    return rows


def load_sorties() -> dict[str, pd.DataFrame]:
    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []

    for row in _manifest_rows(ACLOUD_MANIFEST):
        path = ACLOUD_MANIFEST.parent / row["filename"]
        key = f"acloud/{path.stem}"
        try:
            raw = _read_acloud(path)
            df = _standardize(
                raw,
                sortie_key=key,
                aircraft=row.get("aircraft", ""),
                source="ACLOUD 2017",
                date=_date_from_acloud_name(path),
            )
        except Exception as exc:
            skipped.append((key, f"load failed: {exc}"))
            continue
        if df is None:
            skipped.append((key, "no valid fixes"))
            continue
        kept, reason = apply_sortie_filters(
            df,
            min_dur_min=MIN_DUR_MIN,
            max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT,
            max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((key, reason or "?"))
            continue
        sorties[key] = label_phases(kept)

    for row in _manifest_rows(HALO_AC3_MANIFEST):
        path = HALO_AC3_MANIFEST.parent / row["filename"]
        flight = row.get("flight", path.stem)
        key = f"halo-ac3/{flight}"
        try:
            raw = _read_halo_ac3(path)
            df = _standardize(
                raw,
                sortie_key=key,
                aircraft=row.get("aircraft", ""),
                source="HALO-AC3 2022",
                date=_date_from_halo_flight(flight),
            )
        except Exception as exc:
            skipped.append((key, f"load failed: {exc}"))
            continue
        if df is None:
            skipped.append((key, "no valid fixes"))
            continue
        kept, reason = apply_sortie_filters(
            df,
            min_dur_min=MIN_DUR_MIN,
            max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT,
            max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((key, reason or "?"))
            continue
        sorties[key] = label_phases(kept)

    summary_table(sorties, skipped, source_label="AWI Polar 5/6 PANGAEA")
    if sorties:
        detail = pd.DataFrame([
            {
                "source": df["source"].iloc[0],
                "aircraft": df["aircraft"].iloc[0],
                "n_fixes": len(df),
                "peak_alt_ft": float(df["altitude"].max()),
            }
            for df in sorties.values()
        ])
        print("\nvalid sorties by source / aircraft:")
        print(
            detail.groupby(["source", "aircraft"])
            .agg(sorties=("n_fixes", "count"), fixes=("n_fixes", "sum"), max_alt_ft=("peak_alt_ft", "max"))
            .round(0)
            .to_string()
        )
    return sorties


def _vs_points(
    bins: pd.DataFrame,
    residual: tuple[int, int] | None = None,
    *,
    absolute: bool = False,
) -> list[tuple[int, int]]:
    values = bins["vs_med"].abs() if absolute else bins["vs_med"]
    points = [
        (int(alt), int(round(float(vs))))
        for alt, vs in zip(bins["alt_bin_ft"], values, strict=True)
    ]
    if residual is not None and (not points or points[-1][0] < residual[0]):
        points.append(residual)
    return points


def main() -> None:
    print("AWI Polar 5 / Polar 6 Basler BT-67 calibration")
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
    for sortie in sorties.values():
        t_end = sortie["timestamp"].iloc[-1]
        sub = sortie[sortie["timestamp"] >= t_end - pd.Timedelta(seconds=90)]
        sub = sub[sub["groundspeed"] > 35]
        final_rows.append(sub[["tas_kt"]])
    final = pd.concat(final_rows).dropna() if final_rows else pd.DataFrame()
    approach_kt = float(final["tas_kt"].median()) if len(final) else float("nan")

    peaks = [float(a["altitude"].max()) for a in sorties.values()]
    ceiling_obs = float(np.percentile(peaks, 99))

    rolls = []
    for sortie in sorties.values():
        if "roll_deg" in sortie.columns:
            r = sortie["roll_deg"].abs()
            rolls.append(r[(r > 5.0) & (r < 60.0)])
    roll_p90 = float(pd.concat(rolls).quantile(0.90)) if rolls else float("nan")

    print(f"\nApproach TAS median (last 90 s airborne): {approach_kt:.0f} kt")
    print(f"Observed peak-altitude p99: {ceiling_obs:.0f} ft")
    print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}°")

    from notebooks.calibration._common import apply_calibration_to_profile

    print()
    print("=" * 70)
    path = apply_calibration_to_profile(
        "awi_basler_bt67",
        # Keep 25 kft Basler BT-67 service ceiling; public sortie p99 is lower.
        service_ceiling_ft=25000,
        approach_speed_kt=max(90, int(round(approach_kt / 5) * 5)),
        climb_pts=klms,
        cruise_pts=cs,
        descent_pts=ds,
        climb_profile_pts=_vs_points(climb_bins, (25000, 250)),
        descent_profile_pts=_vs_points(desc_bins, (20000, 700), absolute=True),
        max_bank_deg=max(30.0, round(roll_p90)),
    )
    print(f"Wrote calibrated profile to {path}")
    print(f"  fit n_sorties={len(sorties)}")


if __name__ == "__main__":
    main()
