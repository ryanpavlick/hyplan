"""HIAPER (NCAR_GV, N677F) performance calibration from DC3 RAF-NAV.

Loads every DC3 NSF-GV NAV ICARTT file under ``data/HIAPER/dc3-seac4rs/``,
derives true airspeed from wind-corrected groundspeed (the LaRC-archived
DC3 NAV product omits TASX), and runs the standard per-aircraft
calibration recipe to produce a paste-ready ``NCAR_GV()`` constructor
block.

Run from repo root::

    python -m notebooks.calibration.NCAR_GV.calibrate

Or directly::

    python notebooks/calibration/NCAR_GV/calibrate.py

The calibration recipe mirrors
``notebooks/calibration/NASA_GV/_build_notebook.py`` (the calibrated NASA_GV)
so HIAPER values are directly comparable.  Differences between the two
calibrations should reflect operational differences (HIAPER routinely
cruises FL410-FL510 on long atmospheric campaigns; NASA_GV cruises
similar altitudes on shorter ASP missions) rather than methodological
ones.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Common helpers in notebooks/calibration/_common.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (
    label_phases,
    apply_sortie_filters,
    per_bin,
    tas_per_bin,
    schedule_pts,
)

from hyplan.aircraft.icartt import load_icartt


HIAPER_NAV_GLOB = "data/HIAPER/dc3-seac4rs/DC3-RAF-NAV_GV_*.ICT"

# Calibration parameters (mirrors notebooks/calibration/NASA_GV/_build_notebook.py
# where applicable).  HIAPER routinely operates higher than NASA_GV
# (FL410-FL510 on HIPPO/SOCRATES) — bins extend to 51 kft.
ACTIVE_VS_THR_FPM = 1500.0
MIN_DUR_MIN = 60
MAX_DUR_MIN = 720         # HIPPO/SOCRATES sorties are long; this is short
                          # for those but DC3 sorties topped out around
                          # 5 hours.
MIN_PEAK_ALT_FT = 30000   # GV-class operations
MAX_PEAK_ALT_FT = 55000
TARGET_ALTS_FT = (0, 5000, 10000, 20000, 30000, 41000, 49000)


def derive_tas_from_heading(
    df: pd.DataFrame,
) -> pd.Series:
    """Return TAS (kt) from groundspeed + heading + wind (wind triangle).

    The DC3 RAF-NAV ICARTT subset doesn't ship TASX, but it does ship
    aircraft-grade groundspeed (``GGSPD``, m/s converted to kt by the
    loader), true heading (``THDG``), and wind speed + direction
    (``WSC``/``WDC``).

    Wind triangle:

    * Air velocity vector points along the heading direction with
      magnitude TAS.
    * Wind vector, by convention, blows *from* ``wind_dir`` toward
      ``wind_dir + 180°``.
    * Ground velocity = air + wind.

    Solving ``|ground| = GGSPD`` for TAS gives a quadratic with one
    physically valid root::

        TAS = -wind_along_hdg + sqrt(wind_along_hdg² + GS² - |W|²)

    where ``wind_along_hdg`` is the projection of the wind vector onto
    the heading axis (positive = tailwind from the aircraft's frame).

    Returns a Series of ``tas_kt`` values; NaN where any input is
    missing or where the wind exceeds groundspeed (unsolvable).
    """
    gs = df["groundspeed"].to_numpy(dtype=float)
    hdg = np.radians(df["true_heading"].to_numpy(dtype=float))
    ws = df["wind_speed_kt"].to_numpy(dtype=float)
    wdir = np.radians(df["wind_direction_deg"].to_numpy(dtype=float))

    # Wind components (kt, east + north).  ``wind_direction_deg`` is the
    # direction the wind is coming *from*, so the wind vector points
    # toward (wind_dir + 180°), giving the negative-sign convention.
    wind_e = -ws * np.sin(wdir)
    wind_n = -ws * np.cos(wdir)

    # Wind component along the aircraft heading.  Positive means the
    # wind is pushing the aircraft forward (tailwind in the heading
    # frame).
    wind_along_hdg = wind_e * np.sin(hdg) + wind_n * np.cos(hdg)

    w_sq = wind_e * wind_e + wind_n * wind_n
    disc = wind_along_hdg * wind_along_hdg + gs * gs - w_sq

    tas = np.where(disc >= 0, -wind_along_hdg + np.sqrt(np.maximum(disc, 0.0)), np.nan)
    return pd.Series(tas, index=df.index, name="tas_kt")


def load_sorties() -> dict[str, pd.DataFrame]:
    """Load every HIAPER DC3 NAV file, derive TAS, return per-flight
    DataFrames keyed by flight date."""
    paths = sorted(Path(".").glob(HIAPER_NAV_GLOB))
    if not paths:
        raise SystemExit(
            f"No files found under {HIAPER_NAV_GLOB!r}.  Run\n"
            f"  python -m notebooks.calibration.NCAR_GV._fetch_larc_asd\n"
            f"first to populate data/HIAPER/dc3-seac4rs/."
        )

    sorties: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        df = load_icartt(p)
        df["tas_kt"] = derive_tas_from_heading(df)
        # Filename:  DC3-RAF-NAV_GV_20120518_R2 — extract YYYYMMDD.
        import re as _re
        m = _re.search(r"_(20\d{6})_", p.stem)
        date = m.group(1) if m else p.stem
        kept, reason = apply_sortie_filters(
            df,
            min_dur_min=MIN_DUR_MIN,
            max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT,
            max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if kept is None:
            skipped.append((date, reason or ""))
            continue
        kept = label_phases(kept)
        sorties[date] = kept

    print(f"  loaded {len(sorties)} sorties from {len(paths)} files")
    for d, why in skipped:
        print(f"    skipped {d}: {why}")
    return sorties


def main():
    print("HIAPER (NCAR_GV) calibration from DC3 RAF-NAV")
    print("=" * 70)
    sorties = load_sorties()
    if not sorties:
        raise SystemExit("no sorties survived filtering")

    # ── Per-altitude-bin medians ─────────────────────────────────────
    climb_bins = per_bin(
        sorties, phase="climb", sign=+1,
        active_thr_fpm=ACTIVE_VS_THR_FPM, n_min=30,
    )
    desc_bins = per_bin(
        sorties, phase="descent", sign=-1,
        active_thr_fpm=ACTIVE_VS_THR_FPM, n_min=30,
    )
    cruise_tas = tas_per_bin(sorties, phases=("cruise",), n_min=200)
    climb_tas = tas_per_bin(sorties, phases=("climb",), n_min=200)
    desc_tas = tas_per_bin(sorties, phases=("descent",), n_min=200)

    print()
    print("Climb VS bins (active, ≥30 fixes/bin):")
    print(climb_bins.to_string(index=False))
    print()
    print("Descent VS bins (active, ≥30 fixes/bin):")
    print(desc_bins.to_string(index=False))
    print()
    print("Cruise TAS bins (≥200 fixes/bin):")
    print(cruise_tas.to_string(index=False))

    # ── Schedule breakpoints ─────────────────────────────────────────
    cruise_schedule = schedule_pts(cruise_tas, TARGET_ALTS_FT, n_min=200)
    climb_schedule = schedule_pts(climb_tas, TARGET_ALTS_FT[:5], n_min=200)
    desc_schedule = schedule_pts(desc_tas, TARGET_ALTS_FT[:5], n_min=200)

    print()
    print(f"Cruise TAS schedule:  {cruise_schedule}")
    print(f"Climb  TAS schedule:  {climb_schedule}")
    print(f"Desc.  TAS schedule:  {desc_schedule}")

    # ── Bank angle (turns) ───────────────────────────────────────────
    rolls = []
    for a in sorties.values():
        if "roll_deg" in a.columns:
            r = a["roll_deg"].abs()
            rolls.append(r[r > 5.0])
    if rolls:
        roll_p90 = float(pd.concat(rolls).quantile(0.90))
        roll_p99 = float(pd.concat(rolls).quantile(0.99))
        print()
        print(f"Bank angle p90 / p99 (|roll| > 5°): {roll_p90:.1f}° / {roll_p99:.1f}°")
    else:
        roll_p90 = roll_p99 = float("nan")

    # ── Approach speed ───────────────────────────────────────────────
    final_rows = []
    for a in sorties.values():
        sub = a[(a["altitude"] < 3000) & (a["vertical_rate"] < -200)]
        final_rows.append(sub[["tas_kt"]])
    final_tas = pd.concat(final_rows).dropna()
    approach_kt = float(final_tas["tas_kt"].median())
    print(f"Approach TAS median (final < 3000 ft AGL, descent): "
          f"{approach_kt:.0f} kt  (n={len(final_tas)})")

    # ── Service ceiling (operational p99 of per-sortie peak) ─────────
    peaks = [float(a["altitude"].max()) for a in sorties.values()]
    service_ceil = float(np.percentile(peaks, 99))
    print(f"Service ceiling (op-p99 of sortie peaks): {service_ceil:.0f} ft")

    # ── Paste-ready constructor block ───────────────────────────────
    print()
    print("=" * 70)
    print("PASTE-READY NCAR_GV() PERFORMANCE BLOCK")
    print("=" * 70)
    _print_constructor(
        cruise_schedule=cruise_schedule,
        climb_bins=climb_bins,
        desc_bins=desc_bins,
        approach_kt=approach_kt,
        service_ceil=service_ceil,
        roll_p90=roll_p90,
        n_sorties=len(sorties),
    )


def _print_constructor(*, cruise_schedule, climb_bins, desc_bins,
                       approach_kt, service_ceil, roll_p90, n_sorties):
    def _vs_pts(bins):
        return [
            (int(row["alt_bin_ft"]), int(round(row["vs_med"])))
            for _, row in bins.iterrows()
        ]

    climb_pts = _vs_pts(climb_bins)
    desc_pts = _vs_pts(desc_bins)

    print("class NCAR_GV(Aircraft):")
    print('    """NSF/NCAR HIAPER Gulfstream V research aircraft (N677F)."""')
    print()
    print("    def __init__(self):")
    print("        cruise = TasSchedule(points=[")
    for alt, tas in cruise_schedule:
        print(f"            ({alt:>5} * ureg.feet, {tas:>3} * ureg.knot),")
    print("        ])")
    print("        super().__init__(")
    print(f'            aircraft_type="Gulfstream V",')
    print(f'            tail_number="N677F",')
    print(f'            operator="NSF/NCAR EOL",')
    print(f"            service_ceiling={int(round(service_ceil/100)*100)} * ureg.feet,")
    print(f"            approach_speed={int(round(approach_kt))} * ureg.knot,")
    print("            climb_schedule=cruise,")
    print("            cruise_schedule=cruise,")
    print("            descent_schedule=_descent_schedule_from_cruise(cruise, 49),")
    print("            climb_profile=VerticalProfile(points=[")
    for alt, vs in climb_pts:
        print(f"                ({alt:>5} * ureg.feet, {vs:>5} * ureg.feet / ureg.minute),")
    print("            ]),")
    print("            descent_profile=VerticalProfile(points=[")
    for alt, vs in desc_pts:
        print(f"                ({alt:>5} * ureg.feet, {abs(vs):>5} * ureg.feet / ureg.minute),")
    print("            ]),")
    print(f"            turn_model=TurnModel(max_bank_deg={max(30.0, round(roll_p90))}),")
    print('            engine_type="jet",')
    print("            confidence=PerformanceConfidence(")
    print("                climb=0.85, cruise=0.85, descent=0.85, turns=0.8,")
    print("            ),")
    print("            sources=[SourceRecord(")
    print('                source_type="icartt",')
    print('                citation=(')
    print('                    "NCAR/RAF NSF-GV ICARTT NAV files from NASA LaRC ASD '
          'archive (DC3, '
          f'{n_sorties} sorties); calibrated climb/descent VS bin medians '
          '(active VS >= 1500 fpm), and TAS schedule from wind-derived '
          'groundspeed (DC3 RAF-NAV omits TASX; TAS reconstructed via '
          'heading + wind triangle from in-situ WSC/WDC fields)."')
    print('                ),')
    print('                confidence=0.85,')
    print("            )],")
    print('            calibration_status="calibrated",')
    print("        )")


if __name__ == "__main__":
    main()
