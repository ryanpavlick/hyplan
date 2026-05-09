"""Generate companion calibration notebooks from calibrate.py modules.

Each per-aircraft ``calibrate.py`` already implements the full recipe
(load → filter → phase-label → bin → emit constructor block).  The
companion ``calibration.ipynb`` is a thin interactive wrapper:
imports, a load step, the per-altitude bin tables and TAS schedules,
the approach / ceiling / bank summary, and a paste-ready cell.

Usage::

    python -m notebooks.calibration._make_notebook \\
        --module notebooks.calibration.FAAM_BAe146.calibrate \\
        --aircraft FAAM_BAe146 \\
        --title "FAAM BAe-146 calibration" \\
        --source "FAAM Core Data Product 1 Hz from CEDA" \\
        --out notebooks/calibration/FAAM_BAe146/calibration.ipynb

If invoked without arguments, builds notebooks for every aircraft
listed in :data:`AIRCRAFT_CONFIGS`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-aircraft configuration
# ---------------------------------------------------------------------------

AIRCRAFT_CONFIGS = [
    dict(
        module="notebooks.calibration.FAAM_BAe146.calibrate",
        aircraft="FAAM_BAe146",
        title="FAAM BAe-146 calibration",
        source=(
            "FAAM Core Data Product 1 Hz NetCDFs from CEDA "
            "(2017-2024 ASMM-tagged science campaigns)."
        ),
        notes=(
            "FAAM files are unusually clean: every nav variable carries "
            "units, ``WOW_IND`` gives a perfect on-ground filter, and "
            "``HGT_RADR`` feeds the radar-altimeter approach gate."
        ),
        out="notebooks/calibration/FAAM_BAe146/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.SAFIRE_ATR42.calibrate",
        aircraft="SAFIRE_ATR42",
        title="SAFIRE ATR-42 calibration",
        source=(
            "Two archives: CEDA EUFAR (28 flights across 7 transnational-"
            "access projects) + AERIS EUREC4A 2020 (19 flights, native TAS)."
        ),
        notes=(
            "CEDA EUFAR files lack TAS; reconstructed via wind triangle "
            "from position derivatives + wind components.  AERIS EUREC4A "
            "files ship native TAS and are used directly."
        ),
        out="notebooks/calibration/SAFIRE_ATR42/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.DLR_HALO.calibrate",
        aircraft="DLR_HALO",
        title="DLR HALO (G550) calibration",
        source=(
            "DLR HALO BAHAMAS 10 Hz NetCDFs (HALO-AC3 March-April 2022, "
            "Arctic), downsampled to 1 Hz for binning."
        ),
        notes=(
            "Single-campaign dataset (n=18) so confidence is set to 0.7. "
            "BAHAMAS ships native TAS, IRS-derived attitude, and vertical "
            "velocity — no reconstruction needed."
        ),
        out="notebooks/calibration/DLR_HALO/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.AWI_BaslerBT67.calibrate",
        aircraft="AWI_BaslerBT67",
        title="AWI Polar 5 / Polar 6 (Basler BT-67) calibration",
        source=(
            "Two public PANGAEA wind/temperature products: ACLOUD 2017 "
            "(DOI 10.1594/PANGAEA.902849) and HALO-AC3 2022 (DOI "
            "10.1594/PANGAEA.968911), both Polar 5 + Polar 6 tails."
        ),
        notes=(
            "Public files store altitude at whole-metre precision; the "
            "calibration uses a central-difference smoother before "
            "binning to recover meaningful vertical-rate medians."
        ),
        out="notebooks/calibration/AWI_BaslerBT67/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.BAS_TwinOtter.calibrate",
        aircraft="BAS_TwinOtter",
        title="BAS Twin Otter calibration (5 CEDA archives)",
        source=(
            "Five CEDA archives spanning 2010-2022: OFCAP (sub-Antarctic "
            "Falklands), ACCACIA (high Arctic), ORCHESTRA (Southern Ocean), "
            "IGP (Iceland-Greenland Seas), ArcticCyclones (summer Arctic)."
        ),
        notes=(
            "Variable name suffix differs by GPS unit (``_JAVAD`` for OFCAP "
            "2010-2011, ``_OXTS`` for ACCACIA / ORCHESTRA).  IGP and "
            "ArcticCyclones ship the QC subset only — TAS reconstructed "
            "via wind triangle and VS from gps_alt finite difference."
        ),
        out="notebooks/calibration/BAS_TwinOtter/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NERC_DO228.calibrate",
        aircraft="NERC_DO228",
        title="NERC DO228 (D-CALM) calibration",
        source=(
            "Two CEDA archives: ACTIVE 2005-2006 (``active-package_arsf-"
            "dornier_*.nc``) and Eyjafjallajökull 2010 (``arsf_uk_*_1Hz.csv``)."
        ),
        notes=(
            "Neither archive ships native TAS or attitude — both use a "
            "wind-triangle reconstruction from position derivatives + "
            "U/V wind components."
        ),
        out="notebooks/calibration/NERC_DO228/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NOAA_WP3D.calibrate",
        aircraft="NOAA_WP3D",
        title="NOAA WP-3D Orion calibration",
        source=(
            "NOAA CSL ICARTT archive: ARCPAC 2008, CalNex 2010, SENEX 2013, "
            "SONGNEX 2015 (12 + 27 + 20 + 19 sortie-dates).  Each sortie "
            "merges three ICARTT files (AircraftMet + AircraftPos + "
            "AircraftMis) on the AOCTimewave UTC-seconds-past-midnight key."
        ),
        notes=(
            "NOAA WP-3D (``Hurricane Hunter`` chemistry P-3) flies a "
            "different mission profile than NASA's ``NASA_P3`` (LaRC tail), "
            "warranting its own calibration class."
        ),
        out="notebooks/calibration/NOAA_WP3D/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NCAR_GV.calibrate",
        aircraft="NCAR_GV",
        title="NCAR HIAPER (GV) calibration",
        source=(
            "NSF/NCAR HIAPER ICARTT NAV files from LaRC ASD (DC3 2012). "
            "NetCDF-derived ICARTT format."
        ),
        notes=(
            "HIAPER files lack native TAS; reconstructed via wind triangle "
            "from groundspeed and U/V wind components."
        ),
        out="notebooks/calibration/NCAR_GV/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NOAA_TwinOtter.calibrate",
        aircraft="NOAA_TwinOtter",
        title="NOAA Twin Otter (N48RF + N46RF) calibration",
        source=(
            "Two ICARTT sources combined under ``data/NOAA_TwinOtter/``: "
            "FIREX-AQ 2019 (N48RF, 17 sorties) plus six NOAA CSL "
            "campaigns on N46RF (TopDown 2014, UWFPS 2017, CalFiDE 2022, "
            "AEROMMA 2023, AMMBEC 2024, USOS 2024).  164 sorties total."
        ),
        notes=(
            "Per-file unit detection handles inconsistent m/s vs kt "
            "labeling between PIs (some campaigns advertise "
            "``TrueAirSpd, m/s`` while values are in kt; the aircraft's "
            "physical envelope at ~85 m/s lets us flag and re-interpret "
            "the mislabeled files).  Active-VS threshold is 500 fpm — "
            "lower than jets / turboprops, matching the slow Twin Otter "
            "climb gradient."
        ),
        out="notebooks/calibration/NOAA_TwinOtter/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NASA_P3.calibrate",
        aircraft="NASA_P3",
        title="NASA P-3 Orion (NASA 426, LaRC) calibration",
        source=(
            "Per-sortie IWG1 files under ``data/NASA_P3/``: local NASA "
            "delivery (``p3_*.txt``) plus NASA ASP archive "
            "(``n426_*.txt`` from ``asp-archive.arc.nasa.gov/N426NA``)."
        ),
        notes=(
            "Distinct from the NOAA WP-3D 'Hurricane Hunter' (different "
            "operator, different mission profile, separate calibration "
            "class ``NOAA_WP3D``).  Cruise-phase schedule restricted to "
            "FL150-FL280."
        ),
        out="notebooks/calibration/NASA_P3/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NASA_GV.calibrate",
        aircraft="NASA_GV",
        title="NASA G-V (NASA 95, JSC) calibration",
        source=(
            "Per-sortie IWG1 files split from the local "
            "``n95na_alltracks*.csv`` deliveries.  Files prefix "
            "``n95_*.txt`` under ``data/NASA_GV/``."
        ),
        notes=(
            "Cruise-phase schedule restricted to FL410-FL510 (typical "
            "G-V operating band).  Service ceiling op-p99 reaches "
            "~FL450 in this sample even though FL510 is the airframe "
            "certified ceiling — payload weight typically caps mission "
            "peaks below the AFM number."
        ),
        out="notebooks/calibration/NASA_GV/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NASA_C130.calibrate",
        aircraft="NASA_C130",
        title="NASA C-130H (NASA 436 + 439, Wallops) calibration",
        source=(
            "Per-sortie IWG1 files split from the local "
            "``n43[69]NA_alltracks.csv`` deliveries (ACT-America "
            "2016-2018, NASA Wallops C-130H archive).  Both tails land "
            "in ``data/NASA_C130/`` (prefix ``n436_*.txt`` / "
            "``n439_*.txt``)."
        ),
        notes=(
            "Single-operator (NASA Wallops) calibration; USAF / NRL / "
            "NCAR C-130 variants would need their own classes.  Cruise-"
            "phase schedule restricted to FL200-FL280 (the typical "
            "operational band)."
        ),
        out="notebooks/calibration/NASA_C130/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.KingAirB200.calibrate",
        aircraft="KingAirB200",
        title="KingAirB200 multi-campaign calibration",
        source=(
            "Six ICARTT campaign archives under ``data/KingAirB200/``: "
            "ACT-AMERICA housekeeping, three DISCOVER-AQ deployments "
            "(CA / CO / TX), KORUS-AQ NAV, LMOS NAV."
        ),
        notes=(
            "Active-VS threshold lowered to 1000 fpm (vs the 1500 fpm "
            "used for jets) since B-200 climb rates drop below 1500 fpm "
            "above FL150.  Ground-taxi trim falls back to altitude-based "
            "trim when groundspeed is missing from the ICARTT file."
        ),
        out="notebooks/calibration/KingAirB200/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NASA_WB57.calibrate",
        aircraft="NASA_WB57",
        title="NASA WB-57 (NASA 926 + 927, JSC) calibration",
        source=(
            "Per-sortie IWG1 files split from the local "
            "``n92[67]NA_alltracks.csv`` deliveries.  Both tails land "
            "in ``data/NASA_WB57/`` (prefix ``n926_*.txt`` / "
            "``n927_*.txt``)."
        ),
        notes=(
            "High-altitude reconnaissance twin-jet, typical cruise "
            "FL500-FL620, brochure ceiling FL650.  Cruise-phase "
            "schedule restricted to FL500-FL620; FL400 / FL450 anchors "
            "in ``_models.py`` are hand-curated additions outside the "
            "calibrate.py output."
        ),
        out="notebooks/calibration/NASA_WB57/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NASA_GIII.calibrate",
        aircraft="NASA_GIII",
        title="NASA G-III (NASA 520, LaRC) calibration",
        source=(
            "NASA ASP archive ``asp-archive.arc.nasa.gov/N520NA`` "
            "(per-sortie IWG1 files prefixed ``n520_*.txt``) plus the "
            "local ``n520NA_g3_alltracks.csv`` delivery split into "
            "per-sortie files (prefix ``g3ih_*.txt``).  Both land in "
            "``data/NASA_GIII/`` and are loaded together by glob."
        ),
        notes=(
            "Per-phase TAS schedule targets differ — climb anchored at "
            "SL with rotation TAS, cruise restricted to FL250-FL400 "
            "(the typical operating band), descent anchored at SL "
            "approach TAS.  See ``calibrate.py`` ``main()`` for the "
            "per-phase target altitudes; the notebook uses the unified "
            "``TARGET_ALTS_FT`` for display."
        ),
        out="notebooks/calibration/NASA_GIII/calibration.ipynb",
    ),
    dict(
        module="notebooks.calibration.NOAA_GIV.calibrate",
        aircraft="NOAA_GIV",
        title="NOAA G-IV-SP 'Gonzo' (N49RF) calibration",
        source=(
            "NOAA HRD AOML hurricane field-program archive 2021-2025: "
            "1-second flight-level text files (``*N*.1sec.txt``) per "
            "synoptic-surveillance sortie."
        ),
        notes=(
            "Same file format as the NOAA P-3 H/I files, loaded via the "
            "shared ``_hrd_loader.load_p3_1sec``.  Format does not ship "
            "roll, so ``max_bank_deg=30`` (AFM normal-ops floor)."
        ),
        out="notebooks/calibration/NOAA_GIV/calibration.ipynb",
    ),
]


# ---------------------------------------------------------------------------
# Notebook cell helpers
# ---------------------------------------------------------------------------

def _md(s: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": s.lstrip("\n").splitlines(keepends=True),
    }


def _code(s: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s.lstrip("\n").splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def build_notebook(module: str, aircraft: str, title: str,
                   source: str, notes: str, out: str) -> None:
    cells = []

    cells.append(_md(f"""
# {title}

**Aircraft class:** `hyplan.aircraft.{aircraft}`

**Source:** {source}

{notes}

This notebook is a thin interactive companion to
[`{module.split('.')[-1]}.py`](./calibrate.py).  All loading, filtering, and
binning logic lives in that module so the standalone `python -m {module}`
invocation and this notebook produce identical numbers.  Edit the script
to change the recipe; re-run this notebook to inspect intermediate state
or regenerate the paste-ready constructor block for `_models.py`.
"""))

    cells.append(_md("## 1. Setup"))
    cells.append(_code(f"""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run from the repo root so glob paths resolve.
sys.path.insert(0, str(Path("..").resolve().parents[1]))
sys.path.insert(0, str(Path("..").resolve()))
import {module} as cal
from _common import per_bin, tas_per_bin, schedule_pts, summary_table

print("active-VS threshold:", cal.ACTIVE_VS_THR_FPM, "fpm")
print("duration filter:    ", cal.MIN_DUR_MIN, "..", cal.MAX_DUR_MIN, "min")
print("peak-altitude filter:", cal.MIN_PEAK_ALT_FT, "..", cal.MAX_PEAK_ALT_FT, "ft")
print("TAS schedule targets:", cal.TARGET_ALTS_FT)
"""))

    cells.append(_md("""## 2. Load every sortie

Reads all matching files, applies per-source loaders (with TAS
reconstruction where needed), filters out short / low-altitude / failed
sorties, and phase-labels by vertical-rate threshold."""))
    cells.append(_code("""
import os
os.chdir(Path("..").resolve().parents[1])  # repo root for glob patterns
sorties = cal.load_sorties()
print(f"\\n{len(sorties)} sorties loaded")
"""))

    cells.append(_md("""## 3. Per-altitude-bin VS medians

Active-VS gate (per-aircraft threshold) on phase-labeled fixes,
binned in 5-kft steps with n>=30/bin.  Bins below the floor are
dropped as too thin to trust."""))
    cells.append(_code("""
climb_bins = per_bin(sorties, "climb", +1, cal.ACTIVE_VS_THR_FPM, n_min=30)
desc_bins = per_bin(sorties, "descent", -1, cal.ACTIVE_VS_THR_FPM, n_min=30)
print("CLIMB VS bins:")
print(climb_bins.to_string(index=False))
print("\\nDESCENT VS bins:")
print(desc_bins.to_string(index=False))
"""))

    cells.append(_md("""## 4. TAS schedules per phase

Median TAS per altitude bin for cruise, climb, and descent fixes
(n>=200/bin).  Schedule breakpoints are picked at the per-aircraft
target altitudes."""))
    cells.append(_code("""
cruise_tas = tas_per_bin(sorties, ("cruise",), n_min=200)
climb_tas  = tas_per_bin(sorties, ("climb",),  n_min=200)
desc_tas   = tas_per_bin(sorties, ("descent",), n_min=200)
print("CRUISE TAS bins:")
print(cruise_tas.to_string(index=False))

cs   = schedule_pts(cruise_tas, cal.TARGET_ALTS_FT, n_min=200)
klms = schedule_pts(climb_tas,  cal.TARGET_ALTS_FT, n_min=200)
ds   = schedule_pts(desc_tas,   cal.TARGET_ALTS_FT, n_min=200)
print(f"\\nCruise TAS schedule: {cs}")
print(f"Climb  TAS schedule: {klms}")
print(f"Desc.  TAS schedule: {ds}")
"""))

    cells.append(_md("""## 5. Climb-VS profile sanity plot"""))
    cells.append(_code("""
fig, ax = plt.subplots(figsize=(7, 5))
if not climb_bins.empty:
    ax.fill_betweenx(climb_bins["alt_bin_ft"], climb_bins["vs_p25"],
                      climb_bins["vs_p75"], alpha=0.25, label="IQR")
    ax.plot(climb_bins["vs_med"], climb_bins["alt_bin_ft"], "o-", label="climb median")
if not desc_bins.empty:
    ax.fill_betweenx(desc_bins["alt_bin_ft"], desc_bins["vs_p25"],
                      desc_bins["vs_p75"], alpha=0.25, color="C1", label="descent IQR")
    ax.plot(desc_bins["vs_med"], desc_bins["alt_bin_ft"], "s-", color="C1", label="descent median")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("vertical rate (fpm)")
ax.set_ylabel("altitude bin (ft)")
ax.set_title("Per-bin active-VS medians (IQR shaded)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
"""))

    cells.append(_md("""## 6. Approach speed, service ceiling, bank

* **Approach speed:** median TAS in the final 60 s of each sortie's airborne segment, where VS < -200 fpm.  AGL/MSL altitude alone doesn't isolate landing because many of these aircraft fly low-altitude science surveys at cruise speed.
* **Service ceiling:** operational p99 of per-sortie peak altitudes, NOT the certified airframe ceiling at MTOW.
* **Bank angle:** p90 of |roll| during turn-state fixes (>5° gate), capped at the AFM normal-ops 30° floor."""))
    cells.append(_code("""
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

print(f"Approach TAS median:        {approach_kt:.0f} kt")
print(f"Service ceiling (op-p99):   {ceiling:.0f} ft")
if rolls:
    print(f"Bank angle p90 (|roll|>5°): {roll_p90:.1f}°")
else:
    print(f"Bank angle: not available; AFM default 30° floor applies")
"""))

    cells.append(_md(f"""## 7. Paste-ready constructor block

The block below mirrors what `python -m {module}` prints — keep them in
sync.  Paste into `hyplan/aircraft/_models.py` for the
`{aircraft}` class."""))
    cells.append(_code(f"""
def _vs_pts(bins):
    return [(int(r["alt_bin_ft"]), int(round(r["vs_med"])))
            for _, r in bins.iterrows()]

print(f"# Calibrated against {{len(sorties)}} sorties.")
print(f"service_ceiling={{int(round(ceiling/100)*100)}} * ureg.feet,")
print(f"approach_speed={{int(round(approach_kt))}} * ureg.knot,")
print(f"climb_schedule=TasSchedule(points={{klms!r}}),")
print(f"cruise_schedule=TasSchedule(points={{cs!r}}),")
print(f"descent_schedule=TasSchedule(points={{ds!r}}),")
print(f"climb_profile=VerticalProfile(points={{_vs_pts(climb_bins)!r}}),")
print(f"descent_profile=VerticalProfile(points={{_vs_pts(desc_bins)!r}}),")
if rolls:
    print(f"max_bank_deg={{max(30.0, round(roll_p90))}},")
else:
    print(f"max_bank_deg=30.0,  # AFM default; no roll data in source")
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notebook, indent=1))
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module")
    ap.add_argument("--aircraft")
    ap.add_argument("--title")
    ap.add_argument("--source")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.module:
        build_notebook(args.module, args.aircraft, args.title,
                       args.source, args.notes, args.out)
        return
    print(f"Building notebooks for {len(AIRCRAFT_CONFIGS)} aircraft...")
    for cfg in AIRCRAFT_CONFIGS:
        build_notebook(**cfg)


if __name__ == "__main__":
    main()
