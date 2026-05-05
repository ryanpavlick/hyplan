"""One-shot builder for the B-200 ICARTT calibration notebook."""
from __future__ import annotations
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append({"cell_type":"markdown","metadata":{},"source":s.lstrip("\n").splitlines(keepends=True)})
def code(s): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.lstrip("\n").splitlines(keepends=True)})


md(r"""
# King Air B-200 calibration from NASA ICARTT field campaigns

Calibrates `KingAirB200()` from public NASA ICARTT (.ict) airborne
in-situ data across multiple field campaigns.  Source folders under
`data/KingAirB200/`:

* ACTAMERICA — NASA LaRC B-200 housekeeping data, 2016-2020 (~176 files)
* DISCOVER-AQ California / Colorado / Texas — APPLANIX nav (~90 files)
* KORUS-AQ — B-200 NAV (~28 files)
* LMOS — UC-12 (B-200 military variant) NAV (~27 files)
* ACTIVATE — METNAV; not used (only GPS altitude + heading, no TAS)

Each campaign uses a different ICARTT column-name convention; the
loader (`hyplan.aircraft.icartt.load_icartt`) maps them to a
canonical IWG1-style schema so the rest of the pipeline matches the
other calibration notebooks.
""")

code(r"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyplan import ureg
from hyplan.aircraft import KingAirB200
from hyplan.aircraft.icartt import load_icartt
from hyplan.aircraft.iwg1 import trim_ground_taxi  # works on the canonical-schema DataFrame

# Shared helpers live one directory up (notebooks/calibration/_common.py).
sys.path.insert(0, str(Path("..").resolve()))
from _common import (
    label_phases, apply_sortie_filters, per_bin, tas_per_bin,
    schedule_pts, evaluate_profile, summary_table,
)

# Campaign directories to ingest.  Each .ict file in these dirs
# becomes one candidate sortie.
CAMPAIGN_DIRS = [
    Path("../../../data/KingAirB200/ACTAMERICA_B200_Hskping").resolve(),
    Path("../../../data/KingAirB200/DISCOVERAQ_California_B200_APPLANIX").resolve(),
    Path("../../../data/KingAirB200/DISCOVERAQ_Colorado_B200_APPLANIX").resolve(),
    Path("../../../data/KingAirB200/DISCOVERAQ_Texas_B200_APPLANIX").resolve(),
    Path("../../../data/KingAirB200/KORUSAQ_B200_NAV").resolve(),
    Path("../../../data/KingAirB200/LMOS_UC12_NAV").resolve(),
]

# Phase-label thresholds (same as other calibration notebooks).
CLIMB_FPM   = 300.0
DESCENT_FPM = -300.0

# Sortie filters.
MIN_DUR_MIN     = 60.0
MAX_DUR_MIN     = 600.0
MIN_PEAK_ALT_FT = 8000   # B-200 cruises FL150-FL280 typically; some
                         # boundary-layer campaigns peak around FL080
                         # (real flights, not test/abort)
MAX_PEAK_ALT_FT = 35000  # B-200 brochure ceiling
""")


md("""## 1. Load + phase-label every sortie""")
code(r"""
sorties = {}
skipped = []
for cdir in CAMPAIGN_DIRS:
    if not cdir.is_dir():
        continue
    for p in sorted(cdir.glob("*.ict")):
        try:
            raw = load_icartt(p)
        except Exception as e:
            skipped.append((p.stem, f"load failed ({type(e).__name__})"))
            continue
        # Try standard groundspeed-based trim first; fall back to
        # altitude-only when groundspeed is missing from the ICT.
        if raw["groundspeed"].notna().sum() > 100:
            a = trim_ground_taxi(raw)
        else:
            alt = raw["altitude"]
            if alt.dropna().empty:
                skipped.append((p.stem, "no airborne fixes"))
                continue
            airborne = (alt - alt.min()) > 200
            if not airborne.any():
                skipped.append((p.stem, "no airborne fixes"))
                continue
            first = int(airborne.values.argmax())
            last = int(len(airborne) - 1 - airborne.values[::-1].argmax())
            a = raw.iloc[first:last + 1].reset_index(drop=True)
        a, reason = apply_sortie_filters(
            a, min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if reason is not None:
            skipped.append((p.stem, reason))
            continue
        sorties[p.stem] = label_phases(a, climb_fpm=CLIMB_FPM, descent_fpm=DESCENT_FPM)

summary_table(sorties, skipped, source_label="multi-campaign B-200 ICARTT")
""")


md("""## 2. Per-sortie altitude profiles""")
code(r"""
fig, ax = plt.subplots(figsize=(13, 5))
for name, a in sorties.items():
    t_min = (a["timestamp"] - a["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    ax.plot(t_min, a["altitude"] / 1000, lw=0.4, alpha=0.3, color="steelblue")
ax.set_xlabel("minutes from takeoff")
ax.set_ylabel("altitude (kft)")
ax.set_title(f"NASA B-200 altitude profiles — {len(sorties)} sorties")
ax.grid(alpha=0.3)
ax.set_ylim(0, 35)
plt.tight_layout(); plt.show()
""")


md("""## 3. Climb / descent: per-altitude-bin medians""")
code(r"""
# B-200 climb rates drop below 1500 fpm above FL150; lowering
# the active threshold to 1000 fpm extends climb-bin coverage
# through FL200-FL280.  Same trade-off as the Twin Otter
# (where the threshold is 500 fpm).
ACTIVE_VS_THR_FPM = 1000.0
BIN_FT = 5000

climb_bins   = per_bin(sorties, "climb",   +1, ACTIVE_VS_THR_FPM, bin_ft=BIN_FT,
                       extra_cols=("tas_kt",))
descent_bins = per_bin(sorties, "descent", -1, ACTIVE_VS_THR_FPM, bin_ft=BIN_FT,
                       extra_cols=("tas_kt",))
print("Active CLIMB:");   print(climb_bins.to_string(index=False))
print()
print("Active DESCENT:"); print(descent_bins.to_string(index=False))
""")


md("""## 4. climb_profile / descent_profile breakpoints (per-bin medians)""")
code(r"""
fixed = []
for _, r in climb_bins.iterrows():
    if 0 <= r["alt_bin_ft"] < 35000:
        fixed.append((int(r["alt_bin_ft"]), float(r["vs_med"])))
fixed.append((35000, 500))

fixed_desc = []
for _, r in descent_bins.iterrows():
    if 0 <= r["alt_bin_ft"] < 35000:
        fixed_desc.append((int(r["alt_bin_ft"]), abs(float(r["vs_med"]))))

print("climb_profile:")
for alt, vs in fixed:
    print(f"  ({alt:>5d}, {vs:6.0f})")
print()
print("descent_profile:")
for alt, vs in fixed_desc:
    print(f"  ({alt:>5d}, {vs:6.0f})")
""")


md("""## 5. TAS schedules (per-phase 5-kft bin medians)""")
code(r"""
climb_tas   = tas_per_bin(sorties, ["climb"],   bin_ft=BIN_FT, n_min=200)
cruise_tas  = tas_per_bin(sorties, ["cruise"],  bin_ft=BIN_FT, n_min=200)
descent_tas = tas_per_bin(sorties, ["descent"], bin_ft=BIN_FT, n_min=200)
print("Climb TAS:");   print(climb_tas.to_string(index=False))
print()
print("Cruise TAS:");  print(cruise_tas.to_string(index=False))
print()
print("Descent TAS:"); print(descent_tas.to_string(index=False))

ROTATION_TAS_KT = 110
climb_pts = [(0, ROTATION_TAS_KT)] + schedule_pts(climb_tas, [5000, 10000, 15000, 20000, 25000], n_min=200)
cruise_pts = schedule_pts(cruise_tas, [10000, 15000, 20000, 25000, 28000], n_min=200)
descent_pts = [(0, 130)] + schedule_pts(descent_tas, [5000, 10000, 15000, 20000, 25000], n_min=200)
print()
print("Climb schedule:  ", climb_pts)
print("Cruise schedule: ", cruise_pts)
print("Descent schedule:", descent_pts)
""")


md("""## 6. Validation: IQR + median + shipping""")
code(r"""
alt_grid = np.arange(0, 36000, 500)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.fill_betweenx(climb_bins["alt_bin_ft"]/1000,
                 climb_bins["vs_p25"], climb_bins["vs_p75"],
                 alpha=0.25, color="steelblue", label="active-climb IQR")
ax.plot(climb_bins["vs_med"], climb_bins["alt_bin_ft"]/1000,
        "o", color="steelblue", label="active-climb median")
ax.plot(evaluate_profile(fixed, alt_grid), alt_grid/1000, color="C1", lw=2, label="proposed")
ax.set_xlabel("VS (fpm)"); ax.set_ylabel("altitude (kft)")
ax.set_title("Climb"); ax.legend(loc="upper right")
ax.grid(alpha=0.3); ax.set_xlim(0, 5000)

ax = axes[1]
ax.fill_betweenx(descent_bins["alt_bin_ft"]/1000,
                 (-descent_bins["vs_p75"]).abs(),
                 (-descent_bins["vs_p25"]).abs(),
                 alpha=0.25, color="firebrick", label="active-descent IQR")
ax.plot((-descent_bins["vs_med"]).abs(), descent_bins["alt_bin_ft"]/1000,
        "o", color="firebrick", label="active-descent median")
ax.plot(evaluate_profile(fixed_desc, alt_grid), alt_grid/1000, color="C1", lw=2, label="proposed")
ax.set_xlabel("|VS| (fpm)"); ax.set_ylabel("altitude (kft)")
ax.set_title("Descent"); ax.legend(loc="upper right")
ax.grid(alpha=0.3); ax.set_xlim(0, 5000)
plt.tight_layout(); plt.show()
""")


md("""## 6b. Operational vs aircraft-intrinsic framing

Approach TAS and per-sortie peak altitude that follow describe
**operational** behavior across this multi-campaign sortie set:
peaks reflect actual mission profiles flown (FL150–FL280
boundary-layer sampling, FL280 transit) rather than the airframe
35 kft brochure ceiling under MTOW; approach TAS is the median
final-approach speed for the mission mix.  Aircraft-intrinsic
performance (climb / descent / cruise schedules in §4–§5, bank in
§7) is what the planner consumes; the §7 ceiling / approach
numbers are reviewer-facing context.""")


md("""## 7. Bank, approach, peak altitude""")
code(r"""
banks = pd.concat([a["roll_deg"].abs() for a in sorties.values()]).dropna()
banks = banks[banks > 5.0]
print(f"|Roll| n={len(banks):,}, median {banks.median():.1f}°, p75 {banks.quantile(.75):.1f}°, p90 {banks.quantile(.90):.1f}°")

approach_tas = []
for a in sorties.values():
    floor = a["altitude"].min()
    sub = a[(a["altitude"] - floor < 500) & (a["altitude"] - floor > 50) & (a["vertical_rate"] < -200)]
    if not sub.empty and sub["tas_kt"].notna().any():
        approach_tas.append(sub["tas_kt"].median())
ap_s = pd.Series(approach_tas).dropna()
print(f"Approach TAS: n={len(ap_s)}, median {ap_s.median():.0f} kt, IQR {ap_s.quantile(.25):.0f}-{ap_s.quantile(.75):.0f}")

peaks = pd.Series([a["altitude"].max() for a in sorties.values()])
print(f"Peak alt: median {peaks.median():.0f} ft, p99 {peaks.quantile(0.99):.0f} ft, max {peaks.max():.0f} ft")
""")


md("""## 8. Paste-ready KingAirB200 constructor""")
code(r"""
print("# Paste into hyplan/aircraft/_models.py KingAirB200.__init__")
print()
print("climb_profile=VerticalProfile(points=[")
for alt, vs in fixed:
    print(f"    ({alt:>5d} * ureg.feet, {vs:6.0f} * ureg.feet / ureg.minute),")
print("]),")
print()
print("descent_profile=VerticalProfile(points=[")
for alt, vs in fixed_desc:
    print(f"    ({alt:>5d} * ureg.feet, {vs:6.0f} * ureg.feet / ureg.minute),")
print("]),")
print()
print("climb_schedule=TasSchedule(points=[")
for alt, tas in climb_pts:
    print(f"    ({alt:>5d} * ureg.feet, {tas:3d} * ureg.knot),")
print("]),")
print()
print("cruise_schedule=TasSchedule(points=[")
for alt, tas in cruise_pts:
    print(f"    ({alt:>5d} * ureg.feet, {tas:3d} * ureg.knot),")
print("]),")
print()
print("descent_schedule=TasSchedule(points=[")
for alt, tas in descent_pts:
    print(f"    ({alt:>5d} * ureg.feet, {tas:3d} * ureg.knot),")
print("]),")
print()
print(f"# Median final-approach TAS across {len(ap_s)} sorties.")
print(f"approach_speed={int(round(ap_s.median()))} * ureg.knot,")
print()
print(f"# Operational p99 of per-sortie peak altitudes ({len(sorties)} sorties).")
print(f"# Brochure service ceiling is 35 kft; this number reflects the")
print(f"# mission mix flown, not the airframe ceiling under MTOW.")
print(f"service_ceiling={int(round(peaks.quantile(0.99) / 1000) * 1000)} * ureg.feet,")
print()
print(f"# AFM normal-ops 30°; data p90={banks.quantile(.90):.0f}° agrees.")
print(f"turn_model=TurnModel(max_bank_deg=30.0),")
print()
print(f'sources=[SourceRecord(')
print(f'    source_type="icartt",')
print(f'    reference="Multi-campaign B-200 ICARTT calibration, n={len(sorties)} sorties (ACTAMERICA, DISCOVER-AQ, KORUS-AQ, LMOS)",')
print(f'    confidence=0.85,')
print(f')],')
""")


nb = {"cells": CELLS, "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "calibration.ipynb"
with out.open("w") as f: json.dump(nb, f, indent=1)
print(f"wrote {out} with {len(CELLS)} cells")
