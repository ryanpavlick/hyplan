"""One-shot builder for the Twin Otter ICARTT calibration notebook."""
from __future__ import annotations
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append({"cell_type":"markdown","metadata":{},"source":s.lstrip("\n").splitlines(keepends=True)})
def code(s): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.lstrip("\n").splitlines(keepends=True)})


md(r"""
# Twin Otter calibration from NASA / NOAA ICARTT

Calibrates `TwinOtter()` from FIREX-AQ flight data — N48RF Twin Otter
(NOAA), 19 sorties, summer 2019.  AIMSS Probe in-situ data
(TAS / Roll / Pitch / Heading / wind / pressure) in ICARTT format.
Same load + filter pipeline as the B-200 calibration; different
campaign, different airframe.
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
from hyplan.aircraft import TwinOtter
from hyplan.aircraft.icartt import load_icartt
from hyplan.aircraft.iwg1 import trim_ground_taxi

# Shared helpers live one directory up (notebooks/calibration/_common.py).
sys.path.insert(0, str(Path("..").resolve()))
from _common import (
    label_phases, apply_sortie_filters, per_bin, tas_per_bin,
    schedule_pts, evaluate_profile, summary_table,
)

CAMPAIGN_DIRS = [
    Path("../../../data/TwinOtter/FIREXAQ_TwinOtter_N48_FLIGHTDATA").resolve(),
]
CLIMB_FPM = 300.0
DESCENT_FPM = -300.0
MIN_DUR_MIN = 60.0
MAX_DUR_MIN = 600.0
MIN_PEAK_ALT_FT = 4000   # Twin Otter cruises FL080-FL120
MAX_PEAK_ALT_FT = 25000  # Twin Otter brochure ceiling 25 kft
""")


md("""## 1. Load + phase-label every sortie""")
code(r"""
sorties = {}
skipped = []
for cdir in CAMPAIGN_DIRS:
    if not cdir.is_dir(): continue
    for p in sorted(cdir.glob("*.ict")):
        try:
            raw = load_icartt(p)
        except Exception as e:
            skipped.append((p.stem, f"load failed ({type(e).__name__})"))
            continue
        # No groundspeed in this delivery; fall back to altitude-only
        # airborne detection.
        if raw["groundspeed"].notna().sum() > 100:
            a = trim_ground_taxi(raw)
        else:
            alt = raw["altitude"]
            if alt.dropna().empty:
                skipped.append((p.stem, "no airborne fixes")); continue
            airborne = (alt - alt.min()) > 200
            if not airborne.any():
                skipped.append((p.stem, "no airborne fixes")); continue
            first = int(airborne.values.argmax())
            last = int(len(airborne) - 1 - airborne.values[::-1].argmax())
            a = raw.iloc[first:last+1].reset_index(drop=True)
        a, reason = apply_sortie_filters(
            a, min_dur_min=MIN_DUR_MIN, max_dur_min=MAX_DUR_MIN,
            min_peak_alt_ft=MIN_PEAK_ALT_FT, max_peak_alt_ft=MAX_PEAK_ALT_FT,
        )
        if reason is not None:
            skipped.append((p.stem, reason))
            continue
        sorties[p.stem] = label_phases(a, climb_fpm=CLIMB_FPM, descent_fpm=DESCENT_FPM)

summary_table(sorties, skipped, source_label="FIREX-AQ N48RF Twin Otter")
""")


md("""## 2. Per-altitude-bin medians""")
code(r"""
ACTIVE = 500.0  # Twin Otter is a slow climber; the 1500 fpm threshold
                # used for jets / turboprops loses everything above FL050,
                # and 800 fpm still cuts off above FL100.  500 fpm captures
                # the operational climb envelope through FL150.
BIN_FT = 5000

climb_bins   = per_bin(sorties, "climb",   +1, ACTIVE, bin_ft=BIN_FT,
                       extra_cols=("tas_kt",))
descent_bins = per_bin(sorties, "descent", -1, ACTIVE, bin_ft=BIN_FT,
                       extra_cols=("tas_kt",))
print("Active CLIMB:");   print(climb_bins.to_string(index=False))
print()
print("Active DESCENT:"); print(descent_bins.to_string(index=False))
""")


md("""## 3. Profile + schedule breakpoints""")
code(r"""
fixed = []
for _, r in climb_bins.iterrows():
    if 0 <= r["alt_bin_ft"] < 25000:
        fixed.append((int(r["alt_bin_ft"]), float(r["vs_med"])))
fixed.append((25000, 200))

fixed_desc = []
for _, r in descent_bins.iterrows():
    if 0 <= r["alt_bin_ft"] < 25000:
        fixed_desc.append((int(r["alt_bin_ft"]), abs(float(r["vs_med"]))))


climb_tas   = tas_per_bin(sorties, ["climb"],   bin_ft=BIN_FT, n_min=100)
cruise_tas  = tas_per_bin(sorties, ["cruise"],  bin_ft=BIN_FT, n_min=100)
descent_tas = tas_per_bin(sorties, ["descent"], bin_ft=BIN_FT, n_min=100)
print("Climb TAS:");   print(climb_tas.to_string(index=False))
print()
print("Cruise TAS:");  print(cruise_tas.to_string(index=False))
print()
print("Descent TAS:"); print(descent_tas.to_string(index=False))


# Twin Otter rotation TAS ~70 kt; cruise band FL050-FL120.
climb_pts   = [(0, 70)] + schedule_pts(climb_tas,   [5000, 10000, 12000], n_min=100)
cruise_pts  = schedule_pts(cruise_tas, [5000, 8000, 10000, 12000], n_min=100)
descent_pts = [(0, 90)] + schedule_pts(descent_tas, [5000, 10000, 12000], n_min=100)
print()
print("climb:  ", climb_pts)
print("cruise: ", cruise_pts)
print("descent:", descent_pts)
""")


md("""## 4. Validation IQR + median + shipping""")
code(r"""
alt_grid = np.arange(0, 25000, 250)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.fill_betweenx(climb_bins["alt_bin_ft"]/1000, climb_bins["vs_p25"], climb_bins["vs_p75"],
                 alpha=0.25, color="steelblue", label="active-climb IQR")
ax.plot(climb_bins["vs_med"], climb_bins["alt_bin_ft"]/1000, "o", color="steelblue", label="median")
ax.plot(evaluate_profile(fixed, alt_grid), alt_grid/1000, color="C1", lw=2, label="proposed")
ax.set_xlabel("VS (fpm)"); ax.set_ylabel("altitude (kft)"); ax.set_title("Climb")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)

ax = axes[1]
ax.fill_betweenx(descent_bins["alt_bin_ft"]/1000, (-descent_bins["vs_p75"]).abs(),
                 (-descent_bins["vs_p25"]).abs(),
                 alpha=0.25, color="firebrick", label="active-descent IQR")
ax.plot((-descent_bins["vs_med"]).abs(), descent_bins["alt_bin_ft"]/1000, "o",
        color="firebrick", label="median")
ax.plot(evaluate_profile(fixed_desc, alt_grid), alt_grid/1000, color="C1", lw=2, label="proposed")
ax.set_xlabel("|VS| (fpm)"); ax.set_ylabel("altitude (kft)"); ax.set_title("Descent")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")


md("""## 4b. Operational vs aircraft-intrinsic framing

Approach TAS and per-sortie peak altitude that follow describe
**operational** behavior across the FIREX-AQ N48RF mission set:
peaks reflect actual mission profiles flown (boundary-layer plus
free-troposphere transit) rather than the airframe 25 kft brochure
ceiling under MTOW; approach TAS is the median final-approach
speed for the FIREX-AQ mission mix.  Aircraft-intrinsic
performance (climb / descent / cruise schedules in §2–§3, bank in
§5) is what the planner consumes; the §5 ceiling / approach
numbers are reviewer-facing context.""")


md("""## 5. Bank, approach, peak""")
code(r"""
banks = pd.concat([a["roll_deg"].abs() for a in sorties.values()]).dropna()
banks = banks[banks > 5.0]
print(f"|Roll| n={len(banks):,}, median {banks.median():.1f}°, p90 {banks.quantile(.90):.1f}°")

approach_tas = []
for a in sorties.values():
    floor = a["altitude"].min()
    sub = a[(a["altitude"] - floor < 500) & (a["altitude"] - floor > 50) & (a["vertical_rate"] < -200)]
    if not sub.empty and sub["tas_kt"].notna().any():
        approach_tas.append(sub["tas_kt"].median())
ap_s = pd.Series(approach_tas).dropna()
print(f"Approach TAS: n={len(ap_s)}, median {ap_s.median():.0f} kt")

peaks = pd.Series([a["altitude"].max() for a in sorties.values()])
print(f"Peak: median {peaks.median():.0f}, p99 {peaks.quantile(0.99):.0f}, max {peaks.max():.0f} ft")
""")


md("""## 6. Paste-ready""")
code(r"""
print("# Paste into hyplan/aircraft/_models.py TwinOtter.__init__")
print()
print("climb_profile=VerticalProfile(points=[")
for alt, vs in fixed:
    print(f"    ({alt:>5d} * ureg.feet, {vs:5.0f} * ureg.feet / ureg.minute),")
print("]),")
print()
print("descent_profile=VerticalProfile(points=[")
for alt, vs in fixed_desc:
    print(f"    ({alt:>5d} * ureg.feet, {vs:5.0f} * ureg.feet / ureg.minute),")
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
print(f"# Median final-approach TAS across {len(ap_s)} FIREX-AQ sorties.")
print(f"# DHC-6 typical AFM approach is 80-90 kt; the higher value here")
print(f"# reflects the NOAA mission profile.")
print(f"approach_speed={int(round(ap_s.median()))} * ureg.knot,")
print()
print(f"# Operational p99 of per-sortie peak altitudes ({len(sorties)} sorties).")
print(f"# Brochure service ceiling is 25 kft; the operational p99 here")
print(f"# reflects the FIREX-AQ mission mix, not the airframe ceiling.")
print(f"service_ceiling={int(round(peaks.quantile(0.99) / 1000) * 1000)} * ureg.feet,")
print()
print(f"# AFM normal-ops 30°; data p90={banks.quantile(.90):.0f}°.")
print(f"turn_model=TurnModel(max_bank_deg=30.0),")
print()
print(f'sources=[SourceRecord(')
print(f'    source_type="icartt",')
print(f'    reference="NOAA N48RF Twin Otter, FIREX-AQ ICARTT calibration, n={len(sorties)} sorties",')
print(f'    confidence=0.85,')
print(f')],')
""")


nb = {"cells": CELLS, "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "calibration.ipynb"
with out.open("w") as f: json.dump(nb, f, indent=1)
print(f"wrote {out} with {len(CELLS)} cells")
