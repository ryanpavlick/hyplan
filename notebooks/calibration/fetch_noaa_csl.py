"""Fetch nav-bearing ICARTT files from NOAA CSL field-project archives.

The NOAA Chemical Sciences Laboratory hosts campaign data at
``https://csl.noaa.gov/groups/csl7/measurements/<MISSION>/<PLATFORM>/``
behind a cookie-based data-policy agreement (see
``_noaa_csl_fetch.py`` for the protocol).

v1.6 deliverable: AEROMMA 2023 TwinOtter (N46RF), 17 sortie-dates
(33 ICARTT files including L1/L2/L3 segment splits) of
CUPiDS-AircraftData with directly-measured TAS, heading, wind,
attitude, position.  Complements the existing TwinOtter calibration
(17 N48RF sorties from FIREX-AQ).

**Unit-labeling note (2026-05-08)**: the CUPiDS-AircraftData ICT
header advertises ``TrueAirSpd, m/s`` and ``WindSpd, m/s`` but the
actual values are in **knots** (Twin Otter physical max TAS is
~85 m/s ≈ 165 kt; raw values are 80-135 in the file, ergo kt
not m/s).  Calibration code must override the loader's unit
conversion for these columns or the resulting schedules will be
~1.94× too high.  See ``notebooks/calibration/NOAA_TwinOtter/`` (TODO)
for the calibration recipe handling this.

Run from repo root::

    python -m notebooks.calibration.fetch_noaa_csl

Files land under ``data/twin_otter/<mission>/`` to keep them next
to the existing FIREX-AQ data hierarchy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noaa_csl_fetch import fetch_files, date_from_filename


# (mission, platform, hyplan-aircraft-dir, filename_filter, label)
DATASETS = [
    (
        "2023aeromma", "TwinOtter", "twin_otter",
        # CUPiDS-AircraftData_TO_<date>_<rev>_L<n>.ict — the AIMSS Probe
        # met / nav product (TAS, heading, wind, attitude, position).
        lambda n: n.startswith("CUPiDS-AircraftData_TO_") and n.endswith(".ict"),
        "AEROMMA 2023 NOAA Twin Otter (N46RF)",
    ),
]


def main():
    print("NOAA CSL fetch — campaign-published ICARTT nav files")
    print("=" * 70)
    summary = []
    for mission, platform, aircraft_dir, fname_filter, label in DATASETS:
        print(f"\n  {label}")
        out_dir = Path("data") / aircraft_dir / mission
        files = fetch_files(
            mission=mission,
            platform=platform,
            out_dir=out_dir,
            filename_filter=fname_filter,
            label=f"{mission}/{platform}",
        )
        dates = sorted({date_from_filename(p.name) or "" for p in files})
        dates = [d for d in dates if d]
        date_range = (
            f"{dates[0]}–{dates[-1]} ({len(dates)} dates)" if dates else "no dates"
        )
        summary.append((label, aircraft_dir, len(files), date_range))

    print()
    print("=" * 70)
    print(f"  {'Mission':<40}  {'Aircraft':<12}  {'Files':>5}  Date range")
    print("  " + "-" * 80)
    for label, aircraft, count, drange in summary:
        print(f"  {label:<40}  {aircraft:<12}  {count:>5}  {drange}")


if __name__ == "__main__":
    main()
