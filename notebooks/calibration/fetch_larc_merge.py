"""Fetch nav-rich PI-MERGE ICARTT files from older NASA LaRC ASD missions.

For older campaigns (pre-2024) the LaRC archive doesn't ship dedicated
``MetNav_*.ict`` products — but their PI-merged 1-second files
typically include the standard navigation set (lat/lon, altitude,
TAS, ground speed, heading, attitude, wind speed/direction).  These
are usable for HyPlan calibration when the campaign isn't already
covered by the AFRC IWG1 daily-file archive.

DISCOVER-AQ 2011 P-3B is the v1.6 deliverable — fills the 2003-2016
gap in ``data/p3/n426_*`` daily files with 16 well-instrumented
sorties.  Other candidates (ARCTAS 2008, DEVOTE 2011) had insufficient
nav variables on the LaRC archive at probe time (2026-05-07) and are
deferred.

Run from repo root::

    python -m notebooks.calibration.fetch_larc_merge

Files land under ``data/<aircraft>/<mission>/`` (e.g.
``data/p3/discover-aq/discoveraq-mrg01-p3b_merge_20110701_R4.ict``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _larc_asd_fetch import fetch_files, date_from_filename


# (mission, button, hyplan-dir-name, filename_filter, label)
DATASETS = [
    (
        "discover-aq", "Merge", "p3",
        # 1-sec merges (mrg01) only.  Skip 60-sec and grand-merge
        # files (those have different time bases or aggregate across
        # the campaign and don't match our per-sortie binning model).
        lambda n: (
            n.startswith("discoveraq-mrg01-p3b_merge_")
            and "_thru" not in n  # exclude grand-merge
            and n.endswith(".ict")
        ),
        "DISCOVER-AQ 2011 P-3B (1-sec merge)",
    ),
]


def main():
    print("LaRC ASD MERGE fetch — older missions with nav-rich PI merges")
    print("=" * 70)
    summary = []
    for mission, button, aircraft_dir, fname_filter, label in DATASETS:
        print(f"\n  {label}")
        out_dir = Path("data") / aircraft_dir / mission
        files = fetch_files(
            mission=mission,
            button=button,
            out_dir=out_dir,
            filename_filter=fname_filter,
            label=f"{mission}/{button}",
            extensions=(".ict", ".ICT"),
        )
        dates = sorted({date_from_filename(p.name) or "" for p in files})
        dates = [d for d in dates if d]
        date_range = (
            f"{dates[0]}–{dates[-1]} ({len(dates)} dates)" if dates else "no dates"
        )
        summary.append((label, aircraft_dir, len(files), date_range))

    print()
    print("=" * 70)
    print(f"  {'Mission':<40}  {'Aircraft':<6}  {'Files':>5}  Date range")
    print("  " + "-" * 78)
    for label, aircraft, count, drange in summary:
        print(f"  {label:<40}  {aircraft:<6}  {count:>5}  {drange}")


if __name__ == "__main__":
    main()
