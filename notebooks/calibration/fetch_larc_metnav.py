"""DEPRECATED — modern LaRC ASD MetNav files are redundant with the
AFRC IWG1 daily-file archive that already backs the existing
NASA_P3, NASA_GIII, and NASA_ER2 calibrations.

Verified 2026-05-07: every date in
* ARCSIX P-3B (2024-05-17 .. 2024-08-16, 24 dates)
* ARCSIX G-III (2024-05-28 .. 2024-08-16, 17 dates)
* PACEPAX ER-2 (2024-08-28 .. 2024-09-30, 15 dates)
* WHyMSIE ER-2 (2024-10-18 .. 2024-11-13, 10 dates) — registered
* WHyMSIE G-III (2024-10-27 .. 2024-11-18, 11 dates) — registered

…has a matching ``data/<aircraft>/<tail>_YYYY-MM-DD.txt`` IWG1
file on disk.  The IWG1 source is the upstream feed for the
LaRC-published MetNav files, so fetching from LaRC is duplicate
data at best, with mild risk of subtle re-derivation differences.

This driver is kept as code so the URL scheme stays documented
and so future LaRC-only campaigns (whose IWG1 isn't archived in
the AFRC daily-file source) can be added.  The DATASETS list is
empty until such a campaign is identified.

For HIAPER (NSF/NCAR GV, separate tail) the LaRC archive IS the
upstream source — see ``notebooks/calibration/NCAR_GV/`` for that
path.

For older missions (ARCTAS 2008 P-3 / B-200, DEVOTE 2011 B-200 /
UC-12, etc.) where the AFRC IWG1 archive has gaps, nav data lives
in MERGE files.  See ``fetch_larc_merge.py`` for that path
(TODO).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _larc_asd_fetch import fetch_files, date_from_filename


# (mission, button, hyplan-dir-name, filename_filter, comment).
# filename_filter: substring required in filename, plus excluded
# substrings (e.g. "10Hz" — high-rate variant we don't need for
# 5-kft binned calibration).
DATASETS: list = [
    # All modern LaRC MetNav datasets are redundant with the AFRC
    # IWG1 daily-file source (see module docstring).  Add new
    # entries here only when a campaign's nav is at LaRC but NOT
    # in the AFRC archive.
]


def main():
    print("LaRC ASD MetNav fetch — modern (2024+) missions")
    print("=" * 70)
    if not DATASETS:
        print("  no datasets configured (all modern missions are redundant")
        print("  with the AFRC IWG1 daily-file archive; see module docstring)")
        return
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
