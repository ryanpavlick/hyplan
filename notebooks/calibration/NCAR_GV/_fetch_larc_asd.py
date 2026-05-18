"""Fetch HIAPER (NSF/NCAR GV, N677F) ICARTT navigation files from the
NASA LaRC Airborne Science Data archive.

DC3 (May-Jun 2012) is the only LaRC-archived mission with NSF-GV
RAF-NAV files; other HIAPER campaigns (HIPPO/SOCRATES/ORCAS/
ATTREX/WE-CAN) are at NCAR EOL behind the ORDER queue.

Run from repo root::

    python -m notebooks.calibration.NCAR_GV._fetch_larc_asd

Files land under ``data/HIAPER/<mission>/``.

This module is a thin wrapper around
:func:`notebooks.calibration._larc_asd_fetch.fetch_files`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _larc_asd_fetch import fetch_files

# (mission, button, label).  Filename filter is "RAF-NAV but not
# RAF-NAV-HRT" — HRT is the 25-Hz variant which doesn't help with
# 5-kft binned calibration.
KNOWN_MISSIONS = (
    ("dc3-seac4rs", "GV", "DC3 NSF-GV RAF-NAV"),
)


def main():
    print("HIAPER LaRC ASD fetch")
    print("=" * 70)
    total = 0
    for mission, button, label in KNOWN_MISSIONS:
        print(f"\n  {label}")
        files = fetch_files(
            mission=mission,
            button=button,
            out_dir=Path("data") / "HIAPER" / mission,
            filename_filter=lambda n: (
                "RAF-NAV" in n and "RAF-NAV-HRT" not in n
                and n.upper().endswith(".ICT")
            ),
            label=f"{mission}/{button}",
        )
        total += len(files)
    print()
    print(f"  total files: {total}")


if __name__ == "__main__":
    main()
