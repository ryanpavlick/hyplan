"""Download missing-year IWG1 sorties for NASA 806 / NASA 809 from ASP.

The existing ``data/er2/`` cache spans 2012-2016 and 2023-2026 but has
no coverage of 2017-2022 — the years Lait's GSFC flight planner tuned
its ER-2 model against (DCOTSS test flights 2021, deployment 2022).

This script closes that gap by pulling every available FY in the
public NASA ASP archive (``asp-archive.arc.nasa.gov``):

* N806NA: FY2017, FY2018, FY2022 (133 sorties)
* N809NA: FY2019, FY2020, FY2021, FY2022 (100 sorties)

Idempotent — files already present in ``data/er2/`` are skipped.

Run from repo root::

    python -m notebooks.calibration.NASA_ER2._fetch_asp
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _asp_fetch import fetch_tail


DEST = Path("data/er2")

# (tail, [fiscal-year subset])
JOBS = [
    ("N806NA", ["FY2017", "FY2018", "FY2022"]),
    ("N809NA", ["FY2019", "FY2020", "FY2021", "FY2022"]),
]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    grand_total = {"new": 0, "skipped": 0, "missing": 0}
    for tail, fys in JOBS:
        print(f"\n=== {tail}: {', '.join(fys)} ===")
        result = fetch_tail(tail, DEST, fys=fys)
        for k in grand_total:
            grand_total[k] += int(result.get(k, 0))
        print(f"  new={result.get('new', 0)} "
              f"skipped={result.get('skipped', 0)} "
              f"missing={result.get('missing', 0)}")
    print(
        f"\nTotal: {grand_total['new']} new, "
        f"{grand_total['skipped']} already present, "
        f"{grand_total['missing']} unavailable / not-in-FY"
    )


if __name__ == "__main__":
    main()
