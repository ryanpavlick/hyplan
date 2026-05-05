"""Download ACT-America C-130H (NASA 436) IWG1 files into data/c130/.

The NASA Airborne Science Project archive at
``asp-archive.arc.nasa.gov`` serves per-sortie IWG1 records under
``ACTAMERICA/N436NA/<YYYY-MM-DD>/IWG1.<DDMmmYYYY-HHMM>``.  This
script enumerates the index page, downloads each per-sortie file,
prepends the canonical 33-column IWG1 HEADER row (the source files
omit it), and writes the result as ``data/c130/n436_<YYYY-MM-DD>.txt``.

Idempotent — skips dates already present.

Run from the repo root: ``python notebooks/calibration/c130/_fetch_act_america.py``.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

INDEX_URL = "https://asp-archive.arc.nasa.gov/ACTAMERICA/N436NA/"
DEST_DIR = Path(__file__).resolve().parents[2] / ".." / "data" / "b200"
DEST_DIR = DEST_DIR.resolve()

CANONICAL_HEADER = (
    "HEADER,TimeStamp,Latitude,Longitude,GPS MSL Altitude,"
    "WGS84 Altitude,Pressure Altitude,Radar Altitude,Ground Speed,"
    "True Airspeed,Indicated Airspeed,Mach Number,Vertical Velocity,"
    "True Heading,Track,Drift,Pitch,Roll,Side Slip,Angle of Attack,"
    "Ambient Temp,Dew Point,Total Air Temp,Static Press,"
    "Dynamic Press,Cabin Press,Wind Speed,Wind Direction,"
    "Vertical Wind Speed,Solar Zenith Angle,Sun Elevation Aircraft,"
    "Sun Azimuth Ground,Sun Azimuth Aircraft\n"
)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def list_sortie_dates() -> list[str]:
    html = _fetch(INDEX_URL).decode("utf-8", errors="replace")
    dates = re.findall(r'href="(\d{4}-\d{2}-\d{2})"', html)
    return sorted(set(dates))


def find_data_filename(date: str) -> str | None:
    """Return the IWG1.<dateStamp> filename for the given sortie date."""
    html = _fetch(INDEX_URL + date + "/").decode("utf-8", errors="replace")
    matches = re.findall(r'href="(IWG1\.[^"]+)"', html)
    # Filter out the schema doc IWG1.xml; keep IWG1.<dateStamp>.
    data_files = [m for m in matches if not m.endswith(".xml")]
    return data_files[0] if data_files else None


def download_sortie(date: str) -> Path | None:
    out = DEST_DIR / f"n436_{date}.txt"
    if out.exists() and out.stat().st_size > 0:
        return None
    fname = find_data_filename(date)
    if fname is None:
        print(f"  {date}: no IWG1.<dateStamp> file found, skipping", file=sys.stderr)
        return None
    body = _fetch(INDEX_URL + date + "/" + fname).decode("utf-8", errors="replace")
    if not body.strip():
        print(f"  {date}: empty body", file=sys.stderr)
        return None
    out.write_text(CANONICAL_HEADER + body)
    return out


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dates = list_sortie_dates()
    print(f"found {len(dates)} sortie dates in {INDEX_URL}")
    new = 0
    skipped = 0
    for date in dates:
        out = download_sortie(date)
        if out is None:
            skipped += 1
        else:
            new += 1
            print(f"  wrote {out.name}")
    print(f"done: {new} new, {skipped} already-present (or missing).")
    print(f"cache: {DEST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
