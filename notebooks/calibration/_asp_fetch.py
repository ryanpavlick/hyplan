"""Shared utility for downloading IWG1 sortie files from the NASA ASP archive.

The NASA Airborne Science Project archive at
``asp-archive.arc.nasa.gov`` serves per-sortie IWG1 records under
``<TAIL>/FYxxxx/<YYYY-MM-DD>/IWG1.<DDMmmYYYY-HHMM>``.  This module
walks every FY/date for a given tail and downloads each per-sortie
file into a target directory, prepending the canonical 33-column
IWG1 HEADER row (the source files omit it) so that
:func:`hyplan.aircraft.iwg1.load_iwg1` can read them directly.

Idempotent — files already present are skipped.

Per-aircraft fetchers in `notebooks/calibration/<aircraft>/_fetch_asp.py`
call :func:`fetch_tail` with the tail and destination directory.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path
from collections.abc import Iterable

ARCHIVE_ROOT = "https://asp-archive.arc.nasa.gov"

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
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _list_hrefs(url: str, pattern: str) -> list[str]:
    try:
        html = _fetch(url).decode("utf-8", errors="replace")
    except urllib.error.HTTPError:
        return []
    return re.findall(pattern, html)


def list_fy_dirs(tail: str) -> list[str]:
    """Return FY directory names available for *tail* (e.g., ['FY2024', 'FY2025'])."""
    return sorted(set(_list_hrefs(f"{ARCHIVE_ROOT}/{tail}/", r'href="(FY\d{4})/?"')))


def list_sortie_dates(tail: str, fy: str) -> list[str]:
    """Return YYYY-MM-DD directory names under <tail>/<fy>/."""
    return sorted(set(_list_hrefs(
        f"{ARCHIVE_ROOT}/{tail}/{fy}/", r'href="(\d{4}-\d{2}-\d{2})"'
    )))


def find_data_filename(tail: str, fy: str, date: str) -> str | None:
    matches = _list_hrefs(
        f"{ARCHIVE_ROOT}/{tail}/{fy}/{date}/", r'href="(IWG1\.[^"]+)"'
    )
    data_files = [m for m in matches if not m.endswith(".xml")]
    return data_files[0] if data_files else None


def fetch_tail(
    tail: str,
    dest_dir: Path | str,
    *,
    file_prefix: str | None = None,
    fys: Iterable[str] | None = None,
) -> dict:
    """Download every available IWG1.<dateStamp> file for *tail* into *dest_dir*.

    Args:
        tail: ASP archive tail identifier, e.g. ``"N426NA"``.
        dest_dir: Where to write the per-sortie files.  Created if missing.
        file_prefix: Filename prefix; defaults to lowercase tail without
            the ``"NA"`` suffix (``N426NA`` → ``n426``).  Files are written as
            ``{prefix}_{YYYY-MM-DD}.txt``.
        fys: Restrict to a subset of fiscal years; default is all available.

    Returns:
        Dict with counts: ``{"new": N, "skipped": M, "missing": K}``.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if file_prefix is None:
        file_prefix = tail.lower().removesuffix("na")

    available_fys = list_fy_dirs(tail)
    if fys is not None:
        available_fys = [fy for fy in available_fys if fy in set(fys)]
    if not available_fys:
        print(f"  no FY directories found for {tail}", file=sys.stderr)
        return {"new": 0, "skipped": 0, "missing": 0}

    new = skipped = missing = 0
    for fy in available_fys:
        dates = list_sortie_dates(tail, fy)
        if not dates:
            continue
        print(f"  {tail} {fy}: {len(dates)} dates")
        for date in dates:
            out = dest / f"{file_prefix}_{date}.txt"
            if out.exists() and out.stat().st_size > 0:
                skipped += 1
                continue
            fname = find_data_filename(tail, fy, date)
            if fname is None:
                missing += 1
                continue
            try:
                body = _fetch(
                    f"{ARCHIVE_ROOT}/{tail}/{fy}/{date}/{fname}"
                ).decode("utf-8", errors="replace")
            except Exception as e:
                print(f"    {date}: download failed ({e})", file=sys.stderr)
                missing += 1
                continue
            if not body.strip():
                missing += 1
                continue
            out.write_text(CANONICAL_HEADER + body)
            new += 1
    return {"new": new, "skipped": skipped, "missing": missing}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: _asp_fetch.py <TAIL> <dest_dir> [file_prefix]")
        sys.exit(2)
    tail = sys.argv[1]
    dest = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else None
    result = fetch_tail(tail, dest, file_prefix=prefix)
    print(f"done: {result}")
