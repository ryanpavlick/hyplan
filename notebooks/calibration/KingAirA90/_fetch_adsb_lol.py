"""Fetch 24-hour ADS-B traces for the King Air A90 fleet from adsb.lol.

For each FAA-registered active 65-A90 / 65-A90-1 (civilian) tail, pulls
the ``trace_full_<hex>.json`` rolling 24-hour file from
``globe.adsb.lol`` and stores per-tail JSON files under
``data/KingAirA90/adsb_lol_traces/``.

Designed to be run daily as a cron job — successive runs accumulate
traces for tails active on each given day.  After ~4-8 weeks of
accumulation we expect coverage of 30-60 distinct A90 tails across
varied operational profiles (skydive, freight, charter), suitable
for ``hyplan.aircraft.adsb.pipeline`` calibration.

Source registry: ``data/KingAirA90/faa_registry.csv``, built by
filtering the FAA Releasable Aircraft bulk download to ACFTREF
codes 1152908 (65-A90) and 1152901 (65-A90-1 civilian).

Run from repo root::

    python -m notebooks.calibration.KingAirA90._fetch_adsb_lol

Idempotent — files already present for today are skipped.  Past-day
files are preserved (each filename embeds the fetch date).
"""
from __future__ import annotations

import csv
import datetime
import gzip
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import contextlib


REGISTRY_PATH = Path("data/KingAirA90/faa_registry.csv")
TRACE_DIR = Path("data/KingAirA90/adsb_lol_traces")
TIMEOUT_S = 20
MAX_WORKERS = 12

USER_AGENT = "Mozilla/5.0 (HyPlan calibration; ryan.pavlick@gmail.com)"


def _fetch_trace(hex_code: str) -> dict | None:
    last2 = hex_code[-2:]
    url = f"https://globe.adsb.lol/data/traces/{last2}/trace_full_{hex_code}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with contextlib.suppress(OSError):
        data = gzip.decompress(data)
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not d.get("trace"):
        return None
    return d


def _save(tail: str, hex_code: str, today: str, payload: dict) -> bool:
    """Write trace JSON to disk; return True if a new file was written."""
    outpath = TRACE_DIR / f"{today}_{tail}_{hex_code}.json"
    if outpath.exists():
        return False
    outpath.write_text(json.dumps(payload))
    return True


def main() -> None:
    if not REGISTRY_PATH.exists():
        print(f"Error: {REGISTRY_PATH} not found.  See module docstring.")
        sys.exit(1)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open() as f:
        rows = list(csv.DictReader(f))

    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    print(f"Fetching adsb.lol traces for {len(rows)} A90 tails (date={today})...")

    def _one(row: dict) -> tuple[str, str, int]:
        hex_code = row["mode_s_hex"].strip().lower()
        tail = row["n_number"].strip()
        if not hex_code:
            return tail, "skip", 0
        d = _fetch_trace(hex_code)
        if d is None:
            return tail, "no_data", 0
        wrote = _save(tail, hex_code, today, d)
        return tail, ("written" if wrote else "exists"), len(d["trace"])

    written = 0
    skipped_existing = 0
    no_data = 0
    total_rows = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for _tail, status, n in ex.map(_one, rows):
            if status == "written":
                written += 1
                total_rows += n
            elif status == "exists":
                skipped_existing += 1
                total_rows += n
            elif status == "no_data":
                no_data += 1

    print(f"  written:           {written}")
    print(f"  already-on-disk:   {skipped_existing}")
    print(f"  no_trace_today:    {no_data}")
    print(f"  total trace rows:  {total_rows:,}")


if __name__ == "__main__":
    main()
