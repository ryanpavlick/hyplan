"""Fetch historical 24-hour ADS-B traces from globe.airplanes.live.

Unlike adsb.lol's live-only ``trace_full_<hex>.json`` endpoint,
airplanes.live publishes ``globe_history/<YYYY-MM-DD>/traces/<last2>/
trace_full_<hex>.json`` covering at least the last ~16 months of
daily archives.  Same readsb-pb trace format as adsb.lol.

This fetcher walks a date range (default: last 30 days) for each
tail in ``data/KingAirA90/faa_registry.csv`` and saves every
non-empty trace under ``data/KingAirA90/adsb_lol_traces/`` so the
existing :mod:`notebooks.calibration.KingAirA90.calibrate` driver
picks them up unchanged.

Run from repo root::

    python -m notebooks.calibration.KingAirA90._fetch_airplanes_live --days 30

Idempotent — skips files already on disk.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import gzip
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REGISTRY_PATH = Path("data/KingAirA90/faa_registry.csv")
TRACE_DIR = Path("data/KingAirA90/adsb_lol_traces")
BASE_URL = "https://globe.airplanes.live/globe_history"
TIMEOUT_S = 30
MAX_WORKERS = 16
USER_AGENT = "Mozilla/5.0 (HyPlan calibration; ryan.pavlick@gmail.com)"


def _fetch(date: str, hex_code: str) -> dict | None:
    last2 = hex_code[-2:]
    url = f"{BASE_URL}/{date}/traces/{last2}/trace_full_{hex_code}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    if data[:1] == b"<":
        return None  # got an HTML error page disguised as 200
    with contextlib.suppress(OSError):
        data = gzip.decompress(data)
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not d.get("trace"):
        return None
    return d


def _save(date: str, tail: str, hex_code: str, payload: dict) -> bool:
    outpath = TRACE_DIR / f"{date}_{tail}_{hex_code}.json"
    if outpath.exists():
        return False
    outpath.write_text(json.dumps(payload))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=30,
                    help="Number of past days to fetch (default 30).")
    ap.add_argument("--tails", type=str, default=None,
                    help="Optional comma-separated subset of N-numbers "
                         "(e.g. N15CT,N41DZ).  Default: all from registry.")
    args = ap.parse_args()

    if not REGISTRY_PATH.exists():
        print(f"Error: {REGISTRY_PATH} not found.")
        sys.exit(1)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open() as f:
        all_rows = list(csv.DictReader(f))
    if args.tails:
        wanted = {t.strip().upper() for t in args.tails.split(",")}
        rows = [r for r in all_rows if r["n_number"].strip().upper() in wanted]
    else:
        rows = all_rows
    rows = [r for r in rows if r["mode_s_hex"].strip()]

    today = datetime.datetime.now(datetime.UTC).date()
    dates = [(today - datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
             for i in range(args.days)]

    jobs = [(date, r["n_number"].strip(), r["mode_s_hex"].strip().lower())
            for date in dates for r in rows]
    print(f"Fetching {len(rows)} tails × {len(dates)} days = {len(jobs)} jobs")

    written = 0
    skipped = 0
    no_data = 0
    total_rows = 0

    def _one(job: tuple[str, str, str]) -> tuple[int, int]:
        date, tail, hex_code = job
        outpath = TRACE_DIR / f"{date}_{tail}_{hex_code}.json"
        if outpath.exists():
            return (0, 0)  # written, no_data
        d = _fetch(date, hex_code)
        if d is None:
            return (0, 1)
        _save(date, tail, hex_code, d)
        return (len(d["trace"]), 0)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for done, f in enumerate(as_completed(futures), start=1):
            n_rows, nd = f.result()
            if n_rows > 0:
                written += 1
                total_rows += n_rows
            elif nd:
                no_data += 1
            else:
                skipped += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(jobs)}  written={written}  "
                      f"no_data={no_data}  rows={total_rows:,}")

    print()
    print(f"  written:    {written}")
    print(f"  exists:     {skipped}")
    print(f"  no_data:    {no_data}")
    print(f"  total rows: {total_rows:,}")


if __name__ == "__main__":
    main()
