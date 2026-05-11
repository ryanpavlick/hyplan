"""Fetch NOAA HRD/AOML flight-level ARWO data for WP-3D + G-IV-SP.

NOAA AOML hosts public 1-second flight-level data per storm at::

    https://www.aoml.noaa.gov/ftp/hrd/data/flightlevel/<YYYY>/<storm>/

Two filename conventions in the same directory:

* ``<YYYYMMDD>U<#>.01.txt`` + ``.SUM.txt``  — G-IV-SP (N49RF), ARWO format
* ``<YYYYMMDD>H<#>.1sec.txt`` + ``_FDlog.txt`` — P-3 N42RF "Kermit"
* ``<YYYYMMDD>I<#>.1sec.txt`` + ``_FDlog.txt`` — P-3 N43RF "Miss Piggy"

Downloads land at ``data/HRD/<tail>/<YYYY>/<storm>/<filename>`` so each
aircraft has a clean per-tail subtree for the calibration loader.
Idempotent: existing files matching server size are skipped.

Run from the repo root::

    python -m notebooks.calibration._hrd_fetch --years 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (file is in notebooks/calibration/)
DATA_DIR = ROOT / "data" / "HRD"
LISTING_BASE = "https://www.aoml.noaa.gov/ftp/hrd/data/flightlevel"

TAIL_NAMES = {
    "U": "USAF_WC130J",       # USAF 53rd WRS Hurricane Hunters (low-altitude penetration)
    "H": "P-3_N42RF",          # NOAA "Kermit"
    "I": "P-3_N43RF",          # NOAA "Miss Piggy"
    "N": "G-IV-SP_N49RF",      # NOAA G-IV-SP "Gonzo" (synoptic surveillance, FL420+)
}

# Filename patterns we want — full 1-sec data files (NOT the SUM/FDlog).
DATA_SUFFIXES = {
    "U": ".01.txt",      # USAF WC-130J ARWO 1-sec
    "H": ".1sec.txt",    # NOAA P-3 1-sec
    "I": ".1sec.txt",
    "N": ".1sec.txt",    # NOAA G-IV-SP 1-sec
}


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href":
                    self.links.append(v)


def _list(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []
    p = _LinkParser()
    p.feed(text)
    return p.links


def _content_length(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def discover(years: list[int], tails: tuple[str, ...]) -> list[dict]:
    entries: list[dict] = []
    for yr in years:
        storms = [
            link.rstrip("/") for link in _list(f"{LISTING_BASE}/{yr}/")
            if link.endswith("/") and not link.startswith("?") and not link.startswith("/")
        ]
        for storm in storms:
            files = _list(f"{LISTING_BASE}/{yr}/{storm}/")
            for fn in files:
                m = re.match(r"^(\d{8})([A-Z])(\d+)", fn)
                if not m:
                    continue
                tail = m.group(2)
                if tail not in tails:
                    continue
                if not fn.endswith(DATA_SUFFIXES[tail]):
                    continue
                entries.append({
                    "tail": tail,
                    "year": yr,
                    "storm": storm,
                    "filename": fn,
                    "url": f"{LISTING_BASE}/{yr}/{storm}/{fn}",
                })
        print(f"  {yr}: {sum(1 for e in entries if e['year']==yr)} matching files", file=sys.stderr, flush=True)
    return entries


def _download_one(entry: dict) -> tuple[dict, str, int]:
    tail_dir = DATA_DIR / TAIL_NAMES[entry["tail"]] / str(entry["year"]) / entry["storm"]
    tail_dir.mkdir(parents=True, exist_ok=True)
    out = tail_dir / entry["filename"]
    if out.exists() and out.stat().st_size > 0:
        return entry, "skip", out.stat().st_size
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(entry["url"], timeout=120) as r:
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        size = tmp.stat().st_size
        tmp.rename(out)
        return entry, "ok", size
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return entry, f"err:{type(e).__name__}:{e}", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2021, 2026)))
    ap.add_argument("--tails", nargs="+", default=["U", "H", "I"],
                    help="Tail letters to fetch (U=G-IV, H/I=P-3). Default skips N.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Discovering HRD files for years {args.years}, tails {args.tails}...", file=sys.stderr)
    entries = discover(args.years, tuple(args.tails))
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=1))
    print(f"manifest: {len(entries)} files", file=sys.stderr)
    if args.list_only:
        return 0

    stats = {"ok": 0, "skip": 0, "err": 0, "bytes": 0}
    errors: list[tuple[dict, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, e): e for e in entries}
        for i, fut in enumerate(as_completed(futs)):
            entry, status, size = fut.result()
            if status == "ok":
                stats["ok"] += 1
                stats["bytes"] += size
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["err"] += 1
                errors.append((entry, status))
            if (i + 1) % 50 == 0 or i + 1 == len(entries):
                gb = stats["bytes"] / 1e9
                print(
                    f"  {i+1}/{len(entries)}  ok={stats['ok']} skip={stats['skip']} "
                    f"err={stats['err']}  {gb:.2f} GB  ({time.time()-t0:.0f}s)",
                    flush=True,
                )
    if errors:
        print(f"\n{len(errors)} errors:", file=sys.stderr)
        for e, msg in errors[:5]:
            print(f"  {e['year']}/{e['storm']}/{e['filename']}: {msg}", file=sys.stderr)
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
