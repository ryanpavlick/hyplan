"""Fetch FAAM 1 Hz core_processed NetCDFs from CEDA.

Auth: CEDA archive access token via ~/.ceda_token (chmod 600). Generate at
https://services.ceda.ac.uk/account/token/.

Endpoint: https://dap.ceda.ac.uk/badc/faam/data/<YYYY>/<flight_dir>/core_processed/
core_faam_<YYYYMMDD>_v###_r#_<flight>_1hz.nc

Files land under data/FAAM/<YYYY>/<flight_dir>/, mirroring the archive layout.
Idempotent: existing files with the expected size are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent  # repo root
DATA_DIR = ROOT / "data" / "FAAM"
TOKEN_PATH = Path.home() / ".ceda_token"

ARCHIVE_BASE = "https://dap.ceda.ac.uk/badc/faam/data"
LISTING_BASE = "https://data.ceda.ac.uk/badc/faam/data"


def _token() -> str:
    return TOKEN_PATH.read_text().strip()


def _list_year(year: int, headers: dict[str, str]) -> list[dict]:
    url = f"{LISTING_BASE}/{year}?json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r).get("items", [])


def _list_core_processed(year: int, flight_dir: str, headers: dict[str, str]) -> list[dict]:
    url = f"{LISTING_BASE}/{year}/{flight_dir}/core_processed?json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return json.load(r).get("items", [])
    except urllib.error.HTTPError:
        return []


def _pick_1hz(items: list[dict]) -> tuple[str, int] | None:
    cands = [
        (it["name"], it.get("size", 0))
        for it in items
        if it["name"].startswith("core_faam_") and it["name"].endswith("_1hz.nc")
    ]
    if not cands:
        return None
    cands.sort()  # highest version last
    return cands[-1]


def build_manifest(years: list[int]) -> list[dict]:
    headers = {"Authorization": f"Bearer {_token()}"}
    manifest: list[dict] = []
    for yr in years:
        flights = [it["name"] for it in _list_year(yr, headers) if it.get("type") == "dir"]
        for fd in flights:
            items = _list_core_processed(yr, fd, headers)
            picked = _pick_1hz(items)
            if picked:
                fn, sz = picked
                manifest.append({"year": yr, "flight_dir": fd, "filename": fn, "size": sz})
        print(f"  {yr}: {sum(1 for m in manifest if m['year']==yr)} flights", file=sys.stderr)
    return manifest


def _download_one(entry: dict, headers: dict[str, str]) -> tuple[dict, str]:
    yr, fd, fn, sz = entry["year"], entry["flight_dir"], entry["filename"], entry["size"]
    out_dir = DATA_DIR / str(yr) / fd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fn
    if out_path.exists() and out_path.stat().st_size == sz:
        return entry, "skip"
    url = f"{ARCHIVE_BASE}/{yr}/{fd}/core_processed/{fn}"
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        tmp.rename(out_path)
        return entry, "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return entry, f"err:{type(e).__name__}:{e}"


def download(manifest: list[dict], workers: int = 8) -> dict:
    headers = {"Authorization": f"Bearer {_token()}"}
    stats = {"ok": 0, "skip": 0, "err": 0, "bytes": 0}
    errors: list[tuple[dict, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, e, headers): e for e in manifest}
        for i, f in enumerate(as_completed(futs)):
            entry, status = f.result()
            if status == "ok":
                stats["ok"] += 1
                stats["bytes"] += entry["size"]
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["err"] += 1
                errors.append((entry, status))
            if (i + 1) % 25 == 0 or i + 1 == len(manifest):
                dt = time.time() - t0
                gb = stats["bytes"] / 1e9
                print(
                    f"  {i+1}/{len(manifest)}  ok={stats['ok']} skip={stats['skip']} "
                    f"err={stats['err']}  {gb:.2f} GB  ({dt:.0f}s)",
                    flush=True,
                )
    if errors:
        print(f"\n{len(errors)} errors:", file=sys.stderr)
        for e, msg in errors[:10]:
            print(f"  {e['year']}/{e['flight_dir']}/{e['filename']}: {msg}", file=sys.stderr)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2025)))
    ap.add_argument("--manifest", type=Path, default=DATA_DIR / "manifest.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    if not TOKEN_PATH.exists():
        print(f"ERROR: {TOKEN_PATH} missing. Generate at services.ceda.ac.uk/account/token/", file=sys.stderr)
        return 2
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building manifest for years {args.years}...", file=sys.stderr)
    manifest = build_manifest(args.years)
    args.manifest.write_text(json.dumps(manifest, indent=1))
    total_gb = sum(m["size"] for m in manifest) / 1e9
    print(f"manifest: {len(manifest)} flights, {total_gb:.1f} GB total", file=sys.stderr)
    if args.list_only:
        return 0
    stats = download(manifest, workers=args.workers)
    print(f"\ndone: {stats}", file=sys.stderr)
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
