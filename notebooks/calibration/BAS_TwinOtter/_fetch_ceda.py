"""Fetch BAS Twin Otter MASIN core nav NetCDFs from CEDA.

Three campaign archives:
* arcticcyclones (2022 Arctic Summer-time Cyclones) — only `core_masin_*_asc-qc.nc`
  is provided; resample to 1 Hz in calibration. ~82 MB/flight.
* ofcap (2010–2011 OFCAP Falklands) — both `bas-core_masin_*_1hz.nc` and
  `_50hz.nc`; we keep 1 Hz only. ~2.5 MB/flight.
* mac/twinotter has no core nav (only MAN-2DS particle imagery). Skipped.

Lands at data/BAS_TwinOtter/<archive>/<flight_dir>/<filename>.nc.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
DATA_DIR = ROOT / "data" / "BAS_TwinOtter"
TOKEN_PATH = Path.home() / ".ceda_token"
LISTING_BASE = "https://data.ceda.ac.uk"
ARCHIVE_BASE = "https://dap.ceda.ac.uk"

# Each entry: (badc archive name, subdir name under data/).
# Most use "twinotter"; ACCACIA archives under "masin".
ARCHIVES = [
    ("arcticcyclones", "twinotter"),
    ("ofcap", "twinotter"),
    ("accacia", "masin"),
    ("igp", "twinotter"),
    ("orchestra", "twinotter"),
]


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN_PATH.read_text().strip()}"}


def _jget(path: str, headers: dict[str, str]) -> dict | None:
    url = f"{LISTING_BASE}{path}?json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return json.load(r)
    except (urllib.error.URLError, json.JSONDecodeError):
        return None


def _pick_file(items: list[dict]) -> dict | None:
    """Prefer 1Hz; fall back to highest-quality high-rate QC variant
    (asc-qc, igp-qc, orc-qc, etc.)."""
    onehz = [i for i in items if i.get("name", "").endswith(".nc") and "_1hz" in i["name"].lower()]
    if onehz:
        onehz.sort(key=lambda i: i["name"])
        return onehz[-1]
    qc = [i for i in items if i.get("name", "").endswith(".nc") and "-qc" in i["name"].lower()]
    if qc:
        qc.sort(key=lambda i: i["name"])
        return qc[-1]
    return None


def discover() -> list[dict]:
    h = _headers()
    out: list[dict] = []
    for arch, subdir in ARCHIVES:
        d = _jget(f"/badc/{arch}/data/{subdir}", h) or {}
        flights = [i["name"] for i in d.get("items", []) if "flight" in i.get("name", "").lower() and i.get("type") in ("dir", "link")]
        for fd in flights:
            d2 = _jget(f"/badc/{arch}/data/{subdir}/{fd}", h)
            if not d2:
                continue
            picked = _pick_file(d2.get("items", []))
            if picked:
                out.append({
                    "archive": arch,
                    "flight_dir": fd,
                    "filename": picked["name"],
                    "size": picked.get("size", 0),
                    "url": f"{ARCHIVE_BASE}/badc/{arch}/data/{subdir}/{fd}/{picked['name']}",
                })
    return out


def _download_one(entry: dict, headers: dict[str, str]) -> tuple[dict, str]:
    out = DATA_DIR / entry["archive"] / entry["flight_dir"] / entry["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == entry["size"]:
        return entry, "skip"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=headers), timeout=180) as r:
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        tmp.rename(out)
        return entry, "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return entry, f"err:{type(e).__name__}:{e}"


def write_manifest(entries: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["archive", "flight_dir", "filename", "size", "url", "local_path"])
        w.writeheader()
        for e in entries:
            row = dict(e)
            row["local_path"] = str((DATA_DIR / e["archive"] / e["flight_dir"] / e["filename"]).relative_to(ROOT))
            w.writerow(row)


def main() -> int:
    if not TOKEN_PATH.exists():
        print(f"ERROR: {TOKEN_PATH} missing.", file=sys.stderr)
        return 2
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = discover()
    write_manifest(entries)
    total_gb = sum(e["size"] for e in entries) / 1e9
    print(f"manifest: {len(entries)} flights, {total_gb:.2f} GB", file=sys.stderr)
    headers = _headers()
    stats = {"ok": 0, "skip": 0, "err": 0}
    errors: list[tuple[dict, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_download_one, e, headers): e for e in entries}
        for i, fut in enumerate(as_completed(futs)):
            entry, status = fut.result()
            if status in ("ok", "skip"):
                stats[status] += 1
            else:
                stats["err"] += 1
                errors.append((entry, status))
            if (i + 1) % 5 == 0 or i + 1 == len(entries):
                print(f"  {i+1}/{len(entries)}  {stats}  ({time.time()-t0:.0f}s)", flush=True)
    if errors:
        print(f"\n{len(errors)} errors:", file=sys.stderr)
        for e, msg in errors[:5]:
            print(f"  {e['archive']}/{e['flight_dir']}/{e['filename']}: {msg}", file=sys.stderr)
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
