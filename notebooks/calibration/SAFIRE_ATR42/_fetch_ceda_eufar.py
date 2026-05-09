"""Fetch SAFIRE ATR-42 core nav NetCDFs from CEDA EUFAR archive.

Files land under data/ATR42/ceda-eufar/<project>/<flight_dir>/<filename>.nc, with a
manifest.csv at data/ATR42/ceda-eufar/manifest.csv. Idempotent.

Auth: ~/.ceda_token (CEDA archive access token).
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
DATA_DIR = ROOT / "data" / "ATR42" / "ceda-eufar"
TOKEN_PATH = Path.home() / ".ceda_token"
LISTING_BASE = "https://data.ceda.ac.uk"
ARCHIVE_BASE = "https://dap.ceda.ac.uk"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN_PATH.read_text().strip()}"}


def _jget(path: str, headers: dict[str, str]) -> dict | None:
    url = f"{LISTING_BASE}{path}?json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return json.load(r)
    except (urllib.error.URLError, json.JSONDecodeError):
        return None


def discover() -> list[dict]:
    """Walk EUFAR projects, return one file per ATR-42 flight (best revision)."""
    h = _headers()
    projs_d = _jget("/badc/eufar/data/projects", h) or {}
    projs = [i["name"] for i in projs_d.get("items", []) if i.get("type") == "dir"]

    def scan(p: str) -> list[dict]:
        d = _jget(f"/badc/eufar/data/projects/{p}", h)
        if not d:
            return []
        out: list[dict] = []
        for i in d.get("items", []):
            n = i.get("name", "")
            if "safire-atr42" in n.lower() and i.get("type") in ("dir", "link"):
                d2 = _jget(f"/badc/eufar/data/projects/{p}/{n}", h)
                if not d2:
                    continue
                for it in d2.get("items", []):
                    nm = it.get("name", "")
                    if nm.endswith(".nc") and "core" in nm.lower():
                        out.append({
                            "project": p, "flight_dir": n, "filename": nm,
                            "size": it.get("size", 0),
                            "url": f"{ARCHIVE_BASE}/badc/eufar/data/projects/{p}/{n}/{nm}",
                        })
        return out

    hits: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(scan, projs):
            hits.extend(r)

    flights: dict[tuple[str, str], list[dict]] = {}
    for h_ in hits:
        flights.setdefault((h_["project"], h_["flight_dir"]), []).append(h_)
    keep: list[dict] = []
    for files in flights.values():
        files.sort(key=lambda f: (
            "_1hz" in f["filename"].lower() or ("1hz" in f["filename"].lower() and "25hz" not in f["filename"].lower()),
            "_r1" in f["filename"].lower(),
        ), reverse=True)
        keep.append(files[0])
    return keep


def _download_one(entry: dict, headers: dict[str, str]) -> tuple[dict, str]:
    out = DATA_DIR / entry["project"] / entry["flight_dir"] / entry["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == entry["size"]:
        return entry, "skip"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=headers), timeout=120) as r:
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
        w = csv.DictWriter(f, fieldnames=["project", "flight_dir", "filename", "size", "url", "local_path"])
        w.writeheader()
        for e in entries:
            row = dict(e)
            row["local_path"] = str((DATA_DIR / e["project"] / e["flight_dir"] / e["filename"]).relative_to(ROOT))
            w.writerow(row)


def main() -> int:
    if not TOKEN_PATH.exists():
        print(f"ERROR: {TOKEN_PATH} missing.", file=sys.stderr)
        return 2
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = discover()
    write_manifest(entries)
    total_mb = sum(e["size"] for e in entries) / 1e6
    print(f"manifest: {len(entries)} flights, {total_mb:.0f} MB", file=sys.stderr)
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
            print(f"  {e['project']}/{e['flight_dir']}/{e['filename']}: {msg}", file=sys.stderr)
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
