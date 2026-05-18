"""Download WB-57 MMS-1HZ ICARTT files from the ACCLIP campaign archive.

ACCLIP (Asian Summer Monsoon Chemical and CLimate Impact Project) was
the WB-57's 2022 deployment to Osan, South Korea — and the data Lait
explicitly tuned the GSFC flight planner's WB-57 ascent characteristics
against (ChangeLog 2022-08-02 / 2022-08-16: "improved wb57 tuning to
acclip 2022").  27 daily ICARTT files from the MMS (Meteorological
Measurement System) navigation instrument cover 2022-07-14 through
2022-09-14, totalling ~50 MB.

The MMS files are 1-Hz time series with TAS, true heading, attitude,
wind components, ambient pressure / temperature, and GPS position /
altitude.  ``hyplan.aircraft.load_icartt`` reads them directly
(post-:commit:`aea4d96`, with scale-factor and MMS-column-name
support).

Source: NASA LaRC ASDC, collection
``ACCLIP_MetNav_AircraftInSitu_WB57_Data``.  Granules discovered via
CMR (no auth needed for metadata) and downloaded with the
``EARTHDATA_TOKEN`` from the local ``.env`` (LaRC ASDC requires it).

Run from repo root::

    python -m notebooks.calibration.NASA_WB57._fetch_acclip

Idempotent — files already present in ``data/NASA_WB57/`` are skipped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

DEST = Path("data/NASA_WB57")
CMR_URL = (
    "https://cmr.earthdata.nasa.gov/search/granules.json"
    "?short_name=ACCLIP_MetNav_AircraftInSitu_WB57_Data"
    "&page_size=200"
)

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Source ``.env`` from the repo root into ``os.environ`` if present.

    Lightweight parser (no dotenv dep) covering ``KEY=value`` lines and
    optional ``KEY="value"`` quoting.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _list_acclip_mms_1hz() -> list[tuple[str, str]]:
    """Return ``[(filename, download_url), ...]`` for every MMS-1HZ ICT
    granule in the ACCLIP MetNav collection."""
    with urllib.request.urlopen(CMR_URL, timeout=60) as resp:
        payload = json.loads(resp.read())
    out: list[tuple[str, str]] = []
    for entry in payload.get("feed", {}).get("entry", []):
        name = entry.get("producer_granule_id", "")
        if "MMS-1HZ" not in name:
            continue
        if not (name.endswith(".ICT") or name.endswith(".ict")):
            continue
        for link in entry.get("links", []):
            href = link.get("href", "")
            if href.startswith("https://asdc") and (
                href.endswith(".ICT") or href.endswith(".ict")
            ):
                out.append((name, href))
                break
    return sorted(set(out))


def _download(url: str, dest_path: Path, token: str) -> int:
    """Download ``url`` to ``dest_path`` using a Bearer token.

    Returns the number of bytes written.  Raises on HTTP error.
    """
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest_path.write_bytes(data)
    return len(data)


_DATE_RE = re.compile(r"_(\d{8})_")
_SPLIT_RE = re.compile(r"_R\d+_(\d+)\.ict$", re.IGNORECASE)


def _canonical_dest_name(asdc_name: str) -> str:
    """Translate ``ACCLIP-MMS-1HZ_WB57_20220714_R0.ICT`` →
    ``n926_2022-07-14_acclip-mms.ICT`` to match the IWG1 naming
    convention used by ``calibrate.py``'s glob, while preserving the
    ACCLIP / MMS provenance.

    Some sortie days produce multiple files (e.g.
    ``ACCLIP-MMS-1HZ_WB57_20220721_R0_1.ICT`` /
    ``..._R0_2.ICT``); preserve a numeric suffix in those cases so the
    files don't collide on disk.
    """
    m = _DATE_RE.search(asdc_name)
    if not m:
        # Fall back: keep the original ASDC name unchanged.
        return asdc_name
    base = m.group(1)
    date = f"{base[:4]}-{base[4:6]}-{base[6:8]}"
    suffix = ""
    split_m = _SPLIT_RE.search(asdc_name)
    if split_m:
        suffix = f"-part{split_m.group(1)}"
    return f"n926_{date}{suffix}_acclip-mms.ICT"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_dotenv()
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise SystemExit(
            "EARTHDATA_TOKEN is not set.  Add it to .env or your shell "
            "environment.  Register at https://urs.earthdata.nasa.gov to "
            "obtain a token."
        )

    DEST.mkdir(parents=True, exist_ok=True)
    print("querying CMR for ACCLIP WB-57 MMS-1HZ granules…")
    granules = _list_acclip_mms_1hz()
    print(f"found {len(granules)} MMS-1HZ files")

    new = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    for asdc_name, url in granules:
        dest_name = _canonical_dest_name(asdc_name)
        dest_path = DEST / dest_name
        if dest_path.exists():
            skipped += 1
            continue
        try:
            n_bytes = _download(url, dest_path, token)
        except urllib.error.HTTPError as exc:
            failed.append((dest_name, f"HTTP {exc.code}"))
            continue
        except Exception as exc:
            failed.append((dest_name, repr(exc)))
            continue
        new += 1
        print(f"  + {dest_name} ({n_bytes:>9,} bytes)")

    print(f"\nSummary: {new} new, {skipped} already present, {len(failed)} failed")
    for name, why in failed:
        print(f"  ! {name}: {why}")


if __name__ == "__main__":
    main()
