"""Generic NASA LaRC Airborne Science Data archive fetcher.

The LaRC ASD ArcView pages at
``https://www-air.larc.nasa.gov/cgi-bin/ArcView/<mission>?<button>=1``
expose downloadable files behind ``/cgi-bin/enzFile?<token>`` URLs
(no authentication; public).  Each token is a 35-char auth prefix
plus the hex-encoded server-side path; the prefix gives bearer
access to every file in the listing.

This helper lists files for a (mission, button) pair, applies a
filename filter (typically ``"MetNav"`` for the modern missions
that publish dedicated nav products), and downloads idempotently.

Used by per-aircraft calibration directories under
``notebooks/calibration/<aircraft>/_fetch_larc_asd.py`` and the
top-level ``notebooks/calibration/fetch_larc_asd.py`` driver.
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ARCVIEW_ROOT = "https://www-air.larc.nasa.gov/cgi-bin/ArcView"
ENZFILE_ROOT = "https://www-air.larc.nasa.gov/cgi-bin/enzFile"

USER_AGENT = "hyplan-calibration-fetcher/1.1 (mailto:ryan.p.pavlick@nasa.gov)"


def _http_get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_files(
    mission: str,
    button: str,
    filename_filter: Optional[Callable[[str], bool]] = None,
    extensions: Tuple[str, ...] = (".ICT", ".ict"),
) -> List[Tuple[str, str]]:
    """Return ``[(filename, enzFile_token), ...]`` matching the filter.

    Args:
        mission: ArcView mission slug (e.g. ``"arcsix"``,
            ``"dc3-seac4rs"``).
        button: Aircraft button name as it appears in the ArcView
            form (e.g. ``"P3B"``, ``"G3"``, ``"ER2"``, ``"GV"``).
            Probe the mission landing page to find available buttons.
        filename_filter: Predicate over the bare filename.  Default:
            include every file with a known extension.
        extensions: Tuple of accepted file extensions (case-sensitive).
    """
    url = f"{ARCVIEW_ROOT}/{mission}?{button}=1"
    html = _http_get(url).decode("utf-8", errors="replace")

    pattern = re.compile(
        r'href="/cgi-bin/enzFile\?([0-9A-Fa-f]+)"[^>]*>([^<]+)</a>'
    )
    out: List[Tuple[str, str]] = []
    for m in pattern.finditer(html):
        token, anchor = m.group(1), m.group(2).strip()
        if not any(anchor.endswith(ext) for ext in extensions):
            continue
        if filename_filter is not None and not filename_filter(anchor):
            continue
        out.append((anchor, token))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for f, t in out:
        if f in seen:
            continue
        seen.add(f)
        deduped.append((f, t))
    return deduped


def fetch_files(
    mission: str,
    button: str,
    out_dir: str | Path,
    filename_filter: Optional[Callable[[str], bool]] = None,
    sleep_between: float = 0.5,
    replace: bool = False,
    extensions: Tuple[str, ...] = (".ICT", ".ict"),
    label: Optional[str] = None,
) -> List[Path]:
    """Download every matching file from a LaRC ASD listing.

    Idempotent: skips files already present unless ``replace=True``.
    Returns the list of files (downloaded or pre-existing).
    """
    label = label or f"{mission}/{button}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    listing = list_files(mission, button, filename_filter=filename_filter,
                          extensions=extensions)
    if not listing:
        print(f"  [{label}] no files matched")
        return []

    print(f"  [{label}] {len(listing)} files matched")

    landed: List[Path] = []
    for i, (filename, token) in enumerate(listing, 1):
        target = out_path / filename
        if target.exists() and not replace:
            landed.append(target)
            continue
        url = f"{ENZFILE_ROOT}?{token}"
        try:
            data = _http_get(url, timeout=300.0)
        except Exception as exc:
            print(f"    [{i:>3}/{len(listing)}] FAILED: {filename}  ({exc})")
            continue
        target.write_bytes(data)
        landed.append(target)
        print(
            f"    [{i:>3}/{len(listing)}] {filename}  "
            f"({len(data) / 1024:.0f} KiB)"
        )
        if sleep_between > 0:
            time.sleep(sleep_between)

    return landed


def date_from_filename(name: str) -> Optional[str]:
    """Extract a YYYYMMDD substring from an ICARTT filename, or None."""
    m = re.search(r"_(20\d{6})_", name)
    return m.group(1) if m else None
