"""NOAA CSL field-project ICARTT data fetcher.

The NOAA Chemical Sciences Laboratory archive at
``https://csl.noaa.gov/groups/csl7/measurements/<MISSION>/<PLATFORM>/``
hosts ICARTT files behind a cookie-based data-policy agreement.

Workflow:

1. GET the ``DataDownload/`` page to receive the agreement form.
2. POST the agreement (``dataAgreementOK=yes``) to ``checkCookie.php``
   with the ``platform`` / ``mission`` / ``page`` hidden fields.
   Server sets a per-mission ``DataAgreement`` cookie.
3. With the cookie, fetch the search page and parse out
   ``../../data/<...>/<filename>.ict`` hrefs.
4. Resolve each href against the search page URL and download.

Used by per-mission drivers (e.g. ``fetch_noaa_aeromma_twinotter.py``)
where the calibration target is a NOAA-CSL hosted campaign whose
nav data isn't already in the AFRC IWG1 archive.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import re

USER_AGENT = "hyplan-calibration-fetcher/1.1 (mailto:ryan.p.pavlick@nasa.gov)"

CSL_ROOT = "https://csl.noaa.gov"


def _opener(jar: CookieJar) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar)
    )


def _http_get(opener, url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_post(opener, url: str, fields: dict, referer: str,
               timeout: float = 60.0) -> str:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": referer,
        },
    )
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _accept_data_policy(
    opener,
    mission: str,
    platform: str,
    download_url: str,
) -> None:
    """POST the data-policy agreement so the search page is unlocked."""
    page_path = urllib.parse.urlparse(download_url).path
    fields = {
        "dataAgreementOK": "yes",
        "platform": platform,
        "mission": mission,
        "page": page_path,
        "Submit": "Submit",
    }
    cookie_url = f"{CSL_ROOT}/groups/csl7/measurements/checkCookie.php"
    _http_post(opener, cookie_url, fields, referer=download_url)


def list_files(
    mission: str,
    platform: str,
    download_subpath: str = "DataDownload",
    search_page: str = "DataDownloadAllSearch.php",
    filename_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[Tuple[str, str]], CookieJar]:
    """List downloadable ICARTT files in a NOAA CSL mission archive.

    Returns ``([(filename, full_url), ...], cookie_jar)``.  The cookie
    jar must be reused for the actual file downloads.
    """
    jar = CookieJar()
    opener = _opener(jar)

    download_url = (
        f"{CSL_ROOT}/groups/csl7/measurements/{mission}/{platform}/"
        f"{download_subpath}/"
    )
    # 1) Fetch the agreement page to receive the session
    _http_get(opener, download_url)
    # 2) Submit the agreement
    _accept_data_policy(opener, mission, platform, download_url)
    # 3) Fetch the search page now that the cookie is set
    search_url = f"{download_url}{search_page}"
    html = _http_get(opener, search_url)

    # ICARTT hrefs are relative ../../data/<platform>/...
    hrefs = re.findall(r'href="(\.\./\.\./[^"]+\.ict)"', html, re.I)
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href in hrefs:
        full = urllib.parse.urljoin(search_url, href)
        name = Path(urllib.parse.urlparse(full).path).name
        if name in seen:
            continue
        if filename_filter is not None and not filename_filter(name):
            continue
        seen.add(name)
        out.append((name, full))
    return out, jar


def fetch_files(
    mission: str,
    platform: str,
    out_dir: str | Path,
    filename_filter: Optional[Callable[[str], bool]] = None,
    sleep_between: float = 0.5,
    replace: bool = False,
    download_subpath: str = "DataDownload",
    search_page: str = "DataDownloadAllSearch.php",
    label: Optional[str] = None,
) -> List[Path]:
    """Download every matching file, idempotent."""
    label = label or f"{mission}/{platform}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    listing, jar = list_files(
        mission, platform,
        download_subpath=download_subpath,
        search_page=search_page,
        filename_filter=filename_filter,
    )
    if not listing:
        print(f"  [{label}] no files matched")
        return []
    print(f"  [{label}] {len(listing)} files matched")

    opener = _opener(jar)
    landed: List[Path] = []
    for i, (filename, url) in enumerate(listing, 1):
        target = out_path / filename
        if target.exists() and not replace:
            landed.append(target)
            continue
        try:
            data = _http_get(opener, url, timeout=300.0).encode("utf-8", errors="replace")
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
    m = re.search(r"_(20\d{6})_", name)
    return m.group(1) if m else None
