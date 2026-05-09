"""Generate the fleet overview table and calibration data-sources section.

Reads the live ``hyplan.aircraft`` classes and rewrites the marked
regions of ``docs/api/aircraft.md`` and ``docs/calibration.md``.  The
markers are HTML comments (invisible in rendered Sphinx output)::

    <!-- BEGIN AUTOGEN: fleet_overview -->
    ... auto-generated table ...
    <!-- END AUTOGEN: fleet_overview -->

Run from the repo root after any change to aircraft calibration::

    python -m docs._gen_fleet_tables
"""
from __future__ import annotations

import re
from pathlib import Path

from hyplan import aircraft as ha
from hyplan.aircraft._models import __all__ as ALL_AIRCRAFT
from hyplan.units import ureg


REPO_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_MD = REPO_ROOT / "docs" / "api" / "aircraft.md"
CALIBRATION_MD = REPO_ROOT / "docs" / "calibration.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_qty(q, unit: str, fmt: str = ",.0f") -> str:
    if q is None:
        return "—"
    try:
        return format(q.to(unit).m, fmt)
    except Exception:
        return "—"


def _confidence_summary(conf) -> str:
    """Pick the lowest of climb/cruise/descent/turns confidences."""
    if conf is None:
        return "—"
    vals = [getattr(conf, f) for f in ("climb", "cruise", "descent", "turns")
            if getattr(conf, f, None) is not None]
    if not vals:
        return "—"
    return f"{min(vals):.2f}"


def _sortie_count(sources) -> str:
    """Best-effort sortie count parsed from any ``n=NN`` in source references."""
    for s in sources or []:
        m = re.search(r"\bn=(\d+)", s.reference or "")
        if m:
            return m.group(1)
    return "—"


# ---------------------------------------------------------------------------
# Fleet overview table
# ---------------------------------------------------------------------------

def fleet_table() -> str:
    rows = []
    rows.append("| Class | Airframe | Operator | Tail(s) | Ceiling (ft) | Range (nmi) | Endurance (hr) | Engine | Calibration |")
    rows.append("|---|---|---|---|---:|---:|---:|---|---|")
    for name in ALL_AIRCRAFT:
        cls = getattr(ha, name)
        a = cls()
        ceiling = _fmt_qty(a.service_ceiling, "feet")
        rng = _fmt_qty(a.range, "nautical_mile") if getattr(a, "range", None) else "—"
        end = _fmt_qty(a.endurance, "hour", ".1f") if getattr(a, "endurance", None) else "—"
        cal = a.calibration_status
        n = _sortie_count(a.sources)
        if cal == "calibrated" and n != "—":
            cal_str = f"calibrated · n={n}"
        elif cal == "inferred":
            cal_str = "inferred"
        elif cal == "uncalibrated":
            cal_str = "brochure only"
        else:
            cal_str = cal
        rows.append(
            f"| `{name}` | {a.aircraft_type} | {a.operator} | "
            f"{a.tail_number or '—'} | {ceiling} | {rng} | {end} | "
            f"{a.engine_type or '—'} | {cal_str} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Calibration data-sources grouping
# ---------------------------------------------------------------------------

# Each entry: (list of substrings any of which matches, label, note).
# An aircraft can land in multiple groups if its reference text mentions
# more than one source archive.
SOURCE_GROUPS: list[tuple[tuple[str, ...], str, str]] = [
    (("NASA AFRC IWG1",), "NASA AFRC IWG1",
     "Armstrong Flight Research Center IWG1 / nav telemetry. Used for ER-2."),
    (("ASP archive", "NASA 926+927", "NASA 926", "NASA 927"),
     "NASA Airborne Science Program archive",
     "Public asp-archive.arc.nasa.gov IWG1 archive. Used for G-III, G-V, WB-57, P-3, C-130."),
    (("ACTAMERICA", "DISCOVER-AQ", "KORUS-AQ", "LMOS"), "NASA ICARTT campaign archive",
     "Multi-campaign NASA ICARTT data (ACT-America, DISCOVER-AQ, KORUS-AQ, LMOS). Used for KingAirB200."),
    (("NOAA CSL", "TopDown", "UWFPS", "CalFiDE", "AEROMMA", "AMMBEC", "USOS",
      "ARCPAC", "CalNex", "SENEX", "SONGNEX"), "NOAA CSL ICARTT",
     "ICARTT archive of NOAA Chemical Sciences Laboratory chemistry missions. Used for NOAA_TwinOtter (6 campaigns) and NOAA_WP3D (4 campaigns)."),
    (("FIREX-AQ",), "NASA FIREX-AQ",
     "Firefighter EXperiment for Air Quality 2019. Used for the original NOAA_TwinOtter calibration."),
    (("HIAPER", "RAF-NAV"), "NSF/NCAR HIAPER",
     "NSF/NCAR HIAPER (N677F) DC3 ICARTT NAV files via LaRC ASD. TAS reconstructed via wind triangle. Used for NCAR_GV."),
    (("FAAM Core Data",), "CEDA FAAM",
     "FAAM Core Data Product 1 Hz NetCDFs from CEDA, 2017-2024 ASMM-tagged campaigns. Used for FAAM_BAe146."),
    (("CEDA EUFAR", "EUFAR Transnational"), "CEDA EUFAR",
     "European Facility for Airborne Research transnational-access archive. Used for SAFIRE_ATR42 (7 EUFAR projects)."),
    (("EUREC4A",), "AERIS EUREC4A",
     "French AERIS portal — EUREC4A 2020 SAFIRE ATR-42 L2 1 Hz NetCDFs."),
    (("BAS MASIN",), "CEDA BAS MASIN",
     "British Antarctic Survey MASIN core nav across 5 campaign archives (OFCAP, ACCACIA, ORCHESTRA, IGP, ArcticCyclones)."),
    (("BAHAMAS",), "DLR HALO BAHAMAS",
     "DLR HALO native sensor system. Used for DLR_HALO (HALO-AC3)."),
    (("NERC/ARSF", "NERC ARSF",), "CEDA NERC ARSF",
     "NERC Airborne Research and Survey Facility D-CALM Dornier 228 (ACTIVE 2005-2006 + Eyjafjallajökull 2010). TAS reconstructed via wind triangle."),
    (("AWI Polar", "ACLOUD",), "AWI PANGAEA Polar 5/6",
     "Alfred Wegener Institute Basler BT-67 nav/met. Used for AWI_BaslerBT67."),
]


def calibration_sources_section() -> str:
    """One-row-per-archive markdown table grouping aircraft by source."""
    by_group: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    citations: dict[str, list[tuple[str, str]]] = {}  # label → [(url, doi), ...]
    for substrs, label, note in SOURCE_GROUPS:
        by_group[label] = []
        notes[label] = note
        citations[label] = []

    for name in ALL_AIRCRAFT:
        cls = getattr(ha, name)
        a = cls()
        # An aircraft can land in multiple groups (e.g. SAFIRE_ATR42 in
        # both CEDA EUFAR and AERIS EUREC4A); dedupe per (label, name).
        for s in a.sources or []:
            ref_lower = (s.reference or "").lower()
            for substrs, label, _ in SOURCE_GROUPS:
                if any(sub.lower() in ref_lower for sub in substrs):
                    if name not in by_group[label]:
                        by_group[label].append(name)
                    pair = (getattr(s, "url", "") or "", getattr(s, "doi", "") or "")
                    if any(pair) and pair not in citations[label]:
                        citations[label].append(pair)

    lines = []
    lines.append("| Archive / source | Aircraft | Citation | Notes |")
    lines.append("|---|---|---|---|")
    for _, label, _ in SOURCE_GROUPS:
        used = by_group[label]
        if not used:
            continue
        cls_str = ", ".join(f"`{n}`" for n in used)
        cite_parts: list[str] = []
        for url, doi in citations[label]:
            if doi:
                cite_parts.append(f"[doi:{doi}](https://doi.org/{doi})")
            elif url:
                cite_parts.append(f"[link]({url})")
        cite_str = "<br>".join(cite_parts) or "—"
        lines.append(f"| **{label}** | {cls_str} | {cite_str} | {notes[label]} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Marker-aware in-place rewriter
# ---------------------------------------------------------------------------

def _rewrite_block(path: Path, marker: str, body: str) -> None:
    text = path.read_text()
    begin = f"<!-- BEGIN AUTOGEN: {marker} -->"
    end = f"<!-- END AUTOGEN: {marker} -->"
    if begin not in text or end not in text:
        raise SystemExit(
            f"Markers '{begin}' / '{end}' not found in {path}. "
            f"Add the markers to the file before running this script."
        )
    new_block = f"{begin}\n\n{body}\n\n{end}"
    pat = re.compile(re.escape(begin) + ".*?" + re.escape(end), re.DOTALL)
    text2 = pat.sub(new_block, text)
    if text2 != text:
        path.write_text(text2)
        print(f"  updated {path.relative_to(REPO_ROOT)}")
    else:
        print(f"  unchanged {path.relative_to(REPO_ROOT)}")


def main() -> None:
    print("Generating fleet tables...")
    _rewrite_block(AIRCRAFT_MD, "fleet_overview", fleet_table())
    _rewrite_block(CALIBRATION_MD, "data_sources", calibration_sources_section())
    print("done.")


if __name__ == "__main__":
    main()
