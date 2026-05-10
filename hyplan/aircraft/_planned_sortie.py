"""Parse NASA ER-2 planned-sortie products (KML route + Green Card data card).

A planned ER-2 sortie ships as a paired set of artifacts:

* **KML** — ordered ``<Placemark>`` waypoints with lon/lat (and a trailing
  ``<LineString>`` connecting them).  Source of high-precision lateral
  geometry.
* **Green Card** (a.k.a. ER-2 Mission Data Card) — XLSX or PDF table with
  one *waypoint block* (three text rows) per planned fix, plus inline
  sub-rows for ``.level off``, ``.delay``, and ``.descent pt`` events.
  Source of planned altitudes, leg distances, leg/cumulative times,
  TAS / CAS / GS, headings, and remarks.

XLSX Green Cards parse via :mod:`openpyxl`; PDF Green Cards parse via
:mod:`pdfplumber` (install with ``pip install hyplan[planned]``).  The
public :func:`load_planned_sortie` dispatches on file extension.

The output is a single normalized DataFrame keyed by
``order`` (1, 2, 3, …) with one row per planned event (numbered
waypoints *and* sub-rows).  See :func:`load_planned_sortie` for the
column schema.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
import re

import pandas as pd

_KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


@dataclass
class PlannedSortie:
    """Container for one parsed planned-sortie product.

    Attributes:
        header: Metadata dict[Any, Any] from the Green Card header (mission name,
            aircraft id, takeoff/land times, fuel state, sched duration).
        waypoints: One row per planned event; see
            :func:`load_planned_sortie` for the column schema.
    """
    header: dict[Any, Any]
    waypoints: pd.DataFrame


# ---------------------------------------------------------------------------
# Field-level parsers
# ---------------------------------------------------------------------------

def _strip(value: Any) -> str:
    """Return ``value`` as a stripped string, or empty string if ``None``."""
    if value is None:
        return ""
    return str(value).strip()


def _parse_dms(value: Any) -> float | None:
    """Parse ``"N 38 48.35"`` / ``"W104 42.05"`` to decimal degrees.

    Accepts the Green Card's degrees-and-decimal-minutes format.
    Returns ``None`` for empty or unparseable input.
    """
    s = _strip(value)
    if not s:
        return None
    m = re.match(r"^([NSEW])\s*(\d+)\s+(\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return None
    hemi, deg, minutes = m.group(1), int(m.group(2)), float(m.group(3))
    decimal = deg + minutes / 60.0
    if hemi in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_altitude_ft(value: Any) -> float | None:
    """Parse ``"65000M"`` / ``" 6187M"`` to feet.

    The Green Card prints altitudes with a literal ``"M"`` suffix that
    indicates *MSL* (not meters); the value is in feet.  Verified
    against KCOS field elevation (6187 ft) and FL650 cruise (65000 ft).
    """
    s = _strip(value).rstrip("M").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_signed_int(value: Any) -> int | None:
    """Parse ``"+8C"`` / ``"-57C"`` (temp) or stripped numerics to int."""
    s = _strip(value).rstrip("CMTG").strip()
    if not s or s == "N/A":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_speed_kt(value: Any) -> int | None:
    """Parse ``"398 T"`` / ``"220 C"`` / ``"412 G"`` to integer knots.

    Trailing ``T`` / ``C`` / ``G`` denote True / Calibrated / Ground.
    ``"N/A T"`` returns ``None``.
    """
    s = _strip(value).rstrip("TCG").strip()
    if not s or s == "N/A":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_heading_deg(value: Any) -> int | None:
    """Parse ``"134 T"`` / ``"127 M"`` to integer degrees, dropping the suffix."""
    s = _strip(value).rstrip("TM").strip()
    if not s or s == "N/A":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_bank_deg(value: Any) -> int | None:
    """Parse ``"22  °"`` to integer degrees (bank angle)."""
    s = _strip(value).rstrip("°").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_mach(value: Any) -> float | None:
    """Parse ``".55"`` / ``".70"`` / ``"1.2"`` to float; ``"N/A"`` → ``None``."""
    s = _strip(value)
    if not s or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_wind(value: Any) -> tuple[int | None, int | None]:
    """Parse ``"260/004"`` to ``(direction_deg, speed_kt)``.

    Returns ``(None, None)`` for empty cells.
    """
    s = _strip(value)
    m = re.match(r"^(\d+)\s*/\s*(\d+)\s*$", s)
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))


def _parse_distance_nmi(value: Any) -> int | None:
    """Parse ``"   10"`` / ``"  121"`` (leading-space integer) to int."""
    s = _strip(value)
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_leg_time_min(value: Any) -> float | None:
    """Parse a leg-time cell to minutes (float).

    Leg time is always sub-hour: ``"+07.0"``, ``"+11.2"``.
    """
    s = _strip(value).lstrip("+")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_total_time_min(value: Any) -> float | None:
    """Parse a cumulative-time cell to minutes (float).

    The Green Card switches format at 60 min: ``"+47.3"`` below the
    hour, ``"01+20.0"`` / ``"06+33.6"`` above.
    """
    s = _strip(value).lstrip("+")
    if not s:
        return None
    if "+" in s:
        hours_str, mins_str = s.split("+", 1)
        try:
            return int(hours_str) * 60.0 + float(mins_str)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_clock_time(value: Any) -> str | None:
    """Return the clock time string verbatim (``"15:30.0"``)."""
    s = _strip(value)
    return s or None


def _parse_fuel_lb(value: Any) -> int | None:
    """Parse a fuel-cell to integer pounds; empty → ``None``."""
    s = _strip(value)
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_sched_duration_hours(value: Any) -> float | None:
    """Parse ``"06+33+35"`` (HH+MM+SS) to fractional hours."""
    s = _strip(value)
    parts = s.split("+")
    if len(parts) != 3:
        return None
    try:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return h + m / 60.0 + sec / 3600.0


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------

def parse_kml(path: Path | str) -> list[dict[Any, Any]]:
    """Read an ER-2 planned KML route and return ordered placemarks.

    Each placemark becomes ``{"name", "description", "lon", "lat", "alt_m"}``.
    The trailing connector ``<LineString>`` placemark (no ``<name>``) is
    skipped.  Altitudes from ``clampToGround`` placemarks come back as
    ``0.0``; the Green Card is the source of planned altitudes.
    """
    tree = ET.parse(Path(path))
    root = tree.getroot()
    out: list[dict[Any, Any]] = []
    for pm in root.iter(f"{{{_KML_NS['k']}}}Placemark"):
        name_el = pm.find("k:name", _KML_NS)
        coord_el = pm.find(".//k:Point/k:coordinates", _KML_NS)
        if name_el is None or coord_el is None:
            # The trailing route-connector LineString placemark has no
            # name and no Point — skip it.
            continue
        coords = (coord_el.text or "").strip().split(",")
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        alt_m = float(coords[2]) if len(coords) >= 3 else 0.0
        desc_el = pm.find("k:description", _KML_NS)
        out.append({
            "name": (name_el.text or "").strip(),
            "description": (desc_el.text or "").strip() if desc_el is not None else "",
            "lon": lon,
            "lat": lat,
            "alt_m": alt_m,
        })
    return out


# ---------------------------------------------------------------------------
# Green Card — XLSX
# ---------------------------------------------------------------------------

# The Green Card lays out one waypoint per 3-row block, starting after the
# (variable-length) header section.  The header ends and the data table
# begins on the row whose column A reads "WP#".  Column positions are
# fixed across all NASA ER-2 cards observed; see [PAIRS.md] for the
# samples this parser was fitted against.
_COL_WP = 0           # A
_COL_FIX = 1          # B (fix name on row 1, description on row 2)
_COL_VOR_TAC = 2      # C (VOR freq on row 1, TAC channel on row 2)
_COL_ALT_LL = 3       # D (altitude row 1, latitude row 2, longitude row 3)
_COL_TC_MC_BK = 4     # E (TC row 1, MC row 2, Bank row 3)
_COL_IM_TEMP_WIND = 5  # F (Mach row 1, Temp row 2, Wind row 3)
_COL_SPEED = 6        # G (TAS row 1, CAS row 2, GS row 3)
_COL_DIST = 7         # H (leg row 1, total row 2)
_COL_TIME = 8         # I (leg row 1, total row 2, clock row 3)
_COL_FUEL = 9         # J (leg row 1, total row 2, FF row 3)
_COL_REMARKS = 12     # M (row 1)


def _find_header_row(ws: Any) -> int:
    """Return the 1-indexed row containing ``"WP#"`` in column A."""
    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row and _strip(row[_COL_WP]) == "WP#":
            return row_idx
    raise ValueError("Green Card XLSX is missing the 'WP#' header row")


def _find_data_start_row(ws: Any, header_row: int) -> int:
    """Return the 1-indexed row of the first waypoint block.

    The header spans 4 rows ("WP# / DTD#" / "Fix/Point Description" /
    blank / blank) so the first data row is normally ``header_row + 4``,
    but the card's header sub-rows vary slightly across templates;
    walk forward until we hit a row whose column A is a numeric WP#.
    """
    for row_idx in range(header_row + 1, ws.max_row + 1):
        cell = ws.cell(row=row_idx, column=_COL_WP + 1).value
        try:
            int(_strip(cell))
            return row_idx
        except ValueError:
            continue
    raise ValueError("Green Card XLSX has no numeric WP# rows after header")


def _read_header(ws: Any) -> dict[Any, Any]:
    """Extract the Green Card header block.

    Pulls the labeled fields above the ``WP#`` row.  Field names map
    directly to the Green Card layout; missing fields come back as
    ``None``.
    """
    header_row = _find_header_row(ws)
    text_blob = "\n".join(
        " ".join(_strip(c) for c in row if c is not None)
        for row in ws.iter_rows(min_row=1, max_row=header_row - 1, values_only=True)
    )
    aircraft_match = re.search(r"NASA\s*\d{3}", text_blob)
    mission_match = re.search(r"ER-2 MISSION DATA CARD\s+(\S.*?)$", text_blob, re.MULTILINE)
    sched_to_match = re.search(r"SCHED T/O.*?(\d{1,2}:\d{2}:\d{2})\s*Z?", text_blob, re.DOTALL)
    takeoff_match = re.search(r"Takeoff Time\s*\(?Z\)?:\s*(\d{1,2}:\d{2}:\d{2})", text_blob)
    land_match = re.search(r"Land Time\s*\(?Z\)?:\s*(\d{1,2}:\d{2}:\d{2})", text_blob)
    sched_dur_match = re.search(r"Sched Duration\s*:\s*(\d{2}\+\d{2}\+\d{2})", text_blob)
    fuel_load_match = re.search(r"Fuel Load:\s*(\d+)", text_blob)
    fuel_used_match = re.search(r"Fuel Used:\s*(\d+)", text_blob)

    return {
        "aircraft_id": aircraft_match.group(0).replace(" ", "") if aircraft_match else None,
        "mission_name": mission_match.group(1).strip() if mission_match else None,
        "sched_takeoff_z": sched_to_match.group(1) if sched_to_match else None,
        "takeoff_time_z": takeoff_match.group(1) if takeoff_match else None,
        "land_time_z": land_match.group(1) if land_match else None,
        "sched_duration_hours": (
            _parse_sched_duration_hours(sched_dur_match.group(1))
            if sched_dur_match else None
        ),
        "fuel_load_lb": int(fuel_load_match.group(1)) if fuel_load_match else None,
        "fuel_used_lb": int(fuel_used_match.group(1)) if fuel_used_match else None,
    }


def _classify_kind(fix_name: str) -> str:
    """Classify a Green Card row by its fix-name token.

    Sub-rows start with ``"."`` (``.level off``, ``.delay``,
    ``.descent pt``); their kind is the lowercased label with
    spaces collapsed to ``"_"``.  Numbered waypoints get
    ``"waypoint"``.
    """
    s = fix_name.strip()
    if s.startswith("."):
        return s.lstrip(".").strip().lower().replace(" ", "_")
    return "waypoint"


def _parse_waypoint_block(rows: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]) -> dict[Any, Any]:
    """Parse three consecutive rows of Green Card data into one record."""
    r1, r2, r3 = rows

    wp_str = _strip(r1[_COL_WP])
    try:
        wp_num: int | None = int(wp_str)
    except ValueError:
        wp_num = None

    fix_name = _strip(r1[_COL_FIX])
    description = _strip(r2[_COL_FIX])
    vor_freq = _strip(r1[_COL_VOR_TAC]) or None
    tac_chan = _strip(r2[_COL_VOR_TAC]) or None
    wind_dir, wind_speed = _parse_wind(r3[_COL_IM_TEMP_WIND])

    return {
        "wp_num": wp_num,
        "fix_name": fix_name,
        "description": description,
        "kind": _classify_kind(fix_name),
        "vor_freq": vor_freq,
        "tac_channel": tac_chan,
        "lat_deg": _parse_dms(r2[_COL_ALT_LL]),
        "lon_deg": _parse_dms(r3[_COL_ALT_LL]),
        "altitude_ft": _parse_altitude_ft(r1[_COL_ALT_LL]),
        "true_course_deg": _parse_heading_deg(r1[_COL_TC_MC_BK]),
        "magnetic_course_deg": _parse_heading_deg(r2[_COL_TC_MC_BK]),
        "bank_deg": _parse_bank_deg(r3[_COL_TC_MC_BK]),
        "mach": _parse_mach(r1[_COL_IM_TEMP_WIND]),
        "temp_c": _parse_signed_int(r2[_COL_IM_TEMP_WIND]),
        "wind_dir_deg": wind_dir,
        "wind_speed_kt": wind_speed,
        "tas_kt": _parse_speed_kt(r1[_COL_SPEED]),
        "cas_kt": _parse_speed_kt(r2[_COL_SPEED]),
        "gs_kt": _parse_speed_kt(r3[_COL_SPEED]),
        "leg_distance_nmi": _parse_distance_nmi(r1[_COL_DIST]),
        "cumulative_distance_nmi": _parse_distance_nmi(r2[_COL_DIST]),
        "leg_time_min": _parse_leg_time_min(r1[_COL_TIME]),
        "cumulative_time_min": _parse_total_time_min(r2[_COL_TIME]),
        "clock_time": _parse_clock_time(r3[_COL_TIME]),
        "leg_fuel_lb": _parse_fuel_lb(r1[_COL_FUEL]),
        "cumulative_fuel_lb": _parse_fuel_lb(r2[_COL_FUEL]),
        "fuel_flow_pph": _parse_fuel_lb(r3[_COL_FUEL]),
        "remarks": _strip(r1[_COL_REMARKS]) or "",
    }


def parse_green_card_xlsx(path: Path | str) -> PlannedSortie:
    """Parse one ER-2 Green Card XLSX into header + waypoint table.

    Args:
        path: XLSX file path.

    Returns:
        :class:`PlannedSortie` with the extracted header dict[Any, Any] and a
        DataFrame of one row per Green Card event (numbered waypoint
        or sub-row).  See :func:`load_planned_sortie` for the column
        schema.

    Raises:
        ImportError: When ``openpyxl`` is not installed.  Install via
            ``pip install hyplan[planned]``.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "Green Card XLSX parsing requires openpyxl.  Install via "
            "`pip install openpyxl` or `pip install hyplan[planned]`."
        ) from exc

    wb = openpyxl.load_workbook(Path(path), data_only=True)
    ws = wb.active
    header = _read_header(ws)
    data_start = _find_data_start_row(ws, _find_header_row(ws))

    rows: list[tuple[Any, ...]] = list(ws.iter_rows(
        min_row=data_start, max_row=ws.max_row, values_only=True,
    ))

    records: list[dict[Any, Any]] = []
    for i in range(0, len(rows), 3):
        block = rows[i:i + 3]
        if len(block) < 3:
            break  # trailing blank rows
        if all(c is None for c in block[0]):
            break  # padding row at end
        records.append(_parse_waypoint_block((block[0], block[1], block[2])))

    df = pd.DataFrame.from_records(records)
    df.insert(0, "order", range(1, len(df) + 1))
    return PlannedSortie(header=header, waypoints=df)


# ---------------------------------------------------------------------------
# Green Card — PDF
# ---------------------------------------------------------------------------
#
# The PDF Green Card has the same logical layout as the XLSX (one
# 3-row block per planned event, identical column groupings) but each
# logical block is rendered as a single PDF table row with multi-line
# cell values separated by '\n'.  Row 0 of each PDF row maps to the
# (WP#, Fix/Point, VOR, Altitude, Description, TAC, Latitude, Longitude)
# block from XLSX columns A-D; columns E-M (course/wind/speed/dist/
# time/fuel/ATA/remarks) map straightforwardly.

# Column positions in the *normalized* PDF row (see _normalize_pdf_row):
# the first PDF page emits a 10-column table whose column index 1 is
# always None (a layout artifact of the merged-cell header above the
# data); subsequent pages emit a 9-column table without that
# placeholder.  Normalization drops the placeholder so all waypoint
# rows share a 9-column layout.
_PDF_COL_FIX_BLOCK = 0
_PDF_COL_TC_MC_BANK = 1
_PDF_COL_IM_TEMP_WIND = 2
_PDF_COL_SPEED = 3
_PDF_COL_DIST = 4
_PDF_COL_TIME = 5
_PDF_COL_FUEL = 6
_PDF_COL_ATA = 7
_PDF_COL_REMARKS = 8


def _normalize_pdf_row(row: list[Any]) -> list[Any]:
    """Strip the empty placeholder col 1 used by page-1's merged-header layout."""
    if len(row) == 10 and row[1] is None:
        return [row[0]] + list(row[2:])
    return list(row)

# Line 1 of col 0: "[wp#] fix [vor] alt M".  fix may contain spaces
# (".level off", ".descent pt").  vor is a decimal frequency
# (e.g. 112.50) when present.  alt is digits ending in literal "M".
_PDF_LINE1_RE = re.compile(
    r"^\s*(?:(\d+)\s+)?(.+?)\s+(?:(\d+\.\d+)\s+)?(\d+)M\s*$"
)
# Line 2: "[description and/or TAC] N DD MM.MM".
_PDF_LINE2_LAT_RE = re.compile(
    r"^(.*?)\s*([NS]\s+\d+\s+\d+(?:\.\d+)?)\s*$"
)


def _split_lines(cell: object) -> list[str]:
    """Split a (possibly None) PDF cell into a list of stripped lines."""
    if cell is None:
        return []
    return [ln.strip() for ln in str(cell).split("\n")]


def _parse_pdf_fix_block(cell: str) -> dict[Any, Any]:
    """Parse the (WP#/Fix/VOR/Alt; Description/TAC/Lat; Lon) PDF cell.

    Returns a dict with keys ``wp_num``, ``fix_name``, ``description``,
    ``vor_freq``, ``tac_channel``, ``lat_deg``, ``lon_deg``,
    ``altitude_ft``.  Missing fields come back as ``None``.
    """
    lines = _split_lines(cell)
    while len(lines) < 3:
        lines.append("")

    line1, line2, line3 = lines[0], lines[1], lines[2]

    # Line 1: WP# / Fix / VOR / Altitude.
    m1 = _PDF_LINE1_RE.match(line1)
    if m1:
        wp_str, fix_name, vor_str, alt_str = m1.groups()
        wp_num: int | None = int(wp_str) if wp_str else None
        vor_freq: str | None = vor_str if vor_str else None
        altitude_ft: float | None = float(alt_str) if alt_str else None
    else:
        wp_num = None
        fix_name = line1
        vor_freq = None
        altitude_ft = None

    # Line 2: optional description and/or TAC, then latitude.
    m2 = _PDF_LINE2_LAT_RE.match(line2)
    if m2:
        prefix, lat_str = m2.group(1).strip(), m2.group(2).strip()
        lat_deg = _parse_dms(lat_str)
    else:
        prefix = line2
        lat_deg = None

    # Inside ``prefix``: trailing token like "072X" / "114X" / "087Y" is
    # the TAC channel.  Anything else is the description.
    description = ""
    tac_channel: str | None = None
    if prefix:
        tokens = prefix.split()
        if tokens and re.match(r"^\d{3}[A-Z]$", tokens[-1]):
            tac_channel = tokens[-1]
            tokens = tokens[:-1]
        description = " ".join(tokens)

    # Line 3: longitude.
    lon_deg = _parse_dms(line3) if line3 else None

    return {
        "wp_num": wp_num,
        "fix_name": fix_name,
        "description": description,
        "vor_freq": vor_freq,
        "tac_channel": tac_channel,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "altitude_ft": altitude_ft,
    }


def _parse_pdf_waypoint_row(row: list[Any]) -> dict[Any, Any]:
    """Convert one pdfplumber table row into a normalized waypoint record."""
    fix_block = _parse_pdf_fix_block(row[_PDF_COL_FIX_BLOCK])

    tc_lines = _split_lines(row[_PDF_COL_TC_MC_BANK])
    iw_lines = _split_lines(row[_PDF_COL_IM_TEMP_WIND])
    sp_lines = _split_lines(row[_PDF_COL_SPEED])
    di_lines = _split_lines(row[_PDF_COL_DIST])
    ti_lines = _split_lines(row[_PDF_COL_TIME])
    fu_lines = _split_lines(row[_PDF_COL_FUEL])

    for _lst, _n in (
        (tc_lines, 3), (iw_lines, 3), (sp_lines, 3),
        (di_lines, 2), (ti_lines, 3), (fu_lines, 3),
    ):
        while len(_lst) < _n:
            _lst.append("")

    wind_dir, wind_speed = _parse_wind(iw_lines[2])

    return {
        "wp_num": fix_block["wp_num"],
        "fix_name": fix_block["fix_name"],
        "description": fix_block["description"],
        "kind": _classify_kind(fix_block["fix_name"]),
        "vor_freq": fix_block["vor_freq"],
        "tac_channel": fix_block["tac_channel"],
        "lat_deg": fix_block["lat_deg"],
        "lon_deg": fix_block["lon_deg"],
        "altitude_ft": fix_block["altitude_ft"],
        "true_course_deg": _parse_heading_deg(tc_lines[0]),
        "magnetic_course_deg": _parse_heading_deg(tc_lines[1]),
        "bank_deg": _parse_bank_deg(tc_lines[2]),
        "mach": _parse_mach(iw_lines[0]),
        "temp_c": _parse_signed_int(iw_lines[1]),
        "wind_dir_deg": wind_dir,
        "wind_speed_kt": wind_speed,
        "tas_kt": _parse_speed_kt(sp_lines[0]),
        "cas_kt": _parse_speed_kt(sp_lines[1]),
        "gs_kt": _parse_speed_kt(sp_lines[2]),
        "leg_distance_nmi": _parse_distance_nmi(di_lines[0]),
        "cumulative_distance_nmi": _parse_distance_nmi(di_lines[1]),
        "leg_time_min": _parse_leg_time_min(ti_lines[0]),
        "cumulative_time_min": _parse_total_time_min(ti_lines[1]),
        "clock_time": _parse_clock_time(ti_lines[2]),
        "leg_fuel_lb": _parse_fuel_lb(fu_lines[0]),
        "cumulative_fuel_lb": _parse_fuel_lb(fu_lines[1]),
        "fuel_flow_pph": _parse_fuel_lb(fu_lines[2]),
        "remarks": _strip(row[_PDF_COL_REMARKS]) or "",
    }


def _read_pdf_header(text: str) -> dict[Any, Any]:
    """Extract the Green Card header from the page-1 raw text blob."""
    aircraft_match = re.search(r"NASA\s*\d{3}", text)
    # Mission name (e.g., "GEMX") is the short token between
    # "ER-2 MISSION DATA CARD" and "PILOT".  When the PDF has no
    # mission tag (some products go straight from title to PILOT),
    # the match fails and mission_name is None.
    mission_match = re.search(
        r"ER-2 MISSION DATA CARD\s*[\r\n]+\s*(\S{2,30})\s*[\r\n]+\s*PILOT",
        text,
    )
    sched_to_match = re.search(
        r"(\d{1,2}:\d{2}:\d{2})\s*Z", text,
    )
    takeoff_match = re.search(
        r"Takeoff Time\s*\(?Z\)?:\s*(\d{1,2}:\d{2}:\d{2})", text,
    )
    land_match = re.search(
        r"Land Time\s*\(?Z\)?:\s*(\d{1,2}:\d{2}:\d{2})", text,
    )
    sched_dur_match = re.search(
        r"Sched Duration\s*:\s*(\d{2}\+\d{2}\+\d{2})", text,
    )
    fuel_load_match = re.search(r"Fuel Load:\s*(\d+)", text)
    fuel_used_match = re.search(r"Fuel Used:\s*(\d+)", text)

    return {
        "aircraft_id": aircraft_match.group(0).replace(" ", "")
        if aircraft_match else None,
        "mission_name": mission_match.group(1).strip()
        if mission_match else None,
        "sched_takeoff_z": sched_to_match.group(1) if sched_to_match else None,
        "takeoff_time_z": takeoff_match.group(1) if takeoff_match else None,
        "land_time_z": land_match.group(1) if land_match else None,
        "sched_duration_hours": (
            _parse_sched_duration_hours(sched_dur_match.group(1))
            if sched_dur_match else None
        ),
        "fuel_load_lb": int(fuel_load_match.group(1)) if fuel_load_match else None,
        "fuel_used_lb": int(fuel_used_match.group(1)) if fuel_used_match else None,
    }


def parse_green_card_pdf(path: Path | str) -> PlannedSortie:
    """Parse one ER-2 Green Card PDF into header + waypoint table.

    Uses :mod:`pdfplumber`'s table extraction to pull each waypoint
    block as a single row whose cells carry multi-line values.  The
    layout matches the XLSX format byte-for-byte; only the cell-shape
    differs (PDF: 1 row × multi-line cells; XLSX: 3 rows × 1-line cells).

    Args:
        path: PDF file path.

    Returns:
        :class:`PlannedSortie` with the same schema as
        :func:`parse_green_card_xlsx`.

    Raises:
        ImportError: When ``pdfplumber`` is not installed.  Install
            via ``pip install hyplan[planned]``.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "Green Card PDF parsing requires pdfplumber.  Install via "
            "`pip install pdfplumber` or `pip install hyplan[planned]`."
        ) from exc

    records: list[dict[Any, Any]] = []
    header_text = ""
    with pdfplumber.open(Path(path)) as pdf:
        if not pdf.pages:
            raise ValueError(f"Green Card PDF is empty: {path}")
        header_text = pdf.pages[0].extract_text() or ""

        for pi, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for tab in tables:
                for row in tab:
                    if not row:
                        continue
                    norm = _normalize_pdf_row(row)
                    fix_cell = norm[_PDF_COL_FIX_BLOCK]
                    if not fix_cell:
                        continue
                    line1 = str(fix_cell).split("\n", 1)[0]
                    # Skip rows that are part of the page header (no
                    # altitude-with-trailing-M token on line 1).
                    if not _PDF_LINE1_RE.match(line1):
                        continue
                    records.append(_parse_pdf_waypoint_row(norm))

    header = _read_pdf_header(header_text)
    df = pd.DataFrame.from_records(records)
    df.insert(0, "order", range(1, len(df) + 1))
    return PlannedSortie(header=header, waypoints=df)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_planned_sortie(
    kml_path: Path | str,
    gc_path: Path | str,
) -> PlannedSortie:
    """Load and join a planned ER-2 sortie from its KML + Green Card pair.

    The KML provides high-precision lateral coordinates for numbered
    waypoints; the Green Card provides every planned attribute.  When
    both agree on a waypoint's location to within ~0.001° the parser
    upgrades the Green Card's degrees-and-decimal-minutes lat/lon to
    the KML's decimal-degrees value.  When they disagree, the Green
    Card values are kept and a ``kml_disagreement_deg`` column flags
    the discrepancy in degrees of great-circle separation.

    Sub-rows (``.level off`` / ``.delay`` / ``.descent pt``) inherit
    the surrounding waypoint's lat/lon from the Green Card; the KML
    has no corresponding placemarks for them.

    Format dispatch is by Green Card extension: ``.xlsx`` uses
    :mod:`openpyxl`; ``.pdf`` uses :mod:`pdfplumber` (install via
    ``pip install hyplan[planned]``).

    Returns:
        :class:`PlannedSortie` with:

        * ``header``: dict[Any, Any] with ``aircraft_id``, ``mission_name``,
          ``sched_takeoff_z``, ``takeoff_time_z``, ``land_time_z``,
          ``sched_duration_hours``, ``fuel_load_lb``, ``fuel_used_lb``.
        * ``waypoints``: DataFrame with columns ``order``, ``wp_num``,
          ``fix_name``, ``description``, ``kind``, ``vor_freq``,
          ``tac_channel``, ``lat_deg``, ``lon_deg``, ``altitude_ft``,
          ``true_course_deg``, ``magnetic_course_deg``, ``bank_deg``,
          ``mach``, ``temp_c``, ``wind_dir_deg``, ``wind_speed_kt``,
          ``tas_kt``, ``cas_kt``, ``gs_kt``, ``leg_distance_nmi``,
          ``cumulative_distance_nmi``, ``leg_time_min``,
          ``cumulative_time_min``, ``clock_time``, ``leg_fuel_lb``,
          ``cumulative_fuel_lb``, ``fuel_flow_pph``, ``remarks``.
    """
    gc_path = Path(gc_path)
    suffix = gc_path.suffix.lower()
    if suffix == ".xlsx":
        sortie = parse_green_card_xlsx(gc_path)
    elif suffix == ".pdf":
        sortie = parse_green_card_pdf(gc_path)
    else:
        raise ValueError(f"Unsupported Green Card extension: {suffix!r}")

    placemarks = parse_kml(kml_path)

    # Join: match each numbered Green Card row to a KML placemark by
    # *fix name* (KML's <description>) using a first-unused queue.
    # Same-named fixes that appear multiple times in a sortie (e.g.,
    # PUB/R253012 in both climb-out and descent-in) match in order.
    # KML placemarks without a corresponding GC row, or vice versa,
    # are silently skipped — the GC keeps its DMS lat/lon and
    # `kml_disagreement_deg` stays NaN.
    from collections import defaultdict, deque

    df = sortie.waypoints.copy()
    df["kml_disagreement_deg"] = float("nan")

    kml_queues: dict[Any, Any] = defaultdict(deque)
    for pm in placemarks:
        kml_queues[pm["description"]].append(pm)

    for row_idx in df[df["kind"] == "waypoint"].index:
        fix_name = df.at[row_idx, "fix_name"]
        queue = kml_queues.get(fix_name)
        if not queue:
            continue
        pm = queue.popleft()
        gc_lat = df.at[row_idx, "lat_deg"]
        gc_lon = df.at[row_idx, "lon_deg"]
        if gc_lat is None or gc_lon is None:
            df.at[row_idx, "lat_deg"] = pm["lat"]
            df.at[row_idx, "lon_deg"] = pm["lon"]
            continue
        disagree = max(abs(gc_lat - pm["lat"]), abs(gc_lon - pm["lon"]))
        df.at[row_idx, "kml_disagreement_deg"] = disagree
        if disagree < 0.001:
            df.at[row_idx, "lat_deg"] = pm["lat"]
            df.at[row_idx, "lon_deg"] = pm["lon"]

    return PlannedSortie(header=sortie.header, waypoints=df)
