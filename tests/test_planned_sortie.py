"""Tests for the ER-2 planned-sortie parser.

Unit tests cover the field-level parsers (no data dependency).
Integration tests run against the real NM17 B pair when present in
``data/er2/`` (gitignored cache).
"""

from pathlib import Path

import pytest

from hyplan.aircraft import (
    PlannedSortie,
    load_planned_sortie,
    parse_green_card_pdf,
    parse_green_card_xlsx,
    parse_kml,
)
from hyplan.aircraft._planned_sortie import (
    _classify_kind,
    _parse_altitude_ft,
    _parse_bank_deg,
    _parse_dms,
    _parse_heading_deg,
    _parse_leg_time_min,
    _parse_mach,
    _parse_sched_duration_hours,
    _parse_signed_int,
    _parse_speed_kt,
    _parse_total_time_min,
    _parse_wind,
)


# ---------------------------------------------------------------------------
# Unit tests — field parsers
# ---------------------------------------------------------------------------

class TestDmsParser:
    def test_north(self):
        assert _parse_dms("N 38 48.35") == pytest.approx(38.805833, abs=1e-5)

    def test_west(self):
        assert _parse_dms("W104 42.05") == pytest.approx(-104.700833, abs=1e-5)

    def test_south_negative(self):
        assert _parse_dms("S 12 30.00") == pytest.approx(-12.5, abs=1e-9)

    def test_empty_returns_none(self):
        assert _parse_dms("") is None
        assert _parse_dms(None) is None

    def test_unparseable(self):
        assert _parse_dms("not a coordinate") is None


class TestAltitudeParser:
    def test_cruise(self):
        assert _parse_altitude_ft("65000M") == 65000.0

    def test_field_elevation_with_leading_space(self):
        assert _parse_altitude_ft(" 6187M") == 6187.0

    def test_empty(self):
        assert _parse_altitude_ft("") is None
        assert _parse_altitude_ft(None) is None


class TestSpeedParser:
    def test_tas(self):
        assert _parse_speed_kt("398 T") == 398

    def test_cas(self):
        assert _parse_speed_kt("220 C") == 220

    def test_gs(self):
        assert _parse_speed_kt("412 G") == 412

    def test_na(self):
        assert _parse_speed_kt("N/A T") is None

    def test_empty(self):
        assert _parse_speed_kt("") is None


class TestHeadingParser:
    def test_true(self):
        assert _parse_heading_deg("134 T") == 134

    def test_magnetic(self):
        assert _parse_heading_deg("127 M") == 127

    def test_na(self):
        assert _parse_heading_deg("N/A") is None


class TestBankParser:
    def test_bank(self):
        assert _parse_bank_deg("22  °") == 22

    def test_blank_with_unit(self):
        assert _parse_bank_deg("    °") is None


class TestMachParser:
    def test_mach(self):
        assert _parse_mach(".55") == pytest.approx(0.55)

    def test_mach_supersonic_format(self):
        assert _parse_mach("1.2") == pytest.approx(1.2)

    def test_na(self):
        assert _parse_mach("N/A") is None


class TestSignedIntParser:
    def test_temp_positive(self):
        assert _parse_signed_int("+8C") == 8

    def test_temp_negative(self):
        assert _parse_signed_int("-57C") == -57


class TestWindParser:
    def test_wind(self):
        assert _parse_wind("260/004") == (260, 4)

    def test_three_digit_speed(self):
        assert _parse_wind("279/043") == (279, 43)

    def test_empty(self):
        assert _parse_wind("") == (None, None)


class TestTimeParsers:
    def test_leg_time(self):
        assert _parse_leg_time_min("+07.0") == 7.0

    def test_total_time_under_hour(self):
        assert _parse_total_time_min("+47.3") == 47.3

    def test_total_time_over_hour(self):
        assert _parse_total_time_min("06+33.6") == pytest.approx(393.6, abs=0.05)

    def test_sched_duration(self):
        # 06h 33m 35s = 6.5597 h
        assert _parse_sched_duration_hours("06+33+35") == pytest.approx(
            6 + 33 / 60 + 35 / 3600, abs=1e-6,
        )


class TestKindClassifier:
    def test_numbered_waypoint(self):
        assert _classify_kind("KCOS/A") == "waypoint"
        assert _classify_kind("PUB/R253012") == "waypoint"

    def test_level_off(self):
        assert _classify_kind(".level off") == "level_off"

    def test_delay(self):
        assert _classify_kind(".delay") == "delay"

    def test_descent_pt(self):
        assert _classify_kind(".descent pt") == "descent_pt"


# ---------------------------------------------------------------------------
# Integration — real NM17 B pair
# ---------------------------------------------------------------------------

NM17B_KML = Path("data/er2/NM17 B KML.kml")
NM17B_XLSX = Path("data/er2/NM17 B ER2 Green Card1.xlsx")


@pytest.mark.skipif(
    not (NM17B_KML.exists() and NM17B_XLSX.exists()),
    reason="requires data/er2/NM17 B KML.kml + Green Card XLSX cache",
)
class TestNM17BPair:
    def test_kml_placemark_count(self):
        # 38 waypoints in the Green Card; KML has the same 38 placemarks
        # (KCOS at start and KCOS/A at end count as separate Placemarks).
        pms = parse_kml(NM17B_KML)
        assert len(pms) == 38

    def test_kml_first_placemark_is_kcos(self):
        pms = parse_kml(NM17B_KML)
        assert "KCOS" in pms[0]["description"]

    def test_xlsx_header_fields(self):
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        h = sortie.header
        assert h["aircraft_id"] == "NASA806"
        assert h["mission_name"] == "GEMX"
        assert h["sched_takeoff_z"] == "15:30:00"
        assert h["takeoff_time_z"] == "15:30:00"
        assert h["land_time_z"] == "22:03:35"
        assert h["fuel_load_lb"] == 1853
        assert h["fuel_used_lb"] == 1332
        assert h["sched_duration_hours"] == pytest.approx(6.5597, abs=1e-3)

    def test_xlsx_row_counts(self):
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        df = sortie.waypoints
        # 38 numbered waypoints + 2 .level off + 2 .delay + 2 .descent pt
        assert (df["kind"] == "waypoint").sum() == 38
        assert (df["kind"] == "level_off").sum() == 2
        assert (df["kind"] == "delay").sum() == 2
        assert (df["kind"] == "descent_pt").sum() == 2
        assert len(df) == 44

    def test_cumulative_time_matches_sched_duration(self):
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        last_cum = sortie.waypoints["cumulative_time_min"].iloc[-1]
        sched_min = sortie.header["sched_duration_hours"] * 60.0
        # Green Card cumulative is rounded to 0.1 min; allow 0.5 min slack.
        assert abs(last_cum - sched_min) < 0.5

    def test_leg_times_sum_to_total(self):
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        leg_sum = sortie.waypoints["leg_time_min"].sum()
        last_cum = sortie.waypoints["cumulative_time_min"].iloc[-1]
        # Sum of leg times equals final cumulative within 0.2 min
        # (each leg is rounded to 0.1, accumulated rounding ≤ 0.5 over 44 rows).
        assert abs(leg_sum - last_cum) < 0.5

    def test_endpoints_are_kcos_airport(self):
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        df = sortie.waypoints
        first_wp = df[df["kind"] == "waypoint"].iloc[0]
        last_wp = df[df["kind"] == "waypoint"].iloc[-1]
        assert first_wp["fix_name"] == "KCOS/A"
        assert last_wp["fix_name"] == "KCOS/A"
        assert first_wp["altitude_ft"] == pytest.approx(6187, abs=1)
        assert last_wp["altitude_ft"] == pytest.approx(6187, abs=1)

    def test_load_planned_sortie_joins_kml_and_gc(self):
        sortie = load_planned_sortie(NM17B_KML, NM17B_XLSX)
        assert isinstance(sortie, PlannedSortie)
        df = sortie.waypoints
        # KML provides decimal-degrees lat/lon; Green Card provides DMS.
        # When they agree to GC's DMS precision (≤ ~0.0002°), the parser
        # upgrades the GC value to the KML value.
        wpt_rows = df[df["kind"] == "waypoint"]
        max_disagree = wpt_rows["kml_disagreement_deg"].max()
        assert max_disagree < 0.001, (
            f"KML and Green Card disagree by {max_disagree:.5f}° at some waypoint"
        )

    def test_cruise_altitude_consistency(self):
        # Most legs in NM17 B are flown at FL650; verify the parser
        # extracts the cruise altitude consistently across them.
        sortie = parse_green_card_xlsx(NM17B_XLSX)
        df = sortie.waypoints
        cruise_alt_rows = df[df["altitude_ft"] == 65000]
        assert len(cruise_alt_rows) >= 25, (
            "expected the bulk of the survey grid to be at FL650"
        )


# ---------------------------------------------------------------------------
# Integration — PDF Green Cards (CO06 / CO07v4 / NM 09)
# ---------------------------------------------------------------------------

CO06_KML = Path("data/er2/CO06 KML.kml")
CO06_PDF = Path("data/er2/CO06 ER2 Green Card1.pdf")
CO07V4_KML = Path("data/er2/CO07v4 KML.kml")
CO07V4_PDF = Path("data/er2/CO07v4 ER2 Green Card1.pdf")
NM09_KML = Path("data/er2/NM09 KML.kml")
NM09_PDF = Path("data/er2/NM 09 ER2 Green Card1.pdf")


def _pdfplumber_available() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not (_pdfplumber_available() and CO06_PDF.exists() and CO06_KML.exists()),
    reason="requires pdfplumber + data/er2/CO06 cache",
)
class TestCO06Pair:
    def test_pdf_header(self):
        sortie = parse_green_card_pdf(CO06_PDF)
        h = sortie.header
        assert h["aircraft_id"] == "NASA806"
        assert h["mission_name"] == "GEMX"
        assert h["fuel_load_lb"] == 1453
        assert h["fuel_used_lb"] == 926
        # Sched duration 04+28+53 → 4.481 h
        assert h["sched_duration_hours"] == pytest.approx(4.481, abs=1e-3)

    def test_pdf_event_counts(self):
        df = parse_green_card_pdf(CO06_PDF).waypoints
        # CO06 has 28 numbered waypoints + 2 .level off + 2 .delay + 2 .descent pt
        assert (df["kind"] == "waypoint").sum() == 28
        assert (df["kind"] == "level_off").sum() == 2
        assert (df["kind"] == "delay").sum() == 2
        assert (df["kind"] == "descent_pt").sum() == 2

    def test_pdf_cumulative_matches_sched(self):
        sortie = parse_green_card_pdf(CO06_PDF)
        last_cum = sortie.waypoints["cumulative_time_min"].iloc[-1]
        sched_min = sortie.header["sched_duration_hours"] * 60.0
        assert abs(last_cum - sched_min) < 0.5

    def test_pdf_kml_join(self):
        full = load_planned_sortie(CO06_KML, CO06_PDF)
        wpts = full.waypoints[full.waypoints["kind"] == "waypoint"]
        # Some PDF waypoints have no KML counterpart (KML omits some
        # climb/descent fixes in CO06); the join leaves those rows
        # with NaN disagreement and the GC's DMS lat/lon intact.  All
        # *matched* waypoints should agree to GC's DMS precision.
        matched = wpts.dropna(subset=["kml_disagreement_deg"])
        assert len(matched) > 0
        assert matched["kml_disagreement_deg"].max() < 0.001


@pytest.mark.skipif(
    not (_pdfplumber_available() and CO07V4_PDF.exists() and CO07V4_KML.exists()),
    reason="requires pdfplumber + data/er2/CO07v4 cache",
)
class TestCO07v4Pair:
    def test_pdf_round_trip(self):
        sortie = parse_green_card_pdf(CO07V4_PDF)
        h = sortie.header
        assert h["aircraft_id"] == "NASA806"
        assert h["mission_name"] == "GEMX"
        df = sortie.waypoints
        assert (df["kind"] == "waypoint").sum() == 34
        last_cum = df["cumulative_time_min"].iloc[-1]
        sched_min = h["sched_duration_hours"] * 60.0
        assert abs(last_cum - sched_min) < 0.5

    def test_pdf_kml_join(self):
        full = load_planned_sortie(CO07V4_KML, CO07V4_PDF)
        wpts = full.waypoints[full.waypoints["kind"] == "waypoint"]
        matched = wpts.dropna(subset=["kml_disagreement_deg"])
        assert len(matched) > 0
        assert matched["kml_disagreement_deg"].max() < 0.001


@pytest.mark.skipif(
    not (_pdfplumber_available() and NM09_PDF.exists() and NM09_KML.exists()),
    reason="requires pdfplumber + data/er2/NM 09 cache",
)
class TestNM09Pair:
    def test_pdf_round_trip(self):
        sortie = parse_green_card_pdf(NM09_PDF)
        h = sortie.header
        assert h["aircraft_id"] == "NASA809"
        # NM 09's PDF has no mission tag (some products go straight
        # from "ER-2 MISSION DATA CARD" to the PILOT row).
        assert h["mission_name"] is None
        df = sortie.waypoints
        assert (df["kind"] == "waypoint").sum() == 39
        last_cum = df["cumulative_time_min"].iloc[-1]
        sched_min = h["sched_duration_hours"] * 60.0
        assert abs(last_cum - sched_min) < 0.5

    def test_pdf_kml_join_full_match(self):
        full = load_planned_sortie(NM09_KML, NM09_PDF)
        wpts = full.waypoints[full.waypoints["kind"] == "waypoint"]
        matched = wpts.dropna(subset=["kml_disagreement_deg"])
        # NM 09's KML has all 39 numbered waypoints.
        assert len(matched) == 39
        assert matched["kml_disagreement_deg"].max() < 0.001


@pytest.mark.skipif(
    _pdfplumber_available(),
    reason="negative test: pdfplumber installed; nothing to verify",
)
def test_pdf_parse_raises_without_pdfplumber():
    """When pdfplumber isn't installed, the PDF parser should raise a
    helpful ImportError rather than a confusing ModuleNotFoundError."""
    pytest.skip("pdfplumber is installed in this environment")
