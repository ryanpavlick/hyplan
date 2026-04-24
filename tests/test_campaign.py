"""Tests for hyplan.campaign."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
from shapely.geometry import box, Polygon

from hyplan.airspace import Airspace, parse_airspace_items
from hyplan.campaign import Campaign
from hyplan.exceptions import HyPlanRuntimeError, HyPlanValueError
from hyplan.flight_line import FlightLine
from hyplan.units import ureg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BOUNDS = (-118.5, 33.5, -117.5, 34.5)

FAKE_RAW_AIRSPACE_ITEMS = [
    {
        "name": "Test Restricted",
        "type": 1,
        "icaoClass": "",
        "country": "US",
        "lowerLimit": {"value": 0, "unit": 0},
        "upperLimit": {"value": 5000, "unit": 0},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-118.5, 33.5], [-117.5, 33.5],
                [-117.5, 34.5], [-118.5, 34.5],
                [-118.5, 33.5],
            ]],
        },
    },
]


def _make_flight_line(site_name="Test Line"):
    return FlightLine.start_length_azimuth(
        lat1=34.0, lon1=-118.0,
        length=ureg.Quantity(50, "km"), az=90,
        altitude_msl=ureg.Quantity(3000, "meter"),
        site_name=site_name,
    )


def _mock_fetch_raw(bounds, country=None, max_age_hours=24.0):
    """Mock for OpenAIPClient.fetch_airspaces_raw."""
    airspaces = parse_airspace_items(FAKE_RAW_AIRSPACE_ITEMS)
    return airspaces, FAKE_RAW_AIRSPACE_ITEMS


# ---------------------------------------------------------------------------
# TestCampaignInit
# ---------------------------------------------------------------------------


class TestCampaignInit:
    def test_from_bounds(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        assert c.name == "Test"
        assert c.bounds == SAMPLE_BOUNDS
        assert c.polygon is not None
        assert not c.is_fetched

    def test_from_polygon(self):
        poly = Polygon([
            (-118.5, 33.5), (-117.5, 33.5),
            (-117.5, 34.5), (-118.5, 34.5),
        ])
        c = Campaign("Test", polygon=poly)
        assert c.polygon.equals(poly)
        assert c.bounds == poly.bounds

    def test_both_raises(self):
        poly = box(*SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not both"):
            Campaign("Test", bounds=SAMPLE_BOUNDS, polygon=poly)

    def test_neither_raises(self):
        with pytest.raises(HyPlanValueError, match="must be provided"):
            Campaign("Test")

    def test_invalid_bounds_min_gt_max(self):
        with pytest.raises(HyPlanValueError, match="min_lon"):
            Campaign("Test", bounds=(-117.0, 33.0, -118.0, 34.0))

    def test_invalid_bounds_lat(self):
        with pytest.raises(HyPlanValueError, match="Latitude"):
            Campaign("Test", bounds=(-118.0, -100.0, -117.0, 34.0))

    def test_country(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS, country="US")
        assert c.country == "US"

    def test_empty_flight_lines(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        assert c.flight_lines == []
        assert c.groups == []


# ---------------------------------------------------------------------------
# TestCampaignAirspace
# ---------------------------------------------------------------------------


class TestCampaignAirspace:
    def test_fetch_populates_airspaces(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        assert c.is_fetched
        assert len(c.airspaces) == 1
        assert c.airspaces[0].name == "Test Restricted"

    def test_fetch_skips_if_loaded(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")
            c.fetch_airspaces(api_key="test-key")  # second call
            # Should only be called once
            assert mock_instance.fetch_airspaces_raw.call_count == 1

    def test_fetch_force_refetches(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")
            c.fetch_airspaces(api_key="test-key", force=True)
            assert mock_instance.fetch_airspaces_raw.call_count == 2

    def test_fetch_returns_self(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            result = c.fetch_airspaces(api_key="test-key")
        assert result is c

    def test_check_conflicts_delegates(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        fl = _make_flight_line()
        conflicts = c.check_conflicts([fl])
        assert len(conflicts) == 1

    def test_check_conflicts_not_fetched_raises(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanRuntimeError, match="No airspace data"):
            c.check_conflicts()

    def test_check_conflicts_uses_own_lines(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        fl = _make_flight_line()
        c.add_flight_lines([fl])
        conflicts = c.check_conflicts()  # no argument — uses own lines
        assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# TestCampaignFlightLines
# ---------------------------------------------------------------------------


class TestCampaignFlightLines:
    def test_add_flight_lines(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        c.add_flight_lines([fl])
        assert len(c.flight_lines) == 1

    def test_add_creates_group(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        group_id = c.add_flight_lines(
            [fl], group_name="Coastal", group_type="flight_box"
        )
        assert group_id.startswith("group_")
        assert len(c.groups) == 1
        assert c.groups[0]["name"] == "Coastal"
        assert c.groups[0]["type"] == "flight_box"

    def test_group_has_line_ids(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl1 = _make_flight_line("Line 1")
        fl2 = _make_flight_line("Line 2")
        c.add_flight_lines([fl1, fl2], group_name="Pair")
        assert len(c.groups[0]["line_ids"]) == 2
        assert len(c.flight_line_ids) == 2

    def test_generation_params(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        c.add_flight_lines(
            [fl],
            group_type="flight_box",
            generation_params={"method": "box_around_polygon", "azimuth": 315},
        )
        assert c.groups[0]["generation"]["azimuth"] == 315

    def test_remove_group(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        group_id = c.add_flight_lines([fl])
        assert len(c.flight_lines) == 1
        c.remove_group(group_id)
        assert len(c.flight_lines) == 0
        assert len(c.groups) == 0

    def test_remove_nonexistent_raises(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not found"):
            c.remove_group("group_999")

    def test_multiple_groups(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl1 = _make_flight_line("A")
        fl2 = _make_flight_line("B")
        g1 = c.add_flight_lines([fl1], group_name="First")
        c.add_flight_lines([fl2], group_name="Second")
        assert len(c.groups) == 2
        assert len(c.flight_lines) == 2
        c.remove_group(g1)
        assert len(c.groups) == 1
        assert len(c.flight_lines) == 1
        assert c.groups[0]["name"] == "Second"

    def test_remove_flight_line_updates_group_membership(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl1 = _make_flight_line("A")
        fl2 = _make_flight_line("B")
        c.add_flight_lines([fl1, fl2], group_name="Pair")
        assert len(c.flight_lines) == 2
        c.remove_flight_line("line_001")
        assert len(c.flight_lines) == 1
        assert c.groups[0]["line_ids"] == ["line_002"]

    def test_remove_flight_line_removes_empty_group(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        c.add_flight_lines([fl], group_name="Solo")
        c.remove_flight_line("line_001")
        assert len(c.flight_lines) == 0
        assert len(c.groups) == 0

    def test_remove_flight_line_nonexistent_raises(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not found"):
            c.remove_flight_line("line_999")

    def test_replace_flight_line_preserves_id_and_group(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line("Original")
        c.add_flight_lines([fl], group_name="G")
        new_fl = _make_flight_line("Replaced")
        c.replace_flight_line("line_001", new_fl)
        assert c.flight_lines[0].site_name == "Replaced"
        assert c.flight_line_ids == ["line_001"]
        assert c.groups[0]["line_ids"] == ["line_001"]

    def test_replace_flight_line_nonexistent_raises(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not found"):
            c.replace_flight_line("line_999", _make_flight_line())

    def test_flight_lines_to_geojson(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line("GeoLine")
        c.add_flight_lines([fl])
        fc = c.flight_lines_to_geojson()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1
        assert fc["features"][0]["id"] == "line_001"
        assert fc["features"][0]["properties"]["line_id"] == "line_001"
        assert fc["features"][0]["properties"]["site_name"] == "GeoLine"

    def test_revision_increments_on_mutations(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        initial_rev = c.revision
        fl = _make_flight_line()
        c.add_flight_lines([fl])
        assert c.revision == initial_rev + 1
        c.replace_flight_line("line_001", _make_flight_line("New"))
        assert c.revision == initial_rev + 2
        c.remove_flight_line("line_001")
        assert c.revision == initial_rev + 3

    def test_campaign_id_is_stable(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        cid = c.campaign_id
        assert isinstance(cid, str)
        assert len(cid) > 0
        c.add_flight_lines([_make_flight_line()])
        assert c.campaign_id == cid  # unchanged by mutations


# ---------------------------------------------------------------------------
# TestCampaignPersistence
# ---------------------------------------------------------------------------


class TestCampaignPersistence:
    def test_save_creates_folder_structure(self, tmp_path):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = _make_flight_line()
        c.add_flight_lines([fl])

        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)

        assert os.path.isfile(os.path.join(save_path, "campaign.json"))
        assert os.path.isfile(os.path.join(save_path, "domain.geojson"))
        assert os.path.isfile(os.path.join(save_path, "flight_lines", "all_lines.geojson"))
        assert os.path.isfile(os.path.join(save_path, "flight_lines", "groups.json"))

    def test_save_without_airspaces_ok(self, tmp_path):
        """Saving without fetched airspaces should work (no airspaces.json)."""
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)
        assert not os.path.exists(os.path.join(save_path, "airspaces.json"))

    def test_save_load_roundtrip(self, tmp_path):
        c = Campaign("SoCal", bounds=SAMPLE_BOUNDS, country="US")

        # Add airspaces
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        # Add flight lines
        fl1 = _make_flight_line("Line A")
        fl2 = _make_flight_line("Line B")
        c.add_flight_lines([fl1, fl2], group_name="Coastal", group_type="flight_box")
        c.add_flight_lines([_make_flight_line("Line C")], group_name="Inland")

        # Save
        save_path = str(tmp_path / "socal_campaign")
        c.save(save_path)

        # Load
        loaded = Campaign.load(save_path)

        assert loaded.name == "SoCal"
        assert loaded.country == "US"
        assert loaded.bounds == SAMPLE_BOUNDS
        assert loaded.is_fetched
        assert len(loaded.airspaces) == 1
        assert loaded.airspaces[0].name == "Test Restricted"
        assert len(loaded.flight_lines) == 3
        assert len(loaded.groups) == 2
        assert loaded.groups[0]["name"] == "Coastal"
        assert loaded.groups[1]["name"] == "Inland"

    def test_load_no_api_calls(self, tmp_path):
        """Loading should never call the OpenAIP API."""
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)

        with patch("hyplan.campaign.OpenAIPClient") as MockClient2:
            loaded = Campaign.load(save_path)
            MockClient2.assert_not_called()
            assert loaded.is_fetched

    def test_load_reparses_airspaces(self, tmp_path):
        """Raw items are re-parsed on load (not stored as Airspace objects)."""
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")

        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)

        loaded = Campaign.load(save_path)
        # Airspaces are Airspace objects, not raw dicts
        assert isinstance(loaded.airspaces[0], Airspace)

    def test_flight_lines_roundtrip(self, tmp_path):
        """Flight lines preserve geometry and metadata through save/load."""
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(50, "km"), az=45,
            altitude_msl=ureg.Quantity(6000, "meter"),
            site_name="NE Transect",
            investigator="Dr. Smith",
        )
        c.add_flight_lines([fl])

        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)
        loaded = Campaign.load(save_path)

        loaded_fl = loaded.flight_lines[0]
        assert loaded_fl.site_name == "NE Transect"
        assert loaded_fl.investigator == "Dr. Smith"
        assert loaded_fl.altitude_msl.magnitude == pytest.approx(6000, rel=1e-3)
        assert loaded_fl.lat1 == pytest.approx(fl.lat1, abs=1e-4)
        assert loaded_fl.lon1 == pytest.approx(fl.lon1, abs=1e-4)

    def test_groups_roundtrip(self, tmp_path):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines(
            [_make_flight_line()],
            group_name="Box A",
            group_type="flight_box",
            generation_params={"method": "box_around_polygon", "azimuth": 90},
        )

        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)
        loaded = Campaign.load(save_path)

        assert len(loaded.groups) == 1
        g = loaded.groups[0]
        assert g["name"] == "Box A"
        assert g["type"] == "flight_box"
        assert g["generation"]["azimuth"] == 90

    def test_file_format_version(self, tmp_path):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        save_path = str(tmp_path / "test_campaign")
        c.save(save_path)

        with open(os.path.join(save_path, "campaign.json")) as f:
            meta = json.load(f)
        assert meta["version"] == 1

    def test_revision_metadata_roundtrip(self, tmp_path):
        c = Campaign("Rev Test", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        original_id = c.campaign_id
        original_rev = c.revision

        save_path = str(tmp_path / "rev_campaign")
        c.save(save_path)

        loaded = Campaign.load(save_path)
        assert loaded.campaign_id == original_id
        assert loaded.revision == original_rev
        assert loaded.updated_at is not None

    def test_revision_metadata_in_campaign_json(self, tmp_path):
        c = Campaign("Meta Test", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        save_path = str(tmp_path / "meta_campaign")
        c.save(save_path)

        with open(os.path.join(save_path, "campaign.json")) as f:
            meta = json.load(f)
        assert "campaign_id" in meta
        assert "revision" in meta
        assert "updated_at" in meta
        assert meta["revision"] >= 1


# ---------------------------------------------------------------------------
# TestCampaignDisplay
# ---------------------------------------------------------------------------


class TestCampaignDisplay:
    def test_summary_before_fetch(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        s = c.summary()
        assert "Test" in s
        assert "not fetched" in s

    def test_summary_after_fetch(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        with patch("hyplan.campaign.OpenAIPClient") as MockClient:
            MockClient.return_value.fetch_airspaces_raw = MagicMock(
                side_effect=_mock_fetch_raw
            )
            c.fetch_airspaces(api_key="test-key")
        s = c.summary()
        assert "Airspaces: 1" in s

    def test_repr(self):
        c = Campaign("Test", bounds=SAMPLE_BOUNDS)
        r = repr(c)
        assert "Campaign(" in r
        assert "Test" in r


# ---------------------------------------------------------------------------
# TestCampaignPatterns
# ---------------------------------------------------------------------------


class TestCampaignPatterns:
    """Pattern collection: add, remove, replace, lookup, persistence."""

    def _add_rosette(self, c, n_lines=3):
        from hyplan.flight_patterns import rosette
        pat = rosette(
            center=(34.0, -118.0), heading=0.0,
            altitude=ureg.Quantity(3000, "meter"),
            radius=ureg.Quantity(10, "km"),
            n_lines=n_lines,
        )
        return c.add_pattern(pat), pat

    def _add_spiral(self, c):
        from hyplan.flight_patterns import spiral
        pat = spiral(
            center=(34.0, -118.0), heading=0.0,
            altitude_start=ureg.Quantity(1000, "meter"),
            altitude_end=ureg.Quantity(3000, "meter"),
            radius=ureg.Quantity(2, "km"),
            n_turns=2, points_per_turn=18,
        )
        return c.add_pattern(pat), pat

    def test_add_rosette_assigns_pattern_id(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c)
        assert pid == "pattern_001"
        assert pat.pattern_id == "pattern_001"
        assert len(c.patterns) == 1

    def test_add_rosette_rekeys_lines_with_global_ids(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=3)
        assert list(pat.lines.keys()) == ["line_001", "line_002", "line_003"]

    def test_pattern_line_ids_share_namespace_with_free_standing(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()], group_name="standalone")
        # First standalone line is line_001; pattern starts at line_002
        pid, pat = self._add_rosette(c, n_lines=2)
        assert list(pat.lines.keys()) == ["line_002", "line_003"]

    def test_remove_pattern(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, _ = self._add_rosette(c)
        c.remove_pattern(pid)
        assert len(c.patterns) == 0

    def test_remove_pattern_unknown_raises(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not found"):
            c.remove_pattern("pattern_999")

    def test_replace_pattern_preserves_pattern_id(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=3)
        new_pat = pat.regenerate(n_lines=5)
        c.replace_pattern(pid, new_pat)
        replaced = c.get_pattern(pid)
        assert replaced.pattern_id == pid
        assert len(replaced.lines) == 5

    def test_replace_pattern_assigns_fresh_line_ids(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=3)
        old_ids = list(pat.lines.keys())
        new_pat = pat.regenerate(n_lines=3)
        c.replace_pattern(pid, new_pat)
        new_ids = list(c.get_pattern(pid).lines.keys())
        assert set(new_ids).isdisjoint(set(old_ids))

    def test_get_line_finds_pattern_lines(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=2)
        any_id = list(pat.lines.keys())[0]
        fl = c.get_line(any_id)
        assert fl is pat.lines[any_id]

    def test_get_line_finds_free_standing(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        fl = c.get_line("line_001")
        assert fl.site_name == "Test Line"

    def test_get_line_unknown_raises(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        with pytest.raises(HyPlanValueError, match="not found"):
            c.get_line("line_999")

    def test_find_pattern_for_line(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])  # line_001 (free-standing)
        pid, pat = self._add_rosette(c, n_lines=2)
        pattern_line_id = list(pat.lines.keys())[0]
        assert c.find_pattern_for_line(pattern_line_id) is pat
        assert c.find_pattern_for_line("line_001") is None
        assert c.find_pattern_for_line("nonexistent") is None

    def test_all_flight_lines_includes_pattern_lines(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        self._add_rosette(c, n_lines=3)
        assert len(c.all_flight_lines()) == 4

    def test_replace_line_anywhere_routes_to_pattern(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=3)
        target_id = list(pat.lines.keys())[1]
        new_fl = _make_flight_line(site_name="Replaced")
        c.replace_line_anywhere(target_id, new_fl)
        # Pattern still owns the line, with the same id
        assert target_id in c.get_pattern(pid).lines
        assert c.get_pattern(pid).lines[target_id].site_name == "Replaced"

    def test_replace_line_anywhere_routes_to_free_standing(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        new_fl = _make_flight_line(site_name="Updated")
        c.replace_line_anywhere("line_001", new_fl)
        assert c.get_line("line_001").site_name == "Updated"

    def test_remove_line_anywhere_drops_pattern_leg(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=3)
        target_id = list(pat.lines.keys())[1]
        c.remove_line_anywhere(target_id)
        assert target_id not in c.get_pattern(pid).lines
        assert len(c.get_pattern(pid).lines) == 2

    def test_remove_last_pattern_leg_removes_pattern(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_rosette(c, n_lines=1)
        only_id = list(pat.lines.keys())[0]
        c.remove_line_anywhere(only_id)
        assert pid not in c.pattern_ids

    def test_add_spiral_stores_waypoints(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        pid, pat = self._add_spiral(c)
        loaded = c.get_pattern(pid)
        assert loaded.is_waypoint_based
        assert len(loaded.waypoints) == len(pat.waypoints)

    def test_revision_bumped_on_pattern_mutations(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        r0 = c.revision
        pid, pat = self._add_rosette(c)
        assert c.revision == r0 + 1
        c.replace_pattern(pid, pat.regenerate(n_lines=4))
        assert c.revision == r0 + 2
        c.remove_pattern(pid)
        assert c.revision == r0 + 3

    def test_flight_lines_to_geojson_includes_pattern_lines(self):
        c = Campaign("T", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        pid, pat = self._add_rosette(c, n_lines=2)
        fc = c.flight_lines_to_geojson()
        assert len(fc["features"]) == 3
        pattern_features = [f for f in fc["features"]
                            if f["properties"].get("pattern_id") == pid]
        assert len(pattern_features) == 2

    def test_save_and_load_roundtrip_with_patterns(self, tmp_path):
        c = Campaign("RT", bounds=SAMPLE_BOUNDS)
        c.add_flight_lines([_make_flight_line()])
        rid, rosette_pat = self._add_rosette(c, n_lines=2)
        sid, spiral_pat = self._add_spiral(c)

        c.save(str(tmp_path))
        loaded = Campaign.load(str(tmp_path))

        assert loaded.pattern_ids == [rid, sid]
        # Counters are restored, so further additions don't collide
        assert loaded._pattern_counter == 2  # type: ignore[attr-defined]
        # Line counter advances past pattern lines
        loaded.add_flight_lines([_make_flight_line()])
        assert loaded._line_counter > 3  # type: ignore[attr-defined]
        # Rosette lines preserved
        ros = loaded.get_pattern(rid)
        assert len(ros.lines) == len(rosette_pat.lines)
        # Spiral waypoints preserved
        sp = loaded.get_pattern(sid)
        assert len(sp.waypoints) == len(spiral_pat.waypoints)


# ---------------------------------------------------------------------------
# TestComputePlanWithPatterns
# ---------------------------------------------------------------------------


class TestComputePlanWithPatterns:
    """compute_flight_plan accepts Pattern in the sequence and expands it."""

    def test_pattern_alone_in_sequence(self):
        from hyplan.aircraft import KingAirB200
        from hyplan.planning.engine import compute_flight_plan
        from hyplan.flight_patterns import racetrack
        b200 = KingAirB200()
        pat = racetrack(
            center=(34.0, -118.0), heading=0.0,
            altitude=ureg.Quantity(20000, "feet"),
            leg_length=ureg.Quantity(30, "km"),
            n_legs=3, offset=ureg.Quantity(2, "km"),
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[pat])
        assert len(plan) >= 3

    def test_pattern_mixed_with_free_standing_line(self):
        from hyplan.aircraft import KingAirB200
        from hyplan.planning.engine import compute_flight_plan
        from hyplan.flight_patterns import spiral
        b200 = KingAirB200()
        fl = _make_flight_line(site_name="Stand-alone")
        pat = spiral(
            center=(34.5, -118.5), heading=0.0,
            altitude_start=ureg.Quantity(2000, "meter"),
            altitude_end=ureg.Quantity(4000, "meter"),
            radius=ureg.Quantity(2, "km"), n_turns=1,
        )
        plan = compute_flight_plan(aircraft=b200, flight_sequence=[fl, pat])
        # Expanded sequence has at least 1 flight_line + many waypoint legs
        assert len(plan) >= 2
