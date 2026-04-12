"""Tests for hyplan.airspace."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import requests
from shapely.geometry import LineString, Polygon, box

from hyplan.airspace import (
    Airspace,
    OpenAIPClient,
    FAATFRClient,
    NASRAirspaceSource,
    check_airspace_conflicts,
    check_airspace_proximity,
    fetch_and_check,
    classify_severity,
    _parse_airspace_item,
    _resolve_type_filter,
    _cache_key,
    _is_cache_stale,
    _circle_to_polygon,
    _bounds_within_us,
    _extract_entry_exit,
    _is_schedule_active,
    filter_by_schedule,
    OceanicTrack,
    FlightPlanDBClient,
)
from hyplan.exceptions import HyPlanRuntimeError, HyPlanValueError
from hyplan.units import ureg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_airspace(
    name="Test Airspace",
    airspace_class="RESTRICTED",
    airspace_type=1,
    floor_ft=0.0,
    ceiling_ft=60000.0,
    geometry=None,
    country="US",
):
    """Create an Airspace with sensible defaults."""
    if geometry is None:
        geometry = box(-118.5, 33.5, -117.5, 34.5)  # ~1°×1° box over LA
    return Airspace(
        name=name,
        airspace_class=airspace_class,
        airspace_type=airspace_type,
        floor_ft=floor_ft,
        ceiling_ft=ceiling_ft,
        geometry=geometry,
        country=country,
    )


def _make_flight_line(lat1, lon1, lat2, lon2, alt_ft):
    """Create a lightweight flight-line-like object for testing."""
    return SimpleNamespace(
        geometry=LineString([(lon1, lat1), (lon2, lat2)]),
        altitude_msl=ureg.Quantity(alt_ft, "foot"),
    )


# ---------------------------------------------------------------------------
# Airspace dataclass
# ---------------------------------------------------------------------------


class TestAirspaceDataclass:
    def test_construction(self):
        a = _make_airspace(name="R-2508", floor_ft=0, ceiling_ft=18000)
        assert a.name == "R-2508"
        assert a.floor_ft == 0
        assert a.ceiling_ft == 18000
        assert a.airspace_class == "RESTRICTED"
        assert a.source == "openaip"

    def test_defaults(self):
        a = _make_airspace()
        assert a.country == "US"
        assert a.source == "openaip"


# ---------------------------------------------------------------------------
# check_airspace_conflicts
# ---------------------------------------------------------------------------


class TestCheckAirspaceConflicts:
    def test_line_inside_at_overlapping_altitude(self):
        """Line fully inside airspace polygon at matching altitude → conflict."""
        airspace = _make_airspace(floor_ft=0, ceiling_ft=10000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1
        assert conflicts[0].flight_line_index == 0
        assert conflicts[0].airspace is airspace

    def test_line_inside_but_below_floor(self):
        """Line inside polygon but altitude below airspace floor → no conflict."""
        airspace = _make_airspace(floor_ft=5000, ceiling_ft=10000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 3000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 0

    def test_line_inside_but_above_ceiling(self):
        """Line inside polygon but altitude above airspace ceiling → no conflict."""
        airspace = _make_airspace(floor_ft=0, ceiling_ft=5000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 7000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 0

    def test_line_outside_polygon(self):
        """Line entirely outside the airspace polygon → no conflict."""
        airspace = _make_airspace(
            geometry=box(-120.0, 36.0, -119.0, 37.0)  # far away
        )
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 0

    def test_line_at_exact_floor(self):
        """Flight altitude exactly at airspace floor → conflict."""
        airspace = _make_airspace(floor_ft=5000, ceiling_ft=10000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1

    def test_line_at_exact_ceiling(self):
        """Flight altitude exactly at airspace ceiling → conflict."""
        airspace = _make_airspace(floor_ft=0, ceiling_ft=5000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1

    def test_multiple_airspaces_multiple_lines(self):
        """Two lines, two airspaces, only some combinations conflict."""
        a1 = _make_airspace(
            name="A1", floor_ft=0, ceiling_ft=5000,
            geometry=box(-118.5, 33.5, -117.5, 34.5),
        )
        a2 = _make_airspace(
            name="A2", floor_ft=8000, ceiling_ft=15000,
            geometry=box(-118.5, 33.5, -117.5, 34.5),
        )
        fl_low = _make_flight_line(34.0, -118.0, 34.1, -118.0, 3000)
        fl_high = _make_flight_line(34.0, -118.0, 34.1, -118.0, 10000)

        conflicts = check_airspace_conflicts([fl_low, fl_high], [a1, a2])
        # fl_low conflicts with a1 only, fl_high conflicts with a2 only
        assert len(conflicts) == 2
        names = {c.airspace.name for c in conflicts}
        assert names == {"A1", "A2"}

    def test_empty_flight_lines(self):
        conflicts = check_airspace_conflicts([], [_make_airspace()])
        assert conflicts == []

    def test_empty_airspaces(self):
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [])
        assert conflicts == []

    def test_conflict_has_intersection_geometry(self):
        airspace = _make_airspace(floor_ft=0, ceiling_ft=10000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert not conflicts[0].horizontal_intersection.is_empty

    def test_vertical_overlap_values(self):
        airspace = _make_airspace(floor_ft=3000, ceiling_ft=8000)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1
        floor, ceil = conflicts[0].vertical_overlap_ft
        assert floor == 5000
        assert ceil == 5000

    def test_line_partially_intersecting_polygon(self):
        """Line crosses airspace boundary — partial intersection."""
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=10000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        # Line goes from outside to inside
        fl = _make_flight_line(34.25, -118.5, 34.25, -117.7, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# _parse_airspace_item
# ---------------------------------------------------------------------------


class TestParseAirspaceItem:
    def test_valid_item(self):
        item = {
            "name": "Test Zone",
            "icaoClass": "D",
            "type": 4,
            "country": "US",
            "lowerLimit": {"value": 0, "unit": 0, "referenceDatum": 0},
            "upperLimit": {"value": 3000, "unit": 0, "referenceDatum": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-118.0, 34.0], [-117.0, 34.0],
                    [-117.0, 35.0], [-118.0, 35.0],
                    [-118.0, 34.0],
                ]],
            },
        }
        a = _parse_airspace_item(item)
        assert a is not None
        assert a.name == "Test Zone"
        assert a.airspace_class == "D"
        assert a.floor_ft == 0
        assert a.ceiling_ft == pytest.approx(9842.5, rel=1e-3)

    def test_feet_conversion(self):
        item = {
            "name": "FL Zone",
            "type": 1,
            "lowerLimit": {"value": 1000, "unit": 1},  # 1000 feet
            "upperLimit": {"value": 18000, "unit": 1},  # 18000 feet
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [0, 0], [1, 0], [1, 1], [0, 1], [0, 0],
                ]],
            },
        }
        a = _parse_airspace_item(item)
        assert a is not None
        assert a.floor_ft == pytest.approx(1000.0, rel=1e-3)
        assert a.ceiling_ft == pytest.approx(18000.0, rel=1e-3)

    def test_flight_level_conversion(self):
        item = {
            "name": "FL Zone",
            "type": 1,
            "lowerLimit": {"value": 100, "unit": 2},  # FL100
            "upperLimit": {"value": 350, "unit": 2},  # FL350
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [0, 0], [1, 0], [1, 1], [0, 1], [0, 0],
                ]],
            },
        }
        a = _parse_airspace_item(item)
        assert a is not None
        assert a.floor_ft == pytest.approx(10000.0, rel=1e-3)
        assert a.ceiling_ft == pytest.approx(35000.0, rel=1e-3)

    def test_missing_geometry_returns_none(self):
        assert _parse_airspace_item({"name": "No Geom"}) is None

    def test_point_geometry_returns_none(self):
        item = {
            "name": "Point",
            "type": 0,
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        }
        assert _parse_airspace_item(item) is None

    def test_no_icao_class_uses_type_name(self):
        item = {
            "name": "Restricted",
            "type": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
        a = _parse_airspace_item(item)
        assert a.airspace_class == "RESTRICTED"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    def test_cache_key_deterministic(self):
        k1 = _cache_key((-118.0, 33.0, -117.0, 34.0), "US")
        k2 = _cache_key((-118.0, 33.0, -117.0, 34.0), "US")
        assert k1 == k2

    def test_cache_key_differs_by_country(self):
        k1 = _cache_key((-118.0, 33.0, -117.0, 34.0), "US")
        k2 = _cache_key((-118.0, 33.0, -117.0, 34.0), "DE")
        assert k1 != k2

    def test_is_cache_stale_missing_file(self, tmp_path):
        assert _is_cache_stale(str(tmp_path / "nonexistent.json"), 24.0)

    def test_is_cache_stale_fresh_file(self, tmp_path):
        f = tmp_path / "fresh.json"
        f.write_text("{}")
        assert not _is_cache_stale(str(f), 24.0)


# ---------------------------------------------------------------------------
# OpenAIPClient
# ---------------------------------------------------------------------------


class TestOpenAIPClient:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure OPENAIP_API_KEY is not set
            os.environ.pop("OPENAIP_API_KEY", None)
            with pytest.raises(HyPlanValueError, match="API key"):
                OpenAIPClient(api_key="")

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAIP_API_KEY": "test-key-123"}):
            client = OpenAIPClient()
            assert client.api_key == "test-key-123"

    def test_api_key_from_arg(self):
        client = OpenAIPClient(api_key="my-key")
        assert client.api_key == "my-key"

    def test_fetch_uses_cache(self, tmp_path):
        """When cache is fresh, no HTTP request is made."""
        client = OpenAIPClient(api_key="test-key")

        # Write a fake cache file
        cache_items = [{
            "name": "Cached Zone",
            "type": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }]
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            # Write cache
            from hyplan.airspace import _cache_key
            bounds = (-118.0, 33.0, -117.0, 34.0)
            cache_file = os.path.join(cache_dir, _cache_key(bounds, None))
            with open(cache_file, "w") as f:
                json.dump(cache_items, f)

            # Fetch should use cache, not make HTTP request
            with patch("hyplan.airspace.requests.get") as mock_get:
                result = client.fetch_airspaces(bounds)
                mock_get.assert_not_called()
                assert len(result) == 1
                assert result[0].name == "Cached Zone"

    def test_fetch_network_error_raises(self, tmp_path):
        """Network error raises HyPlanRuntimeError."""
        client = OpenAIPClient(api_key="test-key")
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            with patch("hyplan.airspace.requests.get") as mock_get:
                mock_get.side_effect = requests.ConnectionError("no network")
                with pytest.raises(HyPlanRuntimeError, match="request failed"):
                    client.fetch_airspaces((-118.0, 33.0, -117.0, 34.0))

    def test_fetch_successful_api_call(self, tmp_path):
        """Successful API call returns parsed airspaces and writes cache."""
        client = OpenAIPClient(api_key="test-key")
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        api_response = {
            "totalPages": 1,
            "items": [{
                "_id": "abc123",
                "name": "API Zone",
                "type": 3,
                "icaoClass": "B",
                "country": "US",
                "lowerLimit": {"value": 0, "unit": 0},
                "upperLimit": {"value": 5000, "unit": 0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-118, 33], [-117, 33], [-117, 34],
                        [-118, 34], [-118, 33],
                    ]],
                },
            }],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            with patch("hyplan.airspace.requests.get", return_value=mock_resp) as mock_get:
                result = client.fetch_airspaces((-118.0, 33.0, -117.0, 34.0))
                # Tiled fetch makes multiple API calls across the bbox
                assert mock_get.call_count >= 1
                # Same _id returned by every tile → deduped to 1
                assert len(result) == 1
                assert result[0].name == "API Zone"
                assert result[0].airspace_class == "B"


# ---------------------------------------------------------------------------
# fetch_and_check
# ---------------------------------------------------------------------------


class TestFetchAndCheck:
    def test_empty_flight_lines(self):
        assert fetch_and_check([], api_key="key") == []

    def test_integration_us(self):
        """Mocked end-to-end: US flight line uses FAA sources."""
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)

        fake_airspaces = [_make_airspace(floor_ft=0, ceiling_ft=10000)]

        with patch.object(NASRAirspaceSource, "fetch_airspaces", return_value=fake_airspaces), \
             patch.object(NASRAirspaceSource, "fetch_sfras", return_value=[]), \
             patch.object(NASRAirspaceSource, "fetch_class_airspace", return_value=[]), \
             patch.object(FAATFRClient, "fetch_tfrs", return_value=[]):
            conflicts = fetch_and_check([fl])
            assert len(conflicts) == 1
            assert conflicts[0].airspace.name == "Test Airspace"

    def test_integration_international(self):
        """Mocked end-to-end: international flight line uses OpenAIP."""
        fl = _make_flight_line(48.0, 8.0, 48.1, 8.0, 5000)

        fake_airspaces = [_make_airspace(floor_ft=0, ceiling_ft=10000,
                                         geometry=box(7.5, 47.5, 8.5, 48.5))]

        with patch.object(OpenAIPClient, "fetch_airspaces", return_value=fake_airspaces):
            conflicts = fetch_and_check([fl], api_key="test-key")
            assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# Group 1b: Severity classification
# ---------------------------------------------------------------------------


class TestSeverityClassification:
    def test_prohibited_is_hard(self):
        assert classify_severity(3) == "HARD"

    def test_restricted_is_hard(self):
        assert classify_severity(1) == "HARD"

    def test_sfra_is_hard(self):
        assert classify_severity(37) == "HARD"

    def test_danger_is_advisory(self):
        assert classify_severity(2) == "ADVISORY"

    def test_ctr_is_advisory(self):
        assert classify_severity(4) == "ADVISORY"

    def test_tma_is_advisory(self):
        assert classify_severity(7) == "ADVISORY"

    def test_class_b_is_advisory(self):
        assert classify_severity(33) == "ADVISORY"

    def test_class_c_is_advisory(self):
        assert classify_severity(34) == "ADVISORY"

    def test_class_d_is_advisory(self):
        assert classify_severity(35) == "ADVISORY"

    def test_class_e_is_info(self):
        assert classify_severity(36) == "INFO"

    def test_tfr_is_advisory(self):
        assert classify_severity(31) == "ADVISORY"

    def test_fir_is_info(self):
        assert classify_severity(10) == "INFO"

    def test_unknown_is_info(self):
        assert classify_severity(999) == "INFO"

    def test_check_conflicts_populates_severity(self):
        airspace = _make_airspace(floor_ft=0, ceiling_ft=10000, airspace_type=3)
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1
        assert conflicts[0].severity == "HARD"


# ---------------------------------------------------------------------------
# Group 1a: Type filtering
# ---------------------------------------------------------------------------


class TestTypeFilter:
    def test_none_returns_none(self):
        assert _resolve_type_filter(None) is None

    def test_single_int(self):
        assert _resolve_type_filter(1) == {1}

    def test_single_string(self):
        assert _resolve_type_filter("RESTRICTED") == {1}

    def test_case_insensitive(self):
        assert _resolve_type_filter("restricted") == {1}

    def test_list_mixed(self):
        result = _resolve_type_filter([1, "PROHIBITED", 7])
        assert result == {1, 3, 7}

    def test_unknown_string_raises(self):
        with pytest.raises(HyPlanValueError, match="Unknown"):
            _resolve_type_filter("BOGUS_TYPE")

    def test_fetch_airspaces_with_type_filter(self, tmp_path):
        """type_filter should reduce results from fetch_airspaces."""
        client = OpenAIPClient(api_key="test-key")
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_items = [
            {
                "name": "Restricted Zone", "type": 1,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "name": "FIR Zone", "type": 10,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
        ]
        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            bounds = (0.0, 0.0, 1.0, 1.0)
            cache_file = os.path.join(cache_dir, _cache_key(bounds, [None]))
            with open(cache_file, "w") as f:
                json.dump(cache_items, f)

            # No filter: both
            all_result = client.fetch_airspaces(bounds)
            assert len(all_result) == 2

            # Filter to RESTRICTED only
            filtered = client.fetch_airspaces(bounds, type_filter="RESTRICTED")
            assert len(filtered) == 1
            assert filtered[0].name == "Restricted Zone"


# ---------------------------------------------------------------------------
# Group 1c: Ceiling unlimited
# ---------------------------------------------------------------------------


class TestCeilingUnlimited:
    def test_missing_upper_limit(self):
        item = {
            "name": "No Ceiling",
            "type": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
        a = _parse_airspace_item(item)
        assert a is not None
        assert a.ceiling_unlimited is True
        assert a.ceiling_ft == 60000.0

    def test_normal_ceiling(self):
        item = {
            "name": "Low Zone",
            "type": 1,
            "lowerLimit": {"value": 0, "unit": 1},
            "upperLimit": {"value": 5000, "unit": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
        a = _parse_airspace_item(item)
        assert a is not None
        assert a.ceiling_unlimited is False
        assert a.ceiling_ft == 5000.0


# ---------------------------------------------------------------------------
# Group 1d: Multi-country
# ---------------------------------------------------------------------------


class TestMultiCountry:
    def test_cache_key_list(self):
        k1 = _cache_key((-118.0, 33.0, -117.0, 34.0), ["US", "MX"])
        k2 = _cache_key((-118.0, 33.0, -117.0, 34.0), ["MX", "US"])
        # Sorted, so order doesn't matter
        assert k1 == k2

    def test_cache_key_list_vs_single(self):
        k_list = _cache_key((-118.0, 33.0, -117.0, 34.0), ["US"])
        k_str = _cache_key((-118.0, 33.0, -117.0, 34.0), "US")
        assert k_list == k_str


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestCircleToPolygon:
    def test_basic_circle(self):
        poly = _circle_to_polygon(34.0, -118.0, 5.0)
        assert isinstance(poly, Polygon)
        assert poly.is_valid
        # Should roughly contain the center
        from shapely.geometry import Point
        assert poly.contains(Point(-118.0, 34.0))

    def test_radius_scaling(self):
        small = _circle_to_polygon(34.0, -118.0, 1.0)
        large = _circle_to_polygon(34.0, -118.0, 10.0)
        assert large.area > small.area


# ---------------------------------------------------------------------------
# Group 2: FAA TFR client
# ---------------------------------------------------------------------------


class TestFAATFRClient:
    def test_parse_wfs_feature(self):
        """WFS GeoJSON feature + metadata → Airspace."""
        feature = {
            "type": "Feature",
            "id": "V_TFR_LOC.6/1222",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.7, 28.4], [-80.5, 28.4],
                    [-80.5, 28.6], [-80.7, 28.6],
                    [-80.7, 28.4],
                ]],
            },
            "properties": {
                "GID": 123,
                "NOTAM_KEY": "6/1222-1-FDC-F",
                "TITLE": "Cape Canaveral, FL",
                "STATE": "FL",
            },
        }
        meta_map = {
            "6/1222": {
                "notam_id": "6/1222",
                "type": "SPACE OPERATIONS",
                "description": "Cape Canaveral, FL, Thursday",
            },
        }
        result = FAATFRClient._parse_wfs_feature(feature, meta_map)
        assert result is not None
        assert result.source == "faa_tfr"
        assert result.airspace_type == 31
        assert "SPACE OPERATIONS" in result.name
        assert "Cape Canaveral" in result.name
        assert result.floor_ft == 0.0
        assert result.ceiling_ft == 60000.0
        assert result.geometry.is_valid

    def test_parse_wfs_feature_no_geometry(self):
        feature = {"properties": {"NOTAM_KEY": "6/9999"}}
        result = FAATFRClient._parse_wfs_feature(feature, {})
        assert result is None

    def test_parse_wfs_feature_no_metadata(self):
        """WFS feature without matching tfrapi metadata still parses."""
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.7, 28.4], [-80.5, 28.4],
                    [-80.5, 28.6], [-80.7, 28.6],
                    [-80.7, 28.4],
                ]],
            },
            "properties": {
                "NOTAM_KEY": "6/9999-1-FDC-F",
                "TITLE": "Some TFR",
            },
        }
        result = FAATFRClient._parse_wfs_feature(feature, {})
        assert result is not None
        assert result.name == "Some TFR"

    def test_extract_notam_id(self):
        assert FAATFRClient._extract_notam_id("6/1222-1-FDC-F") == "6/1222"
        assert FAATFRClient._extract_notam_id("4/3635-1-FDC-F") == "4/3635"
        assert FAATFRClient._extract_notam_id("") == ""

    def test_cache_roundtrip(self):
        """Airspace → dict → Airspace roundtrip."""
        a = _make_airspace(name="TFR Test", airspace_type=31)
        a.source = "faa_tfr"
        a.effective_start = "2026-01-01"
        d = FAATFRClient._airspace_to_dict(a)
        restored = FAATFRClient._dict_to_airspace(d)
        assert restored is not None
        assert restored.name == "TFR Test"
        assert restored.effective_start == "2026-01-01"

    def test_fetch_network_error(self, tmp_path):
        client = FAATFRClient()
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            with patch("hyplan.airspace.requests.get") as mock_get:
                mock_get.side_effect = requests.ConnectionError("no network")
                with pytest.raises(HyPlanRuntimeError, match="TFR"):
                    client.fetch_tfrs()

    def test_fetch_mocked(self, tmp_path):
        """Mocked WFS + tfrapi response returns parsed TFRs."""
        client = FAATFRClient()
        cache_dir = str(tmp_path / "airspace_cache")
        os.makedirs(cache_dir, exist_ok=True)

        wfs_response = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": "V_TFR_LOC.6/1222",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-80.7, 28.4], [-80.5, 28.4],
                        [-80.5, 28.6], [-80.7, 28.6],
                        [-80.7, 28.4],
                    ]],
                },
                "properties": {
                    "NOTAM_KEY": "6/1222-1-FDC-F",
                    "TITLE": "Cape Canaveral, FL",
                },
            }],
        }
        meta_response = [{
            "notam_id": "6/1222",
            "type": "SPACE OPERATIONS",
            "description": "Cape Canaveral",
        }]

        wfs_resp = MagicMock()
        wfs_resp.json.return_value = wfs_response
        wfs_resp.raise_for_status = MagicMock()

        meta_resp = MagicMock()
        meta_resp.json.return_value = meta_response
        meta_resp.raise_for_status = MagicMock()

        def mock_get(url, **kwargs):
            if "geoserver" in url:
                return wfs_resp
            return meta_resp

        with patch("hyplan.airspace._get_airspace_cache_dir", return_value=cache_dir):
            with patch("hyplan.airspace.requests.get", side_effect=mock_get):
                result = client.fetch_tfrs()
                assert len(result) == 1
                assert "SPACE OPERATIONS" in result[0].name

    def test_parse_date_from_description(self):
        parse = FAATFRClient._parse_date_from_description
        assert parse("Cape Canaveral, FL, Thursday, April 16, 2026 Local") == "2026-04-16"
        assert parse("11NM NW HOT SPRINGS, SD, Sunday, April 12, 2026 through Sunday") == "2026-04-12"
        assert parse("Miami, FL") is None
        assert parse("") is None

    def test_filter_effective_removes_future(self):
        from datetime import date, timedelta
        today = date.today()
        future = (today + timedelta(days=5)).isoformat()
        past = (today - timedelta(days=1)).isoformat()

        a_future = _make_airspace(name="Future TFR")
        a_future.effective_start = future
        a_past = _make_airspace(name="Past TFR")
        a_past.effective_start = past
        a_none = _make_airspace(name="No Date TFR")
        a_none.effective_start = None

        result = FAATFRClient._filter_effective([a_future, a_past, a_none])
        names = [a.name for a in result]
        assert "Past TFR" in names
        assert "No Date TFR" in names
        assert "Future TFR" not in names

    def test_wfs_feature_stores_parsed_date(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.7, 28.4], [-80.5, 28.4],
                    [-80.5, 28.6], [-80.7, 28.6],
                    [-80.7, 28.4],
                ]],
            },
            "properties": {
                "NOTAM_KEY": "6/1222-1-FDC-F",
                "TITLE": "Cape Canaveral, FL, April 16, 2026",
            },
        }
        result = FAATFRClient._parse_wfs_feature(feature, {})
        assert result is not None
        assert result.effective_start == "2026-04-16"


# ---------------------------------------------------------------------------
# Group 3: NASR/SUA
# ---------------------------------------------------------------------------


class TestNASRAirspaceSource:
    def test_feature_to_airspace(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-118, 33], [-117, 33], [-117, 34],
                    [-118, 34], [-118, 33],
                ]],
            },
            "properties": {
                "NAME": "R-2508",
                "TYPE_CODE": "R",
                "LOWER_VAL": 0,
                "UPPER_VAL": 18000,
            },
        }
        result = NASRAirspaceSource._feature_to_airspace(feature)
        assert result is not None
        assert result.name == "R-2508"
        assert result.airspace_type == 1  # RESTRICTED
        assert result.source == "faa_nasr"
        assert result.floor_ft == 0
        assert result.ceiling_ft == 18000

    def test_feature_missing_geometry(self):
        feature = {"properties": {"NAME": "Bad"}}
        result = NASRAirspaceSource._feature_to_airspace(feature)
        assert result is None

    def test_fetch_mocked(self, tmp_path):
        """Mocked ArcGIS response returns parsed airspaces."""
        source = NASRAirspaceSource()
        cache_dir = str(tmp_path / "airspace_cache" / "nasr")
        os.makedirs(cache_dir, exist_ok=True)

        api_response = {
            "features": [{
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-118, 33], [-117, 33], [-117, 34],
                        [-118, 34], [-118, 33],
                    ]],
                },
                "properties": {
                    "NAME": "MOA Test",
                    "TYPE_CODE": "MOA",
                },
            }],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("hyplan.airspace._get_airspace_cache_dir",
                    return_value=str(tmp_path / "airspace_cache")):
            with patch("hyplan.airspace.requests.get", return_value=mock_resp):
                result = source.fetch_airspaces((-118.0, 33.0, -117.0, 34.0))
                assert len(result) == 1
                assert result[0].name == "MOA Test"


# ---------------------------------------------------------------------------
# Group 4: SFRA
# ---------------------------------------------------------------------------


class TestSFRA:
    def test_sfra_feature_to_airspace(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-77.5, 38.5], [-76.5, 38.5],
                    [-76.5, 39.0], [-77.5, 39.0],
                    [-77.5, 38.5],
                ]],
            },
            "properties": {
                "NAME_TXT": "WASHINGTON, DC METROPOLITAN AREA SPECIAL FLIGHT RULES AREA",
                "TYPE_CODE": "SATA",
                "LOCALTYPE_TXT": "SPEC_AT_RULES",
                "DISTVERTUPPER_VAL": 180,
                "DISTVERTUPPER_UOM": "FL",
                "DISTVERTLOWER_VAL": 0,
                "DISTVERTLOWER_UOM": "FT",
            },
        }
        result = NASRAirspaceSource._sfra_feature_to_airspace(feature)
        assert result is not None
        assert result.airspace_class == "SFRA"
        assert result.source == "faa_nasr"
        assert result.floor_ft == 0
        assert result.ceiling_ft == 18000.0  # FL180 → 18000 ft
        assert "WASHINGTON" in result.name

    def test_sfra_feature_ft_units(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "properties": {
                "NAME_TXT": "Test SFRA",
                "DISTVERTUPPER_VAL": 10000,
                "DISTVERTUPPER_UOM": "FT",
                "DISTVERTLOWER_VAL": 0,
                "DISTVERTLOWER_UOM": "FT",
            },
        }
        result = NASRAirspaceSource._sfra_feature_to_airspace(feature)
        assert result is not None
        assert result.ceiling_ft == 10000.0

    def test_sfra_feature_missing_geometry(self):
        feature = {"properties": {"NAME_TXT": "Bad SFRA"}}
        result = NASRAirspaceSource._sfra_feature_to_airspace(feature)
        assert result is None

    def test_fetch_sfras_mocked(self, tmp_path):
        """Mocked ArcGIS response returns parsed SFRAs."""
        source = NASRAirspaceSource()

        api_response = {
            "features": [{
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-77.5, 38.5], [-76.5, 38.5],
                        [-76.5, 39.0], [-77.5, 39.0],
                        [-77.5, 38.5],
                    ]],
                },
                "properties": {
                    "NAME_TXT": "DC SFRA",
                    "TYPE_CODE": "SATA",
                    "DISTVERTUPPER_VAL": 180,
                    "DISTVERTUPPER_UOM": "FL",
                },
            }],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("hyplan.airspace._get_airspace_cache_dir",
                    return_value=str(tmp_path / "airspace_cache")):
            with patch("hyplan.airspace.requests.get", return_value=mock_resp):
                result = source.fetch_sfras((-78.0, 38.0, -76.0, 39.5))
                assert len(result) == 1
                assert result[0].airspace_class == "SFRA"


# ---------------------------------------------------------------------------
# Group 4b: Class airspace
# ---------------------------------------------------------------------------


class TestClassAirspace:
    def test_class_feature_to_airspace(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-118.5, 33.5], [-117.5, 33.5],
                    [-117.5, 34.5], [-118.5, 34.5],
                    [-118.5, 33.5],
                ]],
            },
            "properties": {
                "NAME": "LOS ANGELES CLASS B",
                "CLASS": "B",
                "LOCAL_TYPE": "CLASS_B",
                "LOWER_VAL": 0,
                "LOWER_UOM": "FT",
                "UPPER_VAL": 10000,
                "UPPER_UOM": "FT",
            },
        }
        result = NASRAirspaceSource._class_feature_to_airspace(feature)
        assert result is not None
        assert result.airspace_class == "B"
        assert result.airspace_type == 33  # CLASS_B
        assert result.source == "faa_nasr"
        assert result.floor_ft == 0
        assert result.ceiling_ft == 10000

    def test_class_c_feature(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "properties": {
                "NAME": "ABILENE CLASS C",
                "CLASS": "C",
                "LOCAL_TYPE": "CLASS_C",
                "LOWER_VAL": 0,
                "LOWER_UOM": "FT",
                "UPPER_VAL": 5800,
                "UPPER_UOM": "FT",
            },
        }
        result = NASRAirspaceSource._class_feature_to_airspace(feature)
        assert result is not None
        assert result.airspace_type == 34  # CLASS_C

    def test_class_feature_missing_geometry(self):
        feature = {"properties": {"NAME": "Bad"}}
        result = NASRAirspaceSource._class_feature_to_airspace(feature)
        assert result is None

    def test_fetch_class_airspace_mocked(self, tmp_path):
        source = NASRAirspaceSource()

        api_response = {
            "features": [{
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-118, 33], [-117, 33], [-117, 34],
                        [-118, 34], [-118, 33],
                    ]],
                },
                "properties": {
                    "NAME": "TEST CLASS B",
                    "CLASS": "B",
                    "LOCAL_TYPE": "CLASS_B",
                    "LOWER_VAL": 0,
                    "UPPER_VAL": 10000,
                },
            }],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("hyplan.airspace._get_airspace_cache_dir",
                    return_value=str(tmp_path / "airspace_cache")):
            with patch("hyplan.airspace.requests.get", return_value=mock_resp):
                result = source.fetch_class_airspace((-118.0, 33.0, -117.0, 34.0))
                assert len(result) == 1
                assert result[0].airspace_class == "B"


# ---------------------------------------------------------------------------
# Group 5: US bounds detection and fetch_and_check routing
# ---------------------------------------------------------------------------


class TestBoundsWithinUS:
    def test_conus(self):
        assert _bounds_within_us((-118.0, 33.0, -117.0, 34.0)) is True

    def test_alaska(self):
        assert _bounds_within_us((-150.0, 60.0, -145.0, 65.0)) is True

    def test_hawaii(self):
        assert _bounds_within_us((-160.0, 19.0, -154.0, 22.0)) is True

    def test_europe(self):
        assert _bounds_within_us((5.0, 45.0, 15.0, 55.0)) is False

    def test_cross_atlantic(self):
        # Spans both US and Europe — not within US
        assert _bounds_within_us((-80.0, 30.0, 10.0, 50.0)) is False


class TestFetchAndCheckRouting:
    def test_us_bounds_uses_faa(self):
        """US bounds should call NASRAirspaceSource, not OpenAIPClient."""
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)

        with patch.object(NASRAirspaceSource, "fetch_airspaces", return_value=[]) as mock_nasr, \
             patch.object(NASRAirspaceSource, "fetch_sfras", return_value=[]) as mock_sfra, \
             patch.object(NASRAirspaceSource, "fetch_class_airspace", return_value=[]) as mock_cls, \
             patch.object(FAATFRClient, "fetch_tfrs", return_value=[]) as mock_tfr, \
             patch.object(OpenAIPClient, "fetch_airspaces") as mock_openaip:
            fetch_and_check([fl])
            mock_nasr.assert_called_once()
            mock_sfra.assert_called_once()
            mock_cls.assert_called_once()
            mock_tfr.assert_called_once()
            mock_openaip.assert_not_called()

    def test_international_bounds_uses_openaip(self):
        """International bounds should use OpenAIPClient."""
        fl = _make_flight_line(48.0, 8.0, 48.1, 8.0, 5000)

        with patch.object(OpenAIPClient, "fetch_airspaces", return_value=[]) as mock_openaip, \
             patch.object(NASRAirspaceSource, "fetch_airspaces") as mock_nasr:
            fetch_and_check([fl], api_key="test-key", use_faa=True)
            mock_openaip.assert_called_once()
            mock_nasr.assert_not_called()

    def test_us_bounds_tfr_failure_continues(self):
        """TFR fetch failure should not prevent SUA/SFRA results."""
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        fake_airspace = _make_airspace(floor_ft=0, ceiling_ft=10000)

        with patch.object(NASRAirspaceSource, "fetch_airspaces", return_value=[fake_airspace]), \
             patch.object(NASRAirspaceSource, "fetch_sfras", return_value=[]), \
             patch.object(NASRAirspaceSource, "fetch_class_airspace", return_value=[]), \
             patch.object(FAATFRClient, "fetch_tfrs", side_effect=HyPlanRuntimeError("TFR down")):
            conflicts = fetch_and_check([fl])
            assert len(conflicts) == 1

    def test_type_filter_on_faa_path(self):
        """type_filter should filter results on the FAA path."""
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)
        restricted = _make_airspace(name="R-2508", airspace_type=1, floor_ft=0, ceiling_ft=10000)
        class_b = _make_airspace(name="LAX Class B", airspace_type=33, floor_ft=0, ceiling_ft=10000)

        with patch.object(NASRAirspaceSource, "fetch_airspaces", return_value=[restricted]), \
             patch.object(NASRAirspaceSource, "fetch_sfras", return_value=[]), \
             patch.object(NASRAirspaceSource, "fetch_class_airspace", return_value=[class_b]), \
             patch.object(FAATFRClient, "fetch_tfrs", return_value=[]):
            # No filter: both conflict
            all_conflicts = fetch_and_check([fl])
            assert len(all_conflicts) == 2

            # Filter to RESTRICTED only
            filtered = fetch_and_check([fl], type_filter="RESTRICTED")
            assert len(filtered) == 1
            assert filtered[0].airspace.name == "R-2508"


# ---------------------------------------------------------------------------
# Group 6: Entry/exit points
# ---------------------------------------------------------------------------


class TestEntryExit:
    def test_linestring_intersection(self):
        """LineString intersection has entry and exit."""
        line = LineString([(0, 0), (2, 0)])
        entry, exit_ = _extract_entry_exit(line)
        assert entry == (0, 0)
        assert exit_ == (2, 0)

    def test_point_intersection(self):
        from shapely.geometry import Point
        pt = Point(1, 2)
        entry, exit_ = _extract_entry_exit(pt)
        assert entry == (1.0, 2.0)
        assert exit_ == (1.0, 2.0)

    def test_empty_geometry(self):
        from shapely.geometry import LineString as LS
        empty = LS()
        entry, exit_ = _extract_entry_exit(empty)
        assert entry is None
        assert exit_ is None

    def test_conflicts_have_entry_exit(self):
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=10000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        # Line enters from west, exits east
        fl = _make_flight_line(34.25, -118.5, 34.25, -117.0, 5000)
        conflicts = check_airspace_conflicts([fl], [airspace])
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.entry_point is not None
        assert c.exit_point is not None
        # Entry should be on western boundary (~-118.0)
        assert abs(c.entry_point[0] - (-118.0)) < 0.01
        # Exit should be on eastern boundary (~-117.5)
        assert abs(c.exit_point[0] - (-117.5)) < 0.01


# ---------------------------------------------------------------------------
# Group 7: Near-miss proximity
# ---------------------------------------------------------------------------


class TestProximity:
    def test_near_miss_detected(self):
        """Flight line near but not inside airspace → near-miss."""
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=10000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        # Line 0.005° (~550m) south of the airspace
        fl = _make_flight_line(33.995, -118.0, 33.995, -117.5, 5000)
        near = check_airspace_proximity([fl], [airspace], buffer_m=2000)
        assert len(near) == 1
        assert near[0].severity == "NEAR_MISS"
        assert near[0].distance_to_boundary_m is not None
        assert near[0].distance_to_boundary_m < 2000

    def test_no_near_miss_when_inside(self):
        """Flight line inside airspace → not a near-miss (it's a conflict)."""
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=10000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        fl = _make_flight_line(34.25, -117.8, 34.25, -117.6, 5000)
        near = check_airspace_proximity([fl], [airspace], buffer_m=5000)
        assert len(near) == 0

    def test_no_near_miss_when_far(self):
        """Flight line far from airspace → no near-miss."""
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=10000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        fl = _make_flight_line(30.0, -118.0, 30.0, -117.5, 5000)
        near = check_airspace_proximity([fl], [airspace], buffer_m=1000)
        assert len(near) == 0

    def test_no_near_miss_wrong_altitude(self):
        """Flight line near but above airspace ceiling → no near-miss."""
        airspace = _make_airspace(
            floor_ft=0, ceiling_ft=5000,
            geometry=box(-118.0, 34.0, -117.5, 34.5),
        )
        fl = _make_flight_line(33.995, -118.0, 33.995, -117.5, 10000)
        near = check_airspace_proximity([fl], [airspace], buffer_m=2000)
        assert len(near) == 0


# ---------------------------------------------------------------------------
# Group 8: Schedule filtering
# ---------------------------------------------------------------------------


class TestScheduleFiltering:
    def test_weekday_schedule_active(self):
        from datetime import datetime, timezone
        # Wednesday 10:00 UTC → active during 0700-1800 MON-FRI (UTC-6 = 04:00 local)
        # Actually 10:00 UTC - 6 = 04:00 local → before 0700, so NOT active
        dt = datetime(2026, 4, 15, 10, 0, tzinfo=timezone.utc)  # Wednesday
        assert not _is_schedule_active("0700 - 1800, MON - FRI", -6, 1, dt)

    def test_weekday_schedule_active_midday(self):
        from datetime import datetime, timezone
        # Wednesday 18:00 UTC → 12:00 local (UTC-6) → active
        dt = datetime(2026, 4, 15, 18, 0, tzinfo=timezone.utc)
        assert _is_schedule_active("0700 - 1800, MON - FRI", -6, 1, dt)

    def test_weekday_schedule_weekend(self):
        from datetime import datetime, timezone
        # Saturday 18:00 UTC → 12:00 local → right time, wrong day
        dt = datetime(2026, 4, 18, 18, 0, tzinfo=timezone.utc)
        assert not _is_schedule_active("0700 - 1800, MON - FRI", -6, 1, dt)

    def test_daily_schedule(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)  # Saturday
        assert _is_schedule_active("0600 - 2200, DAILY", 0, 0, dt)

    def test_continuous_always_active(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 18, 3, 0, tzinfo=timezone.utc)
        assert _is_schedule_active("CONTINUOUS", 0, 0, dt)

    def test_h24_always_active(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 18, 3, 0, tzinfo=timezone.utc)
        assert _is_schedule_active("H24", 0, 0, dt)

    def test_by_notam_always_active(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 18, 3, 0, tzinfo=timezone.utc)
        assert _is_schedule_active("OTHER TIMES BY NOTAM", 0, 0, dt)

    def test_filter_by_schedule_keeps_no_schedule(self):
        """Airspaces without schedules are always kept."""
        a = _make_airspace(name="No Schedule")
        result = filter_by_schedule([a])
        assert len(result) == 1

    def test_filter_by_schedule_removes_inactive(self):
        from datetime import datetime, timezone
        a = _make_airspace(name="MOA Test")
        a.schedule = "0700 - 1800, MON - FRI"
        a.gmt_offset = 0
        a.dst_code = 0
        # Saturday 3am UTC → inactive
        dt = datetime(2026, 4, 18, 3, 0, tzinfo=timezone.utc)
        result = filter_by_schedule([a], at_datetime=dt)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Group 9: NASR schedule + floor_reference fields
# ---------------------------------------------------------------------------


class TestNASRScheduleFields:
    def test_feature_preserves_schedule(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-118, 33], [-117, 33], [-117, 34],
                    [-118, 34], [-118, 33],
                ]],
            },
            "properties": {
                "NAME": "R-2508",
                "TYPE_CODE": "R",
                "LOWER_VAL": 0,
                "LOWER_CODE": "SFC",
                "UPPER_VAL": 18000,
                "TIMESOFUSE": "0700 - 1800, MON - FRI",
                "GMTOFFSET": -8,
                "DST_CODE": 1,
            },
        }
        result = NASRAirspaceSource._feature_to_airspace(feature)
        assert result is not None
        assert result.floor_reference == "SFC"
        assert result.schedule == "0700 - 1800, MON - FRI"
        assert result.gmt_offset == -8
        assert result.dst_code == 1

    def test_feature_msl_floor(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "properties": {
                "NAME": "R-Test",
                "TYPE_CODE": "R",
                "LOWER_VAL": 5000,
                "LOWER_CODE": "MSL",
                "UPPER_VAL": 18000,
            },
        }
        result = NASRAirspaceSource._feature_to_airspace(feature)
        assert result is not None
        assert result.floor_reference == "MSL"


# ---------------------------------------------------------------------------
# Group 10: FlightPlanDB oceanic tracks
# ---------------------------------------------------------------------------


class TestFlightPlanDBClient:
    def test_parse_track(self):
        raw = [{
            "ident": "A",
            "validFrom": "2026-04-12T11:30:00.000Z",
            "validTo": "2026-04-12T19:00:00.000Z",
            "route": {
                "nodes": [
                    {"ident": "MALOT", "type": "FIX", "lat": 51.0, "lon": -10.0},
                    {"ident": "5220N", "type": "LATLON", "lat": 52.33, "lon": -20.0},
                    {"ident": "5230N", "type": "LATLON", "lat": 52.5, "lon": -30.0},
                ],
                "eastLevels": ["310", "320", "330"],
                "westLevels": None,
            },
        }]

        client = FlightPlanDBClient()

        with patch("hyplan.airspace._get_airspace_cache_dir") as mock_cache, \
             patch("hyplan.airspace.requests.get") as mock_get:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_cache.return_value = td
                mock_resp = MagicMock()
                mock_resp.json.return_value = raw
                mock_resp.raise_for_status = MagicMock()
                mock_get.return_value = mock_resp

                tracks = client.fetch_nats()
                assert len(tracks) == 1
                t = tracks[0]
                assert t.ident == "A"
                assert t.system == "NAT"
                assert len(t.waypoints) == 3
                assert t.east_levels == ["310", "320", "330"]
                assert t.geometry is not None
                assert t.geometry.is_valid

    def test_fetch_network_error(self):
        client = FlightPlanDBClient()
        with patch("hyplan.airspace._get_airspace_cache_dir") as mock_cache, \
             patch("hyplan.airspace.requests.get") as mock_get:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_cache.return_value = td
                mock_get.side_effect = requests.ConnectionError("no network")
                with pytest.raises(HyPlanRuntimeError, match="FlightPlanDB"):
                    client.fetch_nats()
