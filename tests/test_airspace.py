"""Tests for hyplan.airspace."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import requests
from shapely.geometry import LineString, box

from hyplan.airspace import (
    Airspace,
    OpenAIPClient,
    check_airspace_conflicts,
    fetch_and_check,
    _parse_airspace_item,
    _cache_key,
    _is_cache_stale,
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

    def test_integration(self):
        """Mocked end-to-end: fetch airspaces and detect conflict."""
        fl = _make_flight_line(34.0, -118.0, 34.1, -118.0, 5000)

        fake_airspaces = [_make_airspace(floor_ft=0, ceiling_ft=10000)]

        with patch.object(OpenAIPClient, "fetch_airspaces", return_value=fake_airspaces):
            conflicts = fetch_and_check([fl], api_key="test-key")
            assert len(conflicts) == 1
            assert conflicts[0].airspace.name == "Test Airspace"
