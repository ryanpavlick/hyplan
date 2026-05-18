"""JSON ↔ Aircraft profile round-trip tests.

Each aircraft class is loaded from its bundled JSON profile, re-dumped
to a temporary file, reloaded, and compared field-by-field for parity.
The shipped 22 JSON profiles act as a regression guard against schema
drift; the round-trip ensures dump/load are exact inverses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pint import Quantity

from hyplan.aircraft import (
    NASA_ER2, NASA_GIII, NASA_GIV, NASA_GV, NCAR_GV, NASA_C20A,
    NASA_P3, NOAA_WP3D, NOAA_GIV, NASA_WB57, NASA_B777,
    KingAirA90, KingAirB200, KingAir350,
    NASA_C130, NOAA_TwinOtter, BAS_TwinOtter, FAAM_BAe146,
    SAFIRE_ATR42, NERC_DO228, AWI_BaslerBT67, DLR_HALO,
)
from hyplan.aircraft._base import Aircraft
from hyplan.aircraft._profile_io import (
    dump_aircraft_profile,
    load_aircraft_profile,
    profile_path,
)


# Aircraft class -> JSON filename stem.  Pin every shipped pairing here.
ROSTER: list[tuple[type, str]] = [
    (NASA_ER2, "nasa_er2"),
    (NASA_GIII, "nasa_giii"),
    (NASA_GIV, "nasa_giv"),
    (NASA_GV, "nasa_gv"),
    (NCAR_GV, "ncar_gv"),
    (NASA_C20A, "nasa_c20a"),
    (NASA_P3, "nasa_p3"),
    (NOAA_WP3D, "noaa_wp3d"),
    (NOAA_GIV, "noaa_giv"),
    (NASA_WB57, "nasa_wb57"),
    (NASA_B777, "nasa_b777"),
    (KingAirA90, "king_air_a90"),
    (KingAirB200, "king_air_b200"),
    (KingAir350, "king_air_350"),
    (NASA_C130, "nasa_c130"),
    (NOAA_TwinOtter, "noaa_twin_otter"),
    (BAS_TwinOtter, "bas_twin_otter"),
    (FAAM_BAe146, "faam_bae146"),
    (SAFIRE_ATR42, "safire_atr42"),
    (NERC_DO228, "nerc_do228"),
    (AWI_BaslerBT67, "awi_basler_bt67"),
    (DLR_HALO, "dlr_halo"),
]


COMPARED_FIELDS = [
    "aircraft_type", "tail_number", "operator", "engine_type",
    "calibration_status",
    "service_ceiling", "approach_speed",
    "range", "endurance", "useful_payload", "stall_speed_cas",
    "descent_path_angle_max_deg", "climb_path_angle_max_deg",
    "climb_schedule", "cruise_schedule", "descent_schedule",
    "climb_profile", "descent_profile",
    "turn_model",
    "confidence", "sources",
    "approach_profile", "typical_climb_out",
]


def _approx_equal(a, b, tol=1e-6) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return abs(a.m_as(a.units) - b.to(a.units).m_as(a.units)) <= tol
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and bool(np.allclose(a, b, atol=tol))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_approx_equal(x, y, tol) for x, y in zip(a, b, strict=False))
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        return _approx_equal(a.__dict__, b.__dict__, tol)
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) & set(b.keys())
        return all(_approx_equal(a[k], b[k], tol) for k in keys)
    return a == b


class TestAircraftProfiles:
    """One round-trip parametrize for every aircraft in the roster."""

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_instantiates_from_bundled_json(self, cls, short):
        """Each Aircraft subclass instantiates from its bundled JSON."""
        ac = cls()
        assert isinstance(ac, Aircraft)
        # Shipped JSON exists on disk
        assert profile_path(short).exists(), f"Missing {short}.json"

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_round_trip_through_tmp_json(self, cls, short, tmp_path):
        """dump → load reproduces every compared field."""
        orig = cls()
        tmp_file = tmp_path / f"{short}.json"
        dump_aircraft_profile(orig, tmp_file)

        # Load the dumped file back, bypassing profile_path (which
        # looks under the bundled directory).
        kwargs = _load_from_path(tmp_file)
        restored = Aircraft(**kwargs)

        for field in COMPARED_FIELDS:
            o = getattr(orig, field, None)
            r = getattr(restored, field, None)
            assert _approx_equal(o, r), (
                f"{cls.__name__}.{field} mismatch: {o!r} != {r!r}"
            )

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_bundled_json_matches_class(self, cls, short):
        """The shipped JSON produces an Aircraft equal to the class instance."""
        # cls() and Aircraft(**load_aircraft_profile(short)) should agree
        # field-by-field — guards against drift between the JSON files
        # and the slim _models.py wrappers.
        orig = cls()
        restored = Aircraft(**load_aircraft_profile(short))
        for field in COMPARED_FIELDS:
            o = getattr(orig, field, None)
            r = getattr(restored, field, None)
            assert _approx_equal(o, r), (
                f"{cls.__name__} drift on field {field!r}"
            )


def _load_from_path(path: Path) -> dict:
    """Mirror of load_aircraft_profile but reads from an arbitrary path."""
    from hyplan.aircraft._profile_io import _profile_dict_to_kwargs
    return _profile_dict_to_kwargs(json.loads(path.read_text()))


class TestProfileSchema:
    """Schema-level checks on the bundled JSON files."""

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_required_top_level_keys(self, cls, short):
        raw = json.loads(profile_path(short).read_text())
        for required in (
            "metadata", "scalars",
            "climb_schedule", "cruise_schedule", "descent_schedule",
            "climb_profile", "descent_profile",
            "turn_model",
        ):
            assert required in raw, f"{short}.json missing {required!r}"

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_metadata_keys(self, cls, short):
        meta = json.loads(profile_path(short).read_text())["metadata"]
        assert set(meta) >= {
            "aircraft_type", "tail_number", "operator",
            "engine_type", "calibration_status",
        }
        assert meta["engine_type"] in ("jet", "turboprop", "piston")
        assert meta["calibration_status"] in (
            "calibrated", "inferred", "uncalibrated",
        )

    @pytest.mark.parametrize("cls,short", ROSTER)
    def test_schedule_type_discriminator(self, cls, short):
        raw = json.loads(profile_path(short).read_text())
        for key in ("climb_schedule", "cruise_schedule", "descent_schedule"):
            sched = raw[key]
            assert sched["type"] in ("tas", "cas_mach"), (
                f"{short}.{key}.type = {sched['type']!r}"
            )
