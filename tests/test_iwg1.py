"""Tests for hyplan.aircraft.iwg1.load_iwg1."""

from pathlib import Path

import pandas as pd
import pytest

from hyplan.aircraft import load_iwg1, trim_ground_taxi
from hyplan.exceptions import HyPlanValueError


def _write_iwg1_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a small IWG1-shaped CSV under tmp_path and return the path.

    Each row is a dict of column → value.  Missing columns are filled
    with empty strings (the IWG1 native form).
    """
    columns = [
        "HEADER", "TimeStamp", "Latitude", "Longitude",
        "GPS MSL Altitude", "WGS84 Altitude", "Pressure Altitude",
        "Radar Altitude", "Ground Speed", "True Airspeed",
        "Indicated Airspeed", "Mach Number", "Vertical Velocity",
        "True Heading", "Track", "Drift", "Pitch", "Roll",
        "Side Slip", "Angle of Attack", "Ambient Temp", "Dew Point",
        "Total Air Temp", "Static Press", "Dynamic Press",
        "Cabin Press", "Wind Speed", "Wind Direction",
        "Vertical Wind Speed", "Solar Zenith Angle",
        "Sun Elevation Aircraft", "Sun Azimuth Ground",
        "Sun Azimuth Aircraft",
    ]
    path = tmp_path / "test_iwg1.txt"
    with path.open("w") as fh:
        fh.write(",".join(columns) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in columns) + "\n")
    return path


def _row(ts: str, **kwargs) -> dict:
    base = {"HEADER": "IWG1", "TimeStamp": ts}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Schema + parsing
# ---------------------------------------------------------------------------


class TestSchema:
    def test_returns_expected_columns(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", Latitude=35.0, Longitude=-118.0,
                 **{"Pressure Altitude": 30000, "GPS MSL Altitude": 9100,
                    "True Airspeed": 200, "Mach Number": 0.7,
                    "Wind Speed": 30, "Wind Direction": 270,
                    "Pitch": 0.5, "Roll": 1.0}),
            _row("2024-01-01T00:00:05Z", Latitude=35.01, Longitude=-118.01,
                 **{"Pressure Altitude": 30100, "GPS MSL Altitude": 9130,
                    "True Airspeed": 201, "Mach Number": 0.7,
                    "Wind Speed": 30, "Wind Direction": 270,
                    "Pitch": 0.5, "Roll": 1.0}),
        ])
        df = load_iwg1(p)
        for col in [
            "timestamp", "latitude", "longitude", "altitude",
            "altitude_gps_ft", "tas_kt", "groundspeed", "mach",
            "vertical_rate", "true_heading", "track", "pitch_deg",
            "roll_deg", "wind_speed_kt", "wind_direction_deg",
            "ambient_temp_c", "static_pressure_hpa",
        ]:
            assert col in df.columns, f"missing column {col!r}"

    def test_timestamps_are_tz_naive_utc(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T12:00:00Z"),
            _row("2024-01-01T12:00:05Z"),
        ])
        df = load_iwg1(p)
        assert df["timestamp"].dt.tz is None
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01T12:00:00")

    def test_records_are_sorted_by_timestamp(self, tmp_path):
        # Write out of order; loader should sort.
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T12:00:10Z"),
            _row("2024-01-01T12:00:00Z"),
            _row("2024-01-01T12:00:05Z"),
        ])
        df = load_iwg1(p)
        assert (df["timestamp"].diff().dropna().dt.total_seconds() > 0).all()


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


class TestUnits:
    def test_gps_msl_altitude_meters_to_feet(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", **{"GPS MSL Altitude": 1000}),
            _row("2024-01-01T00:00:05Z", **{"GPS MSL Altitude": 2000}),
        ])
        df = load_iwg1(p)
        # 1000 m = 3280.84 ft
        assert df["altitude_gps_ft"].iloc[0] == pytest.approx(3280.84, abs=0.1)
        assert df["altitude_gps_ft"].iloc[1] == pytest.approx(6561.68, abs=0.1)

    def test_pressure_altitude_kept_as_feet(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", **{"Pressure Altitude": 30000}),
            _row("2024-01-01T00:00:05Z", **{"Pressure Altitude": 30000}),
        ])
        df = load_iwg1(p)
        assert df["altitude"].iloc[0] == pytest.approx(30000.0)

    def test_true_airspeed_mps_to_knots(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", **{"True Airspeed": 100}),
            _row("2024-01-01T00:00:05Z", **{"True Airspeed": 100}),
        ])
        df = load_iwg1(p)
        # 100 m/s = 194.38 kt
        assert df["tas_kt"].iloc[0] == pytest.approx(194.38, abs=0.05)

    def test_wind_speed_mps_to_knots(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", **{"Wind Speed": 25}),
            _row("2024-01-01T00:00:05Z", **{"Wind Speed": 25}),
        ])
        df = load_iwg1(p)
        assert df["wind_speed_kt"].iloc[0] == pytest.approx(48.6, abs=0.05)

    def test_track_normalized_to_0_360(self, tmp_path):
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", Track=-45.0),
            _row("2024-01-01T00:00:05Z", Track=400.0),
        ])
        df = load_iwg1(p)
        assert df["track"].iloc[0] == pytest.approx(315.0)
        assert df["track"].iloc[1] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Vertical-rate derivation
# ---------------------------------------------------------------------------


class TestVerticalRateDerivation:
    def test_climbing_at_constant_rate(self, tmp_path):
        # 5 fixes at 5 s spacing, alt rises by 500 ft each step → 6000 fpm.
        rows = [
            _row(f"2024-01-01T00:00:{5*i:02d}Z", **{"Pressure Altitude": 10000 + 500 * i})
            for i in range(5)
        ]
        p = _write_iwg1_csv(tmp_path, rows)
        df = load_iwg1(p)
        # Centered difference + rolling median: middle fixes should be exactly 6000 fpm.
        middle = df["vertical_rate"].iloc[1:-1]
        assert middle.mean() == pytest.approx(6000.0, abs=10.0)

    def test_descending_gives_negative_rate(self, tmp_path):
        rows = [
            _row(f"2024-01-01T00:00:{5*i:02d}Z", **{"Pressure Altitude": 30000 - 200 * i})
            for i in range(5)
        ]
        p = _write_iwg1_csv(tmp_path, rows)
        df = load_iwg1(p)
        middle = df["vertical_rate"].iloc[1:-1]
        assert middle.mean() == pytest.approx(-2400.0, abs=10.0)

    def test_level_flight_gives_zero_rate(self, tmp_path):
        rows = [
            _row(f"2024-01-01T00:00:{5*i:02d}Z", **{"Pressure Altitude": 60000})
            for i in range(6)
        ]
        p = _write_iwg1_csv(tmp_path, rows)
        df = load_iwg1(p)
        middle = df["vertical_rate"].iloc[1:-1]
        assert middle.abs().max() < 1.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_iwg1(tmp_path / "nonexistent.txt")

    def test_empty_file_raises(self, tmp_path):
        # Writes the header only, no data rows.
        p = _write_iwg1_csv(tmp_path, [])
        with pytest.raises(HyPlanValueError, match="empty"):
            load_iwg1(p)

    def test_missing_timestamp_column_raises(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("HEADER,Latitude,Longitude\nIWG1,35.0,-118.0\n")
        with pytest.raises(HyPlanValueError, match="TimeStamp"):
            load_iwg1(p)

    def test_blank_cells_become_nan(self, tmp_path):
        # Pressure Altitude blank — should land as NaN, not crash.
        p = _write_iwg1_csv(tmp_path, [
            _row("2024-01-01T00:00:00Z", **{"True Airspeed": 200}),
            _row("2024-01-01T00:00:05Z", **{"True Airspeed": 200}),
        ])
        df = load_iwg1(p)
        assert df["altitude"].isna().all()
        assert df["tas_kt"].notna().all()


# ---------------------------------------------------------------------------
# Real-data smoke test (skipped when the cache is empty)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# trim_ground_taxi
# ---------------------------------------------------------------------------


class TestTrimGroundTaxi:
    def _build_synthetic_sortie(self, tmp_path):
        """Synthetic sortie: 3 taxi fixes, takeoff roll, climb to 30 kft,
        cruise, descent, landing rollout, 3 taxi fixes — 50 fixes total."""
        rows = []
        ts = pd.Timestamp("2024-01-01T00:00:00")
        # 3 standing/taxi fixes at ground level, low GS.
        for i in range(3):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000, "Ground Speed": 5},  # 5 m/s ≈ 9.7 kt taxi
            ))
        # Takeoff roll: GS rising 0 → ~80 m/s, altitude near ground.
        for i in range(5):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 3))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000 + 50 * i, "Ground Speed": 30 + 12 * i},
            ))
        # Climb to 30 kft.
        for i in range(15):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 8))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 2000 + 2000 * i, "Ground Speed": 100},
            ))
        # Cruise.
        for i in range(15):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 23))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 30000, "Ground Speed": 100},
            ))
        # Descent.
        for i in range(15):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 38))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 30000 - 2000 * i, "Ground Speed": 80},
            ))
        # Landing rollout: high GS (all > 30 kt threshold), ground level.
        for i in range(5):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 53))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000, "Ground Speed": 60 - 5 * i},
            ))
        # 3 standing/taxi fixes.
        for i in range(3):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 58))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000, "Ground Speed": 3},
            ))
        return _write_iwg1_csv(tmp_path, rows)

    def test_trims_pre_and_post_taxi(self, tmp_path):
        p = self._build_synthetic_sortie(tmp_path)
        df = load_iwg1(p)
        n_before = len(df)
        trimmed = trim_ground_taxi(df)
        # The 3+3 leading/trailing taxi fixes should be gone.
        assert len(trimmed) == n_before - 6
        # First fix should be in the takeoff roll (GS > 30 kt).
        assert trimmed["groundspeed"].iloc[0] > 30
        # Last fix should be in the landing rollout (GS > 30 kt).
        assert trimmed["groundspeed"].iloc[-1] > 30

    def test_keeps_takeoff_rollout_via_groundspeed(self, tmp_path):
        # Takeoff roll has GS > 30 kt but altitude near ground reference.
        p = self._build_synthetic_sortie(tmp_path)
        df = load_iwg1(p)
        trimmed = trim_ground_taxi(df)
        # Verify there are fixes within 200 ft of ground (the rolls).
        ground_ref = trimmed["altitude"].min()
        near_ground = (trimmed["altitude"] - ground_ref) < 200
        assert near_ground.any(), "rollout fixes should be retained"

    def test_keeps_low_altitude_approach(self, tmp_path):
        # Synthesize a long approach below the AGL threshold but with
        # reasonable groundspeed.  Should be retained.
        rows = []
        ts = pd.Timestamp("2024-01-01T00:00:00")
        # 3 taxi fixes (excluded).
        for i in range(3):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000, "Ground Speed": 4},
            ))
        # Approach 100 ft AGL but GS 40 m/s ≈ 78 kt — should be kept.
        for i in range(8):
            rows.append(_row(
                (ts + pd.Timedelta(seconds=5 * (i + 3))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1100, "Ground Speed": 40},
            ))
        p = _write_iwg1_csv(tmp_path, rows)
        df = load_iwg1(p)
        trimmed = trim_ground_taxi(df)
        assert len(trimmed) == 8

    def test_empty_in_empty_out(self):
        empty = pd.DataFrame(columns=["altitude", "groundspeed"])
        out = trim_ground_taxi(empty)
        assert len(out) == 0

    def test_no_airborne_fixes_returns_empty(self, tmp_path):
        # All 5 fixes are slow + ground level — the trace is pure taxi.
        rows = [
            _row(
                (pd.Timestamp("2024-01-01T00:00:00") + pd.Timedelta(seconds=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                **{"Pressure Altitude": 1000, "Ground Speed": 5},
            )
            for i in range(5)
        ]
        p = _write_iwg1_csv(tmp_path, rows)
        df = load_iwg1(p)
        trimmed = trim_ground_taxi(df)
        assert len(trimmed) == 0


@pytest.mark.skipif(
    not list(Path("data/er2").glob("*.txt")),
    reason="requires data/er2/*.txt cache",
)
class TestRealData:
    def test_loads_a_real_file(self):
        p = next(iter(Path("data/er2").glob("*.txt")))
        df = load_iwg1(p)
        assert len(df) > 0
        # Real ER-2 sorties cover a wide TAS range; smoke-test that
        # the loader reproduces it sensibly.
        assert df["tas_kt"].max() > 100  # cruise

    def test_real_file_altitude_gps_matches_pressure_at_low_altitude(self):
        # On the ground, pressure altitude and GPS MSL altitude should
        # agree to within a few hundred feet (atmospheric pressure
        # offset).  Use this as a unit-conversion sanity check.
        p = next(iter(Path("data/er2").glob("*.txt")))
        df = load_iwg1(p)
        low = df[df["altitude"] < 5000].dropna(subset=["altitude_gps_ft"])
        if len(low) >= 10:
            diff = (low["altitude_gps_ft"] - low["altitude"]).abs().median()
            assert diff < 1000, f"GPS vs pressure altitude differ by median {diff:.0f} ft"
