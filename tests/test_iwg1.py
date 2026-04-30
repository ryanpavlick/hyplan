"""Tests for hyplan.aircraft.iwg1.load_iwg1."""

from pathlib import Path

import pandas as pd
import pytest

from hyplan.aircraft import load_iwg1
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
