"""Tests for hyplan.aircraft.icartt — NASA ICARTT (.ict) loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyplan.aircraft.icartt import (
    _convert_to_canonical,
    detect_platform,
    load_icartt,
)
from hyplan.exceptions import HyPlanValueError

# ----------------------------------------------------------------------
# Fixture-builder helpers
# ----------------------------------------------------------------------


def _write_ict(
    tmp_path: Path,
    *,
    columns: list[tuple[str, str]],
    data_rows: list[list[float]],
    base_date: tuple[int, int, int] = (2024, 6, 15),
    extra_header_lines: list[str] | None = None,
    encoding: str = "utf-8",
    sep: str = ",",
    filename: str = "test.ict",
    indep_col: tuple[str, str] = ("Time_UTC", "Seconds"),
    scale_factors: list[float] | None = None,
    missing_values: list[float] | None = None,
) -> Path:
    """Write a minimal but spec-compliant ICARTT FFI 1001 file.

    ``columns`` is a list of (name, unit) for the dependent variables;
    the first independent variable is taken from ``indep_col``.  Per-
    column scale factors and missing-value markers default to 1.0 and
    -9999 respectively, matching the typical ICARTT convention.
    """
    n_dep = len(columns)
    # 12 spec preamble lines + n_dep var lines + 2 comment-count lines
    # + any extras + 1 column line  →  n_header points at the column line
    # (1-indexed).  So:  n_header = 15 + n_dep + len(extras).
    extras = list(extra_header_lines or [])
    n_header = 15 + n_dep + len(extras)
    scales = scale_factors if scale_factors is not None else [1.0] * n_dep
    missings = missing_values if missing_values is not None else [-9999] * n_dep
    header_lines = [
        f"{n_header}, 1001\n",
        "Last, First\n",                                # PI line
        "Institution\n",                                # affiliation
        "Test platform\n",                              # source
        "Test campaign\n",                              # mission
        "1, 1\n",                                       # file vol / # vols
        f"{base_date[0]}, {base_date[1]:02d}, {base_date[2]:02d}, "
        f"{base_date[0]}, {base_date[1]:02d}, {base_date[2]:02d}\n",
        "0\n",                                          # data interval
        f"{indep_col[0]}, {indep_col[1]}\n",            # indep var
        f"{n_dep}\n",                                   # number of dep vars
        ", ".join(str(s) for s in scales) + "\n",        # scale factors
        ", ".join(str(m) for m in missings) + "\n",      # missing values
    ]
    var_lines = [f"{name}, {unit}\n" for name, unit in columns]
    pre_column_meta = [
        "0\n",                                          # # special-comment lines
        "0\n",                                          # # normal-comment lines
    ]
    column_line = sep.join([indep_col[0], *[c[0] for c in columns]]) + "\n"
    data_block = "\n".join(
        sep.join(f"{v}" for v in row) for row in data_rows
    ) + "\n"

    contents = (
        "".join(header_lines)
        + "".join(var_lines)
        + "".join(pre_column_meta)
        + "".join(extras)
        + column_line
        + data_block
    )

    path = tmp_path / filename
    path.write_text(contents, encoding=encoding)
    return path


def _simple_climb(
    tmp_path: Path,
    *,
    n: int = 60,
    sep: str = ",",
    encoding: str = "utf-8",
) -> Path:
    """A 60-second straight climb sortie with full coverage of common fields."""
    rows = []
    for i in range(n):
        t = float(i)
        rows.append([
            t,
            38.0 + 0.0001 * i,           # lat
            -77.0 + 0.0001 * i,          # lon
            float(5_000 + 50 * i),       # pressure altitude ft
            float(5_000 + 50 * i),       # gps altitude ft
            200.0,                       # tas kt
            195.0,                       # ias kt
            0.30,                        # mach
            180.0,                       # true heading deg
            5.0,                         # roll deg
            10.0,                        # wind speed kt
            270.0,                       # wind direction deg
        ])
    return _write_ict(
        tmp_path,
        columns=[
            ("Latitude",         "deg"),
            ("Longitude",        "deg"),
            ("Pressure_Altitude","ft"),
            ("GPS_Altitude",     "ft"),
            ("True_Air_Speed",   "kt"),
            ("Indicated_Air_Speed","kt"),
            ("Mach_Number",      "mach"),
            ("True_Heading",     "deg"),
            ("Roll",             "deg"),
            ("Wind_Speed",       "kt"),
            ("Wind_Direction",   "deg"),
        ],
        data_rows=rows,
        sep=sep,
        encoding=encoding,
    )


# ----------------------------------------------------------------------
# _convert_to_canonical
# ----------------------------------------------------------------------


class TestConvertToCanonical:
    def test_altitude_in_meters_converts_to_feet(self):
        x = pd.Series([300.0])           # metres
        out = _convert_to_canonical("altitude", "m", x)
        assert out.iloc[0] == pytest.approx(984.252, rel=1e-3)  # 300 / 0.3048

    def test_altitude_in_km_converts_to_feet(self):
        x = pd.Series([1.0])             # km
        out = _convert_to_canonical("altitude", "km", x)
        assert out.iloc[0] == pytest.approx(3280.84, rel=1e-3)

    def test_altitude_gps_ft_unit_kept(self):
        x = pd.Series([10000.0])
        out = _convert_to_canonical("altitude_gps_ft", "ft", x)
        assert out.iloc[0] == 10000.0

    def test_longitude_wrap_360_to_pm_180(self):
        # 280 °E should become -80
        x = pd.Series([280.0, 100.0, 359.0])
        out = _convert_to_canonical("longitude", "deg", x)
        assert out.iloc[0] == pytest.approx(-80.0)
        assert out.iloc[1] == pytest.approx(100.0)
        assert out.iloc[2] == pytest.approx(-1.0)

    def test_tas_in_mps_converts_to_kt(self):
        x = pd.Series([100.0])           # m/s
        out = _convert_to_canonical("tas_kt", "m/s", x)
        assert out.iloc[0] == pytest.approx(194.384, rel=1e-3)

    def test_vertical_velocity_in_mps_converts_to_fpm(self):
        x = pd.Series([5.0])             # m/s
        out = _convert_to_canonical("vertical_velocity", "m/s", x)
        assert out.iloc[0] == pytest.approx(984.252, rel=1e-3)

    def test_heading_normalized_to_0_360(self):
        x = pd.Series([-90.0, 450.0])
        out = _convert_to_canonical("true_heading", "deg", x)
        assert out.iloc[0] == pytest.approx(270.0)
        assert out.iloc[1] == pytest.approx(90.0)


# ----------------------------------------------------------------------
# load_icartt
# ----------------------------------------------------------------------


class TestLoadIcartt:
    def test_smoke_basic_climb(self, tmp_path):
        path = _simple_climb(tmp_path)
        df = load_icartt(path)
        assert len(df) == 60
        for col in ("timestamp", "latitude", "longitude", "altitude",
                    "tas_kt", "ias_kt", "vertical_rate"):
            assert col in df.columns

    def test_timestamp_is_base_date_plus_seconds(self, tmp_path):
        path = _simple_climb(tmp_path, n=5)
        df = load_icartt(path)
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-06-15 00:00:00")
        assert df["timestamp"].iloc[-1] == pd.Timestamp("2024-06-15 00:00:04")

    def test_pressure_altitude_used_when_available(self, tmp_path):
        path = _simple_climb(tmp_path)
        df = load_icartt(path)
        # First sample is 5000 ft; 50 ft/s climb for 60 s
        assert df["altitude"].iloc[0] == pytest.approx(5_000)
        assert df["altitude"].iloc[-1] == pytest.approx(5_000 + 50 * 59)

    def test_gps_altitude_fallback_when_pressure_missing(self, tmp_path):
        # Build file with only GPS altitude.
        rows = [[float(i), 38.0, -77.0, float(10_000 + 10 * i)]
                for i in range(20)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",     "deg"),
                ("Longitude",    "deg"),
                ("GPS_Altitude", "ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        assert df["altitude"].notna().any()
        # All samples should be filled from GPS altitude
        assert df["altitude"].iloc[0] == pytest.approx(10_000)

    def test_sentinel_minus9999_becomes_nan(self, tmp_path):
        rows = [
            [0.0, 38.0, -77.0, 5000.0],
            [1.0, -9999.0, -77.0, 5000.0],         # bad lat
            [2.0, 38.0, -9999.0, 5000.0],          # bad lon
            [3.0, 38.0, -77.0, -9999.0],           # bad altitude
        ]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        assert pd.isna(df["latitude"].iloc[1])
        assert pd.isna(df["longitude"].iloc[2])
        assert pd.isna(df["altitude"].iloc[3])

    def test_vertical_rate_climb_is_positive(self, tmp_path):
        # 60 s linear climb at 50 ft/s → 3000 fpm
        path = _simple_climb(tmp_path)
        df = load_icartt(path)
        mid = df["vertical_rate"].iloc[30]
        assert mid == pytest.approx(3000.0, rel=0.1)

    def test_vertical_rate_with_short_file_falls_back_to_nan(self, tmp_path):
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=[[0.0, 38.0, -77.0, 5000.0]],
        )
        df = load_icartt(path)
        assert len(df) == 1
        assert pd.isna(df["vertical_rate"].iloc[0])

    def test_unit_conversion_meters_to_feet_via_loader(self, tmp_path):
        rows = [[float(i), 38.0, -77.0, 1524.0] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",  "deg"),
                ("Longitude", "deg"),
                ("GPS_Altitude", "m"),       # metres on the wire
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        # 1524 m = 5000 ft
        assert df["altitude"].iloc[0] == pytest.approx(5_000, rel=1e-3)

    def test_longitude_0_360_wrapped_to_pm_180(self, tmp_path):
        rows = [[float(i), 38.0, 280.0, 5000.0] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        np.testing.assert_allclose(df["longitude"].to_numpy(), -80.0)

    def test_out_of_range_latitude_becomes_nan(self, tmp_path):
        rows = [[float(i), 95.0 if i == 2 else 38.0, -77.0, 5000.0]
                for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        assert pd.isna(df["latitude"].iloc[2])

    def test_alternative_column_names_via_substring_patterns(self, tmp_path):
        # NCAR/RAF short names: GLAT, GLON, PALTF, TASX, ATX
        rows = [[float(i), 38.0, -77.0, 5000.0, 200.0] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("GLAT",  "deg"),
                ("GLON",  "deg"),
                ("PALTF", "ft"),
                ("TASX",  "kt"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        assert df["latitude"].iloc[0] == pytest.approx(38.0)
        assert df["longitude"].iloc[0] == pytest.approx(-77.0)
        assert df["altitude"].iloc[0] == pytest.approx(5000.0)
        assert df["tas_kt"].iloc[0] == pytest.approx(200.0)

    def test_unmapped_canonical_columns_filled_with_nan(self, tmp_path):
        rows = [[0.0, 38.0, -77.0, 5000.0]]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        # ias_kt, mach, etc. were not in the file; they should still
        # exist as NaN columns so the schema is stable for downstream code.
        for col in ("ias_kt", "mach", "pitch_deg", "roll_deg"):
            assert col in df.columns
            assert pd.isna(df[col].iloc[0])

    def test_latin1_encoded_file_loads(self, tmp_path):
        path = _simple_climb(tmp_path, n=5, encoding="latin-1")
        df = load_icartt(path)
        assert len(df) == 5


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


class TestLoadIcarttErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_icartt(tmp_path / "does_not_exist.ict")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.ict"
        path.write_text("")
        with pytest.raises(HyPlanValueError):
            load_icartt(path)

    def test_no_data_rows_raises(self, tmp_path):
        # Build a header-only ICARTT (column line but no data rows).
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude", "deg"),
                ("Longitude", "deg"),
                ("Pressure_Altitude", "ft"),
            ],
            data_rows=[],
        )
        with pytest.raises(HyPlanValueError):
            load_icartt(path)

    def test_bad_date_in_line7_raises(self, tmp_path):
        path = tmp_path / "bad.ict"
        path.write_text(
            "20, 1001\n"
            "PI\nInst\nSrc\nMission\n1, 1\n"
            "BOGUS-DATE-LINE\n"
            "0\nTime, s\n0\n\n\n"
            "Time, Lat\n0, 38.0\n"
        )
        with pytest.raises(HyPlanValueError):
            load_icartt(path)


# ----------------------------------------------------------------------
# detect_platform
# ----------------------------------------------------------------------


class TestDetectPlatform:
    def test_finds_platform_in_extra_header(self, tmp_path):
        rows = [[float(i), 38.0, -77.0, 5000.0] for i in range(3)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
            extra_header_lines=["PLATFORM: NASA DC-8\n"],
        )
        # n_header counted the extra line, so PLATFORM lives in the
        # header block.  Detect should find it.
        platform = detect_platform(path)
        assert platform is not None
        assert "DC-8" in platform

    def test_returns_none_when_platform_missing(self, tmp_path):
        path = _simple_climb(tmp_path, n=3)
        assert detect_platform(path) is None


# ----------------------------------------------------------------------
# Scale factors + per-column missing values (MMS-style files)
# ----------------------------------------------------------------------


class TestScaleFactors:
    """ICARTT spec line 11 lists per-column scale factors that the reader
    must multiply into each column.  Older campaigns ship 1.0 / column
    (trivial), but high-rate instrument files like NASA MMS on the WB-57
    use scale factors of 0.01 / 0.00001 to fit dynamic range into
    integer storage.  Without scale-factor application the canonical
    columns come out 100x to 100 000x too large and get caught by the
    out-of-range filters.
    """

    def test_scale_factor_applied_to_altitude(self, tmp_path):
        # Altitude column stored as integer metres × 10 (i.e. scale 0.1).
        # Raw 15240 → 1524 m → 5000 ft after m→ft conversion.
        rows = [[float(i), 38.0, -77.0, 15240] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",     "deg"),
                ("Longitude",    "deg"),
                ("GPS_Altitude", "m"),
            ],
            data_rows=rows,
            scale_factors=[1.0, 1.0, 0.1],
        )
        df = load_icartt(path)
        assert df["altitude"].iloc[0] == pytest.approx(5000, rel=1e-3)

    def test_scale_factor_applied_to_latitude_longitude(self, tmp_path):
        # MMS-style: lat/lon stored as integer micro-degrees (scale 1e-5).
        rows = [[float(i), 3800000, -7700000, 5000] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
            scale_factors=[0.00001, 0.00001, 1.0],
        )
        df = load_icartt(path)
        assert df["latitude"].iloc[0] == pytest.approx(38.0)
        assert df["longitude"].iloc[0] == pytest.approx(-77.0)

    def test_scale_factor_applied_to_tas_in_mps(self, tmp_path):
        # TAS_MMS stored as integer cm/s (scale 0.01 → m/s) then
        # converted m/s → kt.  Raw 8916 → 89.16 m/s → ~173 kt.
        rows = [[float(i), 38.0, -77.0, 5000, 8916] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
                ("TAS",              "m/s"),
            ],
            data_rows=rows,
            scale_factors=[1.0, 1.0, 1.0, 0.01],
        )
        df = load_icartt(path)
        assert df["tas_kt"].iloc[0] == pytest.approx(173.3, rel=1e-2)

    def test_default_scale_factor_one_does_not_alter_values(self, tmp_path):
        # Regression: when scale factors line is all 1.0 (typical),
        # the parser should produce the same values it did pre-patch.
        rows = [[float(i), 38.0, -77.0, 5000, 200] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
                ("TAS",              "kt"),
            ],
            data_rows=rows,
            scale_factors=[1.0, 1.0, 1.0, 1.0],
        )
        df = load_icartt(path)
        assert df["altitude"].iloc[0] == pytest.approx(5000)
        assert df["tas_kt"].iloc[0] == pytest.approx(200)
        assert df["latitude"].iloc[0] == pytest.approx(38.0)


class TestPerColumnMissingValues:
    """Per-column missing-value markers (ICARTT line 12).  MMS uses a
    different magnitude per column (e.g. -9999 for TAS, -9999999 for
    latitude in micro-degrees, -99999999 for longitude).
    """

    def test_custom_column_specific_missing_value(self, tmp_path):
        # Use a per-column missing value that's NOT in the global
        # fallback sentinel set (so we know it's being read from line 12).
        rows = [
            [0.0, 38.0, -77.0, 5000.0],
            [1.0, 12345.0, -77.0, 5000.0],  # 12345 is the lat missing
        ]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
            missing_values=[12345, -9999, -9999],
        )
        df = load_icartt(path)
        # First row keeps a valid latitude, second is NaN (matched line-12
        # missing-value marker 12345, not the global fallback).
        assert df["latitude"].iloc[0] == pytest.approx(38.0)
        assert pd.isna(df["latitude"].iloc[1])

    def test_legacy_global_sentinel_minus_9999999_recognised(self, tmp_path):
        # Files in the wild (e.g. MMS lat in micro-degrees) sometimes
        # rely on the global-fallback set for missing values that aren't
        # listed correctly in line 12.  The fallback should catch them.
        rows = [
            [0.0, 38.0, -77.0, 5000.0],
            [1.0, -9999999.0, -77.0, 5000.0],
        ]
        path = _write_ict(
            tmp_path,
            columns=[
                ("Latitude",         "deg"),
                ("Longitude",        "deg"),
                ("Pressure_Altitude","ft"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        # Row 1's lat = -9999999 should be NaN via the global fallback,
        # since -9999999 is one of the extended MMS-era sentinels.
        assert pd.isna(df["latitude"].iloc[1])


class TestMMSColumnPatterns:
    """NASA WB-57 MMS files use short column names like G_LAT_MMS /
    G_LONG_MMS / G_ALT_MMS that must map to the canonical
    latitude / longitude / altitude_gps_ft slots.
    """

    def test_mms_short_column_names_map_correctly(self, tmp_path):
        rows = [[float(i), 38.0, -77.0, 5000.0] for i in range(5)]
        path = _write_ict(
            tmp_path,
            columns=[
                ("G_LAT_MMS",  "deg"),
                ("G_LONG_MMS", "deg"),
                ("G_ALT_MMS",  "m"),
            ],
            data_rows=rows,
        )
        df = load_icartt(path)
        assert df["latitude"].iloc[0] == pytest.approx(38.0)
        assert df["longitude"].iloc[0] == pytest.approx(-77.0)
        # GPS altitude m→ft via canonical conversion
        # (after fallback because Pressure_Altitude is absent).
        assert df["altitude"].iloc[0] == pytest.approx(16404, rel=1e-3)
