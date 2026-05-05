"""NSF/NCAR EOL ICARTT (NASA AMES) flight-level data loader.

NCAR Earth Observing Laboratory publishes HIAPER GV (N677F) and
NCAR C-130 flight-level data through https://data.eol.ucar.edu/.
Two formats are commonly available per project:

* **NASA AMES ICARTT** (.ict) — same FFI 1001 format as the existing
  :mod:`hyplan.aircraft.icartt` loader handles.  Project-specific
  variable names (e.g., HIAPER LRT files use ``GGALT`` for GPS
  altitude, ``ATX`` for ambient temp, ``TASX`` for true airspeed).

* **NetCDF** — 1-Hz LRT and 25-Hz HRT variants.  Same variable
  naming conventions as the ICARTT files.

Most NCAR EOL datasets require a one-line "ORDER" request through
the dataset page at data.eol.ucar.edu, processed via email.  Some
older HIPPO-era datasets are direct-download.

Status: PARTIAL — the existing ICARTT loader already handles HIAPER
LRT files if the column-name patterns are extended.  This module
adds the NCAR-specific name mappings the generic ICARTT loader's
patterns don't currently cover.

Once an NCAR EOL download path is configured, calibration of
NCAR_GV (HIAPER) and NCAR_C130 can use the same notebook structure
as the FIREX-AQ Twin Otter / ACT-America B-200 calibrations.
"""

from __future__ import annotations

# NCAR-specific column-name patterns that complement the generic
# patterns in :mod:`hyplan.aircraft.icartt`._COLUMN_PATTERNS.  These
# are the canonical names from NCAR HIAPER LRT files.
_NCAR_COLUMN_HINTS = {
    # NCAR name → (canonical_name, unit_hint)
    "GGALT":   ("altitude_gps_ft",    "m"),     # GPS altitude
    "PALT":    ("altitude",           "m"),     # Pressure altitude (computed)
    "PALTF":   ("altitude",           "ft"),
    "TASX":    ("tas_kt",             "ms"),    # True airspeed
    "IASX":    ("ias_kt",             "ms"),    # Indicated airspeed
    "MACHX":   ("mach",               "mach"),
    "GSF":     ("groundspeed",        "ms"),
    "WSC":     ("wind_speed_kt",      "ms"),    # Horizontal wind speed (computed)
    "WDC":     ("wind_direction_deg", "deg"),   # Wind direction
    "VSPD":    ("vertical_velocity",  "ms"),    # Vertical speed (m/s)
    "THDG":    ("true_heading",       "deg"),
    "TKAT":    ("track",              "deg"),
    "PITCH":   ("pitch_deg",          "deg"),
    "ROLL":    ("roll_deg",           "deg"),
    "ATTACK":  ("aoa_deg",            "deg"),
    "ATX":     ("ambient_temp_c",     "C"),     # Ambient temp (computed)
    "PSXC":    ("static_pressure_hpa","hPa"),   # Static pressure (computed)
    "GLAT":    ("latitude",           "deg"),
    "GLON":    ("longitude",          "deg"),
}
