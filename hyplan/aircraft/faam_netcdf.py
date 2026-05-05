"""FAAM (UK Met Office) BAe-146 G-LUXE NetCDF flight-data loader.

The Facility for Airborne Atmospheric Measurements (FAAM) operates a
modified BAe-146-301 (G-LUXE) and publishes per-flight "core" data
through CEDA at:

    https://data.ceda.ac.uk/badc/faam/data/<YYYY>/<flight_id>/core_processed/
        core_faam_<YYYYMMDD>_v###_r#_<flight_id>.nc

Files are NetCDF4 with 1-Hz and 32-Hz processed variants.  Variable
naming follows the FAAM Core Data Product spec:
``ALT_GIN`` (geometric altitude), ``PS_RVSM`` (static pressure),
``PALT_RVSM`` (pressure altitude), ``IAS_RVSM`` (indicated airspeed),
``TAS_RVSM`` (true airspeed), ``ROLL_GIN`` / ``PTCH_GIN`` /
``HDG_GIN`` (attitude from the Honeywell GIN), ``LAT_GIN`` /
``LON_GIN`` (position).

Status: SKELETON ONLY — implementation deferred until a CEDA-
registered download path is in place.  The FAAM core_*.nc files
require CEDA login (auth.ceda.ac.uk).  Register at
https://services.ceda.ac.uk/cedasite/register/start/.

The loader returns a DataFrame matching :func:`load_iwg1`'s contract
so calibration notebooks can ingest FAAM data the same way they
ingest IWG1 .txt and ICARTT .ict files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


# Map FAAM core variable names to canonical IWG1-style fields.
# Reference: FAAM Core Data Product spec (https://www.faam.ac.uk/sphinx/coredata/).
_VAR_MAP = {
    "TIME":      ("timestamp",          None),
    "Time":      ("timestamp",          None),
    "LAT_GIN":   ("latitude",           "deg"),
    "LON_GIN":   ("longitude",          "deg"),
    "ALT_GIN":   ("altitude_gps_ft",    "m"),
    "PALT_RVSM": ("altitude",           "m"),  # Pressure altitude
    "IAS_RVSM":  ("ias_kt",             "ms"),
    "TAS_RVSM":  ("tas_kt",             "ms"),
    "GSPD_GIN":  ("groundspeed",        "ms"),
    "VSPD_GIN":  ("vertical_velocity",  "ms"),
    "HDG_GIN":   ("true_heading",       "deg"),
    "TRCK_GIN":  ("track",              "deg"),
    "PTCH_GIN":  ("pitch_deg",          "deg"),
    "ROLL_GIN":  ("roll_deg",           "deg"),
    "TAT_DI_R":  ("ambient_temp_c",     "C"),
    "PS_RVSM":   ("static_pressure_hpa","hPa"),
}


def load_faam_netcdf(path: Union[str, Path]) -> pd.DataFrame:
    """Load one FAAM core NetCDF file into the canonical schema.

    NOT YET IMPLEMENTED — placeholder until CEDA access is set up
    and a representative file can be tested against.
    """
    raise NotImplementedError(
        "FAAM NetCDF loader is a skeleton; implementation deferred "
        "until CEDA-registered access to data.ceda.ac.uk/badc/faam/ "
        "is configured.  See module docstring for the file layout."
    )
