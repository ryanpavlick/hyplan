# `data/er2/` — local NASA ER-2 IWG1 flight log cache

This directory holds **local-only** caches of per-sortie IWG1 in-situ
flight data from NASA ER-2 (`AAF954` / NASA806 and the related airframe).
Everything under [data/er2](.) is gitignored; only this README and
[.gitignore](.gitignore) are tracked.

## What's in an IWG1 file

IWG1 (Inter-agency Working Group 1) is NASA's standard interchange
format for airborne in-situ aircraft data. Each `.txt` file is a
header-prefixed CSV with one record per ~5 s, covering the full sortie
from taxi to taxi. Columns include (units inferred from the data):

| Column              | Unit  |
| ------------------- | ----- |
| `Latitude`          | deg   |
| `Longitude`         | deg   |
| `GPS MSL Altitude`  | m     |
| `WGS84 Altitude`    | m     |
| `Pressure Altitude` | ft    |
| `Radar Altitude`    | ft    |
| `Ground Speed`      | m/s   |
| `True Airspeed`     | m/s   |
| `Indicated Airspeed`| m/s   |
| `Mach Number`       | —     |
| `Vertical Velocity` | m/s   |
| `True Heading`      | deg   |
| `Track`             | deg   |
| `Pitch`, `Roll`     | deg   |
| `Wind Speed`        | m/s   |
| `Wind Direction`    | deg   |
| `Static Press`      | hPa   |
| `Ambient Temp`      | °C    |

Compared with the `data/adsb/` ADS-B cache, IWG1 is much richer:
direct measurements of TAS, wind, attitude, mach, and temperature, so
no reconstruction step is needed and the calibration is more direct.

## Loader

Use [`hyplan.aircraft.iwg1.load_iwg1`](../../hyplan/aircraft/iwg1.py)
to read one file into a normalized DataFrame with HyPlan-conventional
column names and units (timestamps tz-naive UTC, altitudes in ft, TAS
in kt, vertical rate in fpm, etc.).

## Layout

```
data/er2/
├── .gitignore   (tracked; excludes everything below)
├── README.md    (this file)
└── *.txt        (per-sortie IWG1 logs, gitignored)
```

## Distribution

NASA AFRC IWG1 logs are mission/program-specific. Some campaigns
publish openly via the Earthdata DAAC; others remain internal during
the active mission. We keep the raw logs out of git and publish only
calibrated `NASA_ER2()` constants downstream — those are far enough
from the source data to count as code, not derived database content.
