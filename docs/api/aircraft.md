# Aircraft

Aircraft performance models with speed profiles, climb/descent rates,
turn radii, and endurance limits.  22 pre-configured research aircraft
are included; custom aircraft can be created by instantiating
{py:class}`~hyplan.aircraft.Aircraft` directly.  Performance values
for the pre-configured fleet live in editable JSON files at
`hyplan/data/aircraft/<short_name>.json` (see the *Profile I/O*
section below).

## Fleet overview

The table below is auto-generated from the live `hyplan.aircraft`
classes by `python -m docs._gen_fleet_tables`.  Re-run it whenever
calibration sortie counts or performance values change.

<!-- BEGIN AUTOGEN: fleet_overview -->

| Class | Airframe | Operator | Tail(s) | Ceiling (ft) | Range (nmi) | Endurance (hr) | Engine | Calibration |
|---|---|---|---|---:|---:|---:|---|---|
| `NASA_ER2` | ER-2 | NASA AFRC | NASA 806 | 70,000 | 5,000 | 8.0 | jet | calibrated |
| `NASA_GIII` | Gulfstream III | NASA LaRC | NASA 520 | 45,000 | 3,767 | 7.5 | jet | calibrated · n=153 |
| `NASA_GIV` | Gulfstream IV | NASA AFRC | NASA 817 | 45,000 | 5,130 | 7.5 | jet | brochure only |
| `NASA_GV` | Gulfstream V | NASA AFRC | NASA 95 | 51,000 | 5,500 | 13.0 | jet | calibrated · n=101 |
| `NCAR_GV` | Gulfstream V | NSF/NCAR EOL | N677F | 51,000 | 6,500 | 14.0 | jet | calibrated |
| `NASA_C20A` | C-20A | NASA AFRC | NASA 502 | 45,000 | 3,400 | 6.0 | jet | inferred |
| `NASA_P3` | P-3 Orion | NASA WFF | NASA 426 | 27,000 | 3,800 | 12.0 | turboprop | calibrated · n=252 |
| `NOAA_WP3D` | P-3 Orion (WP-3D) | NOAA AOC | N42RF + N43RF | 27,600 | 3,800 | 12.0 | turboprop | calibrated · n=96 |
| `NOAA_GIV` | Gulfstream IV-SP | NOAA AOC | N49RF | 47,500 | 4,220 | 8.5 | jet | calibrated · n=93 |
| `NASA_WB57` | WB-57 | NASA JSC | NASA 926/927 | 63,000 | 2,500 | 6.5 | jet | calibrated · n=100 |
| `NASA_B777` | B777 | NASA LaRC | Unknown | 43,000 | 9,000 | 18.0 | jet | brochure only |
| `KingAirA90` | King Air 90 | Unknown | Unknown | 30,000 | 1,500 | 6.0 | turboprop | brochure only |
| `KingAirB200` | King Air 200 | NASA (multiple) | multi-tail | 30,000 | 1,632 | 6.0 | turboprop | calibrated |
| `KingAir350` | King Air 350 | Unknown | Unknown | 35,000 | 2,100 | 5.0 | turboprop | brochure only |
| `NASA_C130` | C-130H Hercules | NASA WFF | NASA 436 | 28,000 | 2,500 | 10.0 | turboprop | calibrated · n=91 |
| `NOAA_TwinOtter` | DHC-6 Twin Otter | NOAA | N48RF + N46RF | 17,500 | 800 | 6.0 | turboprop | calibrated · n=164 |
| `BAS_TwinOtter` | DHC-6 Twin Otter | BAS | VP-FBL + VP-FBB | 14,600 | 800 | 6.0 | turboprop | calibrated · n=105 |
| `FAAM_BAe146` | BAe-146-301 | FAAM | G-LUXE | 34,500 | 1,800 | 5.0 | jet | calibrated · n=125 |
| `SAFIRE_ATR42` | ATR-42-320 | SAFIRE | F-HMTO | 24,700 | 900 | 5.0 | turboprop | calibrated |
| `NERC_DO228` | Dornier Do228-101 | NERC ARSF | D-CALM | 22,000 | 1,400 | 5.0 | turboprop | calibrated |
| `AWI_BaslerBT67` | Basler BT-67 | AWI | Polar 5 + Polar 6 | 25,000 | 1,600 | 6.5 | turboprop | calibrated |
| `DLR_HALO` | Gulfstream G550 | DLR | D-ADLR | 44,300 | 6,750 | 10.0 | jet | calibrated · n=18 |

<!-- END AUTOGEN: fleet_overview -->

See [`docs/calibration.md`](../calibration.md) for the per-aircraft
calibration provenance and a breakdown of the underlying data archives.

## Base class

```{eval-rst}
.. autoclass:: hyplan.aircraft.Aircraft
   :members:
   :show-inheritance:
```

## Pre-configured aircraft

```{eval-rst}
.. autoclass:: hyplan.aircraft.NASA_ER2
.. autoclass:: hyplan.aircraft.NASA_GIII
.. autoclass:: hyplan.aircraft.NASA_GIV
.. autoclass:: hyplan.aircraft.NASA_GV
.. autoclass:: hyplan.aircraft.NCAR_GV
.. autoclass:: hyplan.aircraft.NASA_C20A
.. autoclass:: hyplan.aircraft.NASA_P3
.. autoclass:: hyplan.aircraft.NASA_WB57
.. autoclass:: hyplan.aircraft.NASA_B777
.. autoclass:: hyplan.aircraft.KingAirA90
.. autoclass:: hyplan.aircraft.KingAirB200
.. autoclass:: hyplan.aircraft.KingAir350
.. autoclass:: hyplan.aircraft.NASA_C130
.. autoclass:: hyplan.aircraft.NOAA_TwinOtter
.. autoclass:: hyplan.aircraft.NOAA_WP3D
.. autoclass:: hyplan.aircraft.NOAA_GIV
.. autoclass:: hyplan.aircraft.BAS_TwinOtter
.. autoclass:: hyplan.aircraft.FAAM_BAe146
.. autoclass:: hyplan.aircraft.SAFIRE_ATR42
.. autoclass:: hyplan.aircraft.NERC_DO228
.. autoclass:: hyplan.aircraft.AWI_BaslerBT67
.. autoclass:: hyplan.aircraft.DLR_HALO
```

## Profile I/O

Each pre-configured aircraft loads its performance values from
`hyplan/data/aircraft/<short_name>.json`.  These functions read,
write, and resolve those files; use them to refresh a calibration
in place or to roll your own aircraft profile externally.

The schema is documented in
[`hyplan/data/aircraft/README.md`](https://github.com/ryanpavlick/hyplan/blob/main/hyplan/data/aircraft/README.md):
compact JSON with units in field names (e.g.
`climb_schedule.points_ft_kt`), discriminated `{"type": "tas" |
"cas_mach"}` speed schedules, and nested `approach_profile` /
`typical_climb_out` blocks for the more complex airframes.

```{eval-rst}
.. autofunction:: hyplan.aircraft.load_aircraft_profile
.. autofunction:: hyplan.aircraft.dump_aircraft_profile
.. autofunction:: hyplan.aircraft.write_calibrated_profile
.. autofunction:: hyplan.aircraft.profile_path
```
