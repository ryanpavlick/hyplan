# Aircraft performance profiles

Each `<short_name>.json` file in this directory holds the calibrated
performance parameters for one research aircraft.  They are loaded by
`hyplan.aircraft._profile_io.load_aircraft_profile()` and consumed by
the thin wrapper classes in `hyplan/aircraft/_models.py`.

## Schema (compact, units in field names)

```jsonc
{
  "metadata": {
    "aircraft_type": "King Air 350",
    "tail_number": "N2UW",
    "operator": "University of Wyoming (UWKA-2)",
    "engine_type": "turboprop",            // "jet" | "turboprop" | "piston"
    "calibration_status": "calibrated"     // "calibrated" | "inferred" | "uncalibrated"
  },
  "scalars": {
    "service_ceiling_ft": 35000,
    "approach_speed_kt": 110,
    "range_nmi": 2100,                     // null if unknown
    "endurance_hr": 5,                     // null if unknown
    "useful_payload_lb": 2970,             // null if unknown
    "stall_speed_cas_kt": 75,              // null if unknown
    "descent_path_angle_max_deg": null,    // null disables steep-descent override
    "climb_path_angle_max_deg": null       // null disables steep-climb override
  },
  "climb_schedule":  { /* see "Schedule types" below */ },
  "cruise_schedule": { /* … */ },
  "descent_schedule":{ /* … */ },
  "climb_profile":  { "points_ft_fpm": [[ft, fpm], …], "source": "" },
  "descent_profile":{ "points_ft_fpm": [[ft, fpm], …], "source": "" },
  "turn_model": {
    "max_bank_deg": 30.0,
    "max_load_factor": 2.5,
    "bank_by_phase": {
      "climb_deg": 14, "cruise_deg": 20,
      "descent_deg": 13, "approach_deg": 11
    }
  },
  "approach_profile": null,                // or {speed_schedule_ft_kt, top_of_approach_agl_ft, glideslope_deg}
  "typical_climb_out": null,               // or {absorbed_in_climb_profile, typical_holds_ft_min,
                                            //     typical_overhead_min, notes,
                                            //     explicit_climb_plan_ft_min}
  "confidence": { "climb": .., "cruise": .., "descent": .., "turns": .. },
  "sources": [
    { "source_type": "adsb", "reference": "…", "confidence": .55,
      "url": "…", "doi": "", "notes": "" }
  ]
}
```

### Schedule types

Speed schedules use a discriminator on the `type` field:

```jsonc
// Piecewise-linear TAS vs altitude (most aircraft)
{ "type": "tas",
  "points_ft_kt": [[5000, 250], [20000, 290], [35000, 318]] }

// Jet CAS/Mach split (jets above the crossover altitude)
{ "type": "cas_mach",
  "cas_kt": 270, "mach": 0.78, "crossover_ft": 28000 }
```

## Editing a profile

Any field can be updated by editing the JSON file in place.  No Python
changes required.  Run `pytest tests/test_aircraft_profiles.py` to
verify the file still loads cleanly.

## Regenerating from calibration

Each `notebooks/calibration/<aircraft>/calibrate.py` script fits the
performance parameters from in-situ flight data (IWG1 / ICARTT /
ADS-B).  To write the result directly to JSON:

```python
from hyplan.aircraft import KingAir350
from hyplan.aircraft._profile_io import dump_aircraft_profile

# Start from the current class, apply your calibrated overrides, dump.
ac = KingAir350()
ac.climb_profile = fit.climb_profile         # from calibrate.py
ac.cruise_schedule = fit.cruise_schedule
# … etc …
dump_aircraft_profile(ac, "hyplan/data/aircraft/king_air_350.json")
```

The next time the package is imported, `KingAir350()` returns an
instance reflecting the new JSON values.

## Adding a new aircraft

1. Create `<short_name>.json` following the schema above.
2. Add a class in `hyplan/aircraft/_models.py`:
   ```python
   class NewAircraft(Aircraft):
       """Operator-facing narrative goes here."""

       def __init__(self) -> None:
           super().__init__(**load_aircraft_profile("<short_name>"))
   ```
3. Re-export the class in `hyplan/aircraft/__init__.py`.
4. Add `(NewAircraft, "<short_name>")` to the `ROSTER` in
   `tests/test_aircraft_profiles.py`.
