"""KingAir350 calibration from airplanes.live ADS-B traces (UWKA-2).

Calibrated against the University of Wyoming King Air (UWKA-2),
N2UW / hex A18F28 — a Beechcraft King Air 350 (Beech B300) used
for atmospheric / boundary-layer / cloud-physics research.  Sparse
sample: UWKA-2 only flies during specific science campaigns.  Over
a 540-day probe window of airplanes.live globe-history archives,
18 days returned non-empty traces (10k trace rows total),
clustered into 5 campaign windows in 2025-01, 2025-03, 2025-06,
2025-07/08, and 2025-10.

This is a single-tail calibration — different from KingAirA90 which
aggregates across the active US fleet.  UWKA-2 has the Blackhawk
XP-67A engine upgrade (PT6A-67A) which slightly outperforms the
stock King Air 350's PT6A-60A, so the calibrated cruise/climb
values may run a few percent above stock factory data.

Run from repo root::

    python -m notebooks.calibration.KingAir350.calibrate
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


TRACE_DIR = Path("data/KingAir350/adsb_lol_traces")

GROUND_GAP_S = 300
AIRBORNE_FLOOR_FT = 500
MIN_DUR_MIN = 5
MIN_PEAK_ALT_FT = 1500


class _Flight:
    def __init__(self, data: pd.DataFrame, icao24: str, callsign: str = "") -> None:
        self.data = data
        self.icao24 = icao24
        self.callsign = callsign


def _trace_to_sorties(path: Path) -> list[_Flight]:
    d = json.loads(path.read_text())
    trace = d.get("trace", [])
    if not trace:
        return []
    icao = d.get("icao", "?")
    callsign = d.get("r", "")
    base_ts = d.get("timestamp", 0)
    rows = []
    for r in trace:
        t = base_ts + r[0]
        lat, lon = r[1], r[2]
        alt = r[3]
        if alt == "ground" or alt is None:
            alt = np.nan
        gs = r[4] if len(r) > 4 and r[4] is not None else np.nan
        track = r[5] if len(r) > 5 and r[5] is not None else np.nan
        vs = r[7] if len(r) > 7 and r[7] is not None else np.nan
        rows.append((t, lat, lon, alt, gs, track, vs))
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "latitude", "longitude", "altitude",
                 "groundspeed", "track", "vertical_rate"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.dropna(subset=["altitude", "groundspeed", "track"])
    if df.empty:
        return []
    floor = df["altitude"].quantile(0.05)
    df = df.sort_values("timestamp").reset_index(drop=True)
    airborne = (df["altitude"] - floor) > AIRBORNE_FLOOR_FT
    df["airborne"] = airborne
    sorties: list[pd.DataFrame] = []
    cur: list[int] = []
    last_t: float | None = None
    t_arr = df["timestamp"].astype("int64").to_numpy() // 1_000_000_000
    for i, ab in enumerate(airborne):
        if ab:
            cur.append(i)
            last_t = float(t_arr[i])
        elif cur and last_t is not None:
            if t_arr[i] - last_t > GROUND_GAP_S:
                sorties.append(df.iloc[cur[0]:cur[-1] + 1].copy())
                cur = []
    if cur:
        sorties.append(df.iloc[cur[0]:cur[-1] + 1].copy())
    flights: list[_Flight] = []
    for sdf in sorties:
        if len(sdf) < 30:
            continue
        dur_min = (sdf["timestamp"].iloc[-1] - sdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur_min < MIN_DUR_MIN:
            continue
        if sdf["altitude"].max() < MIN_PEAK_ALT_FT:
            continue
        sdf = sdf.drop(columns=["airborne"]).reset_index(drop=True)
        flights.append(_Flight(sdf, icao24=icao, callsign=callsign))
    return flights


def main() -> None:
    print("KingAir350 calibration from airplanes.live (N2UW / UWKA-2)")
    print("=" * 70)
    flights: list[_Flight] = []
    for fp in sorted(TRACE_DIR.glob("*.json")):
        flights.extend(_trace_to_sorties(fp))
    if not flights:
        raise SystemExit("no sorties loaded")

    print(f"\nLoaded {len(flights)} sorties:")
    per_date: dict[str, list[float]] = defaultdict(list)
    for f in flights:
        date = f.data["timestamp"].iloc[0].strftime("%Y-%m-%d")
        dur = (f.data["timestamp"].iloc[-1] - f.data["timestamp"].iloc[0]).total_seconds() / 60
        per_date[date].append(dur)
    for date in sorted(per_date):
        ds = per_date[date]
        print(f"  {date}  n={len(ds):2d}  total={sum(ds)/60:5.1f} hr  "
              f"avg={sum(ds)/len(ds):5.1f} min")

    from hyplan.aircraft.adsb.airdata import reconstruct_airdata, resolve_wind_field
    from hyplan.aircraft.adsb.fitting import fit_schedules
    from hyplan.aircraft.adsb.phases import label_phases

    print("\nLabeling phases…")
    phased = [label_phases(f, backend="heuristic") for f in flights]
    print("Wind correction (still_air baseline)…")
    wind = resolve_wind_field("still_air", phased)
    airdata = [reconstruct_airdata(df, wind) for df in phased]
    combined = pd.concat(airdata, ignore_index=True)
    print(f"  combined dataframe: {len(combined):,} rows")
    print("Fitting schedules…")
    fit = fit_schedules(combined, altitude_bin_ft=2000.0, max_schedule_points=6)

    from hyplan.aircraft._profile_io import write_calibrated_profile
    from hyplan.units import ureg

    conf = fit.overall_confidence()
    conf_val = getattr(conf, "summary", conf)
    if not isinstance(conf_val, (int, float)):
        conf_val = float("nan")

    print()
    print("=" * 70)
    path = write_calibrated_profile(
        "king_air_350",
        service_ceiling=int(round(fit.service_ceiling_ft / 100) * 100) * ureg.feet,
        approach_speed=int(round(fit.approach_speed_kt)) * ureg.knot,
        climb_schedule=fit.climb_schedule,
        cruise_schedule=fit.cruise_schedule,
        descent_schedule=fit.descent_schedule,
        climb_profile=fit.climb_profile,
        descent_profile=fit.descent_profile,
    )
    print(f"Wrote calibrated profile to {path}")
    print(f"  fit n_sorties={len(flights)}  overall_confidence={conf_val:.2f}")


if __name__ == "__main__":
    main()
