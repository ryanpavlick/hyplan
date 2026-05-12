"""KingAirA90 calibration from adsb.lol ADS-B traces.

First-pass A90 calibration using a single-day adsb.lol globe-trace
snapshot (2026-05-09).  The 124-tail FAA registry was filtered to
the 11 tails active on that day; ~34 sorties extracted; schedules
fitted via the existing :mod:`hyplan.aircraft.adsb` pipeline with
``wind_source='still_air'`` (groundspeed is treated as TAS — fine for
short sorties at moderate altitudes; v1.6.1 will switch to MERRA-2
wind triangle once the daily-cron sample accumulates).

Sortie sample is dominated by skydive operations (Dynamic Avlease,
Marc Aircraft) staying low (~4-5 kft); only ~12 sorties reach
FL150+ (N15CT, N416MR, N41DZ).  Confidence is set to 0.5
accordingly.

Run from repo root::

    python -m notebooks.calibration.KingAirA90.calibrate
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


TRACE_DIR = Path("data/KingAirA90/adsb_lol_traces")

# Sortie segmentation
GROUND_GAP_S = 300       # >5 min on ground splits sorties
AIRBORNE_FLOOR_FT = 500  # min AGL altitude to count as airborne
MIN_DUR_MIN = 5
MIN_PEAK_ALT_FT = 1500   # below this is pattern work, not a sortie

# Skydive filter: short-duration jump-run profiles dominate the active
# US A90 fleet (SKYDIVE GRAND HAVEN / JOSEPHINE AIR / BRAVO HOTEL etc.).
# Classic signature: climb to drop altitude (FL130-180), brief plateau,
# aggressive descent.  Exclude sorties matching duration + peak-altitude
# bracket plus a sustained-descent gate so cruise/climb/descent
# schedules reflect normal-ops A90 envelope.
SKYDIVE_MAX_DUR_MIN = 25
SKYDIVE_MIN_PEAK_FT = 10000
SKYDIVE_MIN_DESCENT_FPM = 2500   # sustained 60+s segment


class _Flight:
    """Minimal shim mimicking traffic.core.Flight for the ADS-B pipeline."""

    def __init__(self, data: pd.DataFrame, icao24: str, callsign: str = "") -> None:
        self.data = data
        self.icao24 = icao24
        self.callsign = callsign


def _trace_to_sorties(path: Path) -> list[_Flight]:
    """Parse adsb.lol trace_full JSON into per-sortie Flight shims."""
    d = json.loads(path.read_text())
    trace = d.get("trace", [])
    if not trace:
        return []
    icao = d.get("icao", "?")
    callsign = d.get("r", "")
    base_ts = d.get("timestamp", 0)

    rows = []
    for r in trace:
        # [t_offset, lat, lon, alt_baro, gs, track, ?, baro_rate, ...]
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

    # Estimate ground floor; airborne if alt > floor + AIRBORNE_FLOOR_FT
    ground_floor = df["altitude"].quantile(0.05)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Segment by ground gaps
    airborne = (df["altitude"] - ground_floor) > AIRBORNE_FLOOR_FT
    df["airborne"] = airborne
    sorties: list[pd.DataFrame] = []
    cur_idx: list[int] = []
    last_airborne_t = None
    t_arr = df["timestamp"].astype("int64").values // 1_000_000_000

    for i, ab in enumerate(airborne):
        if ab:
            cur_idx.append(i)
            last_airborne_t = t_arr[i]
        elif cur_idx and last_airborne_t is not None:
            if t_arr[i] - last_airborne_t > GROUND_GAP_S:
                sorties.append(df.iloc[cur_idx[0]:cur_idx[-1] + 1].copy())
                cur_idx = []
    if cur_idx:
        sorties.append(df.iloc[cur_idx[0]:cur_idx[-1] + 1].copy())

    flights: list[_Flight] = []
    for sdf in sorties:
        if len(sdf) < 30:
            continue
        dur_min = (sdf["timestamp"].iloc[-1] - sdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur_min < MIN_DUR_MIN:
            continue
        peak_alt = sdf["altitude"].max()
        if peak_alt < MIN_PEAK_ALT_FT:
            continue
        # Skydive filter
        if (dur_min < SKYDIVE_MAX_DUR_MIN
                and peak_alt >= SKYDIVE_MIN_PEAK_FT
                and _has_sustained_descent(sdf, SKYDIVE_MIN_DESCENT_FPM)):
            continue
        sdf = sdf.drop(columns=["airborne"]).reset_index(drop=True)
        flights.append(_Flight(sdf, icao24=icao, callsign=callsign))
    return flights


def _has_sustained_descent(sdf: pd.DataFrame, fpm_threshold: float,
                            window_s: float = 60.0) -> bool:
    """True if the sortie has any 60-second segment with avg VS < -threshold."""
    if "vertical_rate" not in sdf.columns:
        return False
    t = sdf["timestamp"].astype("int64").to_numpy() / 1e9
    vs = sdf["vertical_rate"].to_numpy(dtype=float)
    # Slide a window: for each row, average VS over the last window_s seconds.
    n = len(sdf)
    j = 0
    for i in range(n):
        while j < i and t[i] - t[j] > window_s:
            j += 1
        seg = vs[j:i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 5 and seg.mean() < -fpm_threshold:
            return True
    return False


def load_sorties() -> tuple[list[_Flight], dict[str, int]]:
    flights: list[_Flight] = []
    skydive_dropped = 0
    for fp in sorted(TRACE_DIR.glob("*.json")):
        # Count skydive drops by re-segmenting without the filter
        flights_kept = _trace_to_sorties(fp)
        flights.extend(flights_kept)
        # Count would-be-dropped by relaxing the filter; cheap re-parse
        skydive_dropped += _count_skydive(fp)
    return flights, {"skydive_dropped": skydive_dropped}


def _count_skydive(path: Path) -> int:
    """Count sorties that match the skydive signature (would be dropped)."""
    d = json.loads(path.read_text())
    trace = d.get("trace", [])
    if not trace:
        return 0
    base_ts = d.get("timestamp", 0)
    rows = []
    for r in trace:
        t = base_ts + r[0]
        alt = r[3] if r[3] != "ground" and r[3] is not None else np.nan
        gs = r[4] if len(r) > 4 and r[4] is not None else np.nan
        track = r[5] if len(r) > 5 and r[5] is not None else np.nan
        vs = r[7] if len(r) > 7 and r[7] is not None else np.nan
        rows.append((t, alt, gs, track, vs))
    df = pd.DataFrame(rows, columns=["t", "altitude", "gs", "track", "vertical_rate"])
    df = df.dropna(subset=["altitude", "gs", "track"])
    if df.empty:
        return 0
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    floor = df["altitude"].quantile(0.05)
    df = df.sort_values("timestamp").reset_index(drop=True)
    airborne = (df["altitude"] - floor) > AIRBORNE_FLOOR_FT
    sorties: list[pd.DataFrame] = []
    cur: list[int] = []
    last_t = None
    t_arr = df["t"].to_numpy()
    for i, ab in enumerate(airborne):
        if ab:
            cur.append(i)
            last_t = t_arr[i]
        elif cur and last_t is not None:
            if t_arr[i] - last_t > GROUND_GAP_S:
                sorties.append(df.iloc[cur[0]:cur[-1] + 1].copy())
                cur = []
    if cur:
        sorties.append(df.iloc[cur[0]:cur[-1] + 1].copy())
    n = 0
    for sdf in sorties:
        dur = (sdf["timestamp"].iloc[-1] - sdf["timestamp"].iloc[0]).total_seconds() / 60
        peak = sdf["altitude"].max()
        if (MIN_DUR_MIN <= dur < SKYDIVE_MAX_DUR_MIN
                and peak >= SKYDIVE_MIN_PEAK_FT
                and _has_sustained_descent(sdf, SKYDIVE_MIN_DESCENT_FPM)):
            n += 1
    return n


def main() -> None:
    print("KingAirA90 calibration from adsb.lol traces")
    print("=" * 70)
    flights, stats = load_sorties()
    if not flights:
        raise SystemExit("no sorties loaded")

    # Sortie summary
    print(f"\nLoaded {len(flights)} sorties from {TRACE_DIR} "
          f"(skydive-filtered: {stats['skydive_dropped']} dropped)")
    # Per-tail aggregate
    from collections import defaultdict
    per_tail: dict[str, list[float]] = defaultdict(list)
    for f in flights:
        dur = (f.data["timestamp"].iloc[-1] - f.data["timestamp"].iloc[0]).total_seconds() / 60
        per_tail[f.callsign].append(dur)
    for tail in sorted(per_tail):
        durs = per_tail[tail]
        print(f"  {tail:8s}  n={len(durs):3d}  avg dur={sum(durs)/len(durs):5.1f} min  "
              f"total={sum(durs)/60:5.1f} hr")

    # Run pipeline stages
    from hyplan.aircraft.adsb.phases import label_phases
    from hyplan.aircraft.adsb.airdata import reconstruct_airdata, resolve_wind_field
    from hyplan.aircraft.adsb.fitting import fit_schedules

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
    # PerformanceConfidence has a ``.summary`` float; older versions return
    # a plain float.  Handle both cases.
    conf_val = getattr(conf, "summary", conf)
    if not isinstance(conf_val, (int, float)):
        conf_val = float("nan")

    print()
    print("=" * 70)
    path = write_calibrated_profile(
        "king_air_a90",
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
