# Performance

HyPlan workflows mix compute-bound math (Dubins integration,
ray–terrain intersection, isochrone bisection) with network-bound data
fetches (MERRA-2 / GFS winds, Google Earth Engine cloud queries,
terrain DEM tiles). This page documents what's benchmarked, what
isn't, and how to reproduce the numbers on your own hardware.

## Benchmarked operations

The reproducible cases live in
[`bench/run_isochrone.py`](https://github.com/ryanpavlick/hyplan/blob/main/bench/run_isochrone.py)
and the `pytest -m perf` suite. Run with:

```bash
python -m bench.run_isochrone                  # cases 1–3 (no network)
BENCH_GRIDDED=1 python -m bench.run_isochrone  # adds MERRA-2 fixture
pytest -m perf                                 # CI gate variant
```

| Case | Description | Reference | CI ceiling |
|---|---|---|---|
| 1 | Round-trip still-air isochrone, NASA G-III, 36 rays | 0.32 s | < 4 s |
| 2 | Round-trip constant-wind isochrone, NASA G-III, 72 rays | 0.73 s | < 6 s |
| 3 | Refuel isochrone, B-200 from KEFD, 2 refuel candidates, 36 rays | 2.59 s | < 20 s |
| 4 | Realistic gridded MERRA-2 isochrone (network-gated) | not gated | — |

Reference numbers are from a quiet developer laptop (Apple M2 Pro,
HyPlan v1.6.2, numpy 2.4); your hardware will land within roughly 0.5×
to 2× of these. CI ceilings are deliberately loose for shared
GitHub-hosted runners.

The cumulative benchmark runs in **~3.6 s end-to-end** with no network
access required, so it's reasonable to add to a smoke-test loop after
any planning-engine refactor.

## What's not benchmarked

These workflows have variable costs that depend on external services
or input size; describing them as "fast" or "slow" without context is
misleading.

- **MERRA-2 / GFS / GMAO wind fetches** — dominated by OPeNDAP round
  trip and the requested lat/lon/time/level slab size. A typical 4-D
  slab for a one-day mission over a 10° box runs 2–10 s on a good
  connection, much longer on a poor one. Slabs are cached on the
  `_GriddedWindField` instance, so reuse across rays / pairs amortizes
  the fetch.
- **Google Earth Engine cloud queries** — server-side reduction over a
  polygon × time window. A single ROI × 10-year MODIS climatology
  typically returns in 5–30 s. EE rate-limits and queues; the same
  query later in the day may run faster.
- **Terrain DEM tile fetches** — `hyplan.terrain` lazy-loads SRTM /
  ASTER tiles via `rasterio`. First touch on a 1° tile is around 1 s;
  cached thereafter. Long mountain transits spanning many tiles cost
  proportionally.
- **Polygon-fill flight-line generation** — scales with polygon area
  divided by swath width. Sub-second for typical science boxes;
  multi-second for continental sweeps.
- **Multi-day sortie optimization** — `greedy_optimize` is heuristic
  and runs in seconds for ≤ 100 candidate lines; not benchmarked at
  larger scale.

## When you might hit a wall

The realistic performance ceiling for HyPlan today is the
**isochrone-with-MERRA-2-winds** workflow at fine ray resolution
(360 rays) when the wind slab is large. The 4-D slab fetch plus
per-ray trochoidal integration can run for tens of seconds. If you're
sweeping many (base, recovery) pairs in a parameter study, this
dominates wall time.

Mitigations:

- Reuse a single `_GriddedWindField` across rays and pairs so the slab
  fetch amortizes
- Drop ray resolution to 36–72 for exploratory sweeps; bump to 360
  only for final figures
- Restrict the wind slab pressure range to the aircraft's operating
  envelope via the `pressure_min_hpa` / `pressure_max_hpa` constructor
  arguments

## Profiling

Standard tools work; HyPlan uses no native extensions:

```bash
python -m cProfile -o iso.prof -m bench.run_isochrone
python -m snakeviz iso.prof    # or pyinstrument, py-spy
```

The hottest paths today are `_solve_rays`, `_evaluate_refuel_at_d`, and
`_GriddedWindField._interp4d` (4-D linear interpolation). All three
are vectorized.

## Reporting performance regressions

If `bench/run_isochrone.py` slows down by more than 2× on a known-good
baseline, please open an issue with:

- The full `python -m bench.run_isochrone` output
- `pip freeze | grep -E "numpy|pyproj|shapely|hyplan"`
- Hardware (CPU, RAM, OS)

Fixed-version regressions are usually traceable to a numpy or pyproj
upgrade; the workflow is to bisect against the reference numbers
above.
