# Proposal 0002 — Step-climb (climb-out staging) representation

## Status

Open.  Original framing was about cross-survey altitude drift; revised
to focus on the operationally relevant phenomenon: climb-out staging
during the takeoff phase.

## Terminology

In aviation, **step climb** is a series of short pauses or slow-climb
segments during the climb-out where the aircraft levels off to burn
fuel and reduce gross weight before continuing to climb.  The driving
physics is the **weight-limited ceiling**: a heavily-loaded aircraft
can't reach its target cruise altitude directly, so the climb is
staged.

This is distinct from:

* **Cross-survey altitude drift** — a survey aircraft's cruise
  altitude rising slowly across a multi-hour grid as fuel burns off.
  Driven by similar physics but manifests during cruise, not climb.
  Currently single-altitude `FlightLine` doesn't capture this; for
  most sorties the per-segment timing residual from this is small
  (≤2% on the NM17 B replay).

* **In-flight altitude changes between survey lines** — usually
  driven by sensor or scientific objective, not weight.

This proposal addresses **step climb** specifically.

## Context

For a typical NASA ER-2 sortie planned card, the climb-out from
KCOS to FL650 is staged through several intermediate altitudes:

| event | altitude | duration | mechanism |
|---|---:|---:|---|
| BRK/E | FL240 | ~0 min | level off briefly |
| `.level off` | FL260 | ~0 min | level off briefly |
| PUB/R253012 | FL260 | — | transit |
| PUB/R206014 | FL356 | — | transit |
| `.delay` orbit at PUB | FL356 → FL611 | **25 min** | hold (slow climb in orbit) |
| `.level off` | FL650 | ~0 min | top of climb |
| TBE/E245028 | FL650 | — | first survey waypoint |

The 25-minute orbit at PUB is the dominant term.  During those
25 minutes the aircraft makes zero forward progress while gaining
~25,000 ft.  HyPlan's continuous-climb model integrates the same
altitude band against `climb_profile` and returns a forward distance
that the aircraft *did not* travel.  The +11.5 min residual on the
NM17 B planned-vs-flown comparison (KCOS → first /L) is partially
attributable to this spatial mismatch.

## Empirical evidence

From the [IWG1 sortie-replay notebook](../notebooks/er2_calibration/sortie_replay.ipynb)
(post-Item-1 hybrid planner) and the
[planned-vs-flown notebook](../notebooks/er2_calibration/planned_vs_flown.ipynb):

* Climb / cruise / descent phases: residuals close once the hybrid
  2D-Dubins + integrated-vertical planner lands.
* Survey-grid phase: per-line cruise residuals are <1% with the
  proper trochoidal CCC + IWG1-trace wind.
* **Pre-survey (KCOS → first /L):** +11.5 min residual on NM17 B,
  largely attributable to the unmodeled climb staging.

That pre-survey residual is the gap this proposal addresses.

## Design options

### Option A — `Aircraft.step_climb()` helper *(implemented)*

A simple method on `Aircraft` that takes a starting altitude, ending
altitude, and a list of `(level_off_altitude, hold_duration)` pauses,
returning total time and forward distance.  Each pause is interpreted
as a level orbit at the staging altitude — adds time, no distance.
Climb segments between pauses use the calibrated `climb_profile`.

```python
t, d = aircraft.step_climb(
    start_altitude=6_187 * ureg.foot,    # KCOS
    end_altitude=65_000 * ureg.foot,     # FL650
    pauses=[(35_600 * ureg.foot, 25 * ureg.minute)],
)
```

**Pros:** smallest possible API addition.  Standalone helper —
mission designers compute their own staged climb time/distance and
feed it into custom planning code.  Already shipping.

**Cons:** not integrated with `compute_flight_plan` — the planner
doesn't know about `step_climb`-derived staging, so the takeoff-phase
geometry is unchanged.  Useful for analysis / fuel budgeting but
doesn't close the +11.5 min residual on its own.

### Option B — Climb plan as `compute_flight_plan` parameter

Pass a `climb_pauses` (or richer `ClimbPlan`) parameter to
`compute_flight_plan`.  The takeoff-phase computation in
`Aircraft._hybrid_path` consults it and rebuilds the climb portion
as a sequence of climb-segment-then-orbit blocks at the specified
staging waypoints.

**Pros:** fully integrated planning.  Closes the residual.  Mission
designers can supply pauses derived from a Green Card or from a
generic per-aircraft "typical climb plan."

**Cons:** structural change to the planner.  Needs a clean API for
specifying staging waypoints (lat/lon? altitude only?) and the
geometry of each orbit.

### Option C — Slow-climb-segment model

Refine `step_climb` to take `(level_off_alt, exit_alt, duration)`
3-tuples instead of `(altitude, duration)` 2-tuples.  When `exit_alt
> level_off_alt`, the aircraft climbs from `level_off_alt` to
`exit_alt` during the duration — modeling the .delay's actual
behavior (slow climb in orbit, not a level hold).

**Pros:** more faithful to reality for ER-2-style `.delay` orbits.

**Cons:** redundant with `climb_profile.rate_at()` if the duration
matches the integrated rate over that band — caller has to compute
this consistency themselves.  Adds API complexity.

### Option D — Document and accept

The +11.5 min pre-survey residual is documented and the user
interprets HyPlan's continuous-climb model as a known approximation.
No code changes.

## Recommendation

Option A is shipped.  Whether to do Option B depends on whether the
+11.5 min residual is a planning-fidelity blocker or an analytical
nuisance.  For mission-design / fuel-budget calculations,
`Aircraft.step_climb()` is sufficient — designers can call it
directly to compute staged climb times.  For closing the residual in
`compute_flight_plan` output, Option B is the smallest behavioral
addition, but its complexity isn't justified until the residual is
shown to bind real planning decisions.

## Out of scope

* **Real-time fuel-burn modeling.**  HyPlan's `Aircraft` is
  fuel-agnostic; introducing fuel-state-dependent climb performance
  requires a much larger structural change (BADA-style mass-dependent
  performance), out of scope here.
* **Wind-corrected step-climb altitude.**  Optimal altitude depends on
  wind aloft; that's a routing-optimization layer above HyPlan's
  current scope.
* **Cross-survey altitude drift.**  Documented as a separate
  phenomenon above; for current survey-aircraft replays the
  per-segment residual is small (≤2%) and not addressed by this
  proposal.
