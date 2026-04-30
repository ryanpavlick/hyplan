# Proposal 0002 — Step-cruise representation in `FlightLine`

## Status

Open.  Parked from the planner-fidelity / hybrid-path PR.

## Context

Real high-altitude survey aircraft (ER-2, U-2, WB-57, Global Hawk) step-cruise across a multi-hour mission as fuel burns off and the weight-limited ceiling rises.  A typical ER-2 sortie sees cruise altitude transition through 50 → 55 → 60 → 61 kft over the course of the survey grid, even though the ground track is stepping line-by-line through a single rectangular polygon.

HyPlan's [`FlightLine`](../hyplan/flight_line.py) is a single-altitude representation: both `waypoint1` and `waypoint2` carry the same `altitude_msl`.  When `compute_flight_plan` realizes a real sortie that step-cruised, it can only place the aircraft at the line's nominal altitude.  Per-line timing and TAS will diverge from observation; bottom-line mission duration is approximately self-canceling (lower TAS at lower altitude offsets less climb-rate cost reaching the assumed altitude), so range / fuel planning remains usable, but per-segment operational fidelity drops.

The [n=5 ADS-B archive replay](../notebooks/er2_calibration/) earlier surfaced this as the largest residual: total duration matched within 3% but on-station segment time was off by 200%+ — the model assumed all lines were flown at FL600 cruise TAS while the aircraft was actually at FL500 with lower TAS at the start of the survey.

## Empirical evidence (from IWG1 sortie replay, post-Item-1)

Pending the IWG1 sortie-replay notebook (Item 3 of the planner-redesign plan).  Expected pattern:

* Climb / approach phases: residuals close once the hybrid 2D-Dubins + integrated-vertical planner lands.
* On-station phase: per-line altitude diverges; per-line time diverges by ~10-20% on ER-2 sorties; total mission duration close (self-cancellation).

That divergence is the gap this proposal addresses.

## Design options

### Option A — Multi-altitude `FlightLine`

Add an optional `altitude_schedule: TasSchedule`-shaped field carrying altitude as a function of along-line distance.  Single-altitude usage stays the same; mission designers who want a step-cruise line opt in.

**Pros:** explicit per-line; designer controls intent.

**Cons:** not what mission designers actually do — they don't pre-plan step altitudes; they let the aircraft drift up as fuel burns.  Adds API surface that won't see much organic adoption.

### Option B — Altitude-by-elapsed-time policy on `Aircraft`

Add an optional `cruise_altitude_schedule: VerticalProfile`-shaped field on `Aircraft`, indexed by elapsed mission time, that the planner consults for what cruise altitude to use at each segment.  The schedule is calibrated from sortie data (e.g., IWG1) so it captures the empirical step pattern.

**Pros:** matches operational reality (drift-up by fuel burn / aircraft state); zero burden on mission designer.

**Cons:** elapsed-time semantics are ambiguous (which "mission start" — wheels-up?  first-line entry?).  Needs careful definition.  Couples `Aircraft` to mission-elapsed-time which is unusual.

### Option C — Calibrated `cruise_altitude_offset` per `FlightLine`

Accept that step-cruise is a sortie-planner concern, not an aircraft-model concern.  Tooling helps: a function that takes a list of `FlightLine`s + an `Aircraft` and produces a stepped variant with calibrated altitude offsets.  Mission designers can call it before passing the lines to `compute_flight_plan`.

**Pros:** decouples mission-planning concern from aircraft model.  Can be added without touching `FlightLine` or `Aircraft`.  Designer keeps explicit control.

**Cons:** still requires a "what's the right step pattern" answer, which depends on aircraft + nominal mission length + fuel state.

### Option D — Document and accept

The total-duration self-cancellation makes step-cruise a fidelity issue for visualization and per-segment timing, not for the headline mission-planning numbers.  Document in `FlightLine`'s docstring (already done in [hyplan/flight_line.py](../hyplan/flight_line.py)) and on `Aircraft.cruise_speed_at` that the single-altitude representation is intentional.  Don't add structure.

**Pros:** zero code change; honest about the model's limits.

**Cons:** users wanting per-segment fidelity have to reach for external tooling.

## Recommendation

Defer the structural decision until after the **IWG1 sortie-replay notebook lands** and quantifies the divergence with the post-Item-1 hybrid planner.  If the per-segment residuals at typical mission-design altitudes are <10%, Option D is sufficient.  If the divergence is meaningful for science-mission planning, Option C is the smallest behavioral addition that addresses it.

## Out of scope

* Real-time fuel-burn modeling.  HyPlan's `Aircraft` is fuel-agnostic; introducing fuel-state-dependent altitude requires a much larger structural change (BADA-style mass-dependent performance), out of scope here.
* Wind-corrected step-cruise altitude.  Optimal altitude depends on wind aloft; that's a routing-optimization layer above HyPlan's current scope.
