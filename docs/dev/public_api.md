# Public API conventions

HyPlan exposes three tiers of API surface. Knowing which tier a symbol
belongs to tells you what stability guarantees apply and where to put
new code.

## Public

In a module `__all__`, documented under `docs/api/`, semver-stable.

- **Stability guarantee**: signature changes only at major-version
  boundaries; deprecations get one minor-version release of
  `DeprecationWarning` runway before removal.
- **Import paths**: subpackage-qualified is the canonical contract.
  `hyplan.planning.create_flight_line_record`, `hyplan.exports.to_kml`,
  `hyplan.instruments.GLIHT_HRAC` — these are public; their qualified
  paths are stable.
- **Top-level convenience**: the `hyplan.*` top-level namespace
  re-exports a curated subset of public symbols for ergonomics
  (e.g. `from hyplan import FlightLine`). Not everything documented
  under `docs/api/` is in the top-level `__all__`; that's a deliberate
  curation choice, not a bug.
- **Documentation**: every public symbol has an autodoc entry under
  `docs/api/`. CI enforces this via
  [`docs/_scripts/check_api_coverage.py`](../_scripts/check_api_coverage.py),
  which fails if an autodoc directive points at a name that doesn't
  resolve from its qualified path.

## Public-import policy (the "Policy B" choice)

HyPlan adopts **Policy B**: subpackage-qualified imports are
first-class public surfaces. Both of these are public:

```python
from hyplan.planning import create_flight_line_record  # always works
from hyplan import FlightLine                          # if curated to top
```

Not every public symbol is required to be in `hyplan/__init__.py`'s
top-level `__all__`. New top-level convenience exports are added
deliberately when a symbol earns its way in (heavy use across
notebooks / docs / examples), not automatically for every documented
function.

This is the policy the docs already follow — autodoc directives in
`docs/api/` use qualified paths.

## Advanced

Importable, but **not** in any `__all__`; may move or be renamed.
Documented under `docs/api/advanced/` if at all (this directory
doesn't exist yet; create it when something belongs there). May be
referenced by user code that knows what it's doing — power-user
extension points.

- **Stability guarantee**: best-effort; signature changes can land
  in minor releases with a note in CHANGELOG.
- **Examples**: low-level constraint constructors, internal
  registration helpers exposed for plugin authors.

## Private

Underscore-prefixed (`_foo`), no docs, no stability guarantee. May
change in any release without notice.

- Anything in `hyplan/_*.py` or `hyplan/*/_*.py`.
- Module-private helpers starting with `_`.
- Test-only utilities.

## Test guardrails

[`tests/test_public_api.py`](../../tests/test_public_api.py)
parametrizes over the core installed-by-default surfaces' `__all__`
lists (`hyplan`, `hyplan.instruments`, `hyplan.planning`,
`hyplan.exports`) and asserts every name resolves via `getattr`. This
catches the common drift case where `__all__` advertises a name that's
no longer exported.

Optional-extra surfaces (e.g. `hyplan.winds.*` from the `winds`
extra) get their own per-extra test files when the time comes, gated
on the relevant extra being installed.

## Adding a new public function

1. Add it to its subpackage's `__init__.py` `__all__`.
2. Add an autodoc entry under `docs/api/<subpackage>.md`.
3. Decide if it earns a top-level `hyplan` re-export. Most don't —
   keep the top-level namespace curated.
4. Add tests. The `test_public_api.py` parametrize covers it
   automatically once it's in `__all__`.
5. Mention it in CHANGELOG under `## Unreleased`.

## Renaming or removing a public symbol

1. Deprecate first. Add a `DeprecationWarning` at the old import site
   (e.g. via a PEP-562 `__getattr__` in `__init__.py`) so existing
   code keeps working with a heads-up.
2. Mark the alias for removal in CHANGELOG with a target version.
3. Wait at least one minor release before removing.

This is the pattern v1.11's `ScanningSensor` → `SwathSensor` rename
follows (see [`plans/payload.md`](../../plans/payload.md)).
