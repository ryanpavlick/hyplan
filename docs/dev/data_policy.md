# Data and artifact policy

HyPlan carries four different kinds of file in the repository. Knowing
which kind a file is determines where it goes, whether it's tracked,
and what guarantees apply.

## Categories

### 1. Committed fixtures (`tests/fixtures/`)

Small, read-only, golden inputs for the test suite. Tracked in git.
Tests load these directly and expect their contents to be stable.

**Examples**: representative `flight_plan.gpx` for an exporter
round-trip test; a 10-row CSV of synthetic IWG1 telemetry for a
calibration smoke test.

**Limits**: keep individual files small (< 100 KB ideally); don't
commit binary blobs that would be better as generated outputs.

### 2. Committed example data (`hyplan/data/`, `notebooks/data/`)

Authoritative sample data shipped with the package or tutorials.
Used by the library or notebooks at runtime; required for HyPlan to
function. Tracked in git.

**Examples**: [`hyplan/data/de421.bsp`](../../hyplan/data/de421.bsp)
(JPL planetary ephemeris); aircraft calibration JSON profiles under
[`hyplan/data/aircraft/`](../../hyplan/data/aircraft/);
the bundled GeoJSON regions under `hyplan/data/`.

**Distinction from fixtures**: example data is for *users* to consume
through HyPlan APIs. Fixtures are for *tests* to consume directly.

### 3. Regenerable notebook artifacts

Output files produced by re-running a notebook. **Not tracked**;
added to [`.gitignore`](../../.gitignore). The notebook is the source of
truth — anyone who needs the artifact runs the notebook to produce it.

**Examples**:

- [`notebooks/interactive_export/`](../../notebooks/interactive_export/)
  — produced by `export_formats.ipynb`. Eleven files (GPX, KML, ICT,
  CSV, XLSX, etc.) demonstrating every exporter; regenerated every
  run.
- `notebooks/glint_arc_*.geojson` and `glint_arc_trackair.txt` —
  produced by `glint_arc_planning.ipynb`.

**When to convert into a fixture**: if a test needs a specific output
shape as a golden, *copy* a small representative file from the
regenerated set into `tests/fixtures/` and pin it there. Don't make
the test depend on the notebook running.

### 4. Local / auth-walled caches

Data downloaded at runtime from authenticated APIs (NASA Earthdata,
GEE, OpenAIP) or user-specific raw inputs (IWG1 telemetry archives,
ADS-B alltracks, planned-sortie cards). **Never tracked**; lives
under `~/.cache/hyplan/` or an env-var-controlled location.

**Examples**: NASA Earthdata MERRA-2 GRIB files, GEE-exported imagery
tiles, downloaded ADS-B Globe-history archives.

**Distinction from regenerable artifacts**: regenerable artifacts can
be reproduced by anyone who installs HyPlan. Auth-walled caches can
only be reproduced by users with the right credentials and
permissions, so they're outside the open-source distribution.

## Decision tree

1. Does the test suite consume this file directly?
   → **Fixture** (`tests/fixtures/`), tracked, small.
2. Does HyPlan itself or a notebook consume this file as authoritative
   sample data?
   → **Example data** (`hyplan/data/` or `notebooks/data/`), tracked.
3. Does a notebook **produce** this file as output that's also a
   tutorial artifact?
   → **Regenerable artifact**, gitignored.
4. Does the file come from an authenticated API or user-specific
   raw input?
   → **Cache**, never tracked, lives under `~/.cache/hyplan/` or an
   env-var-controlled path.

## Notebook output tracking — current state

- `notebooks/interactive_export/` (11 files): gitignored (category 3).
  Run `notebooks/export_formats.ipynb` to regenerate locally.
- `notebooks/glint_arc_*` (3 files): gitignored (category 3).
- All other `notebooks/*.ipynb` cell outputs: **tracked**, by
  convention, because the rendered tutorials add pedagogical value
  (charts, tables, computed numbers). This is a deliberate choice;
  `nbstripout` is *not* enforced.

## Adding new tracked files

Before adding a new file to git, run through the decision tree above.
If it's category 3 or 4, add the appropriate `.gitignore` entry first
and verify `git status` no longer lists the file.
