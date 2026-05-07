# Authoring and maintaining notebooks

This is a developer-facing guide.  For end users learning the
library, see [`README.md`](README.md).

The notebooks in this directory serve three different audiences:

1. **User tutorials** — `tutorial.ipynb` and topic primers
   (`isochrone.ipynb`, `winds.ipynb`, `flight_patterns.ipynb`,
   etc.).  These ship as part of the library's documentation
   and are referenced from the README and `docs/`.  Treat them
   as part of the public surface: outputs are committed,
   re-execution is mandatory before pushing changes that touch
   them.

2. **Developer / calibration QA** —
   `calibration/<aircraft>/calibration.ipynb`,
   `calibration/er2/sortie_replay.ipynb`,
   `calibration/er2/planned_vs_flown.ipynb`.  These live next to
   the per-aircraft fit code, are exercised by the maintainers
   when calibration data lands, and validate timing residuals
   against real IWG1 / ICARTT records.  Less polished narrative
   than the user tutorials; outputs change every time
   calibration data lands.

3. **Internal smoke notebooks** — anything in
   `notebooks/_build_notebook.py` style scripts that *generate*
   notebook content from Python.  Edit the script, not the
   resulting `.ipynb`.

## Authoring rules

### Re-execute and commit outputs before pushing

Every notebook in this repo ships its outputs.  When you change
a notebook (by editing source cells, by source-code change that
affects outputs, or by upgrading a dependency that touches
plotting), **re-execute end-to-end and commit the resulting
outputs together with your code change**.  This keeps the
shipped tutorial outputs consistent with the code that
produced them.

```bash
jupyter nbconvert --to notebook --execute notebooks/<file>.ipynb \
  --inplace --ExecutePreprocessor.timeout=300
```

For long-running notebooks (MERRA-2, GFS, GEOS-FP, MODIS Earth
Engine), the timeout may need to be 600 seconds.  Notebooks
that hit auth-walled endpoints will skip those sections when
credentials are missing — that's expected; just commit the
post-skip output.

### Don't commit notebook re-executions in unrelated PRs

If you're doing source code work that doesn't touch a given
notebook's behavior, **don't include that notebook's
re-execution in your PR**.  Notebook output diffs are large,
review-unfriendly, and turn small PRs into reviewer-fatigue
events.  If your change *does* alter notebook outputs (e.g.,
you changed the formula a tutorial demonstrates), include the
re-execution.  Otherwise leave it.

This matters because notebook outputs include execution
timestamps, random matplotlib metadata, and provider response
metadata that all churn even for behavior-equivalent code.  The
v1.5 release surfaced 41 stale notebook diffs that had
accumulated this way; we'd rather avoid that next time.

### Use the `_show_plot()` Agg-backend pattern

When a notebook is executed under `nbconvert` (or in CI), the
matplotlib backend is `Agg` — figures don't display via
`plt.show()`.  Notebooks that mix interactive use and CI
execution should use the small helper pattern already in
`aircraft_performance.ipynb`:

```python
import matplotlib
from IPython.display import display

def _show_plot():
    """Display matplotlib figures without Agg-backend warnings."""
    if matplotlib.get_backend().lower() == "agg":
        for num in plt.get_fignums():
            display(plt.figure(num))
    else:
        getattr(plt, "show")()
```

Then call `_show_plot()` instead of `plt.show()` at the end of
each plotting cell.

### Spot-check long-running notebooks

Some notebooks routinely take 3–5 minutes when fully executed
(`isochrone.ipynb` with MERRA-2; calibration notebooks with full
sortie replay).  When iterating on a tutorial change:

1. Edit the affected cell(s) and run them in your local kernel
   to verify outputs look right.
2. If your change is structural (added/removed a section,
   changed cell dependencies), do a full nbconvert
   end-to-end pass.
3. If your change is purely cosmetic (a typo, a small markdown
   reword), no re-execute is needed — just commit the source
   edit.

For network-dependent sections gated on `has_earthdata` /
`has_gfs` etc., it's fine to commit a notebook executed without
those credentials; the gated section will print a "skipped"
message.

### Adding new notebooks

1. Place the file in the appropriate category and link it from
   [`README.md`](README.md) in the right table.
2. Use nbviewer URLs in the README link
   (`https://nbviewer.org/github/ryanpavlick/hyplan/blob/main/notebooks/<file>.ipynb`)
   for reliable rendering.
3. If the notebook is a developer / calibration tool rather
   than a user tutorial, put it under `calibration/<aircraft>/`
   and don't link from the user-facing README.

### Stable cell IDs

Cells that are referenced by `_build_notebook.py` scripts or by
external tools (e.g., the imports cell, the section-9 refuel
cell) need stable `id` fields in the JSON.  When you add a new
section, give it a meaningful `id` (`section8-foo`,
`example-bar`) rather than letting `nbconvert` auto-assign one.

## Verifying changes locally

Before pushing notebook changes:

```bash
# 1. Re-execute the notebook(s) you touched.
jupyter nbconvert --to notebook --execute notebooks/<file>.ipynb \
  --inplace --ExecutePreprocessor.timeout=300

# 2. Sanity-check the diff doesn't include unrelated cells.
git diff notebooks/<file>.ipynb | head -80

# 3. Confirm test suite is unaffected.
python -m pytest -q
```

If a notebook touches a section gated on credentials you don't
have (MERRA-2, GFS, GEE), the gated cell will skip — that's the
intended behavior.  Just confirm the surrounding cells still
produce sensible output and commit.
