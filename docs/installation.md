# Installation

Requires Python 3.10+.

## From PyPI (recommended)

```bash
pip install hyplan
```

This pulls the published wheel and the bundled skyfield ephemeris file
(`hyplan/data/de421.bsp`), so satellite-overpass and solar-geometry
features work without a first-run download.

## From source (for development)

```bash
git clone https://github.com/ryanpavlick/hyplan
cd hyplan
pip install -e ".[dev]"
```

The `[dev]` extra pulls `pytest`, `pytest-cov`, `ruff`, and `mypy` so
the test, lint, and type-check toolchain match CI exactly.

## With conda/mamba

For users who prefer mamba to manage the geospatial dependency stack
(`rasterio`, `pyproj`, `cfgrib`/`eccodes`, `netcdf4`):

```bash
mamba env create --name hyplan --file environment.yml
mamba activate hyplan
pip install hyplan          # or: pip install -e ".[dev]" from a clone
```

## Optional extras

HyPlan keeps its core install lightweight and gates niche features behind
optional dependency groups. Install one or more with the usual
`pip install hyplan[<extra>]` syntax (combine multiple in a single bracket
list, e.g. `pip install hyplan[clouds,mag]`).

| Extra | Pulls in | Enables |
|-------|----------|---------|
| `mag` | `geomag` | Magnetic-declination correction in [`hyplan.exports.to_pilot_excel`](api/exports.md) when `include_mag_heading=True`, and {func}`hyplan.geometry.true_to_magnetic`. |
| `clouds` | `earthengine-api`, `seaborn` | The {mod}`hyplan.clouds` module — climatology and time-series cloud queries against Google Earth Engine. Requires a separately-authenticated GEE account. |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` | The full test, lint, and type-check toolchain used in CI. Run `pytest --cov=hyplan` after installing. |

None of the extras are required to run the core flight-planning, swath,
or export workflows. If you only want to read the API reference or run
the tutorial, the base install is enough.

## Versioning

HyPlan uses [setuptools-scm](https://setuptools-scm.readthedocs.io/) to
derive its version automatically from git tags. There is no hardcoded version
string to maintain.

- **PyPI installs** pin to the tagged release (`pip install hyplan==1.6.2`).
- **Tagged commits** in a source clone produce clean versions: `git tag v1.7.0` gives version `1.7.0`.
- **Development installs** between tags produce versions like `1.7.1.dev3+g1a2b3c4`.
- **Check the current version** with `python -c "import hyplan; print(hyplan.__version__)"`.

Releases are published to PyPI automatically by `release.yml` →
`publish.yml` (PyPI Trusted Publishing).  To cut a new release:

```bash
gh workflow run release.yml -f version=1.7.0
```

This bumps `CITATION.cff`, tags `v1.7.0`, creates the GitHub Release,
and triggers the PyPI upload.

## Library logging

HyPlan uses Python's standard ``logging`` module under the ``hyplan``
namespace. Library code attaches no handlers itself, so log messages are
silent by default. Call {func}`hyplan.setup_logging` once from a notebook,
script, or CLI to attach a stream handler:

```{eval-rst}
.. autofunction:: hyplan.setup_logging
```

## Building the documentation

```bash
pip install sphinx myst-parser furo sphinx-autodoc-typehints
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.
