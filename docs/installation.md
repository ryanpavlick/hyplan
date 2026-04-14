# Installation

## From source (pip)

```bash
git clone https://github.com/ryanpavlick/hyplan
cd hyplan
pip install -e .
```

## From source (conda/mamba)

```bash
mamba env create --name hyplan --file environment.yml
mamba activate hyplan
pip install -e .
```

## Optional extras

HyPlan keeps its core install lightweight and gates niche features behind
optional dependency groups. Install one or more with the usual
`pip install hyplan[<extra>]` syntax (combine multiple in a single bracket
list, e.g. `pip install -e .[gui,mag]`).

| Extra | Pulls in | Enables |
|-------|----------|---------|
| `mag` | `geomag` | Magnetic-declination correction in [`hyplan.exports.to_pilot_excel`](api/exports.md) when `include_mag_heading=True`, and {func}`hyplan.geometry.true_to_magnetic`. |
| `clouds` | `earthengine-api`, `seaborn` | The {mod}`hyplan.clouds` module — climatology and time-series cloud queries against Google Earth Engine. Requires a separately-authenticated GEE account. |
| `gui` | `ipyleaflet`, `ipywidgets>=8`, `ipydatagrid` | The {mod}`hyplan.gui` JupyterLab widgets for waypoint editing and flight-line management. |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` | The full test, lint, and type-check toolchain used in CI. Run `pytest --cov=hyplan` after installing. |

None of the extras are required to run the core flight-planning, swath,
or export workflows. If you only want to read the API reference or run
the tutorial, the base install is enough.

## Versioning

HyPlan uses [setuptools-scm](https://setuptools-scm.readthedocs.io/) to
derive its version automatically from git tags. There is no hardcoded version
string to maintain.

- **Tagged commits** produce clean versions: `git tag v0.2.0` gives version `0.2.0`.
- **Development installs** between tags produce versions like `0.2.1.dev3+g1a2b3c4`.
- **Check the current version** with `python -c "import hyplan; print(hyplan.__version__)"`.

To create a new release:

```bash
git tag v0.2.0
git push origin v0.2.0
pip install -e .  # rebuilds _version.py with the new tag
```

## Building the documentation

```bash
pip install sphinx myst-parser furo sphinx-autodoc-typehints
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.
