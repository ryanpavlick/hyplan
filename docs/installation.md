# Installation

## From source (pip)

```bash
git clone https://github.com/ryanpavlick/hyplan
cd hyplan
pip install -e .
```

## From source (conda/mamba)

```bash
mamba env create --file hyplan/hyplan.yml
mamba activate hyplan-env
pip install -e hyplan
```

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
