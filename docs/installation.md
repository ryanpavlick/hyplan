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

## Building the documentation

```bash
pip install sphinx myst-parser furo sphinx-autodoc-typehints
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.
