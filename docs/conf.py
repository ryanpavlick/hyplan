"""Sphinx configuration for HyPlan documentation."""

import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project information ------------------------------------------------------
project = "HyPlan"
copyright = "2025, Ryan Pavlick"
author = "Ryan Pavlick"

from hyplan import __version__
release = __version__

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    # myst_nb is a superset of myst_parser; it adds .ipynb support and
    # supersedes the bare myst_parser extension. Don't enable both.
    "myst_nb",
]

# MyST settings
myst_enable_extensions = [
    "dollarmath",       # $inline$ and $$block$$ math
    "colon_fence",      # ::: directive syntax
]

# myst-nb: render notebooks using their already-committed outputs.
# We never execute at docs build time — committed outputs are the source
# of truth for tutorials, and the nightly notebook smoke-test workflow
# (.github/workflows/notebooks.yml) is what guarantees they keep working.
nb_execution_mode = "off"

# Source file suffixes — myst-nb registers itself as the parser for both
# Markdown and Jupyter notebooks, so we hand both extensions to it.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Mock only truly optional imports — the docs CI workflow now does a real
# `pip install -e .`, so all required dependencies are available at build time.
# Only keep mocks for optional extras and packages not listed in pyproject.toml.
autodoc_mock_imports = [
    "dubins",
    "ee",           # [clouds] extra
    "numba",
    "seaborn",      # [clouds] extra
    "tabulate",
    "pydantic",
    "pydantic_pint",
]

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pint": ("https://pint.readthedocs.io/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output --------------------------------------------------
html_theme = "furo"
html_title = "HyPlan"

html_theme_options = {
    "source_repository": "https://github.com/ryanpavlick/hyplan",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Don't show module names in front of class names
add_module_names = False

exclude_patterns = ["_build"]
