"""Sanity checks on .github/notebooks/manifest.yml.

The manifest is the source of truth for which notebooks run under
which CI tier.  These tests catch the common drift cases:

* A notebook is added to ``notebooks/`` but not listed in the
  manifest — silent gap in CI coverage.
* A manifest entry references a notebook that no longer exists —
  matrix row will fail at execution time.

Calibration notebooks (``notebooks/calibration/...``) are listed in
the manifest with ``group: calibration`` and only execute on manual
``workflow_dispatch`` with ``include_calibration=true``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / ".github" / "notebooks" / "manifest.yml"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


yaml = pytest.importorskip("yaml")


def _manifest_names() -> set[str]:
    """Return the set of `name:` entries in the manifest."""
    data = yaml.safe_load(MANIFEST.read_text())
    return {entry["name"] for entry in data["notebooks"]}


def _on_disk_names() -> set[str]:
    """Return every notebooks/**/*.ipynb path relative to notebooks/."""
    return {
        str(p.relative_to(NOTEBOOKS_DIR))
        for p in NOTEBOOKS_DIR.rglob("*.ipynb")
        # Ignore Jupyter checkpoint files.
        if ".ipynb_checkpoints" not in p.parts
    }


def test_every_notebook_listed_in_manifest() -> None:
    """No silent CI-coverage gaps: every .ipynb under notebooks/ must
    appear in the manifest."""
    on_disk = _on_disk_names()
    listed = _manifest_names()
    missing = on_disk - listed
    assert not missing, (
        f"Notebooks on disk but missing from .github/notebooks/manifest.yml:\n"
        f"  {sorted(missing)}\n"
        f"Add them with an `extras:` / `requires:` / `group:` entry."
    )


def test_every_manifest_entry_exists() -> None:
    """No stale manifest entries: every `name:` must point to a real
    notebook file."""
    on_disk = _on_disk_names()
    listed = _manifest_names()
    stale = listed - on_disk
    assert not stale, (
        f"Manifest entries reference notebooks that don't exist:\n"
        f"  {sorted(stale)}"
    )


def test_manifest_has_no_duplicates() -> None:
    """No duplicate `name:` keys in the manifest."""
    data = yaml.safe_load(MANIFEST.read_text())
    names = [entry["name"] for entry in data["notebooks"]]
    seen: set[str] = set()
    dupes: list[str] = []
    for n in names:
        if n in seen:
            dupes.append(n)
        seen.add(n)
    assert not dupes, f"Duplicate manifest entries: {dupes}"
