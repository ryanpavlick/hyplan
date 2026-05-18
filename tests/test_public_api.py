"""Sanity checks on HyPlan's public API surface.

Ensures every name in the ``__all__`` of a core, installed-by-default
subpackage resolves cleanly via ``getattr``.  Catches the case where
``__all__`` advertises a symbol that's no longer exported from the
module — exactly the kind of drift that would cause
``from hyplan import X`` to fail at import time for users.

Scoped to core surfaces only.  Optional-extra submodules (winds, clouds,
phenology, adsb, mag, planned) are not iterated here — they'd make CI
flaky in any environment that doesn't install the extra.  Per-extra
``__all__`` tests live in their own files gated on the extra being
installed.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

CORE_MODULES: tuple[str, ...] = (
    "hyplan",
    "hyplan.instruments",
    "hyplan.planning",
    "hyplan.exports",
)


def _module_all(module_name: str) -> list[str]:
    """Return ``__all__`` for ``module_name`` or fail the test that uses it."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "__all__"):
        pytest.skip(f"{module_name} has no __all__ to verify")
    names = list(module.__all__)
    if not names:
        pytest.skip(f"{module_name}.__all__ is empty")
    return names


def _iter_module_names() -> list[tuple[str, str]]:
    """Yield ``(module, name)`` tuples for every entry in every core ``__all__``."""
    pairs: list[tuple[str, str]] = []
    for module_name in CORE_MODULES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            pairs.append((module_name, name))
    return pairs


_MODULE_NAME_PAIRS = _iter_module_names()


@pytest.mark.parametrize(
    ("module_name", "name"),
    _MODULE_NAME_PAIRS,
    ids=[f"{m}.{n}" for m, n in _MODULE_NAME_PAIRS],
)
def test_all_entry_is_importable(module_name: str, name: str) -> None:
    """Every name in ``module.__all__`` must resolve via ``getattr``."""
    module = importlib.import_module(module_name)
    assert hasattr(module, name), (
        f"{module_name}.__all__ advertises {name!r} but the module has no "
        f"such attribute — would break `from {module_name} import {name}`."
    )
    obj = getattr(module, name)
    assert obj is not None, f"{module_name}.{name} resolved to None"


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_has_all(module_name: str) -> None:
    """Each core module declares an ``__all__`` — required for the
    importability test above and for documentation tooling."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), (
        f"{module_name} should declare __all__ so its public surface is explicit."
    )
    assert isinstance(module.__all__, (list, tuple)), (
        f"{module_name}.__all__ should be a list or tuple"
    )


def test_no_duplicate_names_in_all() -> None:
    """``__all__`` lists should not contain duplicate names."""
    failures: list[str] = []
    for module_name in CORE_MODULES:
        names: Any = importlib.import_module(module_name).__all__
        seen: set[str] = set()
        dupes: list[str] = []
        for n in names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        if dupes:
            failures.append(f"{module_name}.__all__ duplicates: {dupes}")
    assert not failures, "\n".join(failures)
