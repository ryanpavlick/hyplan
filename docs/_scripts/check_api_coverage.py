"""Fail-fast checks for HyPlan's docs API coverage.

Two checks:

1. **Orphan pages**: every `docs/api/**/*.md` is referenced by at least one
   ``toctree`` block reachable from ``docs/index.md``.  Pages under
   ``docs/dev/`` are checked the same way against the developer-guide
   toctree.

2. **Symbol existence**: every Sphinx autodoc directive in ``docs/**/*.md``
   (``autofunction``, ``autoclass``, ``automethod``, ``autodata``,
   ``autoexception``, ``autoattribute``, ``automodule``) names a Python
   symbol that imports cleanly via its qualified path.

Exits 0 on success, 1 on any failure (with per-failure messages on stderr).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
INDEX = DOCS_ROOT / "index.md"

# Pages that intentionally live outside the toctree (top-level entry points,
# rendered standalone, or generated).  Keep this list as small as possible.
ORPHAN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "index.md",  # the index itself
    }
)

# Autodoc directives whose argument is a qualified Python symbol that we
# can `importlib.import_module` + `getattr` to verify.  `automodule` only
# requires the module to import.
_AUTODOC_DIRECTIVES = (
    "autofunction",
    "autoclass",
    "automethod",
    "autodata",
    "autoexception",
    "autoattribute",
    "automodule",
)

_AUTODOC_RE = re.compile(
    rf"^\.\.\s+({'|'.join(_AUTODOC_DIRECTIVES)})::\s+(\S+)",
    re.MULTILINE,
)

# MyST toctree blocks look like:
#     ```{toctree}
#     :maxdepth: 2
#     :caption: Flight planning
#
#     api/flight_line
#     api/waypoint
#     ```
_TOCTREE_BLOCK_RE = re.compile(
    r"```\{toctree\}(.*?)```",
    re.DOTALL,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _toctree_entries(text: str) -> list[str]:
    """Return every non-option, non-blank line inside a ``{toctree}`` block."""
    entries: list[str] = []
    for block_match in _TOCTREE_BLOCK_RE.finditer(text):
        body = block_match.group(1)
        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith(":"):
                continue
            entries.append(line)
    return entries


def find_orphan_pages() -> list[Path]:
    """Return docs/api/ + docs/dev/ markdown pages not referenced by any toctree."""
    referenced: set[str] = set()
    for md in DOCS_ROOT.rglob("*.md"):
        for entry in _toctree_entries(_read(md)):
            # Resolve entry relative to the file it was found in.
            base = md.parent
            candidate = (base / entry).with_suffix(".md").resolve()
            referenced.add(str(candidate))

    orphans: list[Path] = []
    for area in ("api", "dev"):
        for md in (DOCS_ROOT / area).glob("**/*.md") if (DOCS_ROOT / area).is_dir() else []:
            if md.name in ORPHAN_ALLOWLIST:
                continue
            if str(md.resolve()) not in referenced:
                orphans.append(md)
    return sorted(orphans)


def find_broken_symbols() -> list[tuple[Path, str, str, str]]:
    """Return (doc_path, directive, qualified_name, error) tuples for any
    autodoc directive whose argument can't be imported."""
    broken: list[tuple[Path, str, str, str]] = []
    for md in DOCS_ROOT.rglob("*.md"):
        text = _read(md)
        for directive, qualname in _AUTODOC_RE.findall(text):
            error = _import_or_error(directive, qualname)
            if error is not None:
                broken.append((md, directive, qualname, error))
    return broken


def _import_or_error(directive: str, qualname: str) -> str | None:
    """Return ``None`` on success, an error string otherwise."""
    if directive == "automodule":
        try:
            importlib.import_module(qualname)
        except Exception as exc:
            return f"import {qualname!r} failed: {exc}"
        return None

    # Other directives: split into module + attribute chain.  Walk attr-by-attr
    # so we report a useful error for e.g. ``hyplan.foo.BAR.method`` failures.
    if "." not in qualname:
        return f"{qualname!r} is not a qualified path (expected module.symbol)"

    parts = qualname.split(".")
    # Try the longest importable prefix as the module, then resolve the rest
    # as attributes.
    module = None
    module_path = ""
    for i in range(len(parts), 0, -1):
        candidate_module = ".".join(parts[:i])
        try:
            module = importlib.import_module(candidate_module)
            module_path = candidate_module
            remainder = parts[i:]
            break
        except Exception:
            continue
    if module is None:
        return f"could not import any module prefix of {qualname!r}"

    obj = module
    walked = module_path
    for attr in remainder:
        if not hasattr(obj, attr):
            return f"{walked!r} has no attribute {attr!r}"
        obj = getattr(obj, attr)
        walked = f"{walked}.{attr}"
    return None


def main() -> int:
    errors = 0

    orphans = find_orphan_pages()
    if orphans:
        errors += len(orphans)
        print("Orphan API pages (not referenced by any toctree):", file=sys.stderr)
        for path in orphans:
            print(f"  {path.relative_to(DOCS_ROOT.parent)}", file=sys.stderr)

    broken = find_broken_symbols()
    if broken:
        errors += len(broken)
        print("\nAutodoc directives pointing at unimportable symbols:", file=sys.stderr)
        for path, directive, qualname, err in broken:
            rel = path.relative_to(DOCS_ROOT.parent)
            print(f"  {rel}: .. {directive}:: {qualname}\n    {err}", file=sys.stderr)

    if errors:
        print(
            f"\ncheck_api_coverage.py: {errors} problem(s) found.",
            file=sys.stderr,
        )
        return 1

    print("check_api_coverage.py: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
