"""Build the GitHub Actions matrix for `.github/workflows/notebooks.yml`.

Reads `.github/notebooks/manifest.yml`, applies tier filters, drops
entries whose required auth secrets aren't present, and emits a JSON
matrix on stdout suitable for `matrix: ${{ fromJson(...) }}`.

Usage::

    python build_matrix.py \\
        --tier {pr,nightly,calibration} \\
        [--changed-file path ...] \\
        [--has-earthdata] [--has-openaip] [--has-gee]

Tiers:

* **pr**       — keep only entries whose ``name:`` appears in any of the
                  ``--changed-file`` arguments (relative to repo root).
                  ``group: calibration`` entries are excluded.
* **nightly**  — every ``group: tutorial`` entry, plus every ``group: auth``
                  entry whose ``requires:`` is satisfied by the
                  ``--has-*`` flags.  Calibration excluded.
* **calibration** — every ``group: calibration`` entry.

Output is one line of JSON: ``{"include": [...]}`` where each include
row carries ``name``, ``extras`` (string for ``pip install -e ".[...]"``),
and ``slug`` (filesystem-safe artifact suffix).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover -- CI installs pyyaml in the setup step
    sys.stderr.write(
        "build_matrix.py requires PyYAML. Run: pip install pyyaml\n"
    )
    sys.exit(2)


MANIFEST = Path(__file__).resolve().parent / "manifest.yml"

KNOWN_SECRETS = ("earthdata", "openaip", "gee")
KNOWN_GROUPS = ("tutorial", "auth", "calibration")


def _load_manifest() -> list[dict]:
    with MANIFEST.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "notebooks" not in data:
        raise SystemExit(f"{MANIFEST}: missing top-level `notebooks:` list")
    entries = data["notebooks"]
    if not isinstance(entries, list):
        raise SystemExit(f"{MANIFEST}: `notebooks` must be a list")
    for entry in entries:
        if "name" not in entry:
            raise SystemExit(f"{MANIFEST}: entry missing `name`: {entry}")
        entry.setdefault("extras", [])
        entry.setdefault("requires", [])
        entry.setdefault("group", "tutorial")
        if entry["group"] not in KNOWN_GROUPS:
            raise SystemExit(
                f"{MANIFEST}: {entry['name']!r} has unknown group "
                f"{entry['group']!r}; allowed: {KNOWN_GROUPS}"
            )
        for req in entry["requires"]:
            if req not in KNOWN_SECRETS:
                raise SystemExit(
                    f"{MANIFEST}: {entry['name']!r} has unknown requires "
                    f"{req!r}; allowed: {KNOWN_SECRETS}"
                )
    return entries


def _slug(name: str) -> str:
    """Convert a notebook path into a filesystem-safe matrix-cell id."""
    return name.replace("/", "__").replace(".ipynb", "")


def _extras_arg(extras: list[str]) -> str:
    """Format the comma-separated extras list for `pip install -e ".[...]"`.

    `notebooks` is always present (the tooling extra)."""
    joined = ["notebooks", *extras]
    return ",".join(joined)


def _filter_changed(entries: list[dict], changed: set[str]) -> list[dict]:
    """Keep entries whose `name` field appears among the changed files
    (after prefixing with ``notebooks/``)."""
    return [e for e in entries if f"notebooks/{e['name']}" in changed]


def _filter_secrets(entries: list[dict], present: set[str]) -> list[dict]:
    """Drop entries whose `requires:` lists a secret not in `present`."""
    return [e for e in entries if all(req in present for req in e["requires"])]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("pr", "nightly", "calibration"), required=True)
    p.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="File path relative to repo root; pass once per changed file.",
    )
    for secret in KNOWN_SECRETS:
        p.add_argument(
            f"--has-{secret}",
            action="store_true",
            help=f"Set when the {secret.upper()} secret is present.",
        )
    args = p.parse_args(argv)

    entries = _load_manifest()

    if args.tier == "pr":
        entries = [e for e in entries if e["group"] != "calibration"]
        if args.changed_file:
            entries = _filter_changed(entries, set(args.changed_file))
        else:
            entries = []
    elif args.tier == "nightly":
        entries = [e for e in entries if e["group"] != "calibration"]
        present = {s for s in KNOWN_SECRETS if getattr(args, f"has_{s}")}
        entries = _filter_secrets(entries, present)
    elif args.tier == "calibration":
        entries = [e for e in entries if e["group"] == "calibration"]

    include = [
        {
            "name": e["name"],
            "slug": _slug(e["name"]),
            "extras": _extras_arg(e["extras"]),
        }
        for e in entries
    ]
    print(json.dumps({"include": include}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
