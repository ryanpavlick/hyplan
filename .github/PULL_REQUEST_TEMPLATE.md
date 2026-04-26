## Summary

<!-- 1-3 sentences: what does this PR change and why? -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would change existing API)
- [ ] Documentation only
- [ ] CI / build / tooling

## Test plan

<!-- How did you verify this works? Mark relevant items with [x]. -->

- [ ] `python -m ruff check hyplan tests` — clean
- [ ] `python -m pytest tests/` — all tests pass
- [ ] `python -m sphinx -b html -W docs docs/_build/html` — docs build clean
- [ ] Relevant notebooks re-executed via `papermill`

## CHANGELOG

- [ ] CHANGELOG.md updated under the appropriate `## v1.x.0 (unreleased)`
      section, OR this change is internal-only and CHANGELOG-exempt.

## Backward compatibility

<!-- For changes touching public API: confirm v1.x stability is preserved.
     If not, justify the break and describe the migration path. -->
