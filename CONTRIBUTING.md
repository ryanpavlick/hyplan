# Contributing to HyPlan

Thank you for your interest in contributing to HyPlan! This guide will help you get started.

## Getting Help

Before opening an issue, check the [documentation](https://ryanpavlick.github.io/hyplan/) and the [example notebooks](notebooks/) — most "how do I do X" answers live there.

If that doesn't cover your question:

- **General usage questions** — open a [GitHub issue](https://github.com/ryanpavlick/hyplan/issues/new) and apply the `question` label. Maintainers and other users can respond there.
- **Bug reports** — see [Reporting Bugs](#reporting-bugs) below.
- **Feature requests** — see [Feature Requests](#feature-requests) below.
- **Security disclosures** — follow the process in [SECURITY.md](SECURITY.md).
- **Direct contact** — for things that don't fit the above (collaboration inquiries, calibration data sharing, anything sensitive), email Ryan Pavlick at <ryan.p.pavlick@nasa.gov>.

We aim to respond to issues within a few working days; this is a small project so please be patient.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hyplan.git
   cd hyplan
   ```
3. **Install** in development mode (the `dev` extra pulls in `pytest`, `pytest-cov`, `ruff`, and `mypy` — the same toolchain CI runs):
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b my-feature
   ```

## Development Workflow

### Running Tests

HyPlan uses [pytest](https://docs.pytest.org/) for testing:

```bash
pytest
```

### Before You Push

CI gates every pull request on linting (Ruff), strict type checking (mypy), and the test suite. Run the same checks locally before pushing:

```bash
ruff check hyplan tests
mypy hyplan
python -m pytest
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for function signatures where practical.
- All physical quantities should use [Pint](https://pint.readthedocs.io/) with the shared `hyplan.units.ureg` registry.

### Documentation

- Add docstrings (NumPy/Google style) to public functions and classes.
- Update or add notebooks in `notebooks/` when introducing new features.
- To build the docs locally (the `docs` extra installs Sphinx, MyST-NB, Furo, and sphinx-autodoc-typehints):
  ```bash
  pip install -e ".[docs]"
  cd docs
  make html
  ```

## Submitting Changes

1. Ensure lint, type checks, and tests all pass (see [Before You Push](#before-you-push)).
2. Commit your changes with a clear, descriptive message.
3. Push to your fork and open a **pull request** against `main`.
4. Describe what your PR does and why.

## Reporting Bugs

Open an [issue](https://github.com/ryanpavlick/hyplan/issues) with:

- A clear title and description.
- Steps to reproduce the problem.
- Expected vs. actual behavior.
- Python version and OS.

## Feature Requests

Feature requests are welcome! Please open an [issue](https://github.com/ryanpavlick/hyplan/issues) describing the use case and proposed behavior.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE.md).
