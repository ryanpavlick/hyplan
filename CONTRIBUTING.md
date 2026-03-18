# Contributing to HyPlan

Thank you for your interest in contributing to HyPlan! This guide will help you get started.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hyplan.git
   cd hyplan
   ```
3. **Install** in development mode:
   ```bash
   pip install -e .
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

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for function signatures where practical.
- All physical quantities should use [Pint](https://pint.readthedocs.io/) with the shared `hyplan.units.ureg` registry.

### Documentation

- Add docstrings (NumPy/Google style) to public functions and classes.
- Update or add notebooks in `notebooks/` when introducing new features.
- To build the docs locally:
  ```bash
  pip install sphinx myst-parser furo sphinx-autodoc-typehints
  cd docs
  make html
  ```

## Submitting Changes

1. Ensure all tests pass (`pytest`).
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
