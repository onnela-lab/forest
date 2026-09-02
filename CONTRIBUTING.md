# Contributing to Forest

Thanks for your interest in contributing. Forest is developed and maintained by the
[Onnela Lab](https://www.hsph.harvard.edu/onnela-lab/) in the Department of Biostatistics at the
Harvard T.H. Chan School of Public Health, alongside the
[Beiwe platform](https://github.com/onnela-lab/beiwe-backend).

## Getting started

Install the development version and run the test suite:

```bash
git clone https://github.com/onnela-lab/forest.git
cd forest
pip install -e .
pytest
```

The `develop` branch is the active development trunk. Please branch from `develop` and open
pull requests against it rather than `main`.

## Ways to contribute

- **Report a bug.** Open an issue using the bug report template. Include your Python version,
  your Forest version, and a minimal example that reproduces the problem.
- **Request a feature or a new method.** Open an issue describing the analysis you need and,
  where relevant, the published method it is based on.
- **Submit a change.** Small fixes can go straight to a pull request. For larger changes,
  especially a new tree, please open an issue first so we can discuss the design before you
  invest the effort.

## Adding a new tree

Forest is organized into independent subpackages called trees, each implementing one
methodological pipeline. A new tree should implement a method that has been described in the
peer-reviewed literature, use the shared data structures and conventions provided by the
Poplar utility layer, and ship with tests and documentation.

## Pull request expectations

- Branch from `develop` and keep the change focused on a single concern.
- Include tests covering new or changed behavior.
- Update the documentation under `docs/` when you change user-facing behavior.
- Code should pass the repository's linting and type checking, and follow the import
  conventions used by the surrounding subpackage.
- Describe what the change does and why in the pull request body.

A maintainer will review your pull request. Reviews are done by a small academic team, so
please allow some time for a response.

## Code of conduct

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).
