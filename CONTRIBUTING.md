# Contributing to TimeSeries Studio AI

Thanks for your interest in contributing! This project stays intentionally
lightweight and 100% local — please keep that principle in mind for any PR.

## Getting Started

1. Fork the repo and clone your fork
2. Set up the dev environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

- **Keep it local**: no new dependency should require an API key or network
  call for core functionality. Optional integrations (e.g. Ollama) must
  degrade gracefully when absent.
- **Follow the existing module boundaries**: data loading, feature
  engineering, models, benchmarking, anomaly detection, and explainability
  each live in their own file under `core/`. New models go in `models.py` and
  register in `MODEL_REGISTRY`.
- **Match the existing code style**: type hints on public functions,
  docstrings on classes, dataclasses for structured results.
- **Dashboard changes**: keep the light theme CSS variables at the top of
  `ui/dashboard.py` consistent; don't hardcode colors elsewhere.

## Submitting a Pull Request

1. Ensure your code runs without errors: `python -m py_compile <changed files>`
2. Test the dashboard manually: `streamlit run ui/dashboard.py`
3. Update `CHANGELOG.md` under `[Unreleased]`
4. Open a PR with a clear description of the change and why it's needed
5. Link any related issue

## Reporting Bugs

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Your Python version and OS
- Full error traceback if applicable

## Feature Requests

Open an issue describing the use case first — happy to discuss design before
a PR is written, especially for anything touching the model registry or data
pipeline.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please
be respectful in all interactions.
