\# Python Style \& Quality

\- Python 3.11+, type hints required for all public functions/classes.

\- Docstrings: Google style.

\- Lint/format: ruff + black + isort; code must pass.

\- No copy-paste duplication; extract helpers into utils/.

\- Unit tests: pytest, name `tests/test\_<module>.py`.

\- Raise specific exceptions; avoid bare `except`.

\- I/O boundaries clear; no prints in library code (use logger).

