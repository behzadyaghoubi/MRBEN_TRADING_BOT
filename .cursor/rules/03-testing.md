\# Testing Policy

\- Every new feature includes unit tests and at least one edge-case test.

\- Backtesting functions must have deterministic tests (seeded).

\- For risk manager (ATR SL/TP), test BUY/SELL, low/high ATR, rounding.

\- CI-friendly: tests runnable via `pytest -q`.

