\# Architecture Rules (MR BEN)

\- Project layers:

&nbsp; - data/: data loaders \& MT5 fetchers

&nbsp; - indicators/: TA helpers (ATR, RSI, MACD, Bollinger, ...)

&nbsp; - signals/: rule-based \& fusion logic (TA + PriceAction + AI filter)

&nbsp; - risk\_manager/: SL/TP (ATR-based), position sizing, max risk per trade

&nbsp; - execution/: broker/MT5 integration (order send/cancel, retry)

&nbsp; - utils/: shared helpers (common, config loader, logging utils)

&nbsp; - dashboard/: Flask/Tkinter UI (read-only controls, charts, logs)

\- Dependencies:

&nbsp; - signals -> indicators, utils

&nbsp; - risk\_manager -> indicators, utils

&nbsp; - execution -> utils

&nbsp; - dashboard -> read from logs/DB; no direct coupling to signals

\- Config:

&nbsp; - All tunables in `config/\*.json` or `settings.py`

&nbsp; - No hard-coded API keys. Use .env (python-dotenv).

\- Error handling:

&nbsp; - Fail safe defaults; graceful degrade if data missing (e.g., ATR unavailable)

\- Logging:

&nbsp; - Structured logging; CSV/SQLite optional via config flags

