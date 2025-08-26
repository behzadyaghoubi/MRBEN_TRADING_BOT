\# Security \& Secrets

\- Never commit secrets; use `.env` + python-dotenv.

\- Add `.env` to .gitignore (already done).

\- Validate external inputs; sanitize file paths.

\- When calling MT5/broker APIs, handle timeouts/retries and log failures.

\- No network calls in unit tests; mock external services.
