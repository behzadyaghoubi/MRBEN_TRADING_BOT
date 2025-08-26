# MR BEN Trading System - Makefile
# Provides common development and deployment tasks

.PHONY: help install format lint test smoke backtest live clean

# Default target
help:
	@echo "MR BEN Trading System - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install    - Install dependencies"
	@echo "  format     - Format code with ruff, isort, and black"
	@echo "  lint       - Run linting with ruff and mypy"
	@echo "  test       - Run tests with pytest"
	@echo ""
	@echo "Trading:"
	@echo "  smoke      - Run smoke test (5 minutes, XAUUSD.PRO)"
	@echo "  backtest   - Run backtest (example: XAUUSD.PRO, 2024-01-01 to 2024-01-31)"
	@echo "  live       - Run live trading (paper mode, XAUUSD.PRO)"
	@echo ""
	@echo "AI Agent:"
	@echo "  agent      - Run AI Agent (observe mode)"
	@echo "  agent-paper- Run AI Agent (paper mode)"
	@echo "  agent-halt - Halt trading via AI Agent"
	@echo "  agent-regime - Run AI Agent with regime detection"
	@echo "  agent-no-regime - Run AI Agent without regime detection"
	@echo ""
	@echo "Experiments:"
	@echo "  exp-regime-on  - 60-day backtest with regime detection ON"
	@echo "  exp-regime-off - 60-day backtest with regime detection OFF"
	@echo ""
	@echo "Utilities:"
	@echo "  clean      - Clean generated files and caches"
	@echo "  help       - Show this help message"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Format code
format:
	@echo "🎨 Formatting code..."
	ruff check --fix .
	isort .
	black .
	@echo "✅ Code formatted"

# Run linting
lint:
	@echo "🔍 Running linting..."
	ruff check .
	mypy src/
	@echo "✅ Linting completed"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest -q
	@echo "✅ Tests completed"

# Run smoke test
smoke:
	@echo "🚀 Running smoke test..."
	python src/core/cli.py smoke --minutes 5 --symbol XAUUSD.PRO
	@echo "✅ Smoke test completed"

# Run backtest (example)
backtest:
	@echo "📊 Running backtest..."
	python src/core/cli.py backtest --symbol XAUUSD.PRO --from 2024-01-01 --to 2024-01-31
	@echo "✅ Backtest completed"

# Run live trading (paper mode)
live:
	@echo "🚀 Starting live trading (paper mode)..."
	python src/core/cli.py live --mode paper --symbol XAUUSD.PRO
	@echo "✅ Live trading completed"

# Run AI Agent (observe mode)
agent:
	@echo "🤖 Starting AI Agent in observe mode..."
	python src/core/cli.py agent --mode observe --symbol XAUUSD.PRO
	@echo "✅ AI Agent completed"

# Run AI Agent (paper mode)
agent-paper:
	@echo "🤖 Starting AI Agent in paper mode..."
	python src/core/cli.py agent --mode paper --symbol XAUUSD.PRO
	@echo "✅ AI Agent completed"

# Halt trading via AI Agent
agent-halt:
	@echo "🛑 Halting trading via AI Agent..."
	python src/core/cli.py agent --halt
	@echo "✅ Trading halted"

# Run AI Agent with regime detection enabled
agent-regime:
	@echo "🤖 Starting AI Agent with regime detection..."
	python src/core/cli.py agent --mode observe --symbol XAUUSD.PRO --regime-enabled true
	@echo "✅ AI Agent with regime detection completed"

# Run AI Agent with regime detection disabled
agent-no-regime:
	@echo "🤖 Starting AI Agent without regime detection..."
	python src/core/cli.py agent --mode observe --symbol XAUUSD.PRO --regime-enabled false
	@echo "✅ AI Agent without regime detection completed"

# Regime experiment: 60-day backtest with regime enabled
exp-regime-on:
	@echo "🧪 Running regime experiment: 60-day backtest with regime detection ON..."
	python src/core/cli.py backtest --symbol XAUUSD.PRO --from 2024-01-01 --to 2024-03-01 --regime-enabled true
	@echo "✅ Regime experiment (ON) completed"

# Regime experiment: 60-day backtest with regime disabled
exp-regime-off:
	@echo "🧪 Running regime experiment: 60-day backtest with regime detection OFF..."
	python src/core/cli.py backtest --symbol XAUUSD.PRO --from 2024-01-01 --to 2024-03-01 --regime-enabled false
	@echo "✅ Regime experiment (OFF) completed"

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
	rm -f smoke_config.json
	@echo "✅ Cleanup completed"

# Development workflow
dev: format lint test
	@echo "✅ Development workflow completed"

# Full workflow
all: install format lint test smoke
	@echo "✅ Full workflow completed"

# AI Agent workflow
agent-workflow: format lint test agent
	@echo "✅ AI Agent workflow completed"
