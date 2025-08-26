#!/bin/bash
# MR BEN Trading System - Live Trading Runner
# Runs live or paper trading

set -e

echo "🚀 MR BEN Trading System - Live Trading"
echo "======================================="

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <mode> <symbol> [config_path]"
    echo "Modes: live, paper"
    echo "Example: $0 paper XAUUSD.PRO"
    echo "Example: $0 live XAUUSD.PRO config/production.json"
    echo ""
    echo "⚠️  WARNING: 'live' mode uses real money!"
    echo "    Use 'paper' mode for testing."
    exit 1
fi

MODE=$1
SYMBOL=$2
CONFIG_PATH=${3:-""}

# Validate mode
if [ "$MODE" != "live" ] && [ "$MODE" != "paper" ]; then
    echo "❌ Error: Mode must be 'live' or 'paper'"
    exit 1
fi

# Safety check for live mode
if [ "$MODE" = "live" ]; then
    echo "⚠️  WARNING: You are about to start LIVE trading!"
    echo "    This will use real money and can result in financial losses."
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " -r
    if [ "$REPLY" != "yes" ]; then
        echo "Live trading cancelled."
        exit 0
    fi
    echo ""
fi

echo "Starting $MODE trading for $SYMBOL"
if [ -n "$CONFIG_PATH" ]; then
    echo "Using configuration: $CONFIG_PATH"
fi
echo ""

# Run live trading
if [ -n "$CONFIG_PATH" ]; then
    python src/core/cli.py live --mode "$MODE" --symbol "$SYMBOL" --config "$CONFIG_PATH"
else
    python src/core/cli.py live --mode "$MODE" --symbol "$SYMBOL"
fi

echo ""
echo "✅ Live trading completed"
