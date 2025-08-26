#!/bin/bash
# MR BEN Trading System - Backtest Runner
# Runs backtests with specified parameters

set -e

echo "📊 MR BEN Trading System - Backtest"
echo "===================================="

# Check if required parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <symbol> <start_date> <end_date> [config_path]"
    echo "Example: $0 XAUUSD.PRO 2024-01-01 2024-01-31"
    echo "Example: $0 XAUUSD.PRO 2024-01-01 2024-01-31 config/custom.json"
    exit 1
fi

SYMBOL=$1
START_DATE=$2
END_DATE=$3
CONFIG_PATH=${4:-""}

echo "Running backtest for $SYMBOL from $START_DATE to $END_DATE"
if [ -n "$CONFIG_PATH" ]; then
    echo "Using configuration: $CONFIG_PATH"
fi
echo ""

# Run backtest
if [ -n "$CONFIG_PATH" ]; then
    python src/core/cli.py backtest --symbol "$SYMBOL" --from "$START_DATE" --to "$END_DATE" --config "$CONFIG_PATH"
else
    python src/core/cli.py backtest --symbol "$SYMBOL" --from "$START_DATE" --to "$END_DATE"
fi

echo ""
echo "✅ Backtest completed"
