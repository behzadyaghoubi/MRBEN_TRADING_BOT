#!/bin/bash
# MR BEN Trading System - Smoke Test Runner
# Runs a quick smoke test with sample data

set -e

echo "🚀 MR BEN Trading System - Smoke Test"
echo "======================================"

# Default values
MINUTES=${1:-5}
SYMBOL=${2:-"XAUUSD.PRO"}

echo "Running smoke test for $SYMBOL for $MINUTES minutes..."
echo ""

# Run smoke test
python src/core/cli.py smoke --minutes "$MINUTES" --symbol "$SYMBOL"

echo ""
echo "✅ Smoke test completed"
