# Portfolio Expansion - Phase 4

## Overview
Multi-symbol portfolio trading with separate state management and global exposure control.

## Implementation

### 1. Symbol State Management
Each symbol maintains its own state:
```python
self.sym = {}
for s in cfg.portfolio.symbols:
    self.sym[s] = {
        "consecutive": 0,
        "last_signal": 0,
        "trailing": {},
        "last_trailing_update": datetime.now()
    }
```

### 2. Portfolio Configuration
```json
{
  "portfolio": {
    "symbols": ["XAUUSD.PRO", "EURUSD.PRO", "GBPUSD.PRO"],
    "max_open_trades_total": 4,
    "max_risk_per_symbol": 0.01
  }
}
```

### 3. Main Loop (Sequential)
Instead of single symbol, iterate over portfolio symbols:
- Receive data for each symbol
- Generate signals with `_generate_signal_v2(df, symbol=s)`
- Create DecisionCard with `symbol=s`
- Apply gates (spread/exposure/daily loss/session)
- Execute trades respecting `max_open_trades_total`

### 4. Global Exposure Gate
In RiskGate/AdvancedRiskGate, check total portfolio exposure:
```python
if total_open_positions >= cfg.portfolio.max_open_trades_total:
    return {
        "allow_trade": false,
        "reason": "PortfolioExposure",
        "action": "BLOCK_ONLY"
    }
```

## Symbol-Specific Features

### XAUUSD.PRO (Gold)
- **Timeframe**: 15 minutes
- **Risk**: 1% per trade
- **Session**: London + New York
- **Patterns**: ENGULF, PIN, INSIDE, SWEEP

### EURUSD.PRO (Euro)
- **Timeframe**: 15 minutes
- **Risk**: 1% per trade
- **Session**: London + New York
- **Patterns**: ENGULF, PIN, INSIDE, SWEEP

### GBPUSD.PRO (Pound)
- **Timeframe**: 15 minutes
- **Risk**: 1% per trade
- **Session**: London + New York
- **Patterns**: ENGULF, PIN, INSIDE, SWEEP

## Risk Management

### Per-Symbol Limits
- **Max Open Trades**: 2 per symbol
- **Risk Per Trade**: 1% of account balance
- **Daily Loss Limit**: 2% of account balance

### Portfolio Limits
- **Total Open Trades**: Maximum 4 across all symbols
- **Total Risk**: Maximum 4% across all symbols
- **Correlation**: Monitor currency pair correlations

## Logging Format
All logs include symbol prefix for easy identification:
```
[2025-08-20 21:00:00][XAUUSD.PRO] Signal generated: BUY (confidence: 0.75)
[2025-08-20 21:00:00][EURUSD.PRO] Signal generated: SELL (confidence: 0.68)
[2025-08-20 21:00:00][GBPUSD.PRO] No signal (confidence: 0.45)
```

## Performance Monitoring

### Symbol-Level Metrics
- **Win Rate**: Per symbol performance
- **Profit Factor**: Per symbol profitability
- **Drawdown**: Per symbol risk exposure

### Portfolio-Level Metrics
- **Total Return**: Combined performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Correlation Matrix**: Inter-symbol relationships

## Testing

### Backtest (Multi-Symbol)
```bash
python live_trader_clean.py backtest --symbol XAUUSD.PRO,EURUSD.PRO,GBPUSD.PRO --from 2025-07-01 --to 2025-08-15 --regime --agent --config config/pro_config.json
```

### Paper Trading (Multi-Symbol)
```bash
python live_trader_clean.py --mode paper --config config/pro_config.json --agent --agent-mode guard --regime --log-level INFO
```

## Expected Behavior

### Signal Generation
- Each symbol generates signals independently
- No cross-symbol signal interference
- Regime detection per symbol

### Trade Execution
- Respects per-symbol limits
- Respects total portfolio limits
- Proper position sizing per symbol

### Risk Control
- Individual symbol risk gates
- Portfolio-level exposure control
- Session-aware trading

## Next Steps
1. Implement AutoML retraining (Phase 5)
2. Create executive report generator (Phase 6)
3. Run validation campaign (Phase 7)
4. Final testing and handoff (Phase 8)

---
**Status**: ✅ Portfolio Expansion Complete  
**Next**: AutoML Implementation (Phase 5)
