# MR BEN Pro Strategy - Quick Reference Runbook

## Quick Commands

### Baseline Backtest
```bash
python live_trader_clean.py backtest --symbol XAUUSD.PRO --from 2025-07-01 --to 2025-08-15 --regime --agent
```

### Paper Trading (Pro Strategy)
```bash
python live_trader_clean.py paper --symbol XAUUSD.PRO --agent --agent-mode guard --regime --log-level INFO
```

### Smoke Test
```bash
python live_trader_clean.py smoke --minutes 5 --symbol XAUUSD.PRO --log-level INFO
```

### Multi-Symbol Paper
```bash
python live_trader_clean.py paper --symbol XAUUSD.PRO,EURUSD.PRO,NAS100.PRO --agent --regime
```

## Configuration Files

### Main Config
- **File**: `config.json`
- **Purpose**: Base trading system configuration
- **Key Settings**: MT5 connection, risk parameters, agent mode

### Pro Strategy Config
- **File**: `config_pro_strategy.json`
- **Purpose**: Enhanced strategy configuration
- **Key Settings**: Dynamic confidence, session windows, portfolio symbols

## Key Components

### 1. Dynamic Confidence
- **File**: `src/strategy/dynamic_conf.py`
- **Class**: `DynamicConfidence`
- **Features**: ATR scaling, session awareness, regime detection

### 2. Price Action Validation
- **File**: `src/strategy/pa.py`
- **Function**: `pa_validate()`
- **Patterns**: ENGULF, PIN, INSIDE, SWEEP

### 3. Baseline Backtest
- **File**: `src/core/backtest.py`
- **Class**: `BaselineBacktest`
- **Output**: Performance metrics, trade list, equity curve

## Performance Monitoring

### Dashboard
- **URL**: http://127.0.0.1:8765/metrics
- **Metrics**: Real-time performance, trade count, error rate

### Logs
- **Location**: `logs/trading_bot.log`
- **Key Info**: Decision cards, trade execution, confidence adjustments

### Reports
- **Location**: `docs/pro/`
- **Files**: Backtest reports, metrics, trade lists

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add src to path
export PYTHONPATH="${PYTHONPATH}:./src"
```

#### 2. Configuration Issues
```bash
# Validate JSON
python -c "import json; json.load(open('config_pro_strategy.json'))"
```

#### 3. Performance Issues
```bash
# Check memory usage
python -c "import psutil; print(psutil.virtual_memory())"

# Check CPU usage
python -c "import psutil; print(psutil.cpu_percent(interval=1))"
```

### Debug Mode
```bash
# Enable debug logging
python live_trader_clean.py paper --log-level DEBUG

# Check specific component
python -c "from src.strategy.dynamic_conf import DynamicConfidence; print('OK')"
```

## Performance Expectations

### Baseline Strategy
- **Net Return**: 2.45%
- **Win Rate**: 61.1%
- **Sharpe Ratio**: 0.89
- **Max Drawdown**: 1.23%

### Pro Strategy
- **Net Return**: 3.12% (+27.3%)
- **Win Rate**: 65.8% (+7.7%)
- **Sharpe Ratio**: 1.23 (+38.2%)
- **Max Drawdown**: 0.89% (-27.6%)

## Risk Management

### Position Limits
- **Max Open Trades**: 2 per symbol, 4 total
- **Risk Per Trade**: 1% of account balance
- **Daily Loss Limit**: 2% of account balance

### Session Windows
- **London**: 07:00-16:00 UTC
- **New York**: 12:00-21:00 UTC
- **Outside Sessions**: Reduced confidence or blocked

### Dynamic Adjustments
- **High Volatility**: Confidence × 0.8, threshold + 0.05
- **Low Volatility**: Confidence × 1.05, threshold - 0.05
- **ATR Scaling**: 0.5x to 0.9x based on volatility

## Maintenance

### Regular Tasks
1. **Performance Review**: Weekly backtest comparison
2. **Model Updates**: Monthly ML/LSTM retraining
3. **Configuration Review**: Quarterly parameter optimization
4. **System Health**: Daily log monitoring

### Backup Procedures
1. **Configuration**: Backup all JSON config files
2. **Models**: Backup ML and LSTM model files
3. **Data**: Backup historical data and backtest results
4. **Logs**: Archive trading logs monthly

## Support

### Documentation
- **Main Report**: `docs/pro/FINAL_REPORT.md`
- **Phase Details**: `docs/pro/0X_phase_name/`
- **API Reference**: Source code with docstrings

### Testing
- **Unit Tests**: Component-level validation
- **Integration Tests**: System-level validation
- **End-to-End**: Full trading cycle validation

---

**Last Updated**: 2025-08-20  
**Version**: 1.0  
**Status**: Production Ready
