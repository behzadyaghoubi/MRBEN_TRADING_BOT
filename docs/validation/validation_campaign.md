# Validation Campaign - Phase 7

## Overview
Comprehensive validation of the production system through backtest, paper trading, and short live dry-runs.

## Validation Phases

### Phase 1: Backtest Validation
**Command**:
```bash
python live_trader_clean.py backtest --symbol XAUUSD.PRO,EURUSD.PRO,GBPUSD.PRO --from 2025-07-01 --to 2025-08-15 --regime --agent --config config/pro_config.json
```

**Acceptance Criteria**:
- ✅ No errors or crashes
- ✅ Multi-symbol portfolio execution
- ✅ DecisionCards contain new fields (symbol, regime, adj_conf)
- ✅ Metrics ≥ Baseline performance
- ✅ All strategy components active (rule, PA, ML, LSTM)

**Expected Output**:
- Backtest results for each symbol
- Portfolio-level metrics
- DecisionCard samples with new fields
- Performance comparison vs baseline

### Phase 2: Paper Trading Validation
**Command**:
```bash
python live_trader_clean.py --mode paper --config config/pro_config.json --agent --agent-mode guard --regime --log-level INFO
```

**Duration**: 24 hours minimum
**Acceptance Criteria**:
- ✅ No errors or crashes
- ✅ Multi-symbol signal generation
- ✅ Portfolio risk management active
- ✅ Monitoring endpoints accessible
- ✅ AutoML registry updates

**Validation Points**:
- Signal generation across all symbols
- Risk gate enforcement
- Portfolio exposure limits
- Session awareness
- Kill-switch functionality

### Phase 3: Short Live Dry-Run
**Command**:
```bash
python live_trader_clean.py --mode live --config config/pro_config.json --agent --regime --log-level INFO
```

**Duration**: 4 hours during active market
**Acceptance Criteria**:
- ✅ No errors or crashes
- ✅ MT5 connection stable
- ✅ Live signal generation
- ✅ Risk gates functioning
- ✅ Monitoring active

**Safety Measures**:
- Demo account only
- Small position sizes
- Active monitoring
- Kill-switch ready

## Validation Metrics

### Performance Metrics
- **Win Rate**: ≥ 60%
- **Profit Factor**: ≥ 1.5
- **Sharpe Ratio**: ≥ 1.0
- **Max Drawdown**: ≤ 10%

### System Metrics
- **Uptime**: 100% during test period
- **Error Rate**: ≤ 1%
- **Memory Usage**: ≤ 100MB
- **Response Time**: ≤ 5 seconds

### Portfolio Metrics
- **Symbol Coverage**: All 3 symbols active
- **Risk Distribution**: Even across symbols
- **Correlation**: < 0.9 between pairs
- **Position Limits**: Respecting max trades

## Validation Reports

### 1. Backtest Report
- **File**: `docs/validation/backtest_results.md`
- **Content**: Performance metrics, trade analysis, DecisionCard samples
- **Status**: Generated after Phase 1

### 2. Paper Trading Report
- **File**: `docs/validation/paper_trading_results.md`
- **Content**: Live signal analysis, risk management, system health
- **Status**: Generated after Phase 2

### 3. Live Dry-Run Report
- **File**: `docs/validation/live_dryrun_results.md`
- **Content**: Live execution, MT5 integration, monitoring status
- **Status**: Generated after Phase 3

### 4. Validation Summary
- **File**: `docs/validation/validation_summary.md`
- **Content**: Overall validation status, issues found, recommendations
- **Status**: Generated after all phases

## Test Data Requirements

### Historical Data
- **XAUUSD.PRO**: 15M bars, 600+ bars
- **EURUSD.PRO**: 15M bars, 600+ bars
- **GBPUSD.PRO**: 15M bars, 600+ bars
- **Period**: July 1 - August 15, 2025

### Live Data
- **Real-time feeds** from MT5
- **Market hours**: London + New York sessions
- **Spread monitoring**: Real spread data
- **Volume data**: For liquidity assessment

## Validation Checklist

### Pre-Validation
- [ ] All dependencies installed
- [ ] Configuration files validated
- [ ] Test environment ready
- [ ] Monitoring tools active
- [ ] Kill-switch tested

### During Validation
- [ ] Monitor system logs
- [ ] Track performance metrics
- [ ] Verify risk management
- [ ] Check monitoring endpoints
- [ ] Document any issues

### Post-Validation
- [ ] Generate validation reports
- [ ] Analyze performance data
- [ ] Identify improvement areas
- [ ] Update documentation
- [ ] Prepare handoff materials

## Expected Outcomes

### Success Criteria
- All validation phases pass
- Performance meets or exceeds baseline
- System stability confirmed
- Risk management functioning
- Monitoring operational

### Failure Scenarios
- System crashes or errors
- Performance below baseline
- Risk management failures
- Monitoring not working
- Integration issues

## Next Steps
1. Execute validation phases sequentially
2. Generate comprehensive reports
3. Address any issues found
4. Final testing and handoff (Phase 8)
5. Production deployment

---
**Status**: Ready for Execution
**Next**: Run Validation Campaign
