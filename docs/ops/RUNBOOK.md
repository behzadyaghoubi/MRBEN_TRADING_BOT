# MR BEN Trading System - Operational Runbook

## System Overview
Production-ready trading system with multi-symbol portfolio, AI supervision, and automated machine learning.

## Quick Start Commands

### 1. Start Live Trading
```bash
# Set password environment variable
$env:MT5_PASSWORD="your_password_here"

# Start live trading
python live_trader_clean.py --mode live --config config/pro_config.json --agent --regime --log-level INFO
```

### 2. Start Paper Trading
```bash
python live_trader_clean.py --mode paper --config config/pro_config.json --agent --agent-mode guard --regime --log-level INFO
```

### 3. Run Backtest
```bash
python live_trader_clean.py backtest --symbol XAUUSD.PRO,EURUSD.PRO,GBPUSD.PRO --from 2025-07-01 --to 2025-08-15 --regime --agent --config config/pro_config.json
```

### 4. Generate Executive Report
```bash
python scripts/generate_exec_report.py
```

## Emergency Procedures

### Kill-Switch (Immediate Stop)
```bash
# Create RUN_STOP file to stop system
echo "" > RUN_STOP

# Remove RUN_STOP file to resume
Remove-Item RUN_STOP
```

### Force Stop
```bash
# Find and kill Python processes
Get-Process python | Stop-Process -Force
```

### Restart System
```bash
# Stop current instance
Remove-Item RUN_STOP -ErrorAction SilentlyContinue
Get-Process python | Stop-Process -Force

# Wait 10 seconds
Start-Sleep -Seconds 10

# Restart
python live_trader_clean.py --mode live --config config/pro_config.json --agent --regime --log-level INFO
```

## Monitoring

### Prometheus Metrics
- **Endpoint**: http://127.0.0.1:9100/prom
- **Metrics**: Uptime, cycles, trades, errors, memory

### Grafana Dashboard
- **URL**: http://127.0.0.1:3000
- **Credentials**: admin/admin
- **Dashboard**: MR BEN Trading System

### System Logs
- **Main Log**: logs/trading_bot.log
- **AutoML Logs**: logs/automl_ml.log, logs/automl_lstm.log
- **Kill-Switch Log**: logs/killswitch.log

## Configuration Management

### Production Config
- **File**: config/pro_config.json
- **Key Settings**:
  - Portfolio symbols: XAUUSD.PRO, EURUSD.PRO, GBPUSD.PRO
  - Max open trades: 4 total
  - Risk per trade: 1%
  - Daily loss limit: 2%

### Environment Variables
- **MT5_PASSWORD**: MetaTrader5 account password
- **MT5_LOGIN**: MetaTrader5 account login (in config)
- **MT5_SERVER**: MetaTrader5 server (in config)

## AutoML Management

### Weekly Retraining
```bash
# Manual ML retraining
python -m src.ops.automl.retrain_ml

# Manual LSTM retraining
python -m src.ops.automl.retrain_lstm
```

### Model Registry
- **File**: models/registry.json
- **Purpose**: Track model versions and performance
- **Auto-update**: On successful retraining

### Performance Monitoring
- **AUC**: Target ≥ 0.80
- **F1**: Target ≥ 0.75
- **Calibration**: Target ≥ 0.90

## Risk Management

### Position Limits
- **Per Symbol**: 2 max open trades
- **Total Portfolio**: 4 max open trades
- **Risk Per Trade**: 1% of account balance
- **Daily Loss Limit**: 2% of account balance

### Risk Gates
- **Spread Gate**: Max spread 0.02 (2 pips)
- **Exposure Gate**: Portfolio-level position limits
- **Daily Loss Gate**: Stop trading if daily loss > 2%
- **Session Gate**: Trade only during London + New York

### Emergency Procedures
- **High Loss**: Automatic stop if daily loss > 2%
- **System Error**: Automatic halt on critical errors
- **Connection Loss**: Automatic retry with exponential backoff

## Troubleshooting

### Common Issues

#### 1. MT5 Connection Failed
```bash
# Check MT5 terminal status
# Verify login credentials
# Check server connectivity
# Restart MT5 terminal
```

#### 2. High Memory Usage
```bash
# Check memory usage
Get-Process python | Select-Object ProcessName, WorkingSet

# Restart if > 200MB
Get-Process python | Stop-Process -Force
# Wait and restart
```

#### 3. No Trading Signals
```bash
# Check market hours
# Verify data feeds
# Check strategy configuration
# Review logs for errors
```

#### 4. AutoML Failures
```bash
# Check dependencies
pip list | findstr "tensorflow sklearn xgboost lightgbm"

# Check logs
Get-Content logs/automl_ml.log -Tail 20
Get-Content logs/automl_lstm.log -Tail 20
```

### Debug Mode
```bash
# Enable debug logging
python live_trader_clean.py --mode live --config config/pro_config.json --log-level DEBUG
```

## Performance Monitoring

### Key Metrics
- **Win Rate**: Target ≥ 60%
- **Profit Factor**: Target ≥ 1.5
- **Sharpe Ratio**: Target ≥ 1.0
- **Max Drawdown**: Target ≤ 10%

### Daily Checks
1. **Morning**: Review overnight performance
2. **Midday**: Check system health
3. **Evening**: Review daily summary
4. **Weekly**: Generate executive report

### Alert Thresholds
- **Error Rate**: > 5% triggers alert
- **Memory Usage**: > 100MB triggers alert
- **Daily Loss**: > 1.5% triggers alert
- **No Signals**: > 2 hours triggers alert

## Backup and Recovery

### Configuration Backup
```bash
# Backup production config
Copy-Item config/pro_config.json config/backup/pro_config_$(Get-Date -Format "yyyyMMdd").json

# Backup model registry
Copy-Item models/registry.json models/backup/registry_$(Get-Date -Format "yyyyMMdd").json
```

### Model Backup
```bash
# Backup current models
Copy-Item models/ml_filter.pkl models/backup/
Copy-Item models/lstm_model.h5 models/backup/
```

### Recovery Procedures
```bash
# Restore from backup
Copy-Item config/backup/pro_config_YYYYMMDD.json config/pro_config.json
Copy-Item models/backup/ml_filter.pkl models/
Copy-Item models/backup/lstm_model.h5 models/
```

## Maintenance Schedule

### Daily
- [ ] Review system logs
- [ ] Check performance metrics
- [ ] Verify monitoring status
- [ ] Backup configuration

### Weekly
- [ ] AutoML retraining (Monday 3:00 AM UTC)
- [ ] Performance review
- [ ] Generate executive report
- [ ] Update documentation

### Monthly
- [ ] Full system audit
- [ ] Performance optimization
- [ ] Risk assessment review
- [ ] Backup verification

## Contact Information

### Trading Operations
- **Primary**: Trading Operations Team
- **Escalation**: System Administrator
- **Emergency**: 24/7 On-Call Engineer

### Support Channels
- **Email**: trading-ops@company.com
- **Phone**: +1-555-TRADING
- **Slack**: #trading-alerts

## Compliance and Reporting

### Regulatory Requirements
- **Trade Reporting**: All trades logged and reported
- **Risk Monitoring**: Continuous risk assessment
- **Audit Trail**: Complete transaction history
- **Compliance Checks**: Automated compliance validation

### Reporting Schedule
- **Daily**: Performance summary
- **Weekly**: Executive summary
- **Monthly**: Comprehensive report
- **Quarterly**: Regulatory compliance report

---
**Last Updated**: 2025-08-20  
**Version**: 1.0  
**Status**: Production Ready
