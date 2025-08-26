# MR BEN Live Trading Troubleshooting Matrix

**Timestamp**: 2025-08-18
**Phase**: Live Trading Operations
**Status**: Active

## Troubleshooting Matrix

| Symptom | Likely Cause | Fix | Evidence |
|---------|--------------|-----|----------|
| No /metrics response | Dashboard port in use / not started | Change port in config / check logs | Screenshot + log lines |
| order_send non-DONE | Trade context / volume / freeze level | Log request; retry policy; adjust deviation | Retcode + request dump |
| Spread always high | Off-market hours / wrong symbol | Validate broker symbol; check market sessions | DecisionCard snapshots |
| Agent components fail | Import errors / missing dependencies | Check src/agent imports; verify requirements | Import traceback |
| MT5 connection fails | Terminal not running / wrong credentials | Verify MT5 terminal; check login/password | Connection error logs |
| Paper trades not executing | Consecutive signal requirement | Check consecutive_signals_required config | Signal generation logs |
| Live trades blocked | DEMO_MODE still true | Set flags.demo_mode = false in config.json | Configuration logs |
| Dashboard not accessible | Port conflict / firewall | Change dashboard.port in config | Port binding errors |
| Agent supervision inactive | Agent not started / mode issues | Verify --agent flag; check agent.mode | Agent initialization logs |

## Common Issues & Solutions

### 1. Dashboard Access Issues
**Problem**: Cannot access http://127.0.0.1:8765/metrics
**Solutions**:
- Check if port 8765 is available
- Verify dashboard.enabled = true in config
- Check firewall settings
- Change port in config if needed

### 2. MT5 Connection Problems
**Problem**: MetaTrader5 connection fails
**Solutions**:
- Ensure MT5 terminal is running
- Verify login credentials in config.json
- Check server name and account type
- Restart MT5 terminal if needed

### 3. Agent Initialization Failures
**Problem**: Agent components fail to initialize
**Solutions**:
- Check src/agent module availability
- Verify Python dependencies
- Check agent configuration in config.json
- Review agent initialization logs

### 4. Trade Execution Blocks
**Problem**: Trades not executing despite signals
**Solutions**:
- Check consecutive_signals_required setting
- Verify spread is below max_spread_points
- Check max_open_trades limit
- Review risk gate configurations

### 5. Configuration Issues
**Problem**: System not using expected settings
**Solutions**:
- Verify config.json syntax
- Check for missing required fields
- Restart system after config changes
- Review configuration loading logs

## Debug Commands

### System Status
```bash
# Check Python processes
Get-Process python

# Check MT5 connection
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

# Check dashboard
curl http://127.0.0.1:8765/metrics
```

### Log Analysis
```bash
# Monitor logs in real-time
Get-Content logs/trading_bot.log -Wait

# Check recent errors
Get-Content logs/trading_bot.log | Select-String "ERROR"
```

## Recovery Procedures

### 1. System Restart
```bash
# Stop all Python processes
Get-Process python | Stop-Process -Force

# Restart system
python live_trader_clean.py
```

### 2. Configuration Reset
```bash
# Backup current config
Copy-Item config.json config_backup.json

# Restore working config
Copy-Item config_backup.json config.json
```

### 3. Emergency Stop
```bash
# Create kill-switch
echo "" > RUN_STOP

# Verify system halted
Get-Process python
```

## Performance Monitoring

### Key Metrics to Watch
- **Uptime**: System stability
- **Cycles/sec**: Processing speed
- **Response Time**: System responsiveness
- **Error Rate**: System health
- **Memory Usage**: Resource consumption
- **Trade Success Rate**: Execution quality

### Alert Thresholds
- Error Rate > 0.1 (10%)
- Memory Usage > 1000 MB
- Response Time > 2.0 seconds
- Consecutive Failures > 3
