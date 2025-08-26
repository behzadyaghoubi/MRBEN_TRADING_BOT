# MR BEN Live Trading System - Runbook

## 🚀 One-Command Live Trading

The MR BEN Live Trading System is now a **one-command** system that automatically starts with:

- **Real LIVE trading** (not paper)
- **GPT-5 agent supervision** (default: guard mode)
- **Regime detection** enabled
- **Dashboard** running on port 8765
- **Production preflight checks** and safety rails

### Quick Start

```bash
# Single command to start everything
python live_trader_clean.py
```

**That's it!** The system automatically:
1. Loads configuration from `config.json`
2. Performs production preflight checks
3. Starts the dashboard
4. Initializes the GPT-5 agent in guard mode
5. Begins live trading with full supervision

## 🔧 Environment Overrides

### Agent Mode Override

```bash
# Override agent mode via environment variable
$env:AGENT_MODE="auto"
python live_trader_clean.py

# Available modes:
# - observe: Read-only monitoring
# - guard: Risk-blocking (default)
# - auto: Auto-remediation
```

### TensorFlow Optimization

```bash
# Optional: Disable oneDNN optimizations for better compatibility
$env:TF_ENABLE_ONEDNN_OPTS="0"
python live_trader_clean.py
```

## ⚠️ Safety Requirements

### Configuration Setup

Ensure `config.json` has:

```json
{
  "DEMO_MODE": false,
  "LOGIN": "your_mt5_login",
  "PASSWORD": "your_mt5_password", 
  "SERVER": "your_mt5_server",
  "SYMBOL": "XAUUSD.PRO",
  "dashboard": {
    "enabled": true,
    "port": 8765
  },
  "agent": {
    "mode": "guard"
  }
}
```

### MT5 Credentials

- **DEMO_MODE**: Must be `false` for live trading
- **LOGIN**: Your MT5 account number
- **PASSWORD**: Your MT5 password
- **SERVER**: Your MT5 broker server

## 📊 Dashboard & Monitoring

### Metrics Endpoint

```bash
# Get real-time system metrics
curl http://127.0.0.1:8765/metrics

# Or use PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/metrics"
```

### Dashboard Features

- **Real-time trading statistics**
- **System health metrics**
- **Agent status and actions**
- **Position information**
- **Performance data**

## 🛑 Kill Switch

### Emergency Stop

Create a file named `RUN_STOP` in the project directory:

```bash
# Windows PowerShell
New-Item -ItemType File -Name "RUN_STOP"

# Or manually create RUN_STOP file
```

The system will:
1. **Immediately halt** all trading
2. **Close open positions** (if configured)
3. **Log the stop event**
4. **Clean shutdown**

### Remove Kill Switch

```bash
# Remove the kill switch to resume
Remove-Item "RUN_STOP"
```

## 🔍 System Status

### Running Status

```bash
# Check if system is running
Get-Process python -ErrorAction SilentlyContinue

# Check logs
Get-Content logs\trading_bot.log -Tail 20
```

### Expected Logs

When running successfully, you should see:

```
[INFO] ✅ Production preflight checks passed
[INFO] Dashboard started at http://127.0.0.1:8765/metrics
[INFO] 🤖 Agent mode: guard
[INFO] ✅ Agent started in guard mode
[INFO] 🚀 Starting trading system...
[INFO] 🔄 Trading loop started
[INFO] 📊 Signal: 1 (confidence: 0.700, consecutive: 2/1)
[INFO] ✅ Trade execution approved
[INFO] 🚀 Executing trade...
[INFO] 📝 Paper trade: BUY 1.32 lots at 3336.25
[INFO] ✅ Trade executed successfully
```

## 🚨 Troubleshooting

### Common Issues

#### 1. MT5 Connection Failed

```
❌ Production preflight failed: MT5 credentials missing for LIVE mode
```

**Solution**: Check `config.json` has valid MT5 credentials and `DEMO_MODE: false`

#### 2. Symbol Not Available

```
❌ Production preflight failed: symbol_select failed: XAUUSD.PRO
```

**Solution**: 
- Verify symbol exists in your MT5 terminal
- Check symbol name spelling
- Ensure market is open

#### 3. Dashboard Not Starting

```
⚠️ Dashboard start failed: [error details]
```

**Solution**:
- Check port 8765 is not in use
- Verify firewall settings
- Check `config.json` dashboard section

#### 4. Agent Initialization Failed

```
❌ Failed to initialize agent components: [error details]
```

**Solution**:
- Check `src/agent/` directory exists
- Verify all agent dependencies are installed
- Check agent configuration in `config.json`

### Debug Mode

```bash
# Run with debug logging
python live_trader_clean.py live --mode live --symbol XAUUSD.PRO --agent --regime --log-level DEBUG
```

## 📋 Manual Commands

### Advanced Usage

```bash
# Manual mode selection
python live_trader_clean.py live --mode live --symbol XAUUSD.PRO --agent --regime --log-level INFO

# Paper trading mode
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --regime --log-level INFO

# Agent-only mode
python live_trader_clean.py agent --mode observe --symbol XAUUSD.PRO --log-level INFO
```

### Command Options

- `--mode`: `live` or `paper`
- `--symbol`: Trading symbol (default: XAUUSD.PRO)
- `--agent`: Enable GPT-5 supervision
- `--agent-mode`: `observe`, `guard`, or `auto`
- `--regime`: Enable regime detection
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## 🔒 Security & Compliance

### Risk Management

- **Spread Gate**: Automatically blocks trades when spread > 180 points
- **Exposure Gate**: Limits open positions to configured maximum
- **Agent Supervision**: Real-time risk monitoring and intervention
- **Kill Switch**: Immediate emergency stop capability

### Production Safety

- **Preflight Checks**: Validates all requirements before starting
- **Credential Verification**: Ensures MT5 access before live trading
- **Symbol Validation**: Confirms trading instrument availability
- **Error Storm Protection**: Automatic halt on excessive errors

## 📈 Performance Monitoring

### Key Metrics

- **Uptime**: System running time
- **Cycles/sec**: Trading loop frequency
- **Total Trades**: Number of executed trades
- **Error Rate**: System error frequency
- **Memory Usage**: System resource consumption

### Health Events

The system automatically reports health events to the agent:

- **STALE_DATA**: Insufficient market data
- **ORDER_FAIL**: MT5 order failures
- **ERROR_RATE**: High error frequency
- **MEMORY_HIGH**: Excessive memory usage

## 🎯 Next Steps

### Immediate Actions

1. **Verify Configuration**: Ensure `config.json` is properly configured
2. **Test Connection**: Verify MT5 terminal connectivity
3. **Start System**: Run `python live_trader_clean.py`
4. **Monitor Dashboard**: Check `http://127.0.0.1:8765/metrics`
5. **Review Logs**: Monitor `logs/trading_bot.log`

### Production Deployment

1. **Environment Setup**: Configure production MT5 credentials
2. **Monitoring**: Set up log aggregation and alerting
3. **Backup**: Implement configuration and data backup
4. **Documentation**: Document broker-specific settings

---

**MR BEN Live Trading System** - Ready for production with GPT-5 supervision! 🚀📈
