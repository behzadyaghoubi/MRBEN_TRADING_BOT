# MR BEN Agent Configuration & Usage Guide

## 🎯 **Overview**

The MR BEN GPT-5 Supervision & Auto-Remediation system provides continuous monitoring, risk management, and automated problem resolution for the trading system.

## 🚀 **Operating Modes**

### **1. Observe Mode** (`--agent-mode observe`)
- **Purpose**: Read-only monitoring and logging
- **Actions**: Logs decisions, health events, and system status
- **No Intervention**: Does not block trades or execute playbooks
- **Use Case**: System monitoring, debugging, performance analysis

### **2. Guard Mode** (`--agent-mode guard`) - **DEFAULT**
- **Purpose**: Risk management with manual confirmation
- **Actions**: 
  - Blocks risky trades via RiskGate
  - Proposes fixes and playbooks
  - Requires human confirmation for high-impact actions
- **Use Case**: Production trading with safety controls

### **3. Auto Mode** (`--agent-mode auto`)
- **Purpose**: Fully automated supervision and remediation
- **Actions**:
  - Executes approved playbooks automatically
  - Restarts MT5, cleans memory, adjusts thresholds
  - Operates within strict safety boundaries
- **Use Case**: 24/7 automated trading with self-healing

### **4. Panic Mode** (Manual or Automatic)
- **Purpose**: Emergency shutdown
- **Triggers**: 
  - Manual activation
  - MAX_DAILY_LOSS breached
  - Multiple order failures
  - Critical system errors
- **Actions**: Immediate position closure and system halt

## 🔧 **Configuration**

### **Agent Configuration in config.json**
```json
{
  "agent": {
    "enabled": true,
    "mode": "guard",
    "auto_playbooks": ["MEM_CLEANUP", "RESTART_MT5"],
    "alert": {
      "sink": "telegram",
      "token": "",
      "chat_id": ""
    },
    "error_rate_halt": 5,
    "memory_mb_halt": 2000
  }
}
```

### **Configuration Parameters**
- **enabled**: Enable/disable agent system
- **mode**: Default agent mode (observe/guard/auto)
- **auto_playbooks**: List of playbooks that can run automatically
- **alert**: Alert system configuration
- **error_rate_halt**: Maximum errors per minute before halt
- **memory_mb_halt**: Memory threshold for automatic halt

## 🎮 **Usage Examples**

### **Basic Usage**
```bash
# Observe mode (read-only)
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode observe

# Guard mode (default - risk management)
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode guard

# Auto mode (fully automated)
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode auto
```

### **Environment Variable Override**
```bash
# Override config with environment variable
$env:AGENT_MODE = "auto"
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent
```

### **Agent-Only Mode**
```bash
# Run agent without trading core
python live_trader_clean.py agent --mode observe --symbol XAUUSD.PRO --agent-mode guard
```

## 🛡️ **Risk Gates**

### **1. Spread Gate**
- **Function**: Monitors spread vs. configured limits
- **Dynamic**: Adjusts based on ATR if enabled
- **Action**: Blocks trades if spread > max_spread_points

### **2. Regime Gate**
- **Function**: Regime-based confidence thresholds
- **Modes**: TRENDING, RANGING, VOLATILE, UNKNOWN
- **Action**: Adjusts confidence requirements per regime

### **3. Exposure Gate**
- **Function**: Position and loss limits
- **Checks**: 
  - Open positions < max_open_trades
  - Daily loss < max_daily_loss
- **Action**: Blocks new trades if limits exceeded

### **4. Frequency Gate**
- **Function**: Trading frequency limits
- **Check**: Daily trades < max_trades_per_day
- **Action**: Prevents overtrading

## 🔄 **Playbooks (Auto-Remediation)**

### **1. Restart MT5**
```python
def playbook_restart_mt5(cfg, symbol):
    # Shutdown MT5
    # Reinitialize connection
    # Reselect symbol
    # Verify connectivity
```

### **2. Memory Cleanup**
```python
def playbook_mem_cleanup(memory_monitor):
    # Force garbage collection
    # Monitor memory usage
    # Report cleanup results
```

### **3. Adjust Thresholds**
```python
def playbook_adjust_threshold(state, factor=1.2, cap=0.9):
    # Increase confidence threshold
    # Apply safety caps
    # Log adjustment
```

### **4. Adjust Risk Parameters**
```python
def playbook_adjust_risk(params):
    # Modify risk settings
    # Apply safety limits
    # Update system state
```

## 📊 **Data Contracts**

### **DecisionCard Schema**
```json
{
  "ts": "2025-08-17T17:30:40.911",
  "symbol": "XAUUSD.PRO",
  "cycle": 123,
  "price": 2398.12,
  "sma20": 2397.44,
  "sma50": 2395.90,
  "raw_conf": 0.70,
  "adj_conf": 0.66,
  "threshold": 0.50,
  "allow_trade": true,
  "regime_label": "NORMAL",
  "regime_scores": {"volatility": 0.009},
  "spread_pts": 180,
  "consecutive": 2,
  "open_positions": 1,
  "risk": {"base_risk": 0.01, "max_open_trades": 2},
  "signal_src": "SMA|AI|Ensemble",
  "mode": "live|paper",
  "agent_mode": "observe|guard|auto"
}
```

### **HealthEvent Schema**
```json
{
  "ts": "2025-08-17T17:30:40.911",
  "severity": "INFO|WARN|ERROR|CRITICAL",
  "kind": "EXCEPTION|STALE_DATA|SPREAD_SPIKE|NO_SYMBOL|MT5_DISCONNECT|MEMORY_HIGH|ORDER_FAIL",
  "message": "Description of the event",
  "context": {"trace": "...", "symbol": "XAUUSD.PRO", "last_bar": "2025-08-17T17:30:00"}
}
```

### **AgentAction Schema**
```json
{
  "action": "NONE|HALT|RESUME|ADJUST_THRESHOLD|ADJUST_RISK|RESTART_MT5|MEM_CLEANUP|RELOAD_CONFIG|ROTATE_LOGS",
  "params": {"threshold": 0.55, "base_risk": 0.007},
  "reason": "Why this action was taken",
  "severity": "INFO|WARN|ERROR"
}
```

## 🚨 **Safety Boundaries**

### **Auto Mode Restrictions**
- **No Permanent Changes**: Cannot modify config files
- **Risk Increases**: Cannot lower thresholds or raise risk automatically
- **Confirmation Required**: HALT actions require human approval or tripwire activation

### **Tripwire Rules**
- **MAX_DAILY_LOSS**: Automatic halt if daily loss limit exceeded
- **ORDER_FAIL x3**: Halt after 3 consecutive order failures
- **MEMORY_HIGH**: Automatic memory cleanup if threshold exceeded

### **Emergency Procedures**
- **Panic Mode**: Immediate position closure and system halt
- **Manual Override**: Human operator can force system shutdown
- **Recovery**: System restart with last known good configuration

## 📈 **Monitoring & Alerts**

### **Event Persistence**
- **File**: `data/events.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Types**: Decision events, health events, agent actions

### **Alert System**
- **Telegram**: Webhook-based notifications
- **Slack**: Channel integration
- **Email**: SMTP-based alerts
- **Logging**: Structured logging with severity levels

### **Performance Metrics**
- **Uptime**: System availability tracking
- **Error Rate**: Error frequency monitoring
- **Memory Usage**: Resource consumption tracking
- **Trade Success**: Execution success rate

## 🧪 **Testing**

### **Test Scenarios**

#### **1. Observe Mode Test**
```bash
# Inject synthetic health event
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode observe
# Verify: Events logged, no actions taken
```

#### **2. Guard Mode Test**
```bash
# Test spread spike detection
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode guard
# Verify: Risk gates active, trade blocking works
```

#### **3. Auto Mode Test**
```bash
# Force MT5 disconnect
python live_trader_clean.py live --mode paper --symbol XAUUSD.PRO --agent --agent-mode auto
# Verify: RESTART_MT5 playbook executes automatically
```

### **Validation Checklist**
- [ ] Agent starts in specified mode
- [ ] Risk gates function correctly
- [ ] Health events are captured
- [ ] Playbooks execute as expected
- [ ] Safety boundaries are respected
- [ ] Events are persisted correctly
- [ ] Alerts are sent for critical events

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Agent Not Starting**
- Check `AGENT_AVAILABLE` flag
- Verify agent imports
- Check configuration syntax

#### **Playbooks Not Executing**
- Verify agent mode is "auto"
- Check playbook permissions
- Review safety boundaries

#### **Risk Gates Too Strict**
- Adjust threshold parameters
- Review regime configurations
- Check spread limits

### **Debug Commands**
```bash
# Check agent status
python -c "from src.agent import maybe_start_agent; print('Agent available')"

# Test risk gates
python -c "from src.agent.risk_gate import AdvancedRiskGate; print('Risk gates available')"

# Verify configuration
python -c "import json; cfg=json.load(open('config.json')); print(cfg.get('agent', {}))"
```

## 📚 **Next Steps**

### **Phase 1: Basic Integration** ✅
- [x] Agent bridge implementation
- [x] Risk gate system
- [x] Basic playbooks
- [x] CLI integration

### **Phase 2: Enhanced Features**
- [ ] Advanced playbooks
- [ ] Machine learning integration
- [ ] Predictive maintenance
- [ ] Performance optimization

### **Phase 3: Production Features**
- [ ] Advanced alerting
- [ ] Dashboard integration
- [ ] Multi-system coordination
- [ ] Compliance reporting

## 📞 **Support**

For questions or issues with the agent system:
1. Check the logs in `logs/trading_bot.log`
2. Review event history in `data/events.jsonl`
3. Verify configuration in `config.json`
4. Test with observe mode first
5. Enable DEBUG logging for detailed information

---

**MR BEN Agent System** - Intelligent Trading Supervision & Auto-Remediation 🚀🤖
