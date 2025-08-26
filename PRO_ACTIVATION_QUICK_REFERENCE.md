# MRBEN PRO Mode - Quick Activation Reference

## 🚀 **Step-by-Step Activation**

### **1. Start Shadow A/B Testing (Paper Mode)**
```bash
# Option A: With A/B flag (if supported)
python mrben\main.py start --mode=paper --symbol XAUUSD.PRO --track=pro --ab=on

# Option B: Without A/B flag (if not supported)
python mrben\main.py start --mode=paper --symbol XAUUSD.PRO --track=pro
```

### **2. Monitor for 10-15 Minutes**
Look for these indicators:

#### **✅ What You Should See:**
- **Metrics**: `track="control"` and `track="pro"` labels
- **Logs**: No "legacy mode" or "SMA_Only" messages
- **Decisions**: [PA], [ML], [LSTM], [CONF], [VOTE] logs
- **Blocks**: `mrben_blocks_total{reason="ml_low_conf"}` metrics

#### **❌ What You Should NOT See:**
- "legacy mode" logs
- "SMA_Only" decision logs
- MT5 errors (10030, 10018)

### **3. Go Live (After Verification)**
```bash
# Remove emergency brake
rm halt.flag

# Start live trading
python mrben\main.py start --mode=live --symbol XAUUSD.PRO --track=pro
```

---

## 📊 **Monitoring Commands**

### **Real-time Metrics**
```bash
# PowerShell metrics monitor
.\watch_metrics.ps1

# Health watchdog (auto-halt if needed)
python health_watchdog.py

# Manual metrics check
curl -s http://127.0.0.1:8765/metrics | Select-String '^mrben_'
```

### **System Status**
```bash
# Check system status
python mrben/main.py status

# Check system health
python mrben/main.py health

# Check configuration
python mrben/app.py --config mrben/config/config.yaml --dry-run
```

---

## 🚨 **Emergency Procedures**

### **Immediate Stop**
```bash
echo "" > halt.flag
```

### **Check What Happened**
```bash
# Check logs for errors
Get-Content logs/mrben.log | Select-String "ERROR|WARNING"

# Check metrics for anomalies
curl -s http://127.0.0.1:8765/metrics | Select-String "mrben_blocks_total"
```

### **Recovery**
```bash
# Remove emergency stop
rm halt.flag

# Restart in paper mode
python mrben/main.py start --mode=paper --symbol XAUUSD.PRO --track=pro
```

---

## ⚙️ **Configuration Verification**

### **Environment Variables (Should Be Set)**
```bash
$env:MRBEN__RISK__BASE_R_PCT = "0.05"           # 0.05% risk per trade
$env:MRBEN__GATES__EXPOSURE_MAX_POSITIONS = "1"  # Max 1 position
$env:MRBEN__GATES__DAILY_LOSS_PCT = "0.8"       # 0.8% daily loss limit
$env:MRBEN__CONFIDENCE__THRESHOLD__MIN = "0.62" # 62% min confidence
```

### **Strategy Features (Should Be Enabled)**
- ✅ Price Action: Engulf, Pin, Inside, Sweep patterns
- ✅ ML Filter: ONNX model with 0.58 min probability
- ✅ LSTM: Time series prediction with 0.55 min agreement
- ✅ Dynamic Confidence: Regime + Session + Drawdown adjustments

---

## 🎯 **Acceptance Criteria for Go-Live**

1. ✅ **Dual Track Operation**: Both control and pro tracks active
2. ✅ **No Legacy Mode**: No "SMA_Only" or legacy logs
3. ✅ **Ensemble Decisions**: PA, ML, LSTM all contributing
4. ✅ **Risk Gates**: All 7 gates functioning
5. ✅ **Metrics**: Real-time data collection working
6. ✅ **No MT5 Errors**: No 10030/10018 errors

---

## 📋 **File Locations**

| Purpose | File Path |
|---------|-----------|
| **Main System** | `mrben/main.py` |
| **Configuration** | `mrben/config/config.yaml` |
| **Metrics Monitor** | `watch_metrics.ps1` |
| **Health Watchdog** | `health_watchdog.py` |
| **Canary Activation** | `activate_canary.ps1` |
| **Emergency Stop** | `halt.flag` |

---

## 🔒 **Safety Features Active**

- ✅ `halt.flag` emergency brake
- ✅ Comprehensive risk gates
- ✅ Real-time monitoring
- ✅ Automatic health watchdog
- ✅ Conservative risk parameters

---

**Status**: ✅ **READY FOR PRO MODE ACTIVATION**  
**Next Action**: Start shadow A/B testing in paper mode  
**Timeline**: 10-15 minutes verification, then go live  
**Risk Level**: LOW (conservative settings + emergency brake)
