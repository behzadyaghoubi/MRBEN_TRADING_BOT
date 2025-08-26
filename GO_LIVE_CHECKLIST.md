# MRBEN System - Go-Live Checklist
*Generated: $(Get-Date)*

## Pre-Flight Checklist

### ✅ **Phase 1 Verification Complete**
- [ ] **A/B Tracks Working**: Both `track="control"` and `track="pro"` visible in metrics
- [ ] **Ensemble Labels Confirmed**: [PA], [ML], [LSTM], [CONF], [VOTE] appearing in logs
- [ ] **Metrics Endpoint Healthy**: Port 8765 accessible with Prometheus metrics
- [ ] **No Legacy Mode**: No `legacy` or `SMA_Only` references in logs
- [ ] **No MT5 Errors**: No 10030, 10018, or other trading errors
- [ ] **Paper Mode Success**: Shadow A/B testing completed successfully

### ✅ **System Configuration Verified**
- [ ] **PRO Mode Active**: `mrben/main.py` running with `SystemIntegrator`
- [ ] **Risk Settings**: `base_r_pct=0.05`, `exposure_max_positions=1`, `daily_loss_pct=0.8`
- [ ] **Strategy Components**: All ensemble components enabled and working
- [ ] **Emergency Systems**: Health watchdog and monitoring scripts ready
- [ ] **Broker Connection**: MT5 connection stable and symbol info accessible

## Launch Commands

### **1. Remove Emergency Brake**
```bash
Remove-Item 'halt.flag'
```

### **2. Set Environment Variables**
```bash
$env:MRBEN__RISK__BASE_R_PCT="0.05"
$env:MRBEN__GATES__EXPOSURE_MAX_POSITIONS="1"
$env:MRBEN__GATES__DAILY_LOSS_PCT="0.8"
$env:MRBEN__CONFIDENCE__THRESHOLD__MIN="0.62"
```

### **3. Start Canary Live Trading**
```bash
python mrben\main.py start --mode=live --symbol XAUUSD.PRO --track=pro --ab=on
```

### **4. Verify Launch**
```bash
python mrben\main.py status
curl -s http://127.0.0.1:8765/metrics | findstr "mrben_trades_opened_total"
```

## Initial Monitoring (First 15 Minutes)

### **Performance Metrics**
- [ ] **Slippage p95**: ≤ 20 points (XAUUSD.PRO)
- [ ] **Latency p95**: ≤ 200 ms (order execution)
- [ ] **Drawdown**: < 1.5% (portfolio)
- [ ] **Spread Impact**: ≤ 180 points (risk gate compliance)

### **System Health**
- [ ] **Metrics Generation**: New metrics appearing every 30 seconds
- [ ] **Log Generation**: Ensemble decisions being logged
- [ ] **A/B Tracks**: Both control and pro tracks active
- [ ] **Risk Gates**: Position sizing and loss limits working

### **Trading Activity**
- [ ] **Signal Generation**: Ensemble decisions being made
- [ ] **Order Execution**: MT5 orders being sent successfully
- [ ] **Position Management**: Risk limits being enforced
- [ ] **Error Handling**: No critical errors in logs

## Post-Launch Monitoring

### **Real-Time Monitoring**
```bash
# Watch metrics
.\watch_metrics.ps1

# Monitor logs
Get-Content .\logs\mrben.log -Tail 100 -Wait

# Health check
python health_watchdog.py
```

### **Key Metrics to Track**
- `mrben_trades_opened_total{track="pro"}`
- `mrben_blocks_total{reason="ml_low_conf"}`
- `mrben_decision_score`
- `mrben_confidence_dyn`
- `mrben_drawdown_pct`

### **Alert Thresholds**
- **Drawdown**: Alert if > 1.0%, Stop if > 1.5%
- **Slippage**: Alert if p95 > 15 points, Stop if > 25 points
- **Latency**: Alert if p95 > 150 ms, Stop if > 250 ms
- **Error Rate**: Alert if > 1%, Stop if > 3%

## Emergency Procedures

### **Immediate Stop**
```bash
# Create emergency brake
New-Item -ItemType File -Name "halt.flag" -Force

# Check system status
python mrben\main.py status
```

### **Investigation Commands**
```bash
# Check recent logs
Get-Content .\logs\mrben.log -Tail 100 | Select-String 'error|ERROR|10030|10018'

# Check metrics
curl -s http://127.0.0.1:8765/metrics | findstr "mrben_"

# Check positions
python mrben\main.py status
```

## Success Criteria

### **Phase 2 Success (First Hour)**
- ✅ **No Critical Errors**: All trades executed successfully
- ✅ **Risk Compliance**: Position sizes and loss limits respected
- ✅ **Performance Acceptable**: Slippage ≤ 20 pts, Latency ≤ 200 ms
- ✅ **A/B Testing Working**: Both tracks generating metrics
- ✅ **Ensemble Decisions**: [PA], [ML], [LSTM] labels in logs

### **Ready for Scale-Up**
- ✅ **Stable Performance**: 1+ hour without issues
- ✅ **Risk Management**: All gates working correctly
- ✅ **Monitoring**: Metrics and logs healthy
- ✅ **Documentation**: LAUNCH_DAY report generated

## Post-Trade Actions

### **Generate Launch Report**
```bash
# Create launch day summary
$report = @"
LAUNCH_DAY_REPORT
Date: $(Get-Date)
Mode: Canary Live
Risk: 0.05% base, Exposure=1, Daily Loss=0.8%
Performance: [To be filled]
Issues: [To be filled]
Next Steps: [To be filled]
"@

$report | Out-File -FilePath "LAUNCH_DAY_REPORT.txt"
```

### **Review and Adjust**
- [ ] **Performance Analysis**: Review slippage, latency, drawdown
- [ ] **Risk Assessment**: Verify risk gates effectiveness
- [ ] **Strategy Tuning**: Adjust confidence thresholds if needed
- [ ] **Documentation**: Update procedures based on learnings

---

**Status**: 🟡 **READY FOR PHASE 1 VERIFICATION**
**Next Step**: Complete Phase 1 (Shadow A/B Testing)
**Go-Live**: After Phase 1 success, remove halt.flag and start live trading
**Risk Level**: Conservative (0.05% risk, max 1 position)
