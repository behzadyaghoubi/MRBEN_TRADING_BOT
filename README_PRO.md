# MRBEN PRO Mode - Operational Runbook

## 🚀 PRO Mode Overview

MRBEN PRO is the official execution path for the MRBEN trading system, featuring:
- **Ensemble Strategy**: Combines Rule-based, Price Action (PA), Machine Learning (ML), and LSTM filters
- **A/B Testing**: Shadow testing with control (SMA-only) vs. pro (ensemble) tracks
- **Modular Architecture**: SystemIntegrator orchestrates all components
- **Conservative Risk Management**: 0.05% risk per trade, 1 max position

## ⚠️ IMPORTANT: Legacy Mode Deprecated

**DO NOT USE** these files for PRO mode:
- `live_trader_clean.py` - Legacy SMA-only mode
- `main_runner.py` - Old execution path
- Any direct MT5 order execution outside core modules

## 🔒 Safety Procedures

### Emergency Brake
- **File**: `halt.flag`
- **Effect**: Immediate system halt
- **Usage**: `New-Item -ItemType File -Force .\halt.flag`

### Before Live Trading
1. Remove `halt.flag`
2. Confirm live mode intention
3. Verify all safety gates are active

## 📋 Execution Paths

### Phase 1: Shadow A/B Testing (Paper Mode)
```powershell
# Start A/B testing in paper mode
.\activate_pro.ps1 -Mode paper

# Monitor for 10-15 minutes
.\phase1_snapshot.ps1

# Check metrics
curl -s http://127.0.0.1:8765/metrics | Select-String '^mrben_'
```

**Acceptance Criteria (10-15 minutes):**
- ✅ Logs contain [PA], [ML], [LSTM], [CONF], [VOTE]
- ✅ Metrics show track="control" and track="pro"
- ✅ No "legacy mode" or "SMA_Only" references
- ✅ Gates active: `mrben_blocks_total{reason="ml_low_conf"}`

### Phase 2: Canary Live (Conservative Risk)
```powershell
# Remove emergency brake
Remove-Item .\halt.flag -ErrorAction SilentlyContinue

# Start live trading
.\activate_pro.ps1 -Mode live
```

**Canary Parameters:**
- Risk per trade: 0.05%
- Exposure: 1 position maximum
- Daily loss: 0.8%
- Deviation: 200 points
- Spread gate: ≤120 points
- Session: Asia → A+ setups only

## 🕐 Trading Sessions (Vancouver Time)

| Session | Time | Risk Level | Notes |
|---------|------|------------|-------|
| **London** | 00:00–09:00 | Normal | Best performance |
| **New York** | 05:00–14:00 | Normal | Best performance |
| **Asia** | 16:00–23:00 | A+ Only | Restricted entry |

## 🔧 Configuration

### Core Settings (`mrben/config/config.yaml`)
- **Risk**: 0.05% per trade, 1 max position
- **Gates**: Spread ≤120pts, cooldown 180s
- **Confidence**: Dynamic enabled, threshold 0.68-0.90
- **Ensemble**: PA + ML + LSTM with weighted voting
- **Execution**: Adaptive filling with order_check

### Environment Variables (Optional Override)
```powershell
$env:MRBEN__CONFIDENCE__THRESHOLD__MIN="0.64"
$env:MRBEN__STRATEGY__ML_FILTER__MIN_PROBA="0.58"
$env:MRBEN__STRATEGY__LSTM_FILTER__AGREE_MIN="0.56"
```

## 📊 Monitoring

### Metrics Endpoint
- **URL**: http://127.0.0.1:8765/metrics
- **Key Metrics**:
  - `mrben_trades_opened_total{track="pro"}`
  - `mrben_decision_score{track="pro"}`
  - `mrben_blocks_total{reason="spread"}`
  - `mrben_drawdown_pct`

### Log Files
- **Main Log**: `.\logs\mrben.log`
- **Key Patterns**:
  - `[PA]` - Price Action analysis
  - `[ML]` - ML filter confidence
  - `[LSTM]` - LSTM prediction
  - `[CONF]` - Dynamic confidence
  - `[VOTE]` - Ensemble decision

## 🚨 Emergency Procedures

### Immediate Stop
```powershell
# Create emergency brake
New-Item -ItemType File -Force .\halt.flag

# Stop all processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Rollback
```powershell
# Git rollback
git checkout launch-v1

# Configuration rollback
git checkout HEAD -- mrben/config/config.yaml
```

## 🔍 Quality Improvements (Optional)

### Pullback Entry Filter
- Avoid chasing wave ends on SMA/Rule entries
- Implement in `core/deciders.py`

### Regime-aware SL/TP
- HIGH_VOL → Wider SL, conservative TP
- Implement in `core/position_management.py`

### Partial Close in Market
- Use Bid/Ask + IOC instead of fixed TP
- Implement in `core/order_management.py`

### Conformal Gate
- Apply final threshold to Decision
- Implement in `core/deciders.py`

## ✅ Acceptance Test Checklist

Before proceeding to live trading:

- [ ] `python mrben\main.py --help` shows PRO path
- [ ] `order_send(` only in core modules
- [ ] `config/config.yaml` loads without errors
- [ ] `/metrics` shows Prometheus format with `mrben_*` keys
- [ ] A/B in paper: Ensemble labels in logs + two tracks in metrics
- [ ] Canary live: Selective trades, no repeated errors, gates working

## 📚 Scripts

### `activate_pro.ps1`
- Safe PRO mode activation
- Emergency brake management
- Mode validation (paper/live)

### `phase1_snapshot.ps1`
- Comprehensive Phase 1 reporting
- A/B track verification
- Ensemble strategy validation
- Error monitoring

## 🎯 Daily Operations (Vancouver)

### Pre-Session Checklist
- [ ] `halt.flag` removed
- [ ] `/metrics` accessible on port 8765
- [ ] XAU spread ≤ 120 points
- [ ] All safety gates active

### During Session
- Monitor `mrben_blocks_total` (reasons: spread, exposure, ml_low_conf, pa_low_score)
- Track `mrben_trade_r` and `mrben_drawdown_pct`
- Watch for unusual patterns or errors

### Post-Session
- Review performance metrics
- Check error logs
- Update risk parameters if needed

## 🔄 Troubleshooting

### Common Issues
1. **Port 8765 in use**: Check for other MRBEN instances
2. **Configuration errors**: Validate YAML syntax
3. **Import errors**: Check Python dependencies
4. **MT5 errors**: Verify order_check implementation

### Recovery Steps
1. Stop all processes
2. Check error logs
3. Verify configuration
4. Restart in paper mode
5. Test thoroughly before live

---

**Remember**: Safety first, test thoroughly, monitor continuously. PRO mode is designed for reliability and conservative risk management.
