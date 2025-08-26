# MR BEN Live Trading System - Clean Version

## 🎯 **Overview**

This is a comprehensive, production-ready live trading system for MetaTrader 5 with AI integration, advanced risk management, and professional-grade features.

## ✅ **Recent Fixes Applied**

### 1. **Import Issues Resolved**
- Fixed `tensorflow.keras.models` import resolution for Pyright
- Added proper type hints and TYPE_CHECKING imports
- Improved error handling for missing dependencies
- Added fallback functions when AI libraries are unavailable

### 2. **Code Structure Improvements**
- Removed duplicate `_update_trailing_stops()` method
- Added missing `_initialize_agent_components()` method
- Fixed agent component initialization flow
- Improved error handling and logging

### 3. **Type Safety Enhancements**
- Added comprehensive type hints
- Used TYPE_CHECKING for conditional imports
- Proper type annotations for all functions
- Pyright-compliant code structure

## 🚀 **Features**

### **Core Trading System**
- **MT5 Integration**: Professional MetaTrader 5 connectivity
- **AI-Powered Signals**: LSTM models, ML filters, ensemble methods
- **Risk Management**: Conformal gates, regime detection, spread controls
- **TP Split Strategy**: Partial closes, breakeven stops, full exits
- **Performance Monitoring**: Real-time metrics, memory management

### **Advanced Features**
- **Conformal Prediction**: Statistical trade validation
- **Regime Detection**: Market condition adaptation
- **Dynamic Thresholds**: Adaptive confidence levels
- **Spread Management**: Intelligent spread filtering
- **Event Logging**: Comprehensive trade tracking

### **Safety Features**
- **Production Preflight**: Safety checks before live trading
- **Error Storm Detection**: Automatic system protection
- **Memory Management**: Automatic cleanup and monitoring
- **Health Monitoring**: Real-time system health checks

## 📦 **Installation**

### **Quick Install**
```bash
# Install dependencies
python install_dependencies.py

# Or manually install
pip install -r requirements.txt
```

### **Required Dependencies**
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- tensorflow >= 2.10.0
- scikit-learn >= 1.1.0
- joblib >= 1.2.0
- MetaTrader5 >= 5.0.0

## ⚙️ **Configuration**

### **Basic Configuration**
Create `config.json` with your trading parameters:

```json
{
  "credentials": {
    "login": "YOUR_MT5_LOGIN",
    "password": "YOUR_MT5_PASSWORD",
    "server": "YOUR_MT5_SERVER"
  },
  "flags": {
    "demo_mode": true
  },
  "trading": {
    "symbol": "XAUUSD.PRO",
    "timeframe": 15,
    "magic_number": 20250721
  }
}
```

### **Advanced Configuration**
- **Risk Management**: Position sizing, loss limits
- **AI Settings**: Model paths, confidence thresholds
- **Execution**: Spread limits, filling modes
- **Monitoring**: Dashboard settings, logging levels

## 🎮 **Usage**

### **Start Live Trading**
```bash
# Live mode with AI agent
python live_trader_clean.py live --agent --regime

# Demo mode for testing
python live_trader_clean.py live --mode demo

# Custom configuration
python live_trader_clean.py live --config custom_config.json
```

### **Command Line Options**
```bash
python live_trader_clean.py --help

Options:
  --mode {live,demo}     Trading mode
  --symbol SYMBOL        Trading symbol
  --agent                Enable AI agent
  --regime               Enable regime detection
  --log-level {DEBUG,INFO,WARNING,ERROR}
                         Logging level
  --config CONFIG        Configuration file path
```

### **Dashboard Access**
- **URL**: http://127.0.0.1:8765/metrics
- **Features**: Real-time performance metrics, system health, trade statistics

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Reinstall dependencies
python install_dependencies.py

# Check Python version
python --version  # Should be 3.8+
```

#### **MT5 Connection Issues**
```bash
# Verify MT5 is running
# Check credentials in config.json
# Ensure demo mode is enabled for testing
```

#### **AI Model Issues**
```bash
# Check model files exist in models/ directory
# Verify TensorFlow installation
# Check model compatibility
```

### **Debug Mode**
```bash
# Enable debug logging
python live_trader_clean.py live --log-level DEBUG

# Check logs
tail -f logs/trading_bot.log
```

## 📊 **Performance Monitoring**

### **Real-time Metrics**
- **Uptime**: System running time
- **Cycles/sec**: Trading loop performance
- **Response Time**: Average cycle duration
- **Memory Usage**: System memory consumption
- **Error Rate**: System error frequency

### **Trade Statistics**
- **Total Trades**: Number of executed trades
- **Success Rate**: Trade success percentage
- **Profit/Loss**: Current P&L status
- **Open Positions**: Current market exposure

## 🛡️ **Safety Features**

### **Production Safeguards**
- **Demo Mode**: Test without real money
- **Risk Limits**: Maximum position sizes
- **Daily Loss Limits**: Automatic stop on losses
- **Spread Gates**: Filter high-spread conditions
- **Kill Switch**: Emergency stop capability

### **System Protection**
- **Error Storm Detection**: Automatic halt on errors
- **Memory Management**: Automatic cleanup
- **Health Monitoring**: Continuous system checks
- **Predictive Maintenance**: Proactive issue detection

## 🔮 **AI Integration**

### **Signal Generation**
- **LSTM Models**: Time series prediction
- **ML Filters**: Signal quality assessment
- **Ensemble Methods**: Multi-model combination
- **Regime Detection**: Market condition adaptation

### **Risk Assessment**
- **Conformal Prediction**: Statistical validation
- **Dynamic Thresholds**: Adaptive confidence levels
- **Market Regime**: Condition-based adjustments
- **Portfolio Risk**: Multi-position management

## 📈 **Trading Strategies**

### **Signal Types**
- **SMA Crossover**: Moving average signals
- **AI Prediction**: Machine learning signals
- **Ensemble**: Combined signal approach
- **Regime Adaptive**: Market condition based

### **Risk Management**
- **ATR-based SL/TP**: Dynamic stop levels
- **Position Sizing**: Risk-based volume calculation
- **Correlation Limits**: Portfolio diversification
- **Drawdown Protection**: Loss limit enforcement

## 🎉 **Getting Started**

1. **Install Dependencies**
   ```bash
   python install_dependencies.py
   ```

2. **Configure Trading Parameters**
   ```bash
   # Edit config.json with your settings
   # Start with demo_mode: true
   ```

3. **Test the System**
   ```bash
   python live_trader_clean.py live --mode demo
   ```

4. **Monitor Performance**
   - Check dashboard at http://127.0.0.1:8765/metrics
   - Review logs in logs/ directory
   - Monitor console output

5. **Go Live (When Ready)**
   ```bash
   # Set demo_mode: false in config.json
   python live_trader_clean.py live --agent --regime
   ```

## 📞 **Support**

### **Documentation**
- **System Architecture**: See `MODULAR_ARCHITECTURE.md`
- **AI Implementation**: See `AI_AGENT_README.md`
- **Configuration Guide**: See `agent-config.md`

### **Logs and Debugging**
- **Trading Logs**: `logs/trading_bot.log`
- **Event Logs**: `data/events.jsonl`
- **Performance Data**: Dashboard metrics

### **Common Commands**
```bash
# Check system status
python live_trader_clean.py --help

# Test configuration
python -c "from live_trader_clean import MT5Config; print('Config OK')"

# Verify imports
python -c "import live_trader_clean; print('Imports OK')"
```

---

**🎯 The system is now production-ready with comprehensive error handling, AI integration, and professional-grade features. Happy trading! 🚀**
