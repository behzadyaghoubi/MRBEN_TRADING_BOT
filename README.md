# MR BEN Live Trading System

A comprehensive, modular trading system with AI-powered signal generation, risk management, and automated execution.

## 🚀 Features

- **AI-Powered Trading**: Ensemble system combining LSTM, ML, and technical analysis
- **Advanced Risk Management**: Dynamic stop-loss, take-profit, and position sizing
- **Real-time MT5 Integration**: Direct connection to MetaTrader5 platform
- **Modular Architecture**: Clean, maintainable code structure (in development)
- **Comprehensive Testing**: Unit tests and smoke tests for reliability
- **Performance Monitoring**: Real-time metrics and system health tracking

## 🎯 **Current Entry Point (LIVE TRADING)**

**For immediate use, the system runs from the original entry point:**

```bash
# Live Trading (Current Active Entry Point)
python live_trader_clean.py

# Or use the provided scripts
./start_trader.bat                    # Windows
./run_live_trader.ps1                 # PowerShell
./run_live_trader.bat                 # Windows Batch
```

## 🔮 **Future Modular Entry Point (In Development)**

**The modular system is being developed and will provide:**

```bash
# Future Modular Entry Point (Not yet active)
python -m src.main                    # Main modular entry point
python -m src.core.cli live          # CLI interface for live trading
python -m src.core.cli paper         # CLI interface for paper trading
```

## 📁 Project Structure

```
mrben-trading-system/
├── live_trader_clean.py             # 🎯 CURRENT ACTIVE ENTRY POINT
├── src/                             # 🔮 FUTURE MODULAR SYSTEM
│   ├── __init__.py
│   ├── main.py                      # Future main entry point
│   ├── core/                        # Core system components
│   │   ├── __init__.py
│   │   ├── trader.py                # Main trading orchestrator
│   │   ├── cli.py                   # Unified CLI interface
│   │   ├── metrics.py               # Performance monitoring
│   │   ├── exceptions.py            # Custom exceptions
│   │   └── database.py              # Database operations
│   ├── config/                      # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py              # MT5Config class
│   ├── data/                        # Data management
│   │   ├── __init__.py
│   │   └── manager.py               # MT5DataManager class
│   ├── ai/                          # AI/ML components
│   │   ├── __init__.py
│   │   └── system.py                # MRBENAdvancedAISystem
│   ├── risk/                        # Risk management
│   │   ├── __init__.py
│   │   └── manager.py               # EnhancedRiskManager
│   ├── execution/                   # Trade execution
│   │   ├── __init__.py
│   │   └── executor.py              # EnhancedTradeExecutor
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── helpers.py               # General helpers
│       ├── position_management.py
│       ├── memory.py                # Memory management
│       └── error_handler.py
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── run_all_tests.py             # Comprehensive test runner
│   ├── test_smoke.py                # Smoke tests
│   ├── unit/                        # Unit tests
│   │   ├── test_config.py
│   │   ├── test_trader.py
│   │   ├── test_risk_manager.py
│   │   └── test_ai_system.py
│   └── integration/                 # Integration tests
│       └── test_system_integration.py
├── config/                          # Configuration files
│   ├── default.yaml                 # Default YAML config
│   └── credentials.example.env      # Example environment file
├── scripts/                         # Shell scripts
│   ├── run_smoke.sh                # Smoke test runner
│   ├── run_backtest.sh             # Backtest runner
│   └── run_live.sh                 # Live trading runner
├── Makefile                         # Development tasks
├── pyproject.toml                   # Project configuration
├── requirements.txt                  # Dependencies
├── README.md                        # This file
└── REFACTORING_COMPLETION_REPORT.md # Detailed refactoring report
```

**📖 For detailed information about the refactoring, see [REFACTORING_COMPLETION_REPORT.md](REFACTORING_COMPLETION_REPORT.md)**

**📊 For current system health and status, see [SYSTEM_HEALTH_REPORT.md](SYSTEM_HEALTH_REPORT.md)**

## 🚀 How to Use

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd mrben-trading-system

# Install dependencies
make install
# or manually: pip install -r requirements.txt

# Set up environment (copy and edit)
cp config/credentials.example.env .env
# Edit .env with your actual credentials
```

### 🎯 **Live Trading (Current Active System)**

```bash
# Method 1: Direct execution
python live_trader_clean.py

# Method 2: Using provided scripts
./start_trader.bat                    # Windows
./run_live_trader.ps1                 # PowerShell
./run_live_trader.bat                 # Windows Batch

# Method 3: Using the execution helper
python execute_live_trader.py
```

### 🔮 **Testing Modular System (Development)**

```bash
# Test the modular components
python test_modular_system.py

# Test individual modules
python -c "from src.config import MT5Config; print('Config module works')"
python -c "from src.telemetry import EventLogger; print('Telemetry module works')"
```

### Development Workflow

```bash
# Format and lint code
make format    # Runs ruff + isort + black
make lint      # Runs ruff + mypy

# Run tests
make test      # Runs all tests
make smoke     # Runs smoke tests only

# Development tasks
make clean     # Clean build artifacts
make docs      # Generate documentation
```

## 🔄 **Migration Path**

### **Current State (Phase 1):**
- ✅ `live_trader_clean.py` - Fully functional, all updates applied
- ✅ Modular components created in `src/` directory
- ✅ Backward compatibility maintained

### **Next Phase (Phase 2):**
- 🔄 Complete modular system testing
- 🔄 Performance validation
- 🔄 Switch to modular entry point

### **Final State (Phase 3):**
- 🎯 `python -m src.main` becomes primary entry point
- 🎯 `live_trader_clean.py` becomes legacy/backup
- 🎯 Full modular architecture active

## 📊 **System Status**

| Component | Status | Entry Point |
|-----------|--------|-------------|
| **Live Trading** | ✅ **ACTIVE** | `python live_trader_clean.py` |
| **Modular System** | 🔄 **In Development** | `python -m src.main` (Future) |
| **Configuration** | ✅ **Active** | `config.json` |
| **AI Models** | ✅ **Active** | `models/` directory |

## 🚨 **Important Notes**

1. **For immediate live trading**: Use `python live_trader_clean.py`
2. **Modular system**: Still in development, not yet production-ready
3. **All updates**: Applied to `live_trader_clean.py` for immediate use
4. **Future migration**: Will be seamless when modular system is complete

## 🆘 **Troubleshooting**

### Common Issues

```bash
# Check system status
python check_and_run.py

# Test syntax
python test_syntax.py

# Run smoke tests
make smoke

# Check configuration
python -c "from src.config import MT5Config; config = MT5Config(); print(config.get_config_summary())"
```

### Getting Help

- Check [REFACTORING_COMPLETION_REPORT.md](REFACTORING_COMPLETION_REPORT.md) for detailed status
- Review [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) for future plans
- Run `python test_modular_system.py` to test modular components

## 🤝 **Contributing**

1. **Current development**: Focus on `live_trader_clean.py` for immediate improvements
2. **Modular development**: Work on `src/` modules for future architecture
3. **Testing**: Ensure both systems work correctly
4. **Documentation**: Keep README and reports updated

## 📈 **Performance**

The current system (`live_trader_clean.py`) includes:
- ✅ All latest optimizations
- ✅ Enhanced AI signal generation
- ✅ Improved risk management
- ✅ Better error handling
- ✅ Performance monitoring

## 🔮 **Roadmap**

- **Q1 2024**: Complete modular system testing
- **Q2 2024**: Performance optimization and validation
- **Q3 2024**: Switch to modular entry point
- **Q4 2024**: Advanced features and plugins

---

**🎯 For immediate use: `python live_trader_clean.py`**  
**🔮 For future: `python -m src.main` (in development)**
