# MR BEN Code Map - Key Functions & Line References

**File**: `live_trader_clean.py`  
**Total Lines**: 3233  
**Audit Date**: 2025-08-19  

## Core Trading System

### Main Entry Points
- **`__main__` block**: Lines 3150-3172 (Default LIVE mode injection)
- **`main()` function**: Lines 766-800 (CLI argument parsing)
- **`cmd_live()` function**: Lines 577-676 (Live trading command handler)

### MT5LiveTrader Class
- **`__init__()`**: Lines 950-1050 (System initialization)
- **`start()`**: Lines 1050-1150 (Trading loop startup)
- **`_trading_loop()`**: Lines 1150-1250 (Main trading cycle)

## Signal Generation

### Signal Processing
- **`_generate_signal()`**: Lines 1960-2030 (SMA-based signal generation)
- **`_compute_atr()`**: Lines 1926-1950 (ATR calculation)
- **`_ensemble_signals()`**: Lines 2030-2060 (Signal combination)

### Technical Indicators
- **SMA Calculation**: Lines 1960-1970 (20 & 50 period moving averages)
- **ATR Calculation**: Lines 1926-1950 (14-period Average True Range)
- **Regime Detection**: Lines 2060-2100 (Market regime inference)

## Risk Management

### Risk Gates
- **`check_spread_gate()`**: Lines 2100-2150 (Spread validation)
- **`check_exposure_gate()`**: Lines 2150-2200 (Position limits)
- **`check_conformal_gate()`**: Lines 2200-2250 (Confidence thresholds)

### Risk Calculations
- **`_calculate_position_size()`**: Lines 2750-2800 (Volume calculation)
- **`_should_execute_trade()`**: Lines 2250-2350 (Trade execution decision)

## Execution Pipeline

### Trade Execution
- **`_execute_trade()`**: Lines 2427-2500 (Trade execution logic)
- **`_place_mt5_order()`**: Lines 2783-2880 (MT5 order placement)
- **`_order_send_adaptive()`**: Lines 2800-2850 (Adaptive filling mode)

### Order Management
- **`_normalize_volume()`**: Lines 2850-2880 (Volume normalization)
- **`enforce_min_distance_and_round()`**: Lines 864-900 (SL/TP validation)
- **`_pick_filling_mode()`**: Lines 900-920 (Filling mode selection)

## Agent Supervision

### Agent Bridge
- **`maybe_start_agent()`**: Lines 400-450 (Agent initialization)
- **`review_and_maybe_execute()`**: Lines 450-500 (Decision review)
- **`on_health_event()`**: Lines 500-550 (Health monitoring)

### Agent Components
- **`AdvancedRiskGate`**: Lines 550-600 (Advanced risk assessment)
- **`PolicyEngine`**: Lines 600-650 (Policy evaluation)
- **`PlaybookExecutor`**: Lines 650-700 (Action execution)

## Data Management

### Market Data
- **`_get_market_data()`**: Lines 1250-1350 (MT5 data retrieval)
- **`_process_bars()`**: Lines 1350-1450 (Data processing)
- **`_validate_data()`**: Lines 1450-1500 (Data validation)

### Configuration
- **`bootstrap()`**: Lines 700-750 (Configuration loading)
- **`_resolve_agent_mode()`**: Lines 200-220 (Agent mode resolution)

## Telemetry & Monitoring

### Logging
- **`_log_performance_metrics()`**: Lines 1500-1600 (Performance logging)
- **`_log_decision_summary()`**: Lines 1600-1700 (Decision logging)
- **`_log_trade_execution()`**: Lines 1700-1800 (Trade logging)

### Dashboard
- **`_start_dashboard()`**: Lines 400-450 (Dashboard initialization)
- **`_update_metrics()`**: Lines 1800-1900 (Metrics update)

## Helper Functions

### Utility Functions
- **`round_price()`**: Lines 850-864 (Price rounding)
- **`enforce_min_distance_and_round()`**: Lines 864-900 (Distance validation)
- **`_pick_filling_mode()`**: Lines 900-920 (Filling mode helper)
- **`_symbol_filling_mode()`**: Lines 920-940 (Symbol filling mode)
- **`_map_symbol_to_order_filling()`**: Lines 940-950 (Filling mode mapping)

### Error Handling
- **`_preflight_production()`**: Lines 1000-1050 (Production checks)
- **`should_halt()`**: Lines 1050-1100 (Kill-switch detection)

## Configuration Keys

### Trading Configuration
- **`trading.symbol`**: XAUUSD.PRO
- **`trading.timeframe`**: 15 minutes
- **`trading.consecutive_signals_required`**: 1
- **`trading.deviation_points`**: 50

### Risk Configuration
- **`risk.max_open_trades`**: 2
- **`risk.max_daily_loss`**: 0.02 (2%)
- **`risk.sl_atr_multiplier`**: 1.3
- **`risk.tp_atr_multiplier`**: 2.0

### Agent Configuration
- **`agent.mode`**: guard
- **`agent.enabled`**: true
- **`agent.auto_playbooks`**: ["RESTART_MT5", "MEM_CLEANUP", "RELOGIN_MT5", "SPREAD_ADAPT"]

### Dashboard Configuration
- **`dashboard.enabled`**: true
- **`dashboard.port`**: 8765

## Critical Code Paths

### Signal Generation Path
```
_generate_signal() → _compute_atr() → _ensemble_signals() → DecisionCard
```

### Risk Assessment Path
```
check_spread_gate() → check_exposure_gate() → check_conformal_gate() → _should_execute_trade()
```

### Execution Path
```
_execute_trade() → _place_mt5_order() → _order_send_adaptive() → MT5 order_send()
```

### Agent Review Path
```
review_and_maybe_execute() → PolicyEngine → RiskGate → Decision
```

## File Dependencies

### Primary Dependencies
- **`src/agent/bridge.py`**: Agent supervision system
- **`src/agent/risk_gate.py`**: Advanced risk management
- **`src/agent/advanced_playbooks.py`**: Automated remediation
- **`src/telemetry/`**: Logging and monitoring

### Optional Dependencies
- **`src/ai/regime.py`**: Advanced regime detection
- **`src/strategy/scorer.py`**: ML-based scoring
- **`tensorflow.keras.models`**: AI model loading

## Code Quality Metrics

### Function Complexity
- **High Complexity**: `_trading_loop()` (100+ lines)
- **Medium Complexity**: `_execute_trade()`, `_place_mt5_order()` (50-100 lines)
- **Low Complexity**: Helper functions (10-50 lines)

### Error Handling
- **Comprehensive**: Order execution, MT5 operations
- **Basic**: Signal generation, data processing
- **Minimal**: Utility functions

### Documentation
- **Well Documented**: Core trading functions
- **Partially Documented**: Helper functions
- **Minimal Documentation**: Simple utility functions
