# MR BEN System Data Flow Diagram

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Market Data   │    │  Signal Gen.    │    │  Risk Gates     │
│   (MT5 Bars)    │───▶│  (SMA20/50)     │───▶│  (Spread/Exp.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │  Conformal      │    │  Agent Review   │
         │              │  Prediction     │───▶│  (Guard Mode)   │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Execution     │    │  Order Mgmt.    │    │  Telemetry      │
│   Pipeline      │◀───│  (SL/TP/Vol)    │◀───│  (Logging)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MT5 Order     │    │  Position       │    │  Dashboard      │
│   Placement     │    │  Management     │    │  (/metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Detailed Data Flow

### 1. Data Ingestion Layer
```
MT5 Terminal → Bars (15min, 600) → DataFrame → Validation → Processing
```

**Components**:
- **Source**: MetaTrader5 API
- **Format**: OHLCV bars
- **Frequency**: Every 12 seconds
- **Validation**: Stale data check, NaN handling

### 2. Signal Generation Layer
```
DataFrame → SMA20/50 → Crossover Logic → Confidence (0.7) → Signal Object
```

**Process**:
1. Calculate SMA20 (20-period moving average)
2. Calculate SMA50 (50-period moving average)
3. Compare: SMA20 vs SMA50
4. Generate: BUY (1), SELL (-1), or NO SIGNAL (0)
5. Assign: Fixed confidence 0.7 (70%)

**Signal Object Structure**:
```python
{
    'signal': -1,           # SELL signal
    'confidence': 0.7,      # Fixed confidence
    'atr': 2.58483,        # 14-period ATR
    'source': 'sma',        # Signal source
    'components': {...}      # Component breakdowns
}
```

### 3. Risk Assessment Layer
```
Signal → Spread Gate → Exposure Gate → Conformal Gate → Decision
```

**Risk Gates**:

#### SpreadGate
- **Input**: Current spread in points
- **Threshold**: 180 points (configurable)
- **Logic**: `spread < 180 ? PASS : BLOCK`
- **Formula**: `(ask - bid) / point_value`

#### ExposureGate
- **Input**: Open positions count
- **Threshold**: 2 positions (configurable)
- **Logic**: `positions < 2 ? PASS : BLOCK`

#### ConformalGate
- **Input**: Signal confidence
- **Threshold**: 0.5 (50%, configurable)
- **Logic**: `confidence > threshold ? PASS : BLOCK`

### 4. Agent Supervision Layer
```
Decision → Agent Review → Policy Engine → Risk Assessment → Final Decision
```

**Guard Mode Behavior**:
- **Purpose**: Block risky trades
- **Actions**: Review, approve/block, log
- **No Auto-remediation**: Manual intervention required

**Health Event Handling**:
- **Types**: ORDER_FAIL, SYSTEM_HEALTH, PERFORMANCE
- **Response**: Logging + policy evaluation
- **No External Alerts**: Local only

### 5. Execution Pipeline
```
Approved Signal → ATR Calculation → SL/TP → Volume → Order Request
```

**ATR-Based SL/TP**:
- **ATR Period**: 14 bars
- **SL Multiplier**: 1.3 × ATR
- **TP Multiplier**: 2.0 × ATR

**Volume Calculation**:
- **Method**: Risk-based (if enabled)
- **Constraints**: Min 0.01, Max 2.0 lots
- **Normalization**: Broker volume step compliance

### 6. Order Management
```
Order Request → Adaptive Filling → MT5 order_send → Result Handling
```

**Adaptive Filling Mode**:
1. **Symbol Recommended** (if available)
2. **ORDER_FILLING_RETURN** (fallback 1)
3. **ORDER_FILLING_IOC** (fallback 2)
4. **ORDER_FILLING_FOK** (fallback 3)

**Error Handling**:
- **10030 Error**: Automatic filling mode fallback
- **Other Errors**: HealthEvent + logging
- **Success**: Position registration + management

### 7. Position Management
```
Open Position → TP Split Logic → Breakeven → Trailing Stops
```

**TP Split Policy**:
- **TP1**: 80% of full TP distance
- **TP2**: 150% of full TP distance
- **Volume Split**: 50% at TP1, 50% at TP2
- **Breakeven**: After TP1 hit

### 8. Telemetry & Monitoring
```
All Events → Logging → Dashboard → Metrics → Health Monitoring
```

**Dashboard Metrics**:
- **Endpoint**: `http://127.0.0.1:8765/metrics`
- **Update Rate**: Real-time (every cycle)
- **Key Metrics**: Uptime, cycles/sec, response time, total trades, error rate, memory

**Logging Structure**:
- **Level**: INFO (configurable)
- **Format**: Timestamp + Level + Component + Message
- **Persistence**: JSONL files + console output

## Data Transformation Points

### 1. Raw Data → Processed Data
```
MT5 Bars → DataFrame → Technical Indicators → Signal Objects
```

### 2. Signal → Decision
```
Signal Object → Risk Gates → DecisionCard → Agent Review → Execution Decision
```

### 3. Decision → Order
```
Execution Decision → ATR Calculation → SL/TP → Volume → MT5 Request
```

### 4. Order → Position
```
MT5 Response → Position Registration → Management Logic → Monitoring
```

## Critical Data Paths

### Primary Trading Path
```
Market Data → Signal Generation → Risk Assessment → Agent Review → Execution → Position Management
```

### Risk Management Path
```
Signal → Spread Gate → Exposure Gate → Conformal Gate → Final Decision
```

### Error Handling Path
```
Error → HealthEvent → Agent → Policy Engine → Logging → Dashboard
```

### Monitoring Path
```
All Events → Logging → Dashboard → Metrics → Health Assessment
```

## Data Quality & Validation

### Input Validation
- **Data Freshness**: Stale data detection
- **Completeness**: NaN handling
- **Format**: OHLCV structure validation

### Processing Validation
- **Signal Logic**: SMA calculation verification
- **Risk Gates**: Threshold validation
- **Execution**: Parameter validation

### Output Validation
- **Order Success**: MT5 retcode verification
- **Position Status**: Open position verification
- **Performance**: Metric accuracy validation
