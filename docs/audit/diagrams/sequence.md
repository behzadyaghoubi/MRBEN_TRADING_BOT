# MR BEN Trading Cycle Sequence Diagram

## Complete Trading Cycle

```
Trading Loop → Market Data → Signal Gen → Risk Gates → Agent → Execution → MT5 → Position
```

## Detailed Sequence

### 1. Data Retrieval
```
MT5LiveTrader → MT5 API → Bars (15min, 600) → DataFrame
```

### 2. Signal Generation
```
DataFrame → SMA20/50 → Crossover Logic → Signal Object
```

### 3. Risk Assessment
```
Signal → Spread Gate → Exposure Gate → Conformal Gate → Decision
```

### 4. Agent Review
```
Decision → Agent Bridge → Policy Engine → Final Decision
```

### 5. Execution
```
Decision → ATR Calc → SL/TP → Volume → Order Request
```

### 6. Order Placement
```
Order Request → Adaptive Filling → MT5 order_send → Result
```

### 7. Position Management
```
Result → Position Registration → TP Split Logic → Monitoring
```

## Key Interactions

- **Every 12 seconds**: Complete cycle execution
- **Risk Gates**: Sequential validation (all must pass)
- **Agent**: Always reviews decisions in guard mode
- **MT5**: Adaptive filling mode with automatic fallback
- **Dashboard**: Real-time metrics update every cycle
