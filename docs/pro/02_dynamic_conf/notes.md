# Dynamic Confidence Implementation - Phase 2

## Overview
Implementation of dynamic confidence adjustment based on ATR volatility, market regime, and trading session awareness.

## Components Implemented

### 1. DynamicConfidence Class
- **Location**: `src/strategy/dynamic_conf.py`
- **Purpose**: Central coordinator for confidence adjustments
- **Methods**:
  - `adjust_confidence()`: Main adjustment function
  - `_adjust_by_atr()`: ATR-based volatility adjustment
  - `_adjust_by_session()`: Session-based timing adjustment
  - `_adjust_by_regime()`: Market regime-based adjustment

### 2. Configuration Structure
```json
{
  "strategy": {
    "min_conf": 0.55,
    "atr_conf_scale": [0.5, 0.9],
    "atr_window": 14,
    "regime": {
      "HIGH_VOL": {"conf_mult": 0.8, "thr_add": 0.05},
      "LOW_VOL": {"conf_mult": 1.05, "thr_add": -0.05}
    }
  },
  "session": {
    "windows": {
      "LONDON": ["07:00", "16:00"],
      "NEWYORK": ["12:00", "21:00"]
    },
    "block_outside": true
  }
}
```

## Adjustment Logic

### ATR-Based Adjustment
- **Window**: 14-period ATR calculation
- **Scaling**: Interpolate between 30th and 70th percentiles
- **Range**: [0.5, 0.9] confidence multiplier
- **Purpose**: Reduce confidence in high volatility, increase in low volatility

### Session-Based Adjustment
- **Windows**: London (07:00-16:00 UTC), New York (12:00-21:00 UTC)
- **Outside Sessions**: Block trades or reduce confidence by 20%
- **Configurable**: Can be disabled or modified

### Regime-Based Adjustment
- **Detection**: Based on 20-period volatility
- **HIGH_VOL**: > 0.002, confidence × 0.8, threshold + 0.05
- **LOW_VOL**: < 0.0005, confidence × 1.05, threshold - 0.05
- **NORMAL**: No adjustment

## Integration Points

### Decision Card Fields
- `adj_conf`: Final adjusted confidence
- `atr_value`: Current ATR value
- `session`: Current trading session
- `regime`: Market regime classification
- `reasons`: Detailed adjustment breakdown

### Logging Example
```
🎯 Decision Summary:
   Signal: -1 | Confidence: 0.700 | Consecutive: 2/1
   Price: 3341.96000 | SMA20: 3344.35450 | SMA50: 3345.37460
   Regime: HIGH_VOL | Adj Conf: 0.560
   Threshold: 0.550 | Allow Trade: True
   ATR: 2.33 | Session: LONDON
```

## Usage Example

```python
from src.strategy.dynamic_conf import DynamicConfidence

# Initialize
dyn_conf = DynamicConfidence(config)

# Adjust confidence
adj_conf, threshold, reasons = dyn_conf.adjust_confidence(
    base_conf=0.7,
    df=market_data,
    current_time=datetime.now()
)

print(f"Adjusted Confidence: {adj_conf:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Reasons: {reasons}")
```

## Benefits

1. **Risk Management**: Automatically reduces exposure in high volatility
2. **Session Awareness**: Avoids trading during low-liquidity periods
3. **Regime Adaptation**: Adjusts strategy based on market conditions
4. **Transparency**: Clear logging of all adjustments
5. **Configurability**: Easy to tune parameters

## Next Steps

- Integration with main signal generation
- Testing with live data
- Performance comparison with baseline
- Fine-tuning of adjustment parameters
