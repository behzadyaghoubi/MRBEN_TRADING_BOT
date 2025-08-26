# AutoML Retraining Logs - Phase 5

## Overview
Weekly automated retraining of ML and LSTM models with safe promotion based on performance improvement.

## Components

### 1. Model Registry
- **File**: `models/registry.json`
- **Purpose**: Track model versions, performance, and promotion history
- **Structure**: Separate tracking for ML and LSTM models

### 2. ML Retraining Script
- **File**: `src/ops/automl/retrain_ml.py`
- **Algorithms**: XGBoost, LightGBM, Random Forest
- **Features**: Automatic algorithm selection, calibration, safe promotion
- **Logs**: `logs/automl_ml.log`

### 3. LSTM Retraining Script
- **File**: `src/ops/automl/retrain_lstm.py`
- **Architecture**: LSTM with dropout layers
- **Features**: Sequence modeling, early stopping, safe promotion
- **Logs**: `logs/automl_lstm.log`

## Safe Promotion Logic

### Promotion Criteria
- **AUC Improvement**: > 0.02 (2% improvement)
- **F1 Improvement**: > 0.02 (2% improvement)
- **Calibration**: Maintained or improved

### Promotion Process
1. Train new model on latest data
2. Evaluate against validation set
3. Compare with current model performance
4. Promote only if significant improvement
5. Update registry with new model path

## Weekly Schedule

### Automated Retraining
```python
import schedule

def weekly_job():
    try:
        os.system("python -m src.ops.automl.retrain_ml")
        os.system("python -m src.ops.automl.retrain_lstm")
    except Exception as e:
        logger.error(f"AutoML job failed: {e}")

# Schedule for Monday 3:00 AM UTC
schedule.every().monday.at("03:00").do(weekly_job)
```

### Manual Retraining
```bash
# ML Model
python -m src.ops.automl.retrain_ml

# LSTM Model
python -m src.ops.automl.retrain_lstm
```

## Model Loading

### Safe Model Loading
During system startup, models are loaded from registry:
```python
def load_models_from_registry():
    with open("models/registry.json", "r") as f:
        registry = json.load(f)
    
    ml_model_path = registry["ml"]["current"]
    lstm_model_path = registry["lstm"]["current"]
    
    return ml_model_path, lstm_model_path
```

### Graceful Degradation
If models are unavailable, system falls back to rule-based strategy:
```python
try:
    ml_model = joblib.load(ml_model_path)
    lstm_model = tf.keras.models.load_model(lstm_model_path)
except Exception as e:
    logger.warning(f"ML models unavailable: {e}, using rule-based fallback")
    ml_model = None
    lstm_model = None
```

## Performance Tracking

### Metrics Recorded
- **AUC**: Area Under ROC Curve
- **F1**: F1 Score (harmonic mean of precision/recall)
- **Calibration**: Predicted vs actual positive rate ratio

### Historical Performance
All model versions and performance metrics are stored in registry:
```json
{
  "ml": {
    "history": [
      {
        "timestamp": "2025-08-20T03:00:00Z",
        "model_path": "models/ml_filter_20250820_030000.pkl",
        "metrics": {"auc": 0.85, "f1": 0.78, "calibration": 0.92},
        "promoted": true
      }
    ]
  }
}
```

## Testing

### Test Retraining
```bash
# Test ML retraining
python src/ops/automl/retrain_ml.py

# Test LSTM retraining
python src/ops/automl/retrain_lstm.py
```

### Expected Output
```
✅ ML retraining completed successfully
✅ LSTM retraining completed successfully
```

## Troubleshooting

### Common Issues
1. **Dependency Missing**: Install required ML libraries
2. **Memory Issues**: Reduce dataset size or batch size
3. **Model Corruption**: Check registry and restore from backup

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps
1. Create executive report generator (Phase 6)
2. Run validation campaign (Phase 7)
3. Final testing and handoff (Phase 8)

---
**Status**: ✅ AutoML Implementation Complete  
**Next**: Executive Report Generator (Phase 6)
