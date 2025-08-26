# Monitoring Setup - Phase 3

## Overview
Local monitoring setup with Prometheus adapter and Grafana dashboard for MR BEN Trading System.

## Components

### 1. Prometheus Adapter
- **File**: `src/ops/prom_adapter.py`
- **Endpoint**: http://127.0.0.1:9100/prom
- **Format**: Prometheus metrics (text/plain)
- **Metrics**:
  - `mrben_uptime_seconds`: System uptime
  - `mrben_cycles_total`: Total trading cycles
  - `mrben_total_trades`: Total trades executed
  - `mrben_error_rate`: Error rate percentage
  - `mrben_memory_mb`: Memory usage in MB

### 2. Prometheus Configuration
- **File**: `docs/ops/prometheus.yml`
- **Targets**: 
  - MR BEN Prometheus adapter (port 9100)
  - MR BEN Dashboard (port 8765)
- **Scrape Interval**: 15s for adapter, 30s for dashboard

### 3. Grafana Dashboard
- **File**: `docs/ops/grafana_dashboard.json`
- **Panels**:
  - System Uptime (stat)
  - Trading Cycles (graph)
  - Total Trades (stat)
  - Error Rate (graph)
  - Memory Usage (graph)

## Setup Instructions

### Step 1: Start Prometheus Adapter
The adapter is automatically started when running the trading system:
```python
try:
    from src.ops.prom_adapter import start_prometheus_adapter
    start_prometheus_adapter(port=9100)
    logger.info("Prometheus adapter at http://127.0.0.1:9100/prom")
except Exception as e:
    logger.warning(f"Prom adapter failed: {e}")
```

### Step 2: Install Prometheus (Local)
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.windows-amd64.zip
unzip prometheus-2.45.0.windows-amd64.zip

# Copy config
copy docs/ops/prometheus.yml prometheus-2.45.0.windows-amd64/

# Start Prometheus
cd prometheus-2.45.0.windows-amd64
./prometheus.exe --config.file=prometheus.yml
```

### Step 3: Install Grafana (Local)
```bash
# Download Grafana
wget https://dl.grafana.com/oss/release/grafana-10.0.3.windows-amd64.zip
unzip grafana-10.0.3.windows-amd64.zip

# Start Grafana
cd grafana-10.0.3.windows-amd64
./bin/grafana-server.exe
```

### Step 4: Import Dashboard
1. Open Grafana at http://127.0.0.1:3000
2. Default credentials: admin/admin
3. Add Prometheus data source: http://127.0.0.1:9090
4. Import dashboard from `docs/ops/grafana_dashboard.json`

## Testing

### Test Prometheus Adapter
```bash
curl http://127.0.0.1:9100/prom
```

Expected output:
```
mrben_uptime_seconds 12345
mrben_cycles_total 100
mrben_total_trades 5
mrben_error_rate 0.02
mrben_memory_mb 45.2
# Timestamp: 1692489600
```

### Test Dashboard
```bash
curl http://127.0.0.1:8765/metrics
```

## Monitoring Features

### Real-time Metrics
- **Uptime**: System running time
- **Performance**: Cycles per second, error rate
- **Trading**: Total trades, success rate
- **Resources**: Memory usage, CPU utilization

### Alerts (Future)
- High error rate (>5%)
- Memory usage spike (>100MB)
- System downtime
- Trading anomalies

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change ports in config files
2. **Import errors**: Check Python path and dependencies
3. **Connection refused**: Verify services are running

### Debug Mode
Enable debug logging in the trading system:
```bash
python live_trader_clean.py --mode live --config config/pro_config.json --log-level DEBUG
```

## Next Steps
1. Set up Grafana alerts
2. Add custom metrics
3. Implement log aggregation
4. Create performance dashboards

---
**Status**: ✅ Monitoring Setup Complete  
**Next**: Portfolio Expansion (Phase 4)
