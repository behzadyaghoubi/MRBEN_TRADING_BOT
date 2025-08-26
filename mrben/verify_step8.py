#!/usr/bin/env python3
"""
MR BEN - STEP8 Quick Verification
Simple verification that Performance Metrics system is working
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_step8():
    """Quick verification of STEP8 components"""
    print("🚀 MR BEN - STEP8 Quick Verification")
    print("=" * 40)

    try:
        # Test 1: Configuration
        from core.configx import load_config

        cfg = load_config()
        print("✅ Configuration loaded")

        # Test 2: Metrics config
        assert hasattr(cfg, 'metrics')
        assert hasattr(cfg.metrics, 'port')
        assert hasattr(cfg.metrics, 'enabled')
        assert cfg.metrics.port == 8765
        assert cfg.metrics.enabled == True
        print("✅ Metrics configuration present and correct")

        # Test 3: Import metrics components
        from core.metricsx import (
            get_metrics_summary,
            observe_block,
            observe_order_send,
            observe_risk_gate,
            observe_trade_close,
            observe_trade_open,
            update_context,
        )

        print("✅ All metrics components imported successfully")

        # Test 4: Test basic functionality (without starting server)
        print("\n📊 Testing Metrics Functions...")

        # Test context update
        update_context(
            equity=10000.0,
            balance=10000.0,
            spread_pts=2.0,
            session="london",
            regime="NORMAL",
            dyn_conf=0.75,
            score=0.68,
            open_positions=2,
        )
        print("✅ Context update function working")

        # Test decision blocks
        observe_block("ml_low_conf")
        observe_block("risk_exposure")
        print("✅ Decision block tracking working")

        # Test order execution metrics
        observe_order_send("ioc", 45.2, 1.5)
        observe_order_send("fok", 23.1, 0.8)
        print("✅ Order execution metrics working")

        # Test trade lifecycle
        observe_trade_open("EURUSD", 1, "pro")
        observe_trade_close("EURUSD", 1, "pro", 1.2)
        print("✅ Trade lifecycle tracking working")

        # Test risk gate observations
        observe_risk_gate("spread", True)
        observe_risk_gate("exposure", False)
        print("✅ Risk gate metrics working")

        # Test metrics summary
        summary = get_metrics_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        print("✅ Metrics summary function working")

        print("\n🎉 STEP8: Performance Metrics & Telemetry - VERIFIED SUCCESSFULLY!")
        print("All core components are working correctly:")
        print("✅ Configuration loading")
        print("✅ Metrics system imports")
        print("✅ Context updates")
        print("✅ Decision tracking")
        print("✅ Order metrics")
        print("✅ Trade lifecycle")
        print("✅ Risk gate monitoring")
        print("✅ Metrics summary")

        print(f"\n📈 Metrics server will start on port {cfg.metrics.port}")
        print("🌐 Prometheus endpoint: http://localhost:8765/metrics")
        print("📊 Ready for Grafana dashboard integration")

        return True

    except Exception as e:
        print(f"❌ STEP8 verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_step8()
    sys.exit(0 if success else 1)
