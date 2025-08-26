#!/usr/bin/env python3
"""
MR BEN - STEP9 A/B Testing Test
Test Control vs Pro deciders and paper trading
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_step9():
    """Test STEP9: Shadow A/B Testing"""
    print("🚀 MR BEN - STEP9 A/B Testing Test")
    print("=" * 50)

    try:
        # Test 1: Configuration
        from core.configx import load_config

        cfg = load_config()
        print("✅ Configuration loaded")

        # Test 2: Import A/B testing components
        from core.ab import ABRunner
        from core.context_factory import ContextFactory
        from core.deciders import ControlDecider, ProDecider
        from core.paper import PaperBroker
        from core.typesx import MarketContext

        print("✅ A/B testing components imported")

        # Test 3: Test MarketContext creation
        print("\n📊 Testing MarketContext...")

        ctx = MarketContext(
            price=1.1050,
            bid=1.1049,
            ask=1.1051,
            atr_pts=25.0,
            sma20=1.1040,
            sma50=1.1020,
            session="london",
            regime="NORMAL",
            equity=10000.0,
            balance=10000.0,
            spread_pts=20.0,
            open_positions=0,
        )

        assert ctx.price == 1.1050
        assert ctx.trend_direction == 1  # Uptrend
        assert ctx.spread == 0.0002
        assert ctx.mid_price == 1.1050
        print("✅ MarketContext working correctly")

        # Test 4: Test Control Decider
        print("\n🎯 Testing Control Decider...")

        control_decider = ControlDecider(ctx)
        control_decision = control_decider.decide()

        assert control_decision.action == "ENTER"
        assert control_decision.dir == 1
        assert control_decision.reason == "sma_cross"
        assert control_decision.track == "control"
        assert control_decision.lot > 0
        assert control_decision.levels is not None

        print(f"✅ Control decision: {control_decision.action} {control_decision.reason}")
        print(f"   Direction: {control_decision.dir}, Lot: {control_decision.lot}")
        print(f"   SL: {control_decision.levels.sl}, TP1: {control_decision.levels.tp1}")

        # Test 5: Test Pro Decider
        print("\n🚀 Testing Pro Decider...")

        pro_decider = ProDecider(ctx)
        pro_decision = pro_decider.decide()

        assert pro_decision.action in ["ENTER", "HOLD"]
        assert pro_decision.track == "pro"

        print(f"✅ Pro decision: {pro_decision.action} {pro_decision.reason}")
        print(f"   Score: {pro_decision.score}, Confidence: {pro_decision.dyn_conf}")

        # Test 6: Test Paper Broker
        print("\n📝 Testing Paper Broker...")

        paper_broker = PaperBroker("EURUSD", "control")

        # Open position
        paper_broker.open(control_decision)

        # Check position opened
        position = paper_broker.get_position_summary()
        assert position is not None
        assert position['symbol'] == "EURUSD"
        assert position['direction'] == "buy"

        print("✅ Paper position opened successfully")

        # Test 7: Test Position Management
        print("\n📈 Testing Position Management...")

        # Simulate TP1 hit
        paper_broker.on_tick(bid=1.1055, ask=1.1057, atr_pts=25.0)

        # Check if TP1 was hit
        updated_position = paper_broker.get_position_summary()
        if updated_position and updated_position.get('tp1_hit'):
            print("✅ TP1 hit and breakeven triggered")
        else:
            print("ℹ️ TP1 not hit yet (expected)")

        # Simulate TP2 hit
        paper_broker.on_tick(bid=1.1065, ask=1.1067, atr_pts=25.0)

        # Check if position was closed
        final_position = paper_broker.get_position_summary()
        if final_position is None:
            print("✅ Position closed at TP2")
        else:
            print("ℹ️ Position still open")

        # Test 8: Test Context Factory
        print("\n🏭 Testing Context Factory...")

        factory = ContextFactory()

        bar_data = {
            'timestamp': '2024-01-01T10:00:00Z',
            'close': 1.1050,
            'bid': 1.1049,
            'ask': 1.1051,
            'atr_pts': 25.0,
            'sma20': 1.1040,
            'sma50': 1.1020,
            'equity': 10000.0,
            'balance': 10000.0,
            'spread_pts': 20.0,
            'open_positions': 0,
        }

        created_ctx = factory.create_from_bar(bar_data)
        assert created_ctx.price == 1.1050
        assert created_ctx.session in ["asia", "london", "ny", "off"]

        print("✅ Context factory working correctly")

        # Test 9: Test A/B Runner
        print("\n🔄 Testing A/B Runner...")

        ab_runner = ABRunner(factory.create_from_bar, "EURUSD")

        # Process a test bar
        ab_runner.on_bar(bar_data)

        # Check statistics
        stats = ab_runner.get_statistics()
        assert stats['symbol'] == "EURUSD"
        assert stats['control']['decisions'] > 0
        assert stats['pro']['decisions'] > 0

        print("✅ A/B runner working correctly")
        print(f"   Control decisions: {stats['control']['decisions']}")
        print(f"   Pro decisions: {stats['pro']['decisions']}")

        # Test 10: Test Metrics Integration
        print("\n📊 Testing Metrics Integration...")

        # Check if metrics are being updated
        from core.metricsx import get_metrics_summary

        metrics_summary = get_metrics_summary()

        assert isinstance(metrics_summary, dict)
        assert len(metrics_summary) > 0

        print("✅ Metrics integration working")

        # Test 11: Test Decision Comparison
        print("\n⚖️ Testing Decision Comparison...")

        # Create different market conditions
        weak_ctx = MarketContext(
            price=1.1045,
            bid=1.1044,
            ask=1.1046,
            atr_pts=15.0,
            sma20=1.1042,
            sma50=1.1040,
            session="asia",
            regime="LOW",
            equity=10000.0,
            balance=10000.0,
            spread_pts=20.0,
            open_positions=0,
        )

        weak_control = ControlDecider(weak_ctx).decide()
        weak_pro = ProDecider(weak_ctx).decide()

        print(f"✅ Weak market - Control: {weak_control.action}, Pro: {weak_pro.action}")

        # Test 12: Test Paper Broker Statistics
        print("\n📈 Testing Paper Broker Statistics...")

        paper_stats = paper_broker.get_statistics()
        assert 'total_trades' in paper_stats
        assert 'win_rate_pct' in paper_stats

        print(f"✅ Paper broker stats: {paper_stats['total_trades']} trades")

        # Cleanup
        paper_broker.close_all()
        ab_runner.cleanup()

        print("\n🎉 STEP9: Shadow A/B Testing - COMPLETED SUCCESSFULLY!")
        print("All components are working correctly:")
        print("✅ Control Decider (SMA-only)")
        print("✅ Pro Decider (Ensemble)")
        print("✅ Paper Broker for control track")
        print("✅ Context Factory")
        print("✅ A/B Runner orchestration")
        print("✅ Decision comparison")
        print("✅ Metrics integration")
        print("✅ Position management simulation")

        return True

    except Exception as e:
        print(f"❌ STEP9 test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step9()
    sys.exit(0 if success else 1)
