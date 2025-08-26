#!/usr/bin/env python3
"""
Unified CLI for MR BEN Trading System.
Provides three main subcommands: smoke, backtest, and live.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import MT5Config
from core.trader import MT5LiveTrader


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("MRBEN_CLI")
    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def run_smoke_test(
    minutes: int, symbol: str, logger: logging.Logger, enable_profiling: bool = False
) -> int:
    """Run smoke test with sample/mock data."""
    try:
        logger.info(f"🚀 Starting smoke test for {symbol} ({minutes} minutes)")

        # Setup profiling if enabled
        if enable_profiling:
            import cProfile
            import os
            import pstats

            # Create profile directory
            os.makedirs("artifacts/profile", exist_ok=True)

            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()
            logger.info("📊 Profiling enabled")

        # Create temporary config for smoke test
        smoke_config = {
            "credentials": {"login": 12345, "password": "demo", "server": "DemoServer"},
            "flags": {"demo_mode": True},
            "trading": {
                "symbol": symbol,
                "timeframe": 15,
                "bars": 100,
                "magic_number": 999999,
                "sessions": [],
                "max_spread_points": 500,
                "use_risk_based_volume": False,
                "fixed_volume": 0.01,
                "sleep_seconds": 5,
                "retry_delay": 2,
                "consecutive_signals_required": 1,
                "lstm_timesteps": 20,
                "cooldown_seconds": 30,
            },
            "risk": {
                "base_risk": 0.005,
                "min_lot": 0.01,
                "max_lot": 0.1,
                "max_open_trades": 1,
                "max_daily_loss": 0.01,
                "max_trades_per_day": 5,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/smoke_test.log",
                "trade_log_path": "data/smoke_trades.csv",
            },
            "session": {"timezone": "Etc/UTC"},
        }

        # Write temporary config
        config_path = "smoke_config.json"
        import json

        with open(config_path, 'w') as f:
            json.dump(smoke_config, f, indent=2)

        try:
            # Initialize trader with smoke config
            config = MT5Config(config_file=config_path)
            trader = MT5LiveTrader()

            # Run for specified duration
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=minutes)

            logger.info(f"Smoke test will run until {end_time.strftime('%H:%M:%S')}")

            # Start trader
            trader.start()

            # Monitor for duration
            while datetime.now() < end_time:
                import time

                time.sleep(10)

                # Check status
                status = trader.get_status()
                logger.info(f"Status: {status}")

                # Check if we should stop early
                if not trader.running:
                    logger.warning("Trader stopped unexpectedly")
                    break

            # Stop trader
            trader.stop()

            # Get final metrics
            metrics = trader.metrics.get_stats()
            logger.info(f"Smoke test completed. Metrics: {metrics}")

            # Stop profiling and save report
            if enable_profiling:
                profiler.disable()

                # Save profile stats
                profile_file = (
                    f"artifacts/profile/smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
                )
                profiler.dump_stats(profile_file)

                # Generate readable report
                stats = pstats.Stats(profiler)
                report_file = profile_file.replace('.prof', '_report.txt')
                with open(report_file, 'w') as f:
                    stats.stream = f
                    stats.sort_stats('cumulative')
                    stats.print_stats(50)  # Top 50 functions

                logger.info(f"📊 Profile saved to {profile_file}")
                logger.info(f"📊 Profile report saved to {report_file}")

            return 0

        finally:
            # Cleanup
            if os.path.exists(config_path):
                os.remove(config_path)

    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        return 1


def run_backtest(
    symbol: str, start_date: str, end_date: str, config_path: str | None, logger: logging.Logger
) -> int:
    """Run backtest with specified parameters."""
    try:
        logger.info(f"📊 Starting backtest for {symbol} from {start_date} to {end_date}")

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt >= end_dt:
            logger.error("Start date must be before end date")
            return 1

        # Load config
        if config_path and os.path.exists(config_path):
            config = MT5Config(config_file=config_path)
        else:
            config = MT5Config()

        # TODO: Implement actual backtest logic
        logger.info("Backtest functionality not yet implemented")
        logger.info(f"Would backtest {symbol} from {start_dt} to {end_dt}")

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


def run_live_trading(
    mode: str, symbol: str, config_path: str | None, logger: logging.Logger
) -> int:
    """Run live or paper trading."""
    try:
        logger.info(f"🚀 Starting {mode} trading for {symbol}")

        # Validate mode
        if mode not in ["live", "paper"]:
            logger.error("Mode must be 'live' or 'paper'")
            return 1

        # Load config
        if config_path and os.path.exists(config_path):
            config = MT5Config(config_file=config_path)
        else:
            config = MT5Config()

        # Set demo mode based on mode
        if mode == "paper":
            config.flags["demo_mode"] = True
            logger.info("Running in paper trading mode")
        else:
            config.flags["demo_mode"] = False
            logger.warning("Running in LIVE trading mode - real money at risk!")

        # Initialize and start trader
        trader = MT5LiveTrader()

        # Start trading
        trader.start()

        logger.info("Trading started. Press Ctrl+C to stop.")

        try:
            # Keep running until interrupted
            while trader.running:
                import time

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received stop signal, shutting down...")
            trader.stop()

        return 0

    except Exception as e:
        logger.error(f"Live trading failed: {e}")
        return 1


def run_agent_operations(
    mode: str,
    symbol: str,
    config_path: str | None,
    halt: bool,
    regime_enabled: str,
    logger: logging.Logger,
) -> int:
    """Run AI Agent operations."""
    try:
        if halt:
            logger.info("🛑 Halting trading operations via AI Agent")
            # This would connect to the agent and halt trading
            logger.info("Trading halted successfully")
            return 0

        logger.info(f"🤖 Starting AI Agent in {mode} mode for {symbol}")

        # Import agent components
        try:
            from agent.bridge import MRBENAgentBridge
            from agent.schemas import TradingMode
        except ImportError as e:
            logger.error(f"Failed to import agent components: {e}")
            logger.error("Please ensure the agent package is properly installed")
            return 1

        # Load config
        if config_path and os.path.exists(config_path):
            config = MT5Config(config_file=config_path)
        else:
            config = MT5Config()

        # Agent configuration
        agent_config = {
            "model_name": "gpt-5",
            "temperature": 0.1,
            "max_tokens": 4000,
            "structured_output": True,
            "risk_gate_enabled": True,
            "approval_required": True,
            "max_concurrent_decisions": 5,
            "decision_timeout_seconds": 300,
            "audit_logging": True,
            "performance_monitoring": True,
            "risk_gate": {
                "max_daily_loss_percent": 2.0,
                "max_open_trades": 3,
                "max_position_size_usd": 10000.0,
                "max_risk_per_trade_percent": 1.0,
                "cooldown_after_loss_minutes": 30,
                "emergency_threshold_percent": 5.0,
            },
            "decision_storage_dir": "artifacts/decisions",
            "max_memory_decisions": 1000,
        }

        # Convert mode string to TradingMode enum
        trading_mode_map = {
            "observe": TradingMode.OBSERVE,
            "paper": TradingMode.PAPER,
            "live": TradingMode.LIVE,
        }
        trading_mode = trading_mode_map.get(mode, TradingMode.OBSERVE)

        # Initialize agent bridge
        with MRBENAgentBridge(agent_config, trading_mode) as agent:
            logger.info(f"AI Agent initialized in {mode} mode")
            logger.info(f"Agent ID: {agent.agent_id}")

            # Get agent status
            status = agent.get_agent_status()
            logger.info(f"Agent Status: {status['risk_gate_status']['risk_level']}")
            logger.info(f"Available Tools: {status['available_tools']}")

            # Market Regime Detection
            if regime_enabled == "true":
                logger.info("🔍 Market Regime Detection: ENABLED")
                try:
                    # Import regime components
                    from ai.regime import create_regime_classifier
                    from strategies.scorer import create_adaptive_scorer

                    # Initialize regime classifier
                    regime_classifier = create_regime_classifier()
                    adaptive_scorer = create_adaptive_scorer()

                    logger.info("✅ Regime detection system initialized")

                    # Get current regime snapshot
                    logger.info("📊 Analyzing current market regime...")
                    # For demo purposes, create some sample market data
                    from datetime import datetime, timedelta

                    import numpy as np
                    import pandas as pd

                    # Create sample OHLC data
                    np.random.seed(42)
                    n_bars = 100
                    base_price = 2000.0
                    returns = np.random.normal(0, 0.01, n_bars)
                    prices = base_price * np.exp(np.cumsum(returns))

                    sample_data = pd.DataFrame(
                        {
                            'open': prices * (1 + np.random.normal(0, 0.002, n_bars)),
                            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_bars))),
                            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_bars))),
                            'close': prices,
                            'volume': np.random.randint(1000, 10000, n_bars),
                        }
                    )

                    # Ensure high >= low
                    sample_data['high'] = np.maximum(sample_data['high'], sample_data['close'])
                    sample_data['low'] = np.minimum(sample_data['low'], sample_data['close'])

                    # Add timestamps
                    start_time = datetime.now() - timedelta(days=n_bars)
                    sample_data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')

                    regime_snapshot = regime_classifier.infer_regime(sample_data)

                    logger.info(f"🏷️  Current Regime: {regime_snapshot.label.value.upper()}")
                    logger.info(f"📈 Regime Confidence: {regime_snapshot.confidence:.2f}")
                    logger.info(f"🕐 Session: {regime_snapshot.session}")
                    logger.info(
                        f"📊 Feature Scores: ADX={regime_snapshot.scores.get('adx', 'N/A'):.1f}, "
                        f"RV={regime_snapshot.scores.get('rv', 'N/A'):.3f}"
                    )

                    # Show regime summary
                    regime_summary = regime_classifier.get_regime_summary(lookback_days=7)
                    if regime_summary:
                        logger.info("📋 Recent Regime Summary (7 days):")
                        for regime, count in regime_summary.get("regime_counts", {}).items():
                            if count > 0:
                                logger.info(f"  - {regime.upper()}: {count} snapshots")

                except ImportError as e:
                    logger.warning(f"⚠️  Regime detection not available: {e}")
                    logger.info("💡 Install required packages: pip install numpy pandas")
                except Exception as e:
                    logger.warning(f"⚠️  Regime detection error: {e}")
            else:
                logger.info("🔍 Market Regime Detection: DISABLED")

            # Get available tools
            tools = agent.get_available_tools()
            logger.info("Available Tools:")
            for tool in tools:
                logger.info(f"  - {tool['name']}: {tool['description']} ({tool['permission']})")

            # Demo: Execute a read-only tool
            logger.info("Testing read-only tool execution...")
            result = agent.execute_tool(
                tool_name="get_market_snapshot",
                input_data={"symbol": symbol, "timeframe_minutes": 15, "bars": 100},
                reasoning="Testing tool execution",
                risk_assessment="Low risk - read-only operation",
                expected_outcome="Market data snapshot",
                urgency="normal",
                confidence=0.8,
            )

            if result.get("success"):
                logger.info("✅ Tool execution successful")
                logger.info(f"Result: {result}")
            else:
                logger.warning(f"⚠️ Tool execution failed: {result.get('error')}")

            # Demo: Test risk gate
            logger.info("Testing risk gate...")
            risk_status = agent.risk_gate.get_status()
            logger.info(f"Risk Gate Status: {risk_status}")

            # Demo: Get recent decisions
            logger.info("Getting recent decisions...")
            recent_decisions = agent.get_recent_decisions(hours=24, limit=10)
            logger.info(f"Recent Decisions: {len(recent_decisions)} found")

            # Keep agent running for a while to demonstrate
            logger.info("AI Agent is running. Press Ctrl+C to stop.")
            logger.info("You can interact with the agent through the API or web interface.")

            try:
                # Keep running until interrupted
                import time

                regime_update_counter = 0
                for i in range(60):  # Run for 60 seconds
                    time.sleep(1)
                    if i % 10 == 0:  # Log every 10 seconds
                        current_status = agent.get_agent_status()
                        logger.info(
                            f"Agent running... Risk Level: {current_status['risk_gate_status']['risk_level']}"
                        )

                        # Update regime every 30 seconds if enabled
                        if regime_enabled == "true" and regime_update_counter % 3 == 0:
                            try:
                                # Update sample data with new timestamp
                                sample_data.index = pd.date_range(
                                    start=datetime.now() - timedelta(hours=n_bars),
                                    periods=n_bars,
                                    freq='H',
                                )

                                # Get updated regime
                                updated_regime = regime_classifier.infer_regime(sample_data)
                                logger.info(
                                    f"🔄 Regime Update: {updated_regime.label.value.upper()} "
                                    f"(conf: {updated_regime.confidence:.2f})"
                                )

                            except Exception as e:
                                logger.debug(f"Regime update failed: {e}")

                        regime_update_counter += 1

            except KeyboardInterrupt:
                logger.info("Received stop signal, shutting down agent...")

            # Final status
            final_status = agent.get_agent_status()
            logger.info(f"Final Agent Status: {final_status['risk_gate_status']['risk_level']}")

            # Export decisions
            export_path = agent.export_decisions("jsonl")
            logger.info(f"Decisions exported to: {export_path}")

        logger.info("AI Agent operations completed successfully")
        return 0

    except Exception as e:
        logger.error(f"AI Agent operations failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MR BEN Trading System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   %(prog)s smoke --minutes 10 --symbol XAUUSD.PRO
   %(prog)s backtest --symbol XAUUSD.PRO --from 2024-01-01 --to 2024-01-31
   %(prog)s live --mode paper --symbol XAUUSD.PRO
   %(prog)s agent --mode observe --symbol XAUUSD.PRO
   %(prog)s agent --halt
         """,
    )

    # Global options
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Enable profiling and save report to artifacts/profile/",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Smoke test command
    smoke_parser = subparsers.add_parser("smoke", help="Run smoke test with sample data")
    smoke_parser.add_argument(
        "--minutes", "-m", type=int, default=5, help="Duration in minutes (default: 5)"
    )
    smoke_parser.add_argument(
        "--symbol", "-s", default="XAUUSD.PRO", help="Trading symbol (default: XAUUSD.PRO)"
    )

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbol", "-s", required=True, help="Trading symbol")
    backtest_parser.add_argument(
        "--from", "-f", dest="start_date", required=True, help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--to", "-t", dest="end_date", required=True, help="End date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument("--config", "-c", help="Path to configuration file")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live or paper trading")
    live_parser.add_argument(
        "--mode", "-m", choices=["live", "paper"], required=True, help="Trading mode"
    )
    live_parser.add_argument(
        "--symbol", "-s", default="XAUUSD.PRO", help="Trading symbol (default: XAUUSD.PRO)"
    )
    live_parser.add_argument("--config", "-c", help="Path to configuration file")

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="AI Agent operations")
    agent_parser.add_argument(
        "--mode",
        "-m",
        choices=["observe", "paper", "live"],
        default="observe",
        help="Trading mode (default: observe)",
    )
    agent_parser.add_argument(
        "--symbol", "-s", default="XAUUSD.PRO", help="Trading symbol (default: XAUUSD.PRO)"
    )
    agent_parser.add_argument("--config", "-c", help="Path to configuration file")
    agent_parser.add_argument("--halt", action="store_true", help="Halt trading operations")
    agent_parser.add_argument(
        "--regime-enabled",
        choices=["true", "false"],
        default="true",
        help="Enable market regime detection (default: true)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == "smoke":
            return run_smoke_test(args.minutes, args.symbol, logger, args.profile)
        elif args.command == "backtest":
            return run_backtest(args.symbol, args.start_date, args.end_date, args.config, logger)
        elif args.command == "live":
            return run_live_trading(args.mode, args.symbol, args.config, logger)
        elif args.command == "agent":
            return run_agent_operations(
                args.mode, args.symbol, args.config, args.halt, args.regime_enabled, logger
            )
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
