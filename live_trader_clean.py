#!/usr/bin/env python3
"""
MR BEN Live Trading System - Production-Grade EntryPoint
Unified orchestration with MTF RSI/MACD + Fusion + ATR SL/TP + logging/metrics/reports
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
from src.core.metrics import PerformanceMetrics

# Indicator imports
from src.indicators.atr import compute_atr
from src.indicators.rsi_macd import compute_macd, compute_rsi

# Risk management imports
from src.risk_manager.atr_sl_tp import SLTPResult, calc_sltp_from_atr

# Signal imports
from src.signals.multi_tf_rsi_macd import analyze_multi_tf_rsi_macd
from src.utils.error_handler import error_handler

# Strategy imports

# Data imports
try:
    from src.data.manager import MT5DataManager

    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False

# AI model imports
try:
    import joblib

    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False

# Price Action imports
try:
    from src.strategy.pa import PriceActionValidator

    PRICE_ACTION_AVAILABLE = True
except ImportError:
    PRICE_ACTION_AVAILABLE = False


@dataclass
class AdaptiveResult:
    """Result from fusion scoring"""

    score_buy: float
    score_sell: float
    label: str
    confidence: float


@dataclass
class TradeRecord:
    """Trade execution record"""

    side: str
    entry: float
    sl: float
    tp: float
    timestamp: datetime
    result: str
    used_fallback: bool
    extras: dict[str, Any]


class LiveTraderApp:
    """Production-grade live trading application with full supervision"""

    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace, logger: logging.Logger):
        self.cfg = cfg
        self.args = args
        self.logger = logger

        # Initialize metrics
        self.metrics = PerformanceMetrics()

        # Initialize data manager
        self.data_manager: MT5DataManager | None = None
        if DATA_MANAGER_AVAILABLE:
            try:
                # Convert timeframe string to minutes
                tf_min = self._timeframe_to_minutes(self.args.timeframe)
                self.data_manager = MT5DataManager(self.args.symbol, tf_min)
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize MT5DataManager: {e}, using fallback adapter"
                )
                self.data_manager = None
        else:
            self.logger.warning("MT5DataManager not available, using fallback adapter")

        # Load AI model
        self.ai_model = None
        if AI_MODEL_AVAILABLE:
            try:
                model_path = "mrben_ai_signal_filter_xgb.joblib"
                if os.path.exists(model_path):
                    self.ai_model = joblib.load(model_path)
                    self.logger.info("AI model loaded successfully")
                else:
                    self.logger.warning("AI model file not found, using fallback")
            except Exception as e:
                self.logger.warning(f"Failed to load AI model: {e}, using fallback")

        # Cache configuration parameters
        self.risk_params = cfg.get('risk', {})
        self.mtf_params = cfg.get('multi_tf', {})
        self.fusion_params = cfg.get('fusion', {})
        self.logging_params = cfg.get('logging', {})

        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

        # Initialize logging handlers if configured
        self._setup_logging_handlers()

        self.logger.info(f"LiveTraderApp initialized for {args.symbol} in {args.mode} mode")

        # Initialize start time
        self.start_time = datetime.now()

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        tf_map = {
            'M1': 1,
            'M2': 2,
            'M3': 3,
            'M4': 4,
            'M5': 5,
            'M10': 10,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H2': 120,
            'H3': 180,
            'H4': 240,
            'D1': 1440,
            'W1': 10080,
            'MN1': 43200,
        }
        return tf_map.get(timeframe, 15)  # Default to M15

    def _setup_logging_handlers(self) -> None:
        """Setup additional logging handlers if configured"""
        if self.logging_params.get('enable_csv'):
            try:
                csv_path = self.logging_params.get('csv_path', 'logs/trades.csv')
                csv_handler = logging.FileHandler(csv_path)
                csv_handler.setLevel(logging.INFO)
                self.logger.addHandler(csv_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup CSV logging: {e}")

        if self.logging_params.get('enable_sqlite'):
            try:
                # SQLite logging would be implemented here
                self.logger.info("SQLite logging configured")
            except Exception as e:
                self.logger.warning(f"Failed to setup SQLite logging: {e}")

    def _fetch_multi_tf_ohlc(
        self, _symbol: str, timeframes: list[str], bars: int
    ) -> dict[str, Any]:
        """Fetch OHLC data for multiple timeframes"""
        dfs = {}

        for tf in timeframes:
            try:
                if self.data_manager and hasattr(self.data_manager, 'get_latest_data'):
                    # Use MT5DataManager if available
                    try:
                        df = self.data_manager.get_latest_data(bars)
                    except Exception as e:
                        self.logger.warning(
                            f"MT5DataManager.get_latest_data failed: {e}, using fallback"
                        )
                        df = self._create_dummy_ohlc(bars)
                else:
                    # Fallback adapter - create dummy data for testing
                    df = self._create_dummy_ohlc(bars)

                if df is not None and len(df) >= bars:
                    dfs[tf] = df
                else:
                    self.logger.warning(
                        f"Insufficient data for {tf}: {len(df) if df is not None else 0} bars"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {tf}: {e}")

        return dfs

    def _create_dummy_ohlc(self, bars: int) -> Any:
        """Create dummy OHLC data for testing when data manager is unavailable"""
        import numpy as np
        import pandas as pd

        dates = pd.date_range(end=datetime.now(), periods=bars, freq='15T')
        base_price = 2000.0

        # Generate realistic price movements
        np.random.seed(42)  # For reproducible testing
        changes = np.random.normal(0, 0.001, bars)
        prices = [base_price]

        for change in changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices, strict=False)):
            volatility = np.random.uniform(0.0005, 0.002)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i - 1] if i > 0 else close

            data.append(
                {
                    'time': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': np.random.randint(100, 1000),
                }
            )

        return pd.DataFrame(data)

    def _ta_decisions(self, dfs: dict[str, Any]) -> Mapping[str, str]:
        """Get technical analysis decisions for multiple timeframes"""
        try:
            return analyze_multi_tf_rsi_macd(
                dfs,
                rsi_period=self.mtf_params.get('rsi_period', 14),
                macd_fast=self.mtf_params.get('macd_fast', 12),
                macd_slow=self.mtf_params.get('macd_slow', 26),
                macd_signal=self.mtf_params.get('macd_signal', 9),
                rsi_overbought=self.mtf_params.get('rsi_overbought', 70),
                rsi_oversold=self.mtf_params.get('rsi_oversold', 30),
            )
        except Exception as e:
            self.logger.error(f"Failed to analyze multi-timeframe RSI/MACD: {e}")
            return {}

    def _pa_signal(self, _df_primary: Any) -> str:
        """Get price action signal"""
        if not PRICE_ACTION_AVAILABLE:
            return "neutral"

        try:
            # This is a placeholder - implement actual price action logic
            # For now, return neutral to avoid errors
            return "neutral"
        except Exception as e:
            self.logger.error(f"Failed to get price action signal: {e}")
            return "neutral"

    def _ai_score(self, features: list[float]) -> float:
        """Get AI model score for features"""
        if self.ai_model is None:
            return 0.5  # Fallback score

        try:
            # Ensure features are in expected format
            if len(features) != 10:
                # Pad or truncate to expected length
                if len(features) < 10:
                    features = features + [0.0] * (10 - len(features))
                else:
                    features = features[:10]

            # Get prediction from model
            score = self.ai_model.predict_proba([features])[0]
            return float(score[1])  # Probability of positive class
        except Exception as e:
            self.logger.error(f"Failed to get AI score: {e}")
            return 0.5  # Fallback score

    def _fuse(self, ta_map: Mapping[str, str], pa: str, ai: float) -> AdaptiveResult:
        """Fuse technical analysis, price action, and AI scores"""
        try:
            # Convert ta_map to scores
            ta_buy_count = sum(1 for v in ta_map.values() if v == "buy")
            ta_sell_count = sum(1 for v in ta_map.values() if v == "sell")
            ta_total = len(ta_map)

            ta_buy_score = ta_buy_count / ta_total if ta_total > 0 else 0.5
            ta_sell_score = ta_sell_count / ta_total if ta_total > 0 else 0.5

            # Convert price action to scores
            pa_buy_score = 1.0 if pa == "buy" else 0.0
            pa_sell_score = 1.0 if pa == "sell" else 0.0

            # Get fusion weights
            w_ta = self.fusion_params.get('w_ta', 0.4)
            w_pa = self.fusion_params.get('w_pa', 0.3)
            w_ai = self.fusion_params.get('w_ai', 0.3)

            # Fuse scores
            buy_score = w_ta * ta_buy_score + w_pa * pa_buy_score + w_ai * ai
            sell_score = w_ta * ta_sell_score + w_pa * pa_sell_score + w_ai * (1 - ai)

            # Determine label and confidence
            threshold_buy = self.fusion_params.get('threshold_buy', 0.6)
            threshold_sell = self.fusion_params.get('threshold_sell', 0.6)

            if buy_score >= threshold_buy:
                label = "buy"
                confidence = buy_score
            elif sell_score >= threshold_sell:
                label = "sell"
                confidence = sell_score
            else:
                label = "neutral"
                confidence = max(buy_score, sell_score)

            return AdaptiveResult(
                score_buy=buy_score, score_sell=sell_score, label=label, confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Failed to fuse signals: {e}")
            return AdaptiveResult(score_buy=0.5, score_sell=0.5, label="neutral", confidence=0.5)

    def _atr_sltp(self, side: str, entry: float, ohlc_primary: Any) -> SLTPResult:
        """Calculate ATR-based stop loss and take profit"""
        try:
            # Compute ATR
            atr_value = compute_atr(ohlc_primary, self.risk_params.get('atr_period', 14))

            if atr_value is None or atr_value <= 0:
                self.logger.warning("ATR computation failed, using fallback SL/TP")
                return calc_sltp_from_atr(
                    side=side,  # type: ignore[arg-type]
                    entry_price=entry,
                    atr_value=entry * 0.01,  # 1% fallback
                    rr=self.risk_params.get('rr', 1.5),
                    sl_k=self.risk_params.get('sl_k', 1.0),
                    tp_k=self.risk_params.get('tp_k', 1.5),
                    fallback_sl_pct=self.risk_params.get('fallback_sl_pct', 0.005),
                    fallback_tp_pct=self.risk_params.get('fallback_tp_pct', 0.0075),
                )

            # Use ATR-based calculation
            result = calc_sltp_from_atr(
                side=side,  # type: ignore[arg-type]
                entry_price=entry,
                atr_value=atr_value,
                rr=self.risk_params.get('rr', 1.5),
                sl_k=self.risk_params.get('sl_k', 1.0),
                tp_k=self.risk_params.get('tp_k', 1.5),
                fallback_sl_pct=self.risk_params.get('fallback_sl_pct', 0.005),
                fallback_tp_pct=self.risk_params.get('fallback_tp_pct', 0.0075),
            )

            # Log SL/TP decision
            sltp_logger = logging.getLogger("core.trade.sltp")
            sltp_logger.info(
                f"SL/TP Decision: side={side}, entry={entry:.2f}, ATR={atr_value:.2f}, "
                f"rr={self.risk_params.get('rr', 1.5)}, sl_k={self.risk_params.get('sl_k', 1.0)}, "
                f"tp_k={self.risk_params.get('tp_k', 1.5)}, fallback_sl_pct={self.risk_params.get('fallback_sl_pct', 0.005)}, "
                f"fallback_tp_pct={self.risk_params.get('fallback_tp_pct', 0.0075)}, "
                f"SL={result.sl:.2f}, TP={result.tp:.2f}, used_fallback={result.used_fallback}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to calculate ATR SL/TP: {e}")
            # Return fallback values
            if side == "buy":
                return SLTPResult(
                    sl=entry * (1 - self.risk_params.get('fallback_sl_pct', 0.005)),
                    tp=entry * (1 + self.risk_params.get('fallback_tp_pct', 0.0075)),
                    used_fallback=True,
                )
            else:
                return SLTPResult(
                    sl=entry * (1 + self.risk_params.get('fallback_sl_pct', 0.005)),
                    tp=entry * (1 - self.risk_params.get('fallback_tp_pct', 0.0075)),
                    used_fallback=True,
                )

    def _place_order(self, side: str, sl: float, tp: float, mode: str) -> bool:
        """Place trading order based on mode"""
        try:
            if mode in ["simulate", "demo"]:
                self.logger.info(
                    f"SIMULATED ORDER: {side.upper()} {self.args.symbol} @ {sl:.2f} SL, {tp:.2f} TP"
                )
                return True
            elif mode == "live":
                # TODO: Implement actual order placement
                self.logger.warning("Live order placement not implemented yet")
                return False
            else:
                self.logger.error(f"Unknown trading mode: {mode}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return False

    def _record_trade(
        self,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        result: str,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Record trade for logging/reporting"""
        if extras is None:
            extras = {}

        _trade_record = TradeRecord(
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            timestamp=datetime.now(),
            result=result,
            used_fallback=extras.get('used_fallback', False),
            extras=extras,
        )

        # Log trade
        self.logger.info(
            f"TRADE RECORDED: {side.upper()} {self.args.symbol} @ {entry:.2f}, "
            f"SL: {sl:.2f}, TP: {tp:.2f}, Result: {result}"
        )

        # TODO: Save to CSV/SQLite if configured

    def _mini_report(self) -> None:
        """Print compact status report"""
        print("\n" + "=" * 60)
        print(f"MR BEN STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Symbol: {self.args.symbol} | Mode: {self.args.mode}")
        print(f"Cycles: {self.cycle_count} | Trades: {self.metrics.trade_count}")
        print(f"Errors: {self.metrics.error_count}")

        # Multi-timeframe summary
        if hasattr(self, 'last_ta_decisions'):
            print(f"MTF Consensus: {self.last_ta_decisions}")

        # Fusion scores
        if hasattr(self, 'last_fusion'):
            print(
                f"Fusion: Buy={self.last_fusion.score_buy:.3f}, Sell={self.last_fusion.score_sell:.3f}"
            )
            print(f"Decision: {self.last_fusion.label} (conf: {self.last_fusion.confidence:.3f})")

        # Risk summary
        fallback_count = getattr(self, 'fallback_count', 0)
        print(f"ATR Fallbacks Used: {fallback_count}")
        print("=" * 60)

    def run_loop(self) -> None:
        """Main trading loop"""
        self.cycle_count = 0
        self.fallback_count = 0

        try:
            while True:
                if self.args.max_cycles > 0 and self.cycle_count >= self.args.max_cycles:
                    self.logger.info(f"Reached max cycles ({self.args.max_cycles}), stopping")
                    break

                with error_handler("main-cycle", self.logger, self.metrics):
                    self.cycle_count += 1

                    # Fetch multi-timeframe data
                    timeframes = self.mtf_params.get('timeframes', ['M5', 'M15', 'H1'])
                    dfs = self._fetch_multi_tf_ohlc(self.args.symbol, timeframes, self.args.bars)

                    if not dfs:
                        self.logger.warning("No data available, skipping cycle")
                        continue

                    # Get primary timeframe data
                    primary_tf = self.args.timeframe
                    df_primary = dfs.get(primary_tf)
                    if df_primary is None:
                        self.logger.warning(f"Primary timeframe {primary_tf} data not available")
                        continue

                    # Technical analysis decisions
                    ta_map = self._ta_decisions(dfs)
                    self.last_ta_decisions = ta_map

                    # Check multi-timeframe consensus
                    min_agreement = self.mtf_params.get('min_agreement', 2)
                    buy_count = sum(1 for v in ta_map.values() if v == "buy")
                    sell_count = sum(1 for v in ta_map.values() if v == "sell")

                    if buy_count < min_agreement and sell_count < min_agreement:
                        self.logger.info(
                            f"Insufficient consensus: buy={buy_count}, sell={sell_count}, min={min_agreement}"
                        )
                        continue

                    # Price action signal
                    pa = self._pa_signal(df_primary)

                    # AI score (extract features from primary timeframe)
                    features = self._extract_features(df_primary)
                    ai_score = self._ai_score(features)

                    # Fuse all signals
                    fusion = self._fuse(ta_map, pa, ai_score)
                    self.last_fusion = fusion

                    # Execute trade if signal is clear
                    if fusion.label in ["buy", "sell"] and fusion.confidence >= 0.6:
                        side = fusion.label
                        entry = df_primary['close'].iloc[-1]

                        # Calculate SL/TP
                        sltp = self._atr_sltp(side, entry, df_primary)
                        if sltp.used_fallback:
                            self.fallback_count += 1

                        # Place order
                        ok = self._place_order(side, sltp.sl, sltp.tp, self.args.mode)
                        if ok:
                            self.metrics.record_trade()
                            self._record_trade(
                                side,
                                entry,
                                sltp.sl,
                                sltp.tp,
                                "placed",
                                {
                                    'used_fallback': sltp.used_fallback,
                                    'confidence': fusion.confidence,
                                },
                            )

                    # Print mini report periodically
                    if self.cycle_count % self.args.report_every == 0:
                        self._mini_report()

                    # Sleep between cycles
                    time.sleep(self.cfg.get('trading', {}).get('sleep_seconds', 12))

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping gracefully")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            raise
        finally:
            self.logger.info("Trading loop ended")

    def _extract_features(self, df: Any) -> list[float]:
        """Extract features for AI model from OHLC data"""
        try:
            # Simple feature extraction - adjust based on your model's expectations
            features = []

            # Price-based features
            features.append(float(df['close'].iloc[-1] / df['close'].iloc[-2] - 1))  # Price change
            features.append(float(df['high'].iloc[-1] / df['low'].iloc[-1] - 1))  # High-low ratio

            # Volume-based features (if available)
            if 'volume' in df.columns:
                features.append(float(df['volume'].iloc[-1] / df['volume'].iloc[-5:].mean() - 1))
            else:
                features.append(0.0)

            # Technical indicators
            rsi = compute_rsi(df, 14)
            features.append(float(rsi.iloc[-1] / 100))  # Normalized RSI

            macd_line, signal_line, hist = compute_macd(df)
            features.append(float(macd_line.iloc[-1] / df['close'].iloc[-1]))  # Normalized MACD
            features.append(float(signal_line.iloc[-1] / df['close'].iloc[-1]))  # Normalized Signal

            # ATR-based features
            atr = compute_atr(df, 14)
            if atr is not None:
                features.append(float(atr.iloc[-1] / df['close'].iloc[-1]))  # Normalized ATR
            else:
                features.append(0.0)

            # Time-based features
            features.append(float(datetime.now().hour / 24))  # Hour of day
            features.append(float(datetime.now().weekday() / 7))  # Day of week

            # Ensure we have exactly 10 features
            while len(features) < 10:
                features.append(0.0)
            features = features[:10]

            return features

        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return [0.0] * 10  # Return default features

    def finalize_report(self) -> None:
        """Generate final run report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON report
            report_data = {
                "run_id": f"run_{timestamp}",
                "symbol": self.args.symbol,
                "mode": self.args.mode,
                "metrics": {
                    "trade_count": self.metrics.trade_count,
                    "win_count": 0,  # Not implemented yet
                    "loss_count": 0,  # Not implemented yet
                    "error_count": self.metrics.error_count,
                    "last_error": "Not implemented",
                },
                "multi_tf": {
                    "timeframes": self.mtf_params.get('timeframes', []),
                    "last_decisions": getattr(self, 'last_ta_decisions', {}),
                    "min_agreement": self.mtf_params.get('min_agreement', 2),
                },
                "fusion": {
                    "score_buy": getattr(
                        self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)
                    ).score_buy,
                    "score_sell": getattr(
                        self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)
                    ).score_sell,
                    "label": getattr(self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)).label,
                },
                "risk": {
                    "atr_period": self.risk_params.get('atr_period', 14),
                    "rr": self.risk_params.get('rr', 1.5),
                    "sl_k": self.risk_params.get('sl_k', 1.0),
                    "tp_k": self.risk_params.get('tp_k', 1.5),
                    "fallbacks_used": getattr(self, 'fallback_count', 0),
                },
                "last_order": {
                    "side": getattr(self, 'last_order_side', "none"),
                    "entry": getattr(self, 'last_order_entry', 0.0),
                    "sl": getattr(self, 'last_order_sl', 0.0),
                    "tp": getattr(self, 'last_order_tp', 0.0),
                    "used_fallback": getattr(self, 'last_order_fallback', False),
                },
                "timestamps": {
                    "started": getattr(self, 'start_time', datetime.now()).isoformat(),
                    "ended": datetime.now().isoformat(),
                },
            }

            # Save JSON report
            json_path = f"logs/run_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Generate markdown summary
            md_path = f"logs/run_report_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write("# MR BEN Trading Run Report\n\n")
                f.write(f"**Run ID**: {report_data['run_id']}\n")
                f.write(f"**Symbol**: {report_data['symbol']}\n")
                f.write(f"**Mode**: {report_data['mode']}\n")
                f.write(
                    f"**Duration**: {report_data['timestamps']['started']} to {report_data['timestamps']['ended']}\n\n"
                )

                f.write("## Performance Metrics\n")
                f.write(f"- **Total Trades**: {report_data['metrics']['trade_count']}\n")
                f.write(f"- **Wins**: {report_data['metrics']['win_count']}\n")
                f.write(f"- **Losses**: {report_data['metrics']['loss_count']}\n")
                f.write(f"- **Errors**: {report_data['metrics']['error_count']}\n")
                if report_data['metrics']['last_error']:
                    f.write(f"- **Last Error**: {report_data['metrics']['last_error']}\n")
                f.write("\n")

                f.write("## Multi-Timeframe Analysis\n")
                f.write(f"- **Timeframes**: {', '.join(report_data['multi_tf']['timeframes'])}\n")
                f.write(f"- **Min Agreement**: {report_data['multi_tf']['min_agreement']}\n")
                f.write(f"- **Last Decisions**: {report_data['multi_tf']['last_decisions']}\n\n")

                f.write("## Fusion Scoring\n")
                f.write(f"- **Buy Score**: {report_data['fusion']['score_buy']:.3f}\n")
                f.write(f"- **Sell Score**: {report_data['fusion']['score_sell']:.3f}\n")
                f.write(f"- **Final Decision**: {report_data['fusion']['label']}\n\n")

                f.write("## Risk Management\n")
                f.write(f"- **ATR Period**: {report_data['risk']['atr_period']}\n")
                f.write(f"- **Risk/Reward**: {report_data['risk']['rr']}\n")
                f.write(f"- **SL Multiplier**: {report_data['risk']['sl_k']}\n")
                f.write(f"- **TP Multiplier**: {report_data['risk']['tp_k']}\n")
                f.write(f"- **Fallbacks Used**: {report_data['risk']['fallbacks_used']}\n")

            self.logger.info(f"Final report saved to {json_path} and {md_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(
        description="MR BEN — Production-Grade Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core arguments
    p.add_argument("--symbol", default="XAUUSD.PRO", help="Trading symbol (default: XAUUSD.PRO)")
    p.add_argument("--timeframe", default="M15", help="Primary execution timeframe (default: M15)")

    # Trading mode (mutually exclusive)
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Live trading mode")
    mode_group.add_argument("--demo", action="store_true", help="Demo trading mode (default)")
    mode_group.add_argument("--simulate", action="store_true", help="Simulation mode")

    # Risk and execution
    p.add_argument("--max-orders", type=int, default=1, help="Maximum open orders (default: 1)")
    p.add_argument("--bars", type=int, default=1500, help="Bars for ATR/indicators (default: 1500)")
    p.add_argument("--risk", type=float, default=0.25, help="Risk per trade (default: 0.25)")
    p.add_argument(
        "--max-risk-per-trade",
        type=float,
        default=0.005,
        help="Max risk per trade (default: 0.005)",
    )

    # Logging and reporting
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )
    p.add_argument(
        "--report-every", type=int, default=20, help="Report every N cycles (default: 20)"
    )
    p.add_argument(
        "--max-cycles", type=int, default=0, help="Max cycles (0 = run forever, default: 0)"
    )

    # Configuration
    p.add_argument(
        "--config",
        default="config/pro_config.json",
        help="Configuration file (default: config/pro_config.json)",
    )

    return p.parse_args()


def load_json_config(config_path: str) -> dict[str, Any]:
    """Load and parse JSON configuration file"""
    try:
        with open(config_path) as f:
            config: dict[str, Any] = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}")
        return {}


def build_logger(log_level: str) -> logging.Logger:
    """Build and configure logger"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/live_trader_clean.log")],
    )

    return logging.getLogger("core.trader")


def main() -> None:
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()

        # Set default mode
        if not any([args.live, args.demo, args.simulate]):
            args.demo = True

        args.mode = "live" if args.live else "demo" if args.demo else "simulate"

        # Load configuration
        config = load_json_config(args.config)

        # Build logger
        logger = build_logger(args.log_level)

        logger.info("Starting MR BEN Live Trading System")
        logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}, Mode: {args.mode}")

        # Create and run application
        app = LiveTraderApp(config, args, logger)

        # Run trading loop
        app.run_loop()

        # Generate final report
        app.finalize_report()

        logger.info("MR BEN Live Trading System completed successfully")
        sys.exit(0)

    except Exception as e:
        logger = logging.getLogger("core.trader")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
