#!/usr/bin/env python3
"""
Evaluation System for MR BEN AI Architecture
Tracks MFE/MAE, performance metrics, and generates KPI reports
"""
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class TradeMetrics:
    """Detailed metrics for a single trade"""

    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float | None

    # Execution details
    volume: float
    confidence: float
    regime: str
    conformal_prob: float

    # Outcome metrics
    exit_price: float
    exit_reason: str  # TP1, TP2, SL, MANUAL
    pnl: float
    r_multiple: float

    # Real-time tracking
    mfe: float  # Maximum Favorable Excursion
    mae: float  # Maximum Adverse Excursion
    duration_minutes: int
    max_drawdown_during: float

    # Split TP details
    tp1_hit: bool
    tp2_hit: bool
    breakeven_triggered: bool


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation and KPI tracking system

    Responsibilities:
    - Real-time MFE/MAE tracking during trades
    - Performance metrics calculation
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Regime-based performance analysis
    - Dashboard data generation
    """

    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger("Evaluator")
        self.config = self._load_config(config_path)

        # Trade tracking
        self.active_trades: dict[str, dict] = {}
        self.completed_trades: list[TradeMetrics] = []

        # Performance metrics
        self.daily_metrics = defaultdict(dict)
        self.regime_metrics = defaultdict(dict)

        # Dashboard data
        self.dashboard_data = {
            "last_update": None,
            "summary": {},
            "daily_pnl": [],
            "drawdown_curve": [],
            "trade_distribution": {},
            "regime_performance": {},
        }

        # Initialize logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load evaluation configuration"""
        default_config = {
            "evaluation": {
                "track_mfe_mae": True,
                "update_frequency_seconds": 60,
                "dashboard_update_minutes": 5,
                "save_detailed_logs": True,
                "risk_free_rate": 0.02,  # 2% annual
            },
            "reporting": {
                "daily_report": True,
                "weekly_report": True,
                "monthly_report": True,
                "export_csv": True,
                "export_plots": True,
            },
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, encoding='utf-8') as f:
                    config = json.load(f)
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
                return default_config
        else:
            return default_config

    def _setup_logging(self):
        """Setup evaluation logging"""
        log_dir = "logs/evaluation"
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler for detailed trade logs
        if self.config["evaluation"]["save_detailed_logs"]:
            log_file = os.path.join(log_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def start_trade_tracking(self, trade_data: dict) -> str:
        """
        Start tracking a new trade for MFE/MAE calculation

        Args:
            trade_data: {
                "symbol": str,
                "side": "BUY"|"SELL",
                "entry_price": float,
                "sl_price": float,
                "tp1_price": float,
                "tp2_price": float,
                "volume": float,
                "confidence": float,
                "regime": str,
                "conformal_prob": float
            }

        Returns:
            trade_id: Unique identifier for the trade
        """
        trade_id = f"{trade_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self.active_trades[trade_id] = {
            **trade_data,
            "trade_id": trade_id,
            "start_time": datetime.now(),
            "entry_price": trade_data["entry_price"],
            "current_price": trade_data["entry_price"],
            "mfe": 0.0,
            "mae": 0.0,
            "price_updates": [],
            "tp1_hit": False,
            "tp2_hit": False,
            "breakeven_triggered": False,
        }

        self.logger.info(
            f"Started tracking trade {trade_id}: {trade_data['side']} "
            f"{trade_data['symbol']} @ {trade_data['entry_price']}"
        )

        return trade_id

    def update_trade_price(
        self, trade_id: str, current_price: float, timestamp: datetime | None = None
    ):
        """Update current price for active trade and calculate MFE/MAE"""

        if trade_id not in self.active_trades:
            return

        trade = self.active_trades[trade_id]
        timestamp = timestamp or datetime.now()

        # Store price update
        trade["price_updates"].append({"timestamp": timestamp, "price": current_price})

        trade["current_price"] = current_price

        # Calculate MFE/MAE
        entry_price = trade["entry_price"]
        side_multiplier = 1 if trade["side"] == "BUY" else -1

        # Favorable excursion (profit direction)
        favorable_move = side_multiplier * (current_price - entry_price)
        trade["mfe"] = max(trade["mfe"], favorable_move)

        # Adverse excursion (loss direction)
        adverse_move = -favorable_move
        trade["mae"] = max(trade["mae"], adverse_move)

        # Check TP/SL hits
        self._check_tp_sl_hits(trade_id, current_price)

    def _check_tp_sl_hits(self, trade_id: str, current_price: float):
        """Check if TP or SL levels have been hit"""
        trade = self.active_trades[trade_id]

        if trade["side"] == "BUY":
            # Check TP1
            if not trade["tp1_hit"] and current_price >= trade["tp1_price"]:
                trade["tp1_hit"] = True
                self.logger.info(f"Trade {trade_id}: TP1 hit at {current_price}")

            # Check TP2
            if (
                trade.get("tp2_price")
                and not trade["tp2_hit"]
                and current_price >= trade["tp2_price"]
            ):
                trade["tp2_hit"] = True
                self.logger.info(f"Trade {trade_id}: TP2 hit at {current_price}")

            # Check breakeven after TP1
            if trade["tp1_hit"] and not trade["breakeven_triggered"]:
                # Move SL to breakeven (simplified)
                trade["sl_price"] = trade["entry_price"]
                trade["breakeven_triggered"] = True
                self.logger.info(f"Trade {trade_id}: Breakeven triggered")

        else:  # SELL
            # Check TP1
            if not trade["tp1_hit"] and current_price <= trade["tp1_price"]:
                trade["tp1_hit"] = True
                self.logger.info(f"Trade {trade_id}: TP1 hit at {current_price}")

            # Check TP2
            if (
                trade.get("tp2_price")
                and not trade["tp2_hit"]
                and current_price <= trade["tp2_price"]
            ):
                trade["tp2_hit"] = True
                self.logger.info(f"Trade {trade_id}: TP2 hit at {current_price}")

            # Check breakeven after TP1
            if trade["tp1_hit"] and not trade["breakeven_triggered"]:
                trade["sl_price"] = trade["entry_price"]
                trade["breakeven_triggered"] = True
                self.logger.info(f"Trade {trade_id}: Breakeven triggered")

    def close_trade(
        self, trade_id: str, exit_price: float, exit_reason: str, actual_pnl: float | None = None
    ) -> TradeMetrics:
        """
        Close a trade and calculate final metrics

        Args:
            trade_id: Trade identifier
            exit_price: Final exit price
            exit_reason: Reason for exit (TP1, TP2, SL, MANUAL)
            actual_pnl: Actual PnL if different from calculated

        Returns:
            TradeMetrics object with complete trade data
        """
        if trade_id not in self.active_trades:
            raise ValueError(f"Trade {trade_id} not found in active trades")

        trade = self.active_trades[trade_id]
        end_time = datetime.now()

        # Calculate final metrics
        side_multiplier = 1 if trade["side"] == "BUY" else -1
        price_diff = side_multiplier * (exit_price - trade["entry_price"])

        # Calculate R-multiple
        risk = abs(trade["entry_price"] - trade["sl_price"])
        r_multiple = price_diff / risk if risk > 0 else 0.0

        # Calculate PnL (simplified)
        if actual_pnl is not None:
            pnl = actual_pnl
        else:
            # Simplified PnL calculation
            pnl = price_diff * trade["volume"]

        # Create TradeMetrics object
        trade_metrics = TradeMetrics(
            trade_id=trade_id,
            timestamp=trade["start_time"],
            symbol=trade["symbol"],
            side=trade["side"],
            entry_price=trade["entry_price"],
            sl_price=trade["sl_price"],
            tp1_price=trade["tp1_price"],
            tp2_price=trade.get("tp2_price"),
            volume=trade["volume"],
            confidence=trade["confidence"],
            regime=trade["regime"],
            conformal_prob=trade["conformal_prob"],
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            r_multiple=r_multiple,
            mfe=trade["mfe"],
            mae=trade["mae"],
            duration_minutes=int((end_time - trade["start_time"]).total_seconds() / 60),
            max_drawdown_during=trade["mae"],  # Simplified
            tp1_hit=trade["tp1_hit"],
            tp2_hit=trade["tp2_hit"],
            breakeven_triggered=trade["breakeven_triggered"],
        )

        # Add to completed trades
        self.completed_trades.append(trade_metrics)

        # Remove from active trades
        del self.active_trades[trade_id]

        # Log trade completion
        self.logger.info(
            f"Trade {trade_id} closed: {exit_reason} @ {exit_price:.2f} | "
            f"PnL: {pnl:.2f} | R: {r_multiple:.2f} | "
            f"MFE: {trade['mfe']:.2f} | MAE: {trade['mae']:.2f}"
        )

        # Update performance metrics
        self._update_performance_metrics(trade_metrics)

        return trade_metrics

    def _update_performance_metrics(self, trade: TradeMetrics):
        """Update running performance metrics"""

        # Daily metrics
        date_key = trade.timestamp.date().isoformat()
        if date_key not in self.daily_metrics:
            self.daily_metrics[date_key] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "total_r": 0.0,
                "mfe_sum": 0.0,
                "mae_sum": 0.0,
            }

        day_metrics = self.daily_metrics[date_key]
        day_metrics["trades"] += 1
        day_metrics["total_pnl"] += trade.pnl
        day_metrics["total_r"] += trade.r_multiple
        day_metrics["mfe_sum"] += trade.mfe
        day_metrics["mae_sum"] += trade.mae

        if trade.pnl > 0:
            day_metrics["wins"] += 1
        else:
            day_metrics["losses"] += 1

        # Regime-based metrics
        regime = trade.regime
        if regime not in self.regime_metrics:
            self.regime_metrics[regime] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "avg_confidence": 0.0,
                "avg_conformal": 0.0,
            }

        regime_metrics = self.regime_metrics[regime]
        regime_metrics["trades"] += 1
        regime_metrics["total_pnl"] += trade.pnl
        regime_metrics["avg_confidence"] = (
            regime_metrics["avg_confidence"] * (regime_metrics["trades"] - 1) + trade.confidence
        ) / regime_metrics["trades"]
        regime_metrics["avg_conformal"] = (
            regime_metrics["avg_conformal"] * (regime_metrics["trades"] - 1) + trade.conformal_prob
        ) / regime_metrics["trades"]

        if trade.pnl > 0:
            regime_metrics["wins"] += 1

    def calculate_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""

        if not self.completed_trades:
            return {"error": "No completed trades available"}

        trades_df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "pnl": t.pnl,
                    "r_multiple": t.r_multiple,
                    "mfe": t.mfe,
                    "mae": t.mae,
                    "regime": t.regime,
                    "confidence": t.confidence,
                    "duration": t.duration_minutes,
                }
                for t in self.completed_trades
            ]
        )

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = (
            -avg_win * winning_trades / (avg_loss * losing_trades)
            if avg_loss != 0
            else float('inf')
        )

        # R-multiple metrics
        avg_r = trades_df['r_multiple'].mean()
        expectancy = avg_r

        # Risk metrics
        returns = trades_df['pnl'].values
        if len(returns) > 1:
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            )
            max_drawdown = self._calculate_max_drawdown(returns)
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        # MFE/MAE analysis
        avg_mfe = trades_df['mfe'].mean()
        avg_mae = trades_df['mae'].mean()
        mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

        # Efficiency metrics
        efficiency = total_pnl / trades_df['mae'].sum() if trades_df['mae'].sum() > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_r_multiple": avg_r,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "mfe_mae_ratio": mfe_mae_ratio,
            "efficiency": efficiency,
            "avg_duration_minutes": trades_df['duration'].mean(),
            "regime_breakdown": dict(self.regime_metrics),
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))

    def generate_daily_report(self, date: str | None = None) -> dict:
        """Generate daily performance report"""

        if date is None:
            date = datetime.now().date().isoformat()

        if date not in self.daily_metrics:
            return {"error": f"No data for date {date}"}

        day_data = self.daily_metrics[date]

        # Calculate daily metrics
        win_rate = day_data["wins"] / day_data["trades"] if day_data["trades"] > 0 else 0
        avg_r = day_data["total_r"] / day_data["trades"] if day_data["trades"] > 0 else 0
        avg_mfe = day_data["mfe_sum"] / day_data["trades"] if day_data["trades"] > 0 else 0
        avg_mae = day_data["mae_sum"] / day_data["trades"] if day_data["trades"] > 0 else 0

        return {
            "date": date,
            "total_trades": day_data["trades"],
            "wins": day_data["wins"],
            "losses": day_data["losses"],
            "win_rate": win_rate,
            "total_pnl": day_data["total_pnl"],
            "avg_r_multiple": avg_r,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
        }

    def export_trade_data(self, filepath: str = None):
        """Export trade data to CSV"""

        if not filepath:
            filepath = (
                f"logs/evaluation/trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert trades to DataFrame
        trades_data = []
        for trade in self.completed_trades:
            trades_data.append(
                {
                    "trade_id": trade.trade_id,
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "volume": trade.volume,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                    "mfe": trade.mfe,
                    "mae": trade.mae,
                    "duration_minutes": trade.duration_minutes,
                    "confidence": trade.confidence,
                    "regime": trade.regime,
                    "conformal_prob": trade.conformal_prob,
                    "tp1_hit": trade.tp1_hit,
                    "tp2_hit": trade.tp2_hit,
                    "breakeven_triggered": trade.breakeven_triggered,
                }
            )

        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Trade data exported to {filepath}")

        return filepath

    def get_dashboard_data(self) -> dict:
        """Get data for dashboard visualization"""

        metrics = self.calculate_performance_metrics()

        self.dashboard_data.update(
            {
                "last_update": datetime.now().isoformat(),
                "summary": metrics,
                "active_trades_count": len(self.active_trades),
                "daily_breakdown": dict(self.daily_metrics),
                "regime_performance": dict(self.regime_metrics),
            }
        )

        return self.dashboard_data
