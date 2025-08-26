#!/usr/bin/env python3
"""
MR BEN - Decision Engines for A/B Testing
Control (SMA-only) vs Pro (Ensemble) deciders
"""

from __future__ import annotations
from typing import Optional
from .typesx import DecisionCard, Levels, MarketContext
from .loggingx import logger


class ControlDecider:
    """SMA-only: No PA/ML/LSTM; only common risk gates."""
    
    def __init__(self, ctx: MarketContext):
        self.ctx = ctx
    
    def decide(self) -> DecisionCard:
        """Make decision based on SMA crossover only"""
        # Determine trend direction from SMA
        rule_dir = self.ctx.trend_direction
        
        if rule_dir == 0:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="no_cross",
                score=0.0,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="control"
            )
        
        # Check risk gates (simplified for control)
        if not self._check_risk_gates():
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="risk_gate_block",
                score=0.0,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="control"
            )
        
        # Calculate position size and levels
        lot = self._calculate_position_size(rule_dir)
        levels = self._calculate_levels(rule_dir)
        
        return DecisionCard(
            action="ENTER",
            dir=rule_dir,
            reason="sma_cross",
            score=0.60,  # Fixed score for control
            dyn_conf=self._calculate_dynamic_confidence(),
            lot=lot,
            levels=levels,
            track="control"
        )
    
    def _calculate_dynamic_confidence(self) -> float:
        """Calculate dynamic confidence based on market context"""
        base_conf = 0.70
        
        # Adjust based on regime
        if self.ctx.regime == "HIGH":
            base_conf *= 0.85
        elif self.ctx.regime == "LOW":
            base_conf *= 1.10
        
        # Adjust based on session
        if self.ctx.session == "asia":
            base_conf *= 0.90
        elif self.ctx.session == "london":
            base_conf *= 1.05
        
        return max(0.0, min(1.0, base_conf))
    
    def _check_risk_gates(self) -> bool:
        """Check basic risk gates"""
        # Spread check
        if self.ctx.spread_pts > 180:  # 18 pips
            return False
        
        # Exposure check
        if self.ctx.open_positions >= 2:
            return False
        
        return True
    
    def _calculate_position_size(self, direction: int) -> float:
        """Calculate position size based on ATR and risk"""
        # Base risk: 0.15% of balance
        risk_amount = self.ctx.balance * 0.0015
        
        # ATR-based stop loss
        atr_usd = self.ctx.atr_pts * 10  # Approximate USD per point
        
        if atr_usd <= 0:
            return 0.1  # Minimum lot
        
        # Position size = risk_amount / atr_usd
        lot = risk_amount / atr_usd
        
        # Clamp to min/max
        lot = max(0.1, min(1.0, lot))
        
        return round(lot, 2)
    
    def _calculate_levels(self, direction: int) -> Levels:
        """Calculate SL/TP levels based on ATR"""
        # Entry price
        entry = self.ctx.mid_price
        
        # ATR multiplier
        atr_mult = 1.6
        
        # Stop Loss
        if direction > 0:  # Buy
            sl = entry - (self.ctx.atr_pts * atr_mult)
            tp1 = entry + (self.ctx.atr_pts * 0.8)
            tp2 = entry + (self.ctx.atr_pts * 1.5)
        else:  # Sell
            sl = entry + (self.ctx.atr_pts * atr_mult)
            tp1 = entry - (self.ctx.atr_pts * 0.8)
            tp2 = entry - (self.ctx.atr_pts * 1.5)
        
        return Levels(
            sl=round(sl, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2, 5)
        )


class ProDecider:
    """Ensemble: Rule + PA + ML + LSTM + Dynamic Confidence."""
    
    def __init__(self, ctx: MarketContext):
        self.ctx = ctx
    
    def decide(self) -> DecisionCard:
        """Make ensemble decision with all components"""
        # 1) Rule-based signal (SMA)
        rule_dir = self.ctx.trend_direction
        
        if rule_dir == 0:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="no_trend",
                score=0.0,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 2) Check risk gates
        if not self._check_risk_gates():
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="risk_gate_block",
                score=0.0,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 3) Price Action (simplified for now)
        pa_dir, pa_score = self._check_price_action()
        if pa_score < 0.55:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="pa_low_score",
                score=pa_score,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 4) ML Filter (simplified for now)
        ml_dir, ml_score = self._check_ml_filter()
        if ml_score < 0.58:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="ml_low_conf",
                score=ml_score,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 5) LSTM Filter (simplified for now)
        lstm_dir, lstm_score = self._check_lstm_filter()
        if lstm_score < 0.55:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="lstm_low_conf",
                score=lstm_score,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 6) Ensemble voting
        final_score = self._ensemble_vote(rule_dir, pa_dir, ml_dir, lstm_dir)
        if final_score < 0.55:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="ensemble_low_score",
                score=final_score,
                dyn_conf=self._calculate_dynamic_confidence(),
                track="pro"
            )
        
        # 7) Dynamic confidence check
        dyn_conf = self._calculate_dynamic_confidence()
        if dyn_conf < 0.60:
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="low_dynamic_conf",
                score=final_score,
                dyn_conf=dyn_conf,
                track="pro"
            )
        
        # All checks passed - ENTER
        lot = self._calculate_position_size(rule_dir)
        levels = self._calculate_levels(rule_dir)
        
        return DecisionCard(
            action="ENTER",
            dir=rule_dir,
            reason="ensemble_pass",
            score=final_score,
            dyn_conf=dyn_conf,
            lot=lot,
            levels=levels,
            track="pro"
        )
    
    def _calculate_dynamic_confidence(self) -> float:
        """Calculate dynamic confidence based on market context"""
        base_conf = 0.70
        
        # Adjust based on regime
        if self.ctx.regime == "HIGH":
            base_conf *= 0.85
        elif self.ctx.regime == "LOW":
            base_conf *= 1.10
        
        # Adjust based on session
        if self.ctx.session == "asia":
            base_conf *= 0.90
        elif self.ctx.session == "london":
            base_conf *= 1.05
        
        return max(0.0, min(1.0, base_conf))
    
    def _check_risk_gates(self) -> bool:
        """Check comprehensive risk gates"""
        # Spread check
        if self.ctx.spread_pts > 180:  # 18 pips
            return False
        
        # Exposure check
        if self.ctx.open_positions >= 2:
            return False
        
        return True
    
    def _check_price_action(self) -> tuple[int, float]:
        """Check price action patterns (simplified)"""
        # For now, return a simple score based on trend strength
        trend_strength = abs(self.ctx.sma20 - self.ctx.sma50) / max(self.ctx.sma50, 1e-9)
        
        if trend_strength > 0.001:  # Strong trend
            return self.ctx.trend_direction, 0.65
        elif trend_strength > 0.0005:  # Medium trend
            return self.ctx.trend_direction, 0.58
        else:  # Weak trend
            return 0, 0.45
    
    def _check_ml_filter(self) -> tuple[int, float]:
        """Check ML filter (simplified)"""
        # For now, return a confidence based on trend consistency
        trend_consistency = 0.6 + (0.2 * (self.ctx.atr_pts / 100))  # Higher ATR = higher confidence
        
        return self.ctx.trend_direction, min(0.95, trend_consistency)
    
    def _check_lstm_filter(self) -> tuple[int, float]:
        """Check LSTM filter (simplified)"""
        # For now, return a confidence based on session and regime
        base_conf = 0.6
        
        if self.ctx.session == "london":
            base_conf += 0.1
        elif self.ctx.session == "ny":
            base_conf += 0.05
        
        if self.ctx.regime == "NORMAL":
            base_conf += 0.05
        
        return self.ctx.trend_direction, min(0.95, base_conf)
    
    def _ensemble_vote(self, rule_dir: int, pa_dir: int, ml_dir: int, lstm_dir: int) -> float:
        """Calculate ensemble voting score"""
        weights = (0.50, 0.20, 0.20, 0.10)  # Rule, PA, ML, LSTM
        
        # Calculate weighted direction
        weighted_dir = (
            weights[0] * rule_dir +
            weights[1] * pa_dir +
            weights[2] * ml_dir +
            weights[3] * lstm_dir
        )
        
        # Convert to score [0..1]
        score = abs(weighted_dir)
        
        return min(1.0, score)
    
    def _calculate_position_size(self, direction: int) -> float:
        """Calculate position size based on ATR and risk"""
        # Base risk: 0.15% of balance
        risk_amount = self.ctx.balance * 0.0015
        
        # ATR-based stop loss
        atr_usd = self.ctx.atr_pts * 10  # Approximate USD per point
        
        if atr_usd <= 0:
            return 0.1  # Minimum lot
        
        # Position size = risk_amount / atr_usd
        lot = risk_amount / atr_usd
        
        # Clamp to min/max
        lot = max(0.1, min(1.0, lot))
        
        return round(lot, 2)
    
    def _calculate_levels(self, direction: int) -> Levels:
        """Calculate SL/TP levels based on ATR"""
        # Entry price
        entry = self.ctx.mid_price
        
        # ATR multiplier
        atr_mult = 1.6
        
        # Stop Loss
        if direction > 0:  # Buy
            sl = entry - (self.ctx.atr_pts * atr_mult)
            tp1 = entry + (self.ctx.atr_pts * 0.8)
            tp2 = entry + (self.ctx.atr_pts * 1.5)
        else:  # Sell
            sl = entry + (self.ctx.atr_pts * atr_mult)
            tp1 = entry - (self.ctx.atr_pts * 0.8)
            tp2 = entry - (self.ctx.atr_pts * 1.5)
        
        return Levels(
            sl=round(sl, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2, 5)
        )
