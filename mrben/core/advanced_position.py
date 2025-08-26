#!/usr/bin/env python3
"""
MR BEN - Advanced Position Management System
Enhanced position sizing, dynamic adjustment, and portfolio-level risk management
"""

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .advanced_risk import AdvancedRiskAnalytics
from .metricsx import observe_position_metric
from .typesx import DecisionCard, MarketContext


class PositionStrategy(str, Enum):
    """Position management strategies"""

    FIXED_SIZE = "fixed_size"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    ML_OPTIMIZED = "ml_optimized"
    PORTFOLIO_OPTIMIZED = "portfolio_optimized"


class ScalingMethod(str, Enum):
    """Position scaling methods"""

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    VOLATILITY_BASED = "volatility_based"
    ML_BASED = "ml_based"


class ExitStrategy(str, Enum):
    """Exit strategy types"""

    FIXED_TP_SL = "fixed_tp_sl"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN = "breakeven"
    TIME_BASED = "time_based"
    VOLATILITY_BASED = "volatility_based"
    ML_BASED = "ml_based"


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""

    lot_size: float
    strategy: PositionStrategy
    confidence: float
    risk_reward_ratio: float
    max_loss_usd: float
    position_value_usd: float
    sizing_factors: dict[str, float]
    recommendations: list[str]


@dataclass
class PositionAdjustment:
    """Position adjustment recommendation"""

    action: str  # "increase", "decrease", "close", "hold"
    lot_change: float
    reason: str
    confidence: float
    urgency: str  # "low", "medium", "high", "critical"
    market_conditions: dict[str, Any]


@dataclass
class PortfolioPosition:
    """Portfolio position information"""

    symbol: str
    direction: int  # 1 for long, -1 for short
    lot_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_exposure: float
    correlation_score: float
    volatility_contribution: float


class AdvancedPositionManager:
    """
    Advanced Position Management System

    Provides enhanced position sizing, dynamic adjustment, and
    portfolio-level risk management using ML and advanced analytics.
    """

    def __init__(
        self,
        config_path: str | None = None,
        model_dir: str = "position_models",
        enable_ml: bool = True,
        enable_portfolio_optimization: bool = True,
        enable_dynamic_adjustment: bool = True,
    ):
        self.config_path = config_path or "advanced_position_config.json"
        self.model_dir = Path(model_dir)
        self.enable_ml = enable_ml
        self.enable_portfolio_optimization = enable_portfolio_optimization
        self.enable_dynamic_adjustment = enable_dynamic_adjustment

        # Create model directory
        self.model_dir.mkdir(exist_ok=True)

        # ML models
        self.sizing_model: RandomForestRegressor | None = None
        self.adjustment_model: RandomForestRegressor | None = None
        self.scaler: StandardScaler | None = None

        # Data storage
        self.position_history: deque = deque(maxlen=10000)
        self.portfolio_positions: dict[str, PortfolioPosition] = {}
        self.adjustment_history: deque = deque(maxlen=5000)

        # Performance tracking
        self.model_performance: dict[str, dict[str, float]] = {}
        self.sizing_accuracy: list[float] = []

        # Threading
        self._lock = threading.RLock()
        self._optimization_thread: threading.Thread | None = None
        self._stop_optimization = threading.Event()

        # Advanced risk analytics integration
        self.risk_analytics: AdvancedRiskAnalytics | None = None

        # Load configuration and models
        self._load_config()
        self._load_models()

        logger.bind(evt="POSITION").info(
            "advanced_position_manager_initialized",
            ml_enabled=enable_ml,
            portfolio_optimization=enable_portfolio_optimization,
        )

    def _load_config(self) -> None:
        """Load advanced position management configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    config = json.load(f)

                # Update configuration
                if 'enable_ml' in config:
                    self.enable_ml = config['enable_ml']
                if 'enable_portfolio_optimization' in config:
                    self.enable_portfolio_optimization = config['enable_portfolio_optimization']
                if 'enable_dynamic_adjustment' in config:
                    self.enable_dynamic_adjustment = config['enable_dynamic_adjustment']

                logger.bind(evt="POSITION").info(
                    "advanced_position_config_loaded", config_path=self.config_path
                )
            else:
                logger.bind(evt="POSITION").info(
                    "advanced_position_config_not_found", config_path=self.config_path
                )
        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "advanced_position_config_load_failed", error=str(e)
            )

    def _load_models(self) -> None:
        """Load pre-trained position management models"""
        try:
            if not self.enable_ml:
                return

            # Load sizing model
            sizing_model_path = self.model_dir / "sizing_model.pkl"
            if sizing_model_path.exists():
                self.sizing_model = joblib.load(sizing_model_path)
                logger.bind(evt="POSITION").info("sizing_model_loaded")

            # Load adjustment model
            adjustment_model_path = self.model_dir / "adjustment_model.pkl"
            if adjustment_model_path.exists():
                self.adjustment_model = joblib.load(adjustment_model_path)
                logger.bind(evt="POSITION").info("adjustment_model_loaded")

            # Load scaler
            scaler_path = self.model_dir / "position_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.bind(evt="POSITION").info("position_scaler_loaded")

            # If no models exist, create default ones
            if not self.sizing_model:
                self._create_default_models()

        except Exception as e:
            logger.bind(evt="POSITION").error("model_loading_failed", error=str(e))
            self._create_default_models()

    def _create_default_models(self) -> None:
        """Create default position management models"""
        try:
            # Create sizing model
            self.sizing_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )

            # Create adjustment model
            self.adjustment_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )

            # Create scaler
            self.scaler = StandardScaler()

            logger.bind(evt="POSITION").info("default_position_models_created")

        except Exception as e:
            logger.bind(evt="POSITION").error("default_model_creation_failed", error=str(e))

    def set_risk_analytics(self, risk_analytics: AdvancedRiskAnalytics) -> None:
        """Set advanced risk analytics for integration"""
        self.risk_analytics = risk_analytics
        logger.bind(evt="POSITION").info("risk_analytics_integrated")

    def calculate_position_size(
        self,
        decision: DecisionCard,
        context: MarketContext,
        account_balance: float,
        risk_per_trade: float,
        strategy: PositionStrategy = PositionStrategy.ML_OPTIMIZED,
    ) -> PositionSizingResult:
        """
        Calculate optimal position size using advanced algorithms

        Args:
            decision: Trading decision
            context: Market context
            account_balance: Current account balance
            risk_per_trade: Risk per trade (percentage or USD)
            strategy: Position sizing strategy

        Returns:
            Position sizing result with recommendations
        """
        try:
            with self._lock:
                # Calculate base position size
                if strategy == PositionStrategy.FIXED_SIZE:
                    lot_size = self._calculate_fixed_size(account_balance, risk_per_trade)
                elif strategy == PositionStrategy.KELLY_CRITERION:
                    lot_size = self._calculate_kelly_criterion(decision, context, account_balance)
                elif strategy == PositionStrategy.VOLATILITY_ADJUSTED:
                    lot_size = self._calculate_volatility_adjusted_size(
                        decision, context, account_balance, risk_per_trade
                    )
                elif strategy == PositionStrategy.ML_OPTIMIZED:
                    lot_size = self._calculate_ml_optimized_size(
                        decision, context, account_balance, risk_per_trade
                    )
                elif strategy == PositionStrategy.PORTFOLIO_OPTIMIZED:
                    lot_size = self._calculate_portfolio_optimized_size(
                        decision, context, account_balance, risk_per_trade
                    )
                else:
                    lot_size = self._calculate_fixed_size(account_balance, risk_per_trade)

                # Apply portfolio-level constraints
                if self.enable_portfolio_optimization:
                    lot_size = self._apply_portfolio_constraints(lot_size, decision, context)

                # Calculate risk metrics
                risk_reward_ratio = self._calculate_risk_reward_ratio(decision, context)
                max_loss_usd = self._calculate_max_loss(lot_size, decision, context)
                position_value_usd = lot_size * context.price * 100000  # Standard lot = 100k units

                # Generate sizing factors
                sizing_factors = self._calculate_sizing_factors(decision, context, lot_size)

                # Generate recommendations
                recommendations = self._generate_sizing_recommendations(
                    lot_size, risk_reward_ratio, sizing_factors, context
                )

                # Calculate confidence
                confidence = self._calculate_sizing_confidence(decision, context, strategy)

                # Create result
                result = PositionSizingResult(
                    lot_size=lot_size,
                    strategy=strategy,
                    confidence=confidence,
                    risk_reward_ratio=risk_reward_ratio,
                    max_loss_usd=max_loss_usd,
                    position_value_usd=position_value_usd,
                    sizing_factors=sizing_factors,
                    recommendations=recommendations,
                )

                # Store in history
                self.position_history.append(asdict(result))

                # Record metrics
                observe_position_metric("lot_size", lot_size)
                observe_position_metric("risk_reward_ratio", risk_reward_ratio)
                observe_position_metric("confidence", confidence)

                logger.bind(evt="POSITION").info(
                    "position_size_calculated",
                    strategy=strategy.value,
                    lot_size=lot_size,
                    confidence=confidence,
                )

                return result

        except Exception as e:
            logger.bind(evt="POSITION").error("position_size_calculation_failed", error=str(e))
            # Return default result
            return self._create_default_sizing_result(account_balance, risk_per_trade)

    def _calculate_fixed_size(self, account_balance: float, risk_per_trade: float) -> float:
        """Calculate fixed position size based on risk percentage"""
        try:
            if isinstance(risk_per_trade, str) and '%' in risk_per_trade:
                risk_percentage = float(risk_per_trade.replace('%', '')) / 100
                risk_amount = account_balance * risk_percentage
            else:
                risk_amount = float(risk_per_trade)

            # Convert to lot size (assuming 1 pip = $10 for standard lot)
            lot_size = risk_amount / 100  # Simplified conversion
            return max(0.01, min(lot_size, 10.0))  # Clamp between 0.01 and 10.0

        except Exception as e:
            logger.bind(evt="POSITION").warning("fixed_size_calculation_failed", error=str(e))
            return 0.1  # Default 0.1 lot

    def _calculate_kelly_criterion(
        self, decision: DecisionCard, context: MarketContext, account_balance: float
    ) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where: b = odds received, p = probability of win, q = probability of loss

            # Estimate win probability from decision confidence
            win_probability = decision.dyn_conf
            loss_probability = 1.0 - win_probability

            # Estimate odds (simplified - could be enhanced with historical data)
            if decision.dir > 0:  # Long position
                potential_gain = (decision.levels.tp1 - context.price) / context.price
                potential_loss = (context.price - decision.levels.sl) / context.price
            else:  # Short position
                potential_gain = (context.price - decision.levels.tp1) / context.price
                potential_loss = (decision.levels.sl - context.price) / context.price

            # Calculate Kelly fraction
            if potential_loss > 0:
                kelly_fraction = (
                    win_probability * potential_gain - loss_probability * potential_loss
                ) / potential_gain
                kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.1  # Default 10%

            # Convert to lot size
            lot_size = (kelly_fraction * account_balance) / 10000  # Simplified conversion
            return max(0.01, min(lot_size, 5.0))  # Clamp between 0.01 and 5.0

        except Exception as e:
            logger.bind(evt="POSITION").warning("kelly_criterion_calculation_failed", error=str(e))
            return 0.1  # Default 0.1 lot

    def _calculate_volatility_adjusted_size(
        self,
        decision: DecisionCard,
        context: MarketContext,
        account_balance: float,
        risk_per_trade: float,
    ) -> float:
        """Calculate position size adjusted for volatility"""
        try:
            # Base position size
            base_size = self._calculate_fixed_size(account_balance, risk_per_trade)

            # Volatility adjustment factor
            volatility_factor = 1.0

            # Adjust based on ATR (volatility)
            if context.atr_pts > 100:  # High volatility
                volatility_factor = 0.7  # Reduce position size
            elif context.atr_pts < 20:  # Low volatility
                volatility_factor = 1.2  # Increase position size

            # Adjust based on market regime
            if context.regime == "HIGH":
                volatility_factor *= 0.8  # Further reduce in high volatility regime
            elif context.regime == "LOW":
                volatility_factor *= 1.1  # Further increase in low volatility regime

            # Apply volatility adjustment
            adjusted_size = base_size * volatility_factor

            return max(0.01, min(adjusted_size, 10.0))

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "volatility_adjusted_size_calculation_failed", error=str(e)
            )
            return 0.1  # Default 0.1 lot

    def _calculate_ml_optimized_size(
        self,
        decision: DecisionCard,
        context: MarketContext,
        account_balance: float,
        risk_per_trade: float,
    ) -> float:
        """Calculate position size using ML model"""
        try:
            if not self.sizing_model or not self.scaler:
                # Fallback to volatility adjusted if ML not available
                return self._calculate_volatility_adjusted_size(
                    decision, context, account_balance, risk_per_trade
                )

            # Extract features for ML model
            features = self._extract_sizing_features(decision, context, account_balance)

            if features is None:
                return self._calculate_volatility_adjusted_size(
                    decision, context, account_balance, risk_per_trade
                )

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict optimal size multiplier
            size_multiplier = self.sizing_model.predict(features_scaled)[0]

            # Apply multiplier to base size
            base_size = self._calculate_fixed_size(account_balance, risk_per_trade)
            ml_optimized_size = base_size * size_multiplier

            # Ensure reasonable bounds
            return max(0.01, min(ml_optimized_size, 10.0))

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "ml_optimized_size_calculation_failed", error=str(e)
            )
            return self._calculate_volatility_adjusted_size(
                decision, context, account_balance, risk_per_trade
            )

    def _calculate_portfolio_optimized_size(
        self,
        decision: DecisionCard,
        context: MarketContext,
        account_balance: float,
        risk_per_trade: float,
    ) -> float:
        """Calculate position size optimized for portfolio"""
        try:
            # Start with ML optimized size
            base_size = self._calculate_ml_optimized_size(
                decision, context, account_balance, risk_per_trade
            )

            if not self.enable_portfolio_optimization:
                return base_size

            # Calculate portfolio constraints
            portfolio_factor = self._calculate_portfolio_factor(decision, context)

            # Apply portfolio optimization
            optimized_size = base_size * portfolio_factor

            return max(0.01, min(optimized_size, 10.0))

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "portfolio_optimized_size_calculation_failed", error=str(e)
            )
            return self._calculate_ml_optimized_size(
                decision, context, account_balance, risk_per_trade
            )

    def _apply_portfolio_constraints(
        self, lot_size: float, decision: DecisionCard, context: MarketContext
    ) -> float:
        """Apply portfolio-level constraints to position size"""
        try:
            if not self.portfolio_positions:
                return lot_size

            # Calculate current portfolio exposure
            total_exposure = sum(abs(pos.lot_size) for pos in self.portfolio_positions.values())
            max_portfolio_exposure = 5.0  # Maximum 5.0 lots total

            # Check if adding this position would exceed limits
            if total_exposure + lot_size > max_portfolio_exposure:
                lot_size = max(0.01, max_portfolio_exposure - total_exposure)
                logger.bind(evt="POSITION").warning(
                    "portfolio_exposure_limit_applied",
                    original_size=lot_size,
                    adjusted_size=lot_size,
                )

            # Check correlation with existing positions
            correlation_factor = self._calculate_correlation_factor(decision, context)
            lot_size *= correlation_factor

            return max(0.01, lot_size)

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "portfolio_constraints_application_failed", error=str(e)
            )
            return lot_size

    def _calculate_portfolio_factor(self, decision: DecisionCard, context: MarketContext) -> float:
        """Calculate portfolio optimization factor"""
        try:
            if not self.portfolio_positions:
                return 1.0

            # Calculate portfolio diversification factor
            diversification_factor = 1.0

            # Reduce size if too many positions in same direction
            same_direction_count = sum(
                1 for pos in self.portfolio_positions.values() if pos.direction == decision.dir
            )
            if same_direction_count > 2:
                diversification_factor *= 0.8

            # Reduce size if too many positions in same session
            session_positions = sum(
                1
                for pos in self.portfolio_positions.values()
                if hasattr(pos, 'session') and pos.session == context.session
            )
            if session_positions > 1:
                diversification_factor *= 0.9

            return diversification_factor

        except Exception as e:
            logger.bind(evt="POSITION").warning("portfolio_factor_calculation_failed", error=str(e))
            return 1.0

    def _calculate_correlation_factor(
        self, decision: DecisionCard, context: MarketContext
    ) -> float:
        """Calculate correlation-based adjustment factor"""
        try:
            if not self.portfolio_positions:
                return 1.0

            # Simple correlation calculation (in production, use proper statistical methods)
            correlation_factor = 1.0

            # Check for high correlation with existing positions
            high_correlation_count = 0
            for pos in self.portfolio_positions.values():
                # Simplified correlation check
                if pos.direction == decision.dir and abs(pos.lot_size) > 0.5:
                    high_correlation_count += 1

            if high_correlation_count > 1:
                correlation_factor *= 0.7  # Reduce size for high correlation

            return correlation_factor

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "correlation_factor_calculation_failed", error=str(e)
            )
            return 1.0

    def _calculate_risk_reward_ratio(self, decision: DecisionCard, context: MarketContext) -> float:
        """Calculate risk-reward ratio for the position"""
        try:
            if decision.levels is None:
                return 1.0

            # Calculate potential gain and loss
            if decision.dir > 0:  # Long position
                potential_gain = decision.levels.tp1 - context.price
                potential_loss = context.price - decision.levels.sl
            else:  # Short position
                potential_gain = context.price - decision.levels.tp1
                potential_loss = decision.levels.sl - context.price

            # Calculate risk-reward ratio
            if potential_loss > 0:
                risk_reward_ratio = potential_gain / potential_loss
            else:
                risk_reward_ratio = 1.0

            return max(0.1, min(risk_reward_ratio, 10.0))  # Clamp between 0.1 and 10.0

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "risk_reward_ratio_calculation_failed", error=str(e)
            )
            return 1.0

    def _calculate_max_loss(
        self, lot_size: float, decision: DecisionCard, context: MarketContext
    ) -> float:
        """Calculate maximum potential loss in USD"""
        try:
            if decision.levels is None:
                return 0.0

            # Calculate stop loss distance
            if decision.dir > 0:  # Long position
                sl_distance = context.price - decision.levels.sl
            else:  # Short position
                sl_distance = decision.levels.sl - context.price

            # Convert to USD (assuming 1 pip = $10 for standard lot)
            max_loss_usd = lot_size * sl_distance * 10000  # Simplified conversion

            return max(0.0, max_loss_usd)

        except Exception as e:
            logger.bind(evt="POSITION").warning("max_loss_calculation_failed", error=str(e))
            return 0.0

    def _calculate_sizing_factors(
        self, decision: DecisionCard, context: MarketContext, lot_size: float
    ) -> dict[str, float]:
        """Calculate various factors that influenced position sizing"""
        try:
            factors = {}

            # Decision confidence factor
            factors['decision_confidence'] = decision.dyn_conf

            # Market volatility factor
            factors['volatility_factor'] = min(context.atr_pts / 50.0, 2.0)

            # Market regime factor
            regime_factors = {"LOW": 0.8, "NORMAL": 1.0, "HIGH": 0.6}
            factors['regime_factor'] = regime_factors.get(context.regime, 1.0)

            # Session factor
            session_factors = {
                "london": 1.0,
                "newyork": 1.0,
                "asia": 0.9,
                "overlap": 1.1,
                "weekend": 0.7,
            }
            factors['session_factor'] = session_factors.get(context.session, 1.0)

            # Spread factor
            factors['spread_factor'] = max(0.5, 1.0 - (context.spread_pts / 100.0))

            # Position size factor
            factors['size_factor'] = min(lot_size / 1.0, 2.0)

            return factors

        except Exception as e:
            logger.bind(evt="POSITION").warning("sizing_factors_calculation_failed", error=str(e))
            return {}

    def _generate_sizing_recommendations(
        self,
        lot_size: float,
        risk_reward_ratio: float,
        sizing_factors: dict[str, float],
        context: MarketContext,
    ) -> list[str]:
        """Generate recommendations for position sizing"""
        try:
            recommendations = []

            # Size-based recommendations
            if lot_size > 2.0:
                recommendations.append("Consider reducing position size for better risk management")
            elif lot_size < 0.1:
                recommendations.append(
                    "Position size is very small - consider increasing if confidence is high"
                )

            # Risk-reward recommendations
            if risk_reward_ratio < 1.5:
                recommendations.append("Risk-reward ratio is below recommended 1.5:1 minimum")
            elif risk_reward_ratio > 3.0:
                recommendations.append(
                    "Excellent risk-reward ratio - consider increasing position size"
                )

            # Market condition recommendations
            if context.regime == "HIGH":
                recommendations.append("High volatility regime - consider reducing position size")

            if context.session == "weekend":
                recommendations.append("Weekend session - be cautious with position sizing")

            if context.spread_pts > 30:
                recommendations.append(
                    "Wide spreads detected - consider waiting for better conditions"
                )

            # Factor-based recommendations
            if sizing_factors.get('decision_confidence', 0) < 0.6:
                recommendations.append("Low decision confidence - consider smaller position size")

            if not recommendations:
                recommendations.append("Position sizing looks optimal for current conditions")

            return recommendations

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "sizing_recommendations_generation_failed", error=str(e)
            )
            return ["Position sizing analysis incomplete - proceed with caution"]

    def _calculate_sizing_confidence(
        self, decision: DecisionCard, context: MarketContext, strategy: PositionStrategy
    ) -> float:
        """Calculate confidence in the sizing calculation"""
        try:
            confidence = 0.5  # Base confidence

            # Strategy confidence
            strategy_confidence = {
                PositionStrategy.FIXED_SIZE: 0.6,
                PositionStrategy.KELLY_CRITERION: 0.7,
                PositionStrategy.VOLATILITY_ADJUSTED: 0.8,
                PositionStrategy.ML_OPTIMIZED: 0.85,
                PositionStrategy.PORTFOLIO_OPTIMIZED: 0.9,
            }
            confidence += strategy_confidence.get(strategy, 0.5) * 0.3

            # Decision confidence
            confidence += decision.dyn_conf * 0.2

            # Market condition confidence
            if context.regime == "NORMAL":
                confidence += 0.1
            elif context.regime == "HIGH":
                confidence -= 0.1

            return max(0.1, min(confidence, 1.0))

        except Exception as e:
            logger.bind(evt="POSITION").warning(
                "sizing_confidence_calculation_failed", error=str(e)
            )
            return 0.5

    def _extract_sizing_features(
        self, decision: DecisionCard, context: MarketContext, account_balance: float
    ) -> list[float] | None:
        """Extract features for ML model training"""
        try:
            features = []

            # Decision features
            features.extend(
                [
                    float(decision.score),
                    float(decision.dyn_conf),
                    float(decision.lot) if decision.lot else 0.0,
                ]
            )

            # Market context features
            features.extend(
                [
                    context.atr_pts,
                    context.spread_pts,
                    context.equity,
                    context.balance,
                    context.open_positions,
                ]
            )

            # Session encoding
            session_encoding = {
                'london': 1.0,
                'newyork': 2.0,
                'asia': 3.0,
                'overlap': 4.0,
                'weekend': 5.0,
            }
            features.append(session_encoding.get(context.session, 1.0))

            # Regime encoding
            regime_encoding = {'LOW': 1.0, 'NORMAL': 2.0, 'HIGH': 3.0}
            features.append(regime_encoding.get(context.regime, 2.0))

            # Account features
            features.append(account_balance)

            return features

        except Exception as e:
            logger.bind(evt="POSITION").warning("sizing_features_extraction_failed", error=str(e))
            return None

    def _create_default_sizing_result(
        self, account_balance: float, risk_per_trade: float
    ) -> PositionSizingResult:
        """Create default sizing result when calculation fails"""
        lot_size = self._calculate_fixed_size(account_balance, risk_per_trade)

        return PositionSizingResult(
            lot_size=lot_size,
            strategy=PositionStrategy.FIXED_SIZE,
            confidence=0.3,
            risk_reward_ratio=1.0,
            max_loss_usd=0.0,
            position_value_usd=lot_size * 100000,
            sizing_factors={'default': 1.0},
            recommendations=['Position sizing calculation failed - using default values'],
        )

    def add_portfolio_position(
        self, symbol: str, direction: int, lot_size: float, entry_price: float, current_price: float
    ) -> None:
        """Add or update portfolio position"""
        try:
            with self._lock:
                # Calculate position metrics
                unrealized_pnl = (current_price - entry_price) * direction * lot_size * 100000
                risk_exposure = lot_size * 100000  # Simplified risk calculation

                # Calculate correlation score (simplified)
                correlation_score = 0.0
                if self.portfolio_positions:
                    same_direction_count = sum(
                        1 for pos in self.portfolio_positions.values() if pos.direction == direction
                    )
                    correlation_score = same_direction_count / len(self.portfolio_positions)

                # Calculate volatility contribution (simplified)
                volatility_contribution = lot_size * 0.1  # Simplified calculation

                # Create or update position
                self.portfolio_positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    direction=direction,
                    lot_size=lot_size,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    risk_exposure=risk_exposure,
                    correlation_score=correlation_score,
                    volatility_contribution=volatility_contribution,
                )

                logger.bind(evt="POSITION").info(
                    "portfolio_position_updated",
                    symbol=symbol,
                    lot_size=lot_size,
                    pnl=unrealized_pnl,
                )

        except Exception as e:
            logger.bind(evt="POSITION").error("portfolio_position_update_failed", error=str(e))

    def remove_portfolio_position(self, symbol: str) -> None:
        """Remove portfolio position"""
        try:
            with self._lock:
                if symbol in self.portfolio_positions:
                    del self.portfolio_positions[symbol]
                    logger.bind(evt="POSITION").info("portfolio_position_removed", symbol=symbol)

        except Exception as e:
            logger.bind(evt="POSITION").error("portfolio_position_removal_failed", error=str(e))

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary and analysis"""
        try:
            with self._lock:
                if not self.portfolio_positions:
                    return {"message": "No portfolio positions"}

                # Calculate portfolio metrics
                total_positions = len(self.portfolio_positions)
                total_exposure = sum(abs(pos.lot_size) for pos in self.portfolio_positions.values())
                total_pnl = sum(pos.unrealized_pnl for pos in self.portfolio_positions.values())

                # Calculate correlation metrics
                long_positions = sum(
                    1 for pos in self.portfolio_positions.values() if pos.direction > 0
                )
                short_positions = sum(
                    1 for pos in self.portfolio_positions.values() if pos.direction < 0
                )

                # Calculate risk metrics
                total_risk = sum(pos.risk_exposure for pos in self.portfolio_positions.values())
                avg_correlation = np.mean(
                    [pos.correlation_score for pos in self.portfolio_positions.values()]
                )

                summary = {
                    "total_positions": total_positions,
                    "total_exposure": total_exposure,
                    "total_pnl": total_pnl,
                    "long_positions": long_positions,
                    "short_positions": short_positions,
                    "total_risk": total_risk,
                    "average_correlation": avg_correlation,
                    "positions": [
                        {
                            "symbol": pos.symbol,
                            "direction": pos.direction,
                            "lot_size": pos.lot_size,
                            "entry_price": pos.entry_price,
                            "current_price": pos.current_price,
                            "unrealized_pnl": pos.unrealized_pnl,
                            "risk_exposure": pos.risk_exposure,
                        }
                        for pos in self.portfolio_positions.values()
                    ],
                    "last_updated": datetime.now(UTC).isoformat(),
                }

                return summary

        except Exception as e:
            logger.bind(evt="POSITION").error("portfolio_summary_generation_failed", error=str(e))
            return {"error": str(e)}

    def train_models(self, training_data: list[dict[str, Any]]) -> bool:
        """Train position management models with new data"""
        try:
            if not self.enable_ml or not training_data:
                return False

            logger.bind(evt="POSITION").info(
                "starting_model_training", data_points=len(training_data)
            )

            # Prepare training data for sizing model
            X_sizing = []
            y_sizing = []

            for data_point in training_data:
                features = self._extract_sizing_features(
                    data_point['decision'],
                    data_point['context'],
                    data_point.get('account_balance', 10000),
                )
                if features is not None:
                    X_sizing.append(features)
                    y_sizing.append(data_point.get('optimal_size_multiplier', 1.0))

            if len(X_sizing) < 10:
                logger.bind(evt="POSITION").warning(
                    "insufficient_sizing_training_data", samples=len(X_sizing)
                )
                return False

            X_sizing = np.array(X_sizing)
            y_sizing = np.array(y_sizing)

            # Train sizing model
            if self.sizing_model:
                self.sizing_model.fit(X_sizing, y_sizing)

                # Evaluate model performance
                y_pred = self.sizing_model.predict(X_sizing)
                mse = np.mean((y_sizing - y_pred) ** 2)
                accuracy = 1.0 - mse

                self.model_performance['sizing_model'] = {
                    'mse': mse,
                    'accuracy': accuracy,
                    'last_trained': datetime.now(UTC).isoformat(),
                }

                # Update sizing accuracy
                self.sizing_accuracy.append(accuracy)
                if len(self.sizing_accuracy) > 100:
                    self.sizing_accuracy = self.sizing_accuracy[-100:]

            # Save models
            self._save_models()

            logger.bind(evt="POSITION").info(
                "model_training_completed",
                samples=len(X_sizing),
                accuracy=accuracy if 'accuracy' in locals() else None,
            )

            return True

        except Exception as e:
            logger.bind(evt="POSITION").error("model_training_failed", error=str(e))
            return False

    def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            if self.sizing_model:
                joblib.dump(self.sizing_model, self.model_dir / "sizing_model.pkl")

            if self.adjustment_model:
                joblib.dump(self.adjustment_model, self.model_dir / "adjustment_model.pkl")

            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "position_scaler.pkl")

            logger.bind(evt="POSITION").info("position_models_saved_to_disk")

        except Exception as e:
            logger.bind(evt="POSITION").error("model_saving_failed", error=str(e))

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._stop_optimization.set()
            if self._optimization_thread and self._optimization_thread.is_alive():
                self._optimization_thread.join(timeout=5.0)

            # Save models before cleanup
            self._save_models()

            logger.bind(evt="POSITION").info("advanced_position_manager_cleanup_complete")

        except Exception as e:
            logger.bind(evt="POSITION").error("cleanup_failed", error=str(e))
