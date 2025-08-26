#!/usr/bin/env python3
"""
MR BEN - Advanced Signal Generation System
Enhanced signal generation algorithms, signal fusion, validation, and multi-timeframe alignment
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .metricsx import observe_signal_metric
from .typesx import MarketContext


class SignalType(str, Enum):
    """Enhanced signal types for advanced generation"""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    PATTERN_BASED = "pattern_based"
    MULTI_TIMEFRAME = "multi_timeframe"
    ML_ENHANCED = "ml_enhanced"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class SignalQuality(str, Enum):
    """Signal quality assessment levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"


class SignalFusionMethod(str, Enum):
    """Signal fusion methods"""

    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    NEURAL_FUSION = "neural_fusion"


@dataclass
class SignalComponent:
    """Individual signal component"""

    signal_type: SignalType
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timeframe: str
    source: str
    timestamp: datetime
    metadata: dict[str, Any]


@dataclass
class FusedSignal:
    """Fused signal result"""

    direction: int
    strength: float
    confidence: float
    quality_score: float
    fusion_method: SignalFusionMethod
    component_signals: list[SignalComponent]
    validation_results: dict[str, Any]
    timestamp: datetime


@dataclass
class SignalValidation:
    """Signal validation results"""

    is_valid: bool
    quality_score: float
    risk_score: float
    market_alignment: float
    timeframe_consistency: float
    validation_checks: dict[str, bool]
    recommendations: list[str]


class AdvancedSignalGenerator:
    """
    Advanced Signal Generation System

    Provides enhanced signal generation algorithms, signal fusion,
    validation, and multi-timeframe alignment using ML and statistical methods.
    """

    def __init__(
        self,
        config_path: str | None = None,
        model_dir: str = "signal_models",
        enable_ml: bool = True,
        enable_fusion: bool = True,
        enable_validation: bool = True,
    ):
        self.config_path = config_path or "advanced_signals_config.json"
        self.model_dir = Path(model_dir)
        self.enable_ml = enable_ml
        self.enable_fusion = enable_fusion
        self.enable_validation = enable_validation

        # Create model directory
        self.model_dir.mkdir(exist_ok=True)

        # ML models
        self.signal_classifier: RandomForestClassifier | None = None
        self.quality_predictor: GradientBoostingRegressor | None = None
        self.fusion_model: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None

        # Data storage
        self.signal_history: deque = deque(maxlen=10000)
        self.fusion_history: deque = deque(maxlen=5000)
        self.validation_history: deque = deque(maxlen=5000)

        # Performance tracking
        self.model_performance: dict[str, dict[str, float]] = {}
        self.signal_accuracy: list[float] = []

        # Threading
        self._lock = threading.RLock()
        self._signal_thread: threading.Thread | None = None
        self._stop_signals = threading.Event()

        # Load configuration and models
        self._load_config()
        self._load_models()

        logger.bind(evt="SIGNAL").info(
            "advanced_signal_generator_initialized",
            ml_enabled=enable_ml,
            fusion_enabled=enable_fusion,
        )

    def _load_config(self) -> None:
        """Load advanced signal generation configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    config = json.load(f)

                # Update configuration
                if 'enable_ml' in config:
                    self.enable_ml = config['enable_ml']
                if 'enable_fusion' in config:
                    self.enable_fusion = config['enable_fusion']
                if 'enable_validation' in config:
                    self.enable_validation = config['enable_validation']

                logger.bind(evt="SIGNAL").info(
                    "advanced_signals_config_loaded", config_path=self.config_path
                )
            else:
                logger.bind(evt="SIGNAL").info(
                    "advanced_signals_config_not_found", config_path=self.config_path
                )
        except Exception as e:
            logger.bind(evt="SIGNAL").warning("advanced_signals_config_load_failed", error=str(e))

    def _load_models(self) -> None:
        """Load pre-trained signal generation models"""
        try:
            if not self.enable_ml:
                return

            # Load signal classifier
            classifier_path = self.model_dir / "signal_classifier.pkl"
            if classifier_path.exists():
                self.signal_classifier = joblib.load(classifier_path)
                logger.bind(evt="SIGNAL").info("signal_classifier_loaded")

            # Load quality predictor
            quality_path = self.model_dir / "quality_predictor.pkl"
            if quality_path.exists():
                self.quality_predictor = joblib.load(quality_path)
                logger.bind(evt="SIGNAL").info("quality_predictor_loaded")

            # Load fusion model
            fusion_path = self.model_dir / "fusion_model.pkl"
            if fusion_path.exists():
                self.fusion_model = joblib.load(fusion_path)
                logger.bind(evt="SIGNAL").info("fusion_model_loaded")

            # Load scaler
            scaler_path = self.model_dir / "signal_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.bind(evt="SIGNAL").info("signal_scaler_loaded")

            # If no models exist, create default ones
            if not self.signal_classifier:
                self._create_default_models()

        except Exception as e:
            logger.bind(evt="SIGNAL").error("model_loading_failed", error=str(e))
            self._create_default_models()

    def _create_default_models(self) -> None:
        """Create default signal generation models"""
        try:
            # Create signal classifier
            self.signal_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )

            # Create quality predictor
            self.quality_predictor = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )

            # Create fusion model
            self.fusion_model = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42
            )

            # Create scaler
            self.scaler = StandardScaler()

            logger.bind(evt="SIGNAL").info("default_signal_models_created")

        except Exception as e:
            logger.bind(evt="SIGNAL").error("default_model_creation_failed", error=str(e))

    def generate_trend_following_signal(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None = None
    ) -> SignalComponent:
        """Generate trend following signal based on moving averages and momentum"""
        try:
            # Calculate trend indicators
            price = context.price
            sma20 = context.sma20
            sma50 = context.sma50

            # Trend direction
            if sma20 > sma50 and price > sma20:
                direction = 1  # Uptrend
            elif sma20 < sma50 and price < sma20:
                direction = -1  # Downtrend
            else:
                direction = 0  # No clear trend

            # Trend strength calculation
            trend_strength = self._calculate_trend_strength(context, historical_data)

            # Confidence based on trend consistency
            confidence = self._calculate_trend_confidence(context, historical_data)

            # Create signal component
            signal = SignalComponent(
                signal_type=SignalType.TREND_FOLLOWING,
                direction=direction,
                strength=trend_strength,
                confidence=confidence,
                timeframe="15m",
                source="trend_analysis",
                timestamp=datetime.now(UTC),
                metadata={
                    "sma20": sma20,
                    "sma50": sma50,
                    "price_position": "above" if price > sma20 else "below",
                },
            )

            # Record metrics
            observe_signal_metric("trend_following_strength", trend_strength)
            observe_signal_metric("trend_following_confidence", confidence)

            return signal

        except Exception as e:
            logger.bind(evt="SIGNAL").error(
                "trend_following_signal_generation_failed", error=str(e)
            )
            return self._create_default_signal_component(SignalType.TREND_FOLLOWING)

    def generate_mean_reversion_signal(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None = None
    ) -> SignalComponent:
        """Generate mean reversion signal based on price deviation from moving averages"""
        try:
            # Calculate mean reversion indicators
            price = context.price
            sma20 = context.sma20
            sma50 = context.sma50

            # Price deviation from moving averages
            deviation_20 = abs(price - sma20) / sma20
            deviation_50 = abs(price - sma50) / sma50

            # Mean reversion threshold
            threshold = 0.02  # 2% deviation

            # Signal direction
            if deviation_20 > threshold or deviation_50 > threshold:
                if price > sma20:
                    direction = -1  # Price above MA, expect reversal down
                else:
                    direction = 1  # Price below MA, expect reversal up
            else:
                direction = 0  # No significant deviation

            # Signal strength based on deviation
            strength = min(deviation_20 + deviation_50, 1.0)

            # Confidence based on deviation magnitude
            confidence = min(strength * 2, 1.0)

            # Create signal component
            signal = SignalComponent(
                signal_type=SignalType.MEAN_REVERSION,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe="15m",
                source="mean_reversion_analysis",
                timestamp=datetime.now(UTC),
                metadata={
                    "deviation_20": deviation_20,
                    "deviation_50": deviation_50,
                    "threshold": threshold,
                },
            )

            # Record metrics
            observe_signal_metric("mean_reversion_strength", strength)
            observe_signal_metric("mean_reversion_confidence", confidence)

            return signal

        except Exception as e:
            logger.bind(evt="SIGNAL").error("mean_reversion_signal_generation_failed", error=str(e))
            return self._create_default_signal_component(SignalType.MEAN_REVERSION)

    def generate_breakout_signal(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None = None
    ) -> SignalComponent:
        """Generate breakout signal based on support/resistance levels"""
        try:
            # Identify key levels
            support_levels, resistance_levels = self._identify_key_levels(context, historical_data)

            # Current price position
            price = context.price

            # Find nearest levels
            nearest_support = max([s for s in support_levels if s < price], default=0)
            nearest_resistance = min(
                [r for r in resistance_levels if r > price], default=float('inf')
            )

            # Breakout detection
            breakout_threshold = context.atr_pts * 0.1  # 10% of ATR

            if price > nearest_resistance + breakout_threshold:
                direction = 1  # Bullish breakout
                strength = min((price - nearest_resistance) / context.atr_pts, 1.0)
            elif price < nearest_support - breakout_threshold:
                direction = -1  # Bearish breakout
                strength = min((nearest_support - price) / context.atr_pts, 1.0)
            else:
                direction = 0  # No breakout
                strength = 0.0

            # Confidence based on breakout strength
            confidence = min(strength * 1.5, 1.0)

            # Create signal component
            signal = SignalComponent(
                signal_type=SignalType.BREAKOUT,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe="15m",
                source="breakout_analysis",
                timestamp=datetime.now(UTC),
                metadata={
                    "nearest_support": nearest_support,
                    "nearest_resistance": nearest_resistance,
                    "breakout_threshold": breakout_threshold,
                },
            )

            # Record metrics
            observe_signal_metric("breakout_strength", strength)
            observe_signal_metric("breakout_confidence", confidence)

            return signal

        except Exception as e:
            logger.bind(evt="SIGNAL").error("breakout_signal_generation_failed", error=str(e))
            return self._create_default_signal_component(SignalType.BREAKOUT)

    def generate_momentum_signal(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None = None
    ) -> SignalComponent:
        """Generate momentum signal based on price momentum and volatility"""
        try:
            # Calculate momentum indicators
            price = context.price
            atr_pts = context.atr_pts

            # Simple momentum calculation (if historical data available)
            momentum = 0.0
            if historical_data and len(historical_data) > 5:
                recent_prices = [d.get('close', price) for d in historical_data[-5:]]
                if len(recent_prices) > 1:
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Momentum-based direction
            if abs(momentum) > 0.001:  # 0.1% threshold
                direction = 1 if momentum > 0 else -1
            else:
                direction = 0

            # Signal strength based on momentum magnitude
            strength = min(abs(momentum) * 100, 1.0)

            # Confidence based on volatility (lower volatility = higher confidence)
            volatility_factor = max(0.1, min(1.0, 50.0 / max(atr_pts, 1.0)))
            confidence = strength * volatility_factor

            # Create signal component
            signal = SignalComponent(
                signal_type=SignalType.MOMENTUM,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe="15m",
                source="momentum_analysis",
                timestamp=datetime.now(UTC),
                metadata={
                    "momentum": momentum,
                    "volatility_factor": volatility_factor,
                    "atr_pts": atr_pts,
                },
            )

            # Record metrics
            observe_signal_metric("momentum_strength", strength)
            observe_signal_metric("momentum_confidence", confidence)

            return signal

        except Exception as e:
            logger.bind(evt="SIGNAL").error("momentum_signal_generation_failed", error=str(e))
            return self._create_default_signal_component(SignalType.MOMENTUM)

    def fuse_signals(
        self,
        signals: list[SignalComponent],
        method: SignalFusionMethod = SignalFusionMethod.WEIGHTED_AVERAGE,
    ) -> FusedSignal:
        """Fuse multiple signal components into a single signal"""
        try:
            if not signals:
                return self._create_default_fused_signal()

            # Filter valid signals
            valid_signals = [s for s in signals if s.direction != 0 and s.confidence > 0.1]

            if not valid_signals:
                return self._create_default_fused_signal()

            # Apply fusion method
            if method == SignalFusionMethod.WEIGHTED_AVERAGE:
                fused_signal = self._weighted_average_fusion(valid_signals)
            elif method == SignalFusionMethod.VOTING:
                fused_signal = self._voting_fusion(valid_signals)
            elif method == SignalFusionMethod.STACKING:
                fused_signal = self._stacking_fusion(valid_signals)
            else:
                fused_signal = self._weighted_average_fusion(valid_signals)

            # Validate fused signal
            validation = self.validate_signal(fused_signal, valid_signals)

            # Update fused signal with validation results
            fused_signal.validation_results = asdict(validation)
            fused_signal.quality_score = validation.quality_score

            # Store in history
            self.fusion_history.append(asdict(fused_signal))

            # Record metrics
            observe_signal_metric("fused_signal_quality", fused_signal.quality_score)
            observe_signal_metric("fused_signal_confidence", fused_signal.confidence)

            return fused_signal

        except Exception as e:
            logger.bind(evt="SIGNAL").error("signal_fusion_failed", error=str(e))
            return self._create_default_fused_signal()

    def _weighted_average_fusion(self, signals: list[SignalComponent]) -> FusedSignal:
        """Fuse signals using weighted average method"""
        try:
            # Calculate weights based on confidence
            total_confidence = sum(s.confidence for s in signals)
            weights = [s.confidence / total_confidence for s in signals]

            # Weighted direction (positive = long, negative = short)
            weighted_direction = sum(
                s.direction * w for s, w in zip(signals, weights, strict=False)
            )

            # Determine final direction
            if abs(weighted_direction) > 0.3:
                direction = 1 if weighted_direction > 0 else -1
            else:
                direction = 0

            # Weighted strength and confidence
            strength = sum(s.strength * w for s, w in zip(signals, weights, strict=False))
            confidence = sum(s.confidence * w for s, w in zip(signals, weights, strict=False))

            return FusedSignal(
                direction=direction,
                strength=min(strength, 1.0),
                confidence=min(confidence, 1.0),
                quality_score=0.5,  # Will be updated by validation
                fusion_method=SignalFusionMethod.WEIGHTED_AVERAGE,
                component_signals=signals,
                validation_results={},
                timestamp=datetime.now(UTC),
            )

        except Exception as e:
            logger.bind(evt="SIGNAL").error("weighted_average_fusion_failed", error=str(e))
            return self._create_default_fused_signal()

    def _voting_fusion(self, signals: list[SignalComponent]) -> FusedSignal:
        """Fuse signals using voting method"""
        try:
            # Count votes for each direction
            long_votes = sum(1 for s in signals if s.direction == 1)
            short_votes = sum(1 for s in signals if s.direction == -1)

            # Determine direction by majority vote
            if long_votes > short_votes:
                direction = 1
            elif short_votes > long_votes:
                direction = -1
            else:
                direction = 0

            # Calculate strength and confidence
            direction_signals = [s for s in signals if s.direction == direction]
            if direction_signals:
                strength = np.mean([s.strength for s in direction_signals])
                confidence = np.mean([s.confidence for s in direction_signals])
            else:
                strength = 0.0
                confidence = 0.0

            return FusedSignal(
                direction=direction,
                strength=min(strength, 1.0),
                confidence=min(confidence, 1.0),
                quality_score=0.5,
                fusion_method=SignalFusionMethod.VOTING,
                component_signals=signals,
                validation_results={},
                timestamp=datetime.now(UTC),
            )

        except Exception as e:
            logger.bind(evt="SIGNAL").error("voting_fusion_failed", error=str(e))
            return self._create_default_fused_signal()

    def _stacking_fusion(self, signals: list[SignalComponent]) -> FusedSignal:
        """Fuse signals using stacking method (ML-based)"""
        try:
            if not self.fusion_model or not self.scaler:
                return self._weighted_average_fusion(signals)

            # Extract features for fusion model
            features = self._extract_fusion_features(signals)
            if features is None:
                return self._weighted_average_fusion(signals)

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict fusion result
            prediction = self.fusion_model.predict(features_scaled)[0]

            # Convert prediction to signal
            direction = int(np.sign(prediction))
            strength = min(abs(prediction), 1.0)
            confidence = min(strength * 1.2, 1.0)

            return FusedSignal(
                direction=direction,
                strength=strength,
                confidence=confidence,
                quality_score=0.5,
                fusion_method=SignalFusionMethod.STACKING,
                component_signals=signals,
                validation_results={},
                timestamp=datetime.now(UTC),
            )

        except Exception as e:
            logger.bind(evt="SIGNAL").error("stacking_fusion_failed", error=str(e))
            return self._weighted_average_fusion(signals)

    def validate_signal(
        self, signal: FusedSignal, component_signals: list[SignalComponent]
    ) -> SignalValidation:
        """Validate fused signal quality and risk"""
        try:
            validation_checks = {}
            recommendations = []

            # Check 1: Signal strength validation
            strength_valid = signal.strength > 0.3
            validation_checks["strength"] = strength_valid
            if not strength_valid:
                recommendations.append(
                    "Signal strength too low, consider waiting for stronger confirmation"
                )

            # Check 2: Confidence validation
            confidence_valid = signal.confidence > 0.4
            validation_checks["confidence"] = confidence_valid
            if not confidence_valid:
                recommendations.append("Signal confidence insufficient, review component signals")

            # Check 3: Component consistency
            directions = [s.direction for s in component_signals if s.direction != 0]
            consistency = len([d for d in directions if d == signal.direction]) / max(
                len(directions), 1
            )
            consistency_valid = consistency > 0.6
            validation_checks["consistency"] = consistency_valid
            if not consistency_valid:
                recommendations.append("Component signals show low consistency")

            # Check 4: Market alignment (if context available)
            market_alignment = 0.5  # Default value
            if hasattr(signal, 'market_context'):
                # Calculate market alignment based on context
                market_alignment = self._calculate_market_alignment(signal.market_context)

            # Check 5: Timeframe consistency
            timeframes = [s.timeframe for s in component_signals]
            timeframe_consistency = len(set(timeframes)) / max(len(timeframes), 1)
            timeframe_valid = timeframe_consistency > 0.3
            validation_checks["timeframe_consistency"] = timeframe_valid

            # Calculate quality score
            quality_score = (
                (signal.strength * 0.3)
                + (signal.confidence * 0.3)
                + (consistency * 0.2)
                + (market_alignment * 0.1)
                + (timeframe_consistency * 0.1)
            )

            # Calculate risk score
            risk_score = 1.0 - quality_score

            # Determine overall validation result
            is_valid = all(
                [strength_valid, confidence_valid, consistency_valid, quality_score > 0.5]
            )

            # Add quality-based recommendations
            if quality_score < 0.6:
                recommendations.append(
                    "Signal quality below threshold, consider additional confirmation"
                )
            if risk_score > 0.5:
                recommendations.append(
                    "High risk signal, reduce position size or wait for better conditions"
                )

            return SignalValidation(
                is_valid=is_valid,
                quality_score=quality_score,
                risk_score=risk_score,
                market_alignment=market_alignment,
                timeframe_consistency=timeframe_consistency,
                validation_checks=validation_checks,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.bind(evt="SIGNAL").error("signal_validation_failed", error=str(e))
            return self._create_default_validation()

    def _calculate_trend_strength(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> float:
        """Calculate trend strength based on moving average alignment"""
        try:
            price = context.price
            sma20 = context.sma20
            sma50 = context.sma50

            # Distance from moving averages
            distance_sma20 = abs(price - sma20) / sma20
            distance_sma50 = abs(price - sma50) / sma50

            # Moving average slope (simplified)
            slope_factor = 0.5  # Default slope factor

            # Calculate trend strength
            trend_strength = (
                (1.0 - distance_sma20) * 0.4 + (1.0 - distance_sma50) * 0.3 + slope_factor * 0.3
            )

            return max(0.0, min(trend_strength, 1.0))

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("trend_strength_calculation_failed", error=str(e))
            return 0.5

    def _calculate_trend_confidence(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> float:
        """Calculate trend confidence based on market conditions"""
        try:
            # Base confidence
            confidence = 0.5

            # Volatility adjustment
            if context.atr_pts < 30:
                confidence += 0.1  # Low volatility = higher confidence
            elif context.atr_pts > 100:
                confidence -= 0.1  # High volatility = lower confidence

            # Session adjustment
            if context.session in ['london', 'newyork']:
                confidence += 0.1  # Major sessions = higher confidence

            # Regime adjustment
            if context.regime == 'NORMAL':
                confidence += 0.1  # Normal regime = higher confidence
            elif context.regime == 'HIGH':
                confidence -= 0.1  # High volatility regime = lower confidence

            return max(0.1, min(confidence, 1.0))

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("trend_confidence_calculation_failed", error=str(e))
            return 0.5

    def _identify_key_levels(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> tuple[list[float], list[float]]:
        """Identify support and resistance levels"""
        try:
            support_levels = []
            resistance_levels = []

            # Use moving averages as key levels
            if context.sma20 > 0:
                support_levels.append(context.sma20)
            if context.sma50 > 0:
                support_levels.append(context.sma50)

            # Add current price levels
            current_price = context.price
            support_levels.append(current_price * 0.995)  # 0.5% below current
            resistance_levels.append(current_price * 1.005)  # 0.5% above current

            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))

            return support_levels, resistance_levels

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("key_levels_identification_failed", error=str(e))
            return [], []

    def _extract_fusion_features(self, signals: list[SignalComponent]) -> list[float] | None:
        """Extract features for ML-based fusion"""
        try:
            if not signals:
                return None

            features = []

            # Signal count features
            features.extend(
                [
                    len(signals),
                    len([s for s in signals if s.direction == 1]),  # Long signals
                    len([s for s in signals if s.direction == -1]),  # Short signals
                    len([s for s in signals if s.direction == 0]),  # Neutral signals
                ]
            )

            # Signal strength features
            features.extend(
                [
                    np.mean([s.strength for s in signals]),
                    np.std([s.strength for s in signals]),
                    np.max([s.strength for s in signals]),
                    np.min([s.strength for s in signals]),
                ]
            )

            # Signal confidence features
            features.extend(
                [
                    np.mean([s.confidence for s in signals]),
                    np.std([s.confidence for s in signals]),
                    np.max([s.confidence for s in signals]),
                    np.min([s.confidence for s in signals]),
                ]
            )

            # Direction consistency
            directions = [s.direction for s in signals if s.direction != 0]
            if directions:
                features.extend([np.mean(directions), np.std(directions)])
            else:
                features.extend([0.0, 0.0])

            return features

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("fusion_features_extraction_failed", error=str(e))
            return None

    def _calculate_market_alignment(self, market_context: dict[str, Any]) -> float:
        """Calculate market alignment score"""
        try:
            # Default alignment score
            alignment = 0.5

            # Adjust based on market regime
            if 'regime' in market_context:
                regime = market_context['regime']
                if regime == 'NORMAL':
                    alignment += 0.2
                elif regime == 'HIGH':
                    alignment -= 0.1

            # Adjust based on session
            if 'session' in market_context:
                session = market_context['session']
                if session in ['london', 'newyork']:
                    alignment += 0.1

            return max(0.0, min(alignment, 1.0))

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("market_alignment_calculation_failed", error=str(e))
            return 0.5

    def _create_default_signal_component(self, signal_type: SignalType) -> SignalComponent:
        """Create default signal component when generation fails"""
        return SignalComponent(
            signal_type=signal_type,
            direction=0,
            strength=0.0,
            confidence=0.0,
            timeframe="15m",
            source="default",
            timestamp=datetime.now(UTC),
            metadata={"error": "default_signal"},
        )

    def _create_default_fused_signal(self) -> FusedSignal:
        """Create default fused signal when fusion fails"""
        return FusedSignal(
            direction=0,
            strength=0.0,
            confidence=0.0,
            quality_score=0.0,
            fusion_method=SignalFusionMethod.WEIGHTED_AVERAGE,
            component_signals=[],
            validation_results={},
            timestamp=datetime.now(UTC),
        )

    def _create_default_validation(self) -> SignalValidation:
        """Create default validation when validation fails"""
        return SignalValidation(
            is_valid=False,
            quality_score=0.0,
            risk_score=1.0,
            market_alignment=0.0,
            timeframe_consistency=0.0,
            validation_checks={},
            recommendations=["Signal validation failed, manual review required"],
        )

    def get_signal_summary(self) -> dict[str, Any]:
        """Get comprehensive signal generation summary"""
        try:
            with self._lock:
                if not self.signal_history:
                    return {"message": "No signal data available"}

                recent_signals = list(self.signal_history)[-100:]  # Last 100 signals

                summary = {
                    "total_signals": len(self.signal_history),
                    "recent_signals": len(recent_signals),
                    "signal_type_distribution": {},
                    "average_quality": 0.0,
                    "average_confidence": 0.0,
                    "model_performance": self.model_performance,
                    "last_updated": datetime.now(UTC).isoformat(),
                }

                # Calculate signal type distribution
                for signal in recent_signals:
                    signal_type = signal.get('signal_type', 'unknown')
                    summary["signal_type_distribution"][signal_type] = (
                        summary["signal_type_distribution"].get(signal_type, 0) + 1
                    )

                # Calculate averages
                if recent_signals:
                    summary["average_quality"] = np.mean(
                        [s.get('quality_score', 0.0) for s in recent_signals]
                    )
                    summary["average_confidence"] = np.mean(
                        [s.get('confidence', 0.0) for s in recent_signals]
                    )

                return summary

        except Exception as e:
            logger.bind(evt="SIGNAL").error("signal_summary_generation_failed", error=str(e))
            return {"error": str(e)}

    def train_models(self, training_data: list[dict[str, Any]]) -> bool:
        """Train signal generation models with new data"""
        try:
            if not self.enable_ml or not training_data:
                return False

            logger.bind(evt="SIGNAL").info(
                "starting_signal_model_training", data_points=len(training_data)
            )

            # Prepare training data for signal classifier
            X_signal = []
            y_signal = []

            for data_point in training_data:
                features = self._extract_signal_features(data_point)
                if features is not None:
                    X_signal.append(features)
                    y_signal.append(data_point.get('signal_label', 0))

            if len(X_signal) < 20:
                logger.bind(evt="SIGNAL").warning(
                    "insufficient_signal_training_data", samples=len(X_signal)
                )
                return False

            X_signal = np.array(X_signal)
            y_signal = np.array(y_signal)

            # Train signal classifier
            if self.signal_classifier:
                self.signal_classifier.fit(X_signal, y_signal)

                # Evaluate model performance
                y_pred = self.signal_classifier.predict(X_signal)
                accuracy = accuracy_score(y_signal, y_pred)

                self.model_performance['signal_classifier'] = {
                    'accuracy': accuracy,
                    'last_trained': datetime.now(UTC).isoformat(),
                }

                # Update signal accuracy
                self.signal_accuracy.append(accuracy)
                if len(self.signal_accuracy) > 100:
                    self.signal_accuracy = self.signal_accuracy[-100:]

            # Save models
            self._save_models()

            logger.bind(evt="SIGNAL").info(
                "signal_model_training_completed",
                samples=len(X_signal),
                accuracy=accuracy if 'accuracy' in locals() else None,
            )

            return True

        except Exception as e:
            logger.bind(evt="SIGNAL").error("signal_model_training_failed", error=str(e))
            return False

    def _extract_signal_features(self, data_point: dict[str, Any]) -> list[float] | None:
        """Extract features for signal model training"""
        try:
            features = []

            # Basic signal features
            if 'signal_type' in data_point:
                signal_type = data_point['signal_type']
                # Encode signal type
                type_encoding = {
                    'trend_following': 1.0,
                    'mean_reversion': 2.0,
                    'breakout': 3.0,
                    'momentum': 4.0,
                }
                features.append(type_encoding.get(signal_type, 0.0))

            # Signal strength and confidence
            features.extend(
                [
                    data_point.get('strength', 0.0),
                    data_point.get('confidence', 0.0),
                    data_point.get('direction', 0.0),
                ]
            )

            # Market context features
            if 'market_context' in data_point:
                context = data_point['market_context']
                features.extend(
                    [
                        context.get('atr_pts', 0.0),
                        context.get('spread_pts', 0.0),
                        context.get('open_positions', 0),
                    ]
                )

            return features if len(features) > 0 else None

        except Exception as e:
            logger.bind(evt="SIGNAL").warning("signal_features_extraction_failed", error=str(e))
            return None

    def _save_models(self) -> None:
        """Save trained signal models to disk"""
        try:
            if self.signal_classifier:
                joblib.dump(self.signal_classifier, self.model_dir / "signal_classifier.pkl")

            if self.quality_predictor:
                joblib.dump(self.quality_predictor, self.model_dir / "quality_predictor.pkl")

            if self.fusion_model:
                joblib.dump(self.fusion_model, self.model_dir / "fusion_model.pkl")

            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "signal_scaler.pkl")

            logger.bind(evt="SIGNAL").info("signal_models_saved_to_disk")

        except Exception as e:
            logger.bind(evt="SIGNAL").error("signal_model_saving_failed", error=str(e))

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._stop_signals.set()
            if self._signal_thread and self._signal_thread.is_alive():
                self._signal_thread.join(timeout=5.0)

            # Save models before cleanup
            self._save_models()

            logger.bind(evt="SIGNAL").info("advanced_signal_generator_cleanup_complete")

        except Exception as e:
            logger.bind(evt="SIGNAL").error("cleanup_failed", error=str(e))
