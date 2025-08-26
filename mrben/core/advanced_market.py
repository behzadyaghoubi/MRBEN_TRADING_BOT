#!/usr/bin/env python3
"""
MR BEN - Advanced Market Analysis System
Enhanced market regime detection, multi-timeframe analysis, and pattern recognition
"""

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .metricsx import observe_market_metric
from .typesx import MarketContext


class MarketRegimeType(str, Enum):
    """Enhanced market regime types"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING_HIGH = "ranging_high"
    RANGING_LOW = "ranging_low"
    VOLATILE_BREAKOUT = "volatile_breakout"
    VOLATILE_MEAN_REVERSION = "volatile_mean_reversion"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    NEWS_DRIVEN = "news_driven"
    TECHNICAL_DRIVEN = "technical_driven"


class TimeframeType(str, Enum):
    """Market timeframes for analysis"""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class PatternType(str, Enum):
    """Market pattern types"""

    SUPPORT_RESISTANCE = "support_resistance"
    TREND_LINE = "trend_line"
    CHANNEL = "channel"
    TRIANGLE = "triangle"
    HEAD_SHOULDERS = "head_shoulders"
    DOUBLE_TOP_BOTTOM = "double_top_bottom"
    FLAG_PENNANT = "flag_pennant"
    WEDGE = "wedge"
    CUP_HANDLE = "cup_handle"
    ROUNDING = "rounding"


@dataclass
class MarketRegimeAnalysis:
    """Comprehensive market regime analysis result"""

    regime: MarketRegimeType
    confidence: float
    trend_strength: float
    volatility_level: float
    momentum_score: float
    support_levels: list[float]
    resistance_levels: list[float]
    key_levels: list[float]
    pattern_indicators: list[str]
    timeframe_analysis: dict[str, dict[str, Any]]
    regime_duration: timedelta
    regime_probability: float


@dataclass
class MultiTimeframeAnalysis:
    """Multi-timeframe market analysis result"""

    primary_timeframe: TimeframeType
    secondary_timeframes: list[TimeframeType]
    trend_alignment: dict[str, str]
    momentum_divergence: dict[str, bool]
    support_resistance: dict[str, list[float]]
    volatility_comparison: dict[str, float]
    timeframe_correlation: dict[str, float]


@dataclass
class PatternAnalysis:
    """Market pattern analysis result"""

    pattern_type: PatternType
    confidence: float
    completion_percentage: float
    target_levels: list[float]
    invalidation_levels: list[float]
    time_to_completion: timedelta | None
    pattern_strength: float


class AdvancedMarketAnalyzer:
    """
    Advanced Market Analysis System

    Provides enhanced market regime detection, multi-timeframe analysis,
    and advanced pattern recognition using ML and statistical methods.
    """

    def __init__(
        self,
        config_path: str | None = None,
        model_dir: str = "market_models",
        enable_ml: bool = True,
        enable_multi_timeframe: bool = True,
        enable_pattern_recognition: bool = True,
    ):
        self.config_path = config_path or "advanced_market_config.json"
        self.model_dir = Path(model_dir)
        self.enable_ml = enable_ml
        self.enable_multi_timeframe = enable_multi_timeframe
        self.enable_pattern_recognition = enable_pattern_recognition

        # Create model directory
        self.model_dir.mkdir(exist_ok=True)

        # ML models
        self.regime_classifier: RandomForestClassifier | None = None
        self.volatility_detector: IsolationForest | None = None
        self.pattern_classifier: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None

        # Data storage
        self.market_history: deque = deque(maxlen=10000)
        self.regime_history: deque = deque(maxlen=5000)
        self.pattern_history: deque = deque(maxlen=3000)

        # Performance tracking
        self.model_performance: dict[str, dict[str, float]] = {}
        self.regime_accuracy: list[float] = []

        # Threading
        self._lock = threading.RLock()
        self._analysis_thread: threading.Thread | None = None
        self._stop_analysis = threading.Event()

        # Load configuration and models
        self._load_config()
        self._load_models()

        logger.bind(evt="MARKET").info(
            "advanced_market_analyzer_initialized",
            ml_enabled=enable_ml,
            multi_timeframe=enable_multi_timeframe,
        )

    def _load_config(self) -> None:
        """Load advanced market analysis configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    config = json.load(f)

                # Update configuration
                if 'enable_ml' in config:
                    self.enable_ml = config['enable_ml']
                if 'enable_multi_timeframe' in config:
                    self.enable_multi_timeframe = config['enable_multi_timeframe']
                if 'enable_pattern_recognition' in config:
                    self.enable_pattern_recognition = config['enable_pattern_recognition']

                logger.bind(evt="MARKET").info(
                    "advanced_market_config_loaded", config_path=self.config_path
                )
            else:
                logger.bind(evt="MARKET").info(
                    "advanced_market_config_not_found", config_path=self.config_path
                )
        except Exception as e:
            logger.bind(evt="MARKET").warning("advanced_market_config_load_failed", error=str(e))

    def _load_models(self) -> None:
        """Load pre-trained market analysis models"""
        try:
            if not self.enable_ml:
                return

            # Load regime classifier
            regime_model_path = self.model_dir / "regime_classifier.pkl"
            if regime_model_path.exists():
                self.regime_classifier = joblib.load(regime_model_path)
                logger.bind(evt="MARKET").info("regime_classifier_loaded")

            # Load volatility detector
            volatility_model_path = self.model_dir / "volatility_detector.pkl"
            if volatility_model_path.exists():
                self.volatility_detector = joblib.load(volatility_model_path)
                logger.bind(evt="MARKET").info("volatility_detector_loaded")

            # Load pattern classifier
            pattern_model_path = self.model_dir / "pattern_classifier.pkl"
            if pattern_model_path.exists():
                self.pattern_classifier = joblib.load(pattern_model_path)
                logger.bind(evt="MARKET").info("pattern_classifier_loaded")

            # Load scaler
            scaler_path = self.model_dir / "market_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.bind(evt="MARKET").info("market_scaler_loaded")

            # If no models exist, create default ones
            if not self.regime_classifier:
                self._create_default_models()

        except Exception as e:
            logger.bind(evt="MARKET").error("model_loading_failed", error=str(e))
            self._create_default_models()

    def _create_default_models(self) -> None:
        """Create default market analysis models"""
        try:
            # Create regime classifier
            self.regime_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )

            # Create volatility detector
            self.volatility_detector = IsolationForest(contamination=0.1, random_state=42)

            # Create pattern classifier
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42
            )

            # Create scaler
            self.scaler = StandardScaler()

            logger.bind(evt="MARKET").info("default_market_models_created")

        except Exception as e:
            logger.bind(evt="MARKET").error("default_model_creation_failed", error=str(e))

    def analyze_market_regime(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None = None
    ) -> MarketRegimeAnalysis:
        """
        Analyze current market regime using advanced methods

        Args:
            context: Current market context
            historical_data: Optional historical data for analysis

        Returns:
            Comprehensive market regime analysis
        """
        try:
            with self._lock:
                # Basic regime detection
                basic_regime = self._detect_basic_regime(context)

                # Enhanced regime analysis
                if self.enable_ml and self.regime_classifier:
                    enhanced_regime = self._detect_enhanced_regime(context, historical_data)
                else:
                    enhanced_regime = basic_regime

                # Volatility analysis
                volatility_level = self._analyze_volatility(context, historical_data)

                # Trend strength analysis
                trend_strength = self._analyze_trend_strength(context, historical_data)

                # Momentum analysis
                momentum_score = self._analyze_momentum(context, historical_data)

                # Support/Resistance levels
                support_levels, resistance_levels = self._identify_key_levels(
                    context, historical_data
                )

                # Pattern indicators
                pattern_indicators = self._identify_pattern_indicators(context, historical_data)

                # Multi-timeframe analysis
                timeframe_analysis = self._analyze_timeframes(context, historical_data)

                # Calculate confidence
                confidence = self._calculate_regime_confidence(
                    basic_regime, enhanced_regime, volatility_level, trend_strength
                )

                # Create analysis result
                analysis = MarketRegimeAnalysis(
                    regime=enhanced_regime,
                    confidence=confidence,
                    trend_strength=trend_strength,
                    volatility_level=volatility_level,
                    momentum_score=momentum_score,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    key_levels=support_levels + resistance_levels,
                    pattern_indicators=pattern_indicators,
                    timeframe_analysis=timeframe_analysis,
                    regime_duration=timedelta(hours=1),  # Default duration
                    regime_probability=confidence,
                )

                # Store in history
                self.regime_history.append(asdict(analysis))

                # Record metrics
                observe_market_metric("regime_confidence", confidence)
                observe_market_metric("trend_strength", trend_strength)
                observe_market_metric("volatility_level", volatility_level)

                logger.bind(evt="MARKET").info(
                    "market_regime_analysis_completed",
                    regime=enhanced_regime.value,
                    confidence=confidence,
                )

                return analysis

        except Exception as e:
            logger.bind(evt="MARKET").error("market_regime_analysis_failed", error=str(e))
            # Return default analysis
            return self._create_default_regime_analysis(context)

    def _detect_basic_regime(self, context: MarketContext) -> MarketRegimeType:
        """Detect basic market regime using simple rules"""
        try:
            # Analyze price position relative to moving averages
            price = context.price
            sma20 = context.sma20
            sma50 = context.sma50

            # Calculate basic indicators
            above_sma20 = price > sma20
            above_sma50 = price > sma50
            sma20_above_sma50 = sma20 > sma50

            # Determine basic regime
            if above_sma20 and above_sma50 and sma20_above_sma50:
                return MarketRegimeType.TRENDING_UP
            elif not above_sma20 and not above_sma50 and not sma20_above_sma50:
                return MarketRegimeType.TRENDING_DOWN
            elif above_sma20 and not above_sma50:
                return MarketRegimeType.RANGING_HIGH
            elif not above_sma20 and above_sma50:
                return MarketRegimeType.RANGING_LOW
            else:
                return MarketRegimeType.LOW_VOLATILITY

        except Exception as e:
            logger.bind(evt="MARKET").warning("basic_regime_detection_failed", error=str(e))
            return MarketRegimeType.LOW_VOLATILITY

    def _detect_enhanced_regime(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> MarketRegimeType:
        """Detect enhanced market regime using ML models"""
        try:
            if not self.regime_classifier or not self.scaler:
                return self._detect_basic_regime(context)

            # Extract features for ML model
            features = self._extract_regime_features(context, historical_data)

            if features is None:
                return self._detect_basic_regime(context)

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict regime
            regime_prediction = self.regime_classifier.predict(features_scaled)[0]

            # Convert prediction to regime type
            regime_map = {
                0: MarketRegimeType.TRENDING_UP,
                1: MarketRegimeType.TRENDING_DOWN,
                2: MarketRegimeType.RANGING_HIGH,
                3: MarketRegimeType.RANGING_LOW,
                4: MarketRegimeType.VOLATILE_BREAKOUT,
                5: MarketRegimeType.VOLATILE_MEAN_REVERSION,
                6: MarketRegimeType.LOW_VOLATILITY,
                7: MarketRegimeType.HIGH_VOLATILITY,
            }

            return regime_map.get(regime_prediction, MarketRegimeType.LOW_VOLATILITY)

        except Exception as e:
            logger.bind(evt="MARKET").warning("enhanced_regime_detection_failed", error=str(e))
            return self._detect_basic_regime(context)

    def _analyze_volatility(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> float:
        """Analyze market volatility level"""
        try:
            # Use ATR for volatility assessment
            atr_volatility = min(context.atr_pts / 100.0, 1.0)

            # If historical data available, calculate rolling volatility
            if historical_data and len(historical_data) > 20:
                prices = [d.get('close', 0) for d in historical_data[-20:]]
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    vol_level = min(volatility, 1.0)
                    return (atr_volatility + vol_level) / 2.0

            return atr_volatility

        except Exception as e:
            logger.bind(evt="MARKET").warning("volatility_analysis_failed", error=str(e))
            return 0.5

    def _analyze_trend_strength(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> float:
        """Analyze trend strength"""
        try:
            # Calculate trend strength based on moving average alignment
            price = context.price
            sma20 = context.sma20
            sma50 = context.sma50

            # Distance from moving averages
            distance_sma20 = abs(price - sma20) / sma20
            distance_sma50 = abs(price - sma50) / sma50

            # Moving average slope
            sma20_slope = (
                (sma20 - context.sma20) / context.sma20 if hasattr(context, 'sma20') else 0
            )
            sma50_slope = (
                (sma50 - context.sma50) / context.sma50 if hasattr(context, 'sma50') else 0
            )

            # Calculate trend strength
            trend_strength = (
                (1.0 - distance_sma20) * 0.4
                + (1.0 - distance_sma50) * 0.3
                + abs(sma20_slope) * 0.2
                + abs(sma50_slope) * 0.1
            )

            return max(0.0, min(trend_strength, 1.0))

        except Exception as e:
            logger.bind(evt="MARKET").warning("trend_strength_analysis_failed", error=str(e))
            return 0.5

    def _analyze_momentum(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> float:
        """Analyze market momentum"""
        try:
            # Calculate momentum based on price movement and volume
            momentum_score = 0.5  # Base momentum

            # Price momentum
            if hasattr(context, 'price_change'):
                price_change = context.price_change
                if price_change > 0:
                    momentum_score += 0.2
                elif price_change < 0:
                    momentum_score -= 0.2

            # Moving average momentum
            if context.sma20 > context.sma50:
                momentum_score += 0.1
            else:
                momentum_score -= 0.1

            # Volatility momentum
            if context.atr_pts > 50:  # High volatility
                momentum_score += 0.1

            return max(0.0, min(momentum_score, 1.0))

        except Exception as e:
            logger.bind(evt="MARKET").warning("momentum_analysis_failed", error=str(e))
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
            logger.bind(evt="MARKET").warning("key_levels_identification_failed", error=str(e))
            return [], []

    def _identify_pattern_indicators(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> list[str]:
        """Identify potential pattern indicators"""
        try:
            indicators = []

            # Moving average patterns
            if context.sma20 > context.sma50:
                indicators.append("golden_cross")
            else:
                indicators.append("death_cross")

            # Price position patterns
            if context.price > context.sma20 > context.sma50:
                indicators.append("strong_uptrend")
            elif context.price < context.sma20 < context.sma50:
                indicators.append("strong_downtrend")
            elif context.price > context.sma20 and context.sma20 < context.sma50:
                indicators.append("potential_reversal")

            # Volatility patterns
            if context.atr_pts > 100:
                indicators.append("high_volatility")
            elif context.atr_pts < 20:
                indicators.append("low_volatility")

            return indicators

        except Exception as e:
            logger.bind(evt="MARKET").warning(
                "pattern_indicators_identification_failed", error=str(e)
            )
            return []

    def _analyze_timeframes(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> dict[str, dict[str, Any]]:
        """Analyze market across different timeframes"""
        try:
            timeframe_analysis = {}

            # Analyze current timeframe (assuming M15)
            current_tf = "15m"
            timeframe_analysis[current_tf] = {
                "trend": "bullish" if context.price > context.sma20 else "bearish",
                "strength": self._analyze_trend_strength(context, historical_data),
                "volatility": self._analyze_volatility(context, historical_data),
                "momentum": self._analyze_momentum(context, historical_data),
            }

            # Add other timeframes if historical data available
            if historical_data and len(historical_data) > 100:
                # Analyze higher timeframes
                timeframe_analysis["1h"] = self._analyze_timeframe_data(
                    historical_data, 4
                )  # 4x15min = 1h
                timeframe_analysis["4h"] = self._analyze_timeframe_data(
                    historical_data, 16
                )  # 16x15min = 4h

            return timeframe_analysis

        except Exception as e:
            logger.bind(evt="MARKET").warning("timeframe_analysis_failed", error=str(e))
            return {}

    def _analyze_timeframe_data(
        self, data: list[dict[str, Any]], aggregation_factor: int
    ) -> dict[str, Any]:
        """Analyze data for a specific timeframe"""
        try:
            if len(data) < aggregation_factor:
                return {"trend": "unknown", "strength": 0.5, "volatility": 0.5, "momentum": 0.5}

            # Aggregate data
            aggregated_data = []
            for i in range(0, len(data), aggregation_factor):
                chunk = data[i : i + aggregation_factor]
                if chunk:
                    # Calculate aggregated values
                    avg_price = np.mean([d.get('close', 0) for d in chunk])
                    avg_volume = np.mean([d.get('volume', 0) for d in chunk])
                    aggregated_data.append({'close': avg_price, 'volume': avg_volume})

            if len(aggregated_data) < 2:
                return {"trend": "unknown", "strength": 0.5, "volatility": 0.5, "momentum": 0.5}

            # Calculate indicators for aggregated timeframe
            prices = [d['close'] for d in aggregated_data]
            trend = "bullish" if prices[-1] > prices[0] else "bearish"
            strength = min(abs(prices[-1] - prices[0]) / prices[0], 1.0)
            volatility = np.std(np.diff(np.log(prices))) if len(prices) > 1 else 0.5
            momentum = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0.0

            return {
                "trend": trend,
                "strength": min(strength, 1.0),
                "volatility": min(volatility, 1.0),
                "momentum": max(-1.0, min(momentum, 1.0)),
            }

        except Exception as e:
            logger.bind(evt="MARKET").warning("timeframe_data_analysis_failed", error=str(e))
            return {"trend": "unknown", "strength": 0.5, "volatility": 0.5, "momentum": 0.5}

    def _calculate_regime_confidence(
        self,
        basic_regime: MarketRegimeType,
        enhanced_regime: MarketRegimeType,
        volatility_level: float,
        trend_strength: float,
    ) -> float:
        """Calculate confidence in regime analysis"""
        try:
            confidence = 0.5  # Base confidence

            # Regime consistency
            if basic_regime == enhanced_regime:
                confidence += 0.2

            # Volatility confidence
            if 0.3 <= volatility_level <= 0.7:
                confidence += 0.1  # Moderate volatility is more predictable

            # Trend strength confidence
            if trend_strength > 0.7:
                confidence += 0.1  # Strong trends are more reliable

            # Model confidence
            if self.regime_classifier:
                confidence += 0.1

            return max(0.1, min(confidence, 1.0))

        except Exception as e:
            logger.bind(evt="MARKET").warning("regime_confidence_calculation_failed", error=str(e))
            return 0.5

    def _extract_regime_features(
        self, context: MarketContext, historical_data: list[dict[str, Any]] | None
    ) -> list[float] | None:
        """Extract features for ML model training"""
        try:
            features = []

            # Price features
            features.extend(
                [context.price, context.sma20, context.sma50, context.atr_pts, context.spread_pts]
            )

            # Technical indicators
            if context.sma20 > 0 and context.sma50 > 0:
                features.extend(
                    [
                        context.price / context.sma20,  # Price to SMA20 ratio
                        context.price / context.sma50,  # Price to SMA50 ratio
                        context.sma20 / context.sma50,  # SMA20 to SMA50 ratio
                    ]
                )
            else:
                features.extend([1.0, 1.0, 1.0])

            # Market context features
            features.extend([context.equity, context.balance, context.open_positions])

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

            return features

        except Exception as e:
            logger.bind(evt="MARKET").warning("regime_features_extraction_failed", error=str(e))
            return None

    def _create_default_regime_analysis(self, context: MarketContext) -> MarketRegimeAnalysis:
        """Create default regime analysis when analysis fails"""
        return MarketRegimeAnalysis(
            regime=MarketRegimeType.LOW_VOLATILITY,
            confidence=0.3,
            trend_strength=0.5,
            volatility_level=0.5,
            momentum_score=0.5,
            support_levels=[context.price * 0.995],
            resistance_levels=[context.price * 1.005],
            key_levels=[context.price * 0.995, context.price * 1.005],
            pattern_indicators=['default_analysis'],
            timeframe_analysis={},
            regime_duration=timedelta(hours=1),
            regime_probability=0.3,
        )

    def get_market_summary(self) -> dict[str, Any]:
        """Get comprehensive market analysis summary"""
        try:
            with self._lock:
                if not self.regime_history:
                    return {"message": "No market analysis data available"}

                recent_analyses = list(self.regime_history)[-100:]  # Last 100 analyses

                summary = {
                    "total_analyses": len(self.regime_history),
                    "recent_analyses": len(recent_analyses),
                    "regime_distribution": {},
                    "average_confidence": np.mean([r['confidence'] for r in recent_analyses]),
                    "average_trend_strength": np.mean(
                        [r['trend_strength'] for r in recent_analyses]
                    ),
                    "average_volatility": np.mean([r['volatility_level'] for r in recent_analyses]),
                    "model_performance": self.model_performance,
                    "last_updated": datetime.now(UTC).isoformat(),
                }

                # Calculate regime distribution
                for analysis in recent_analyses:
                    regime = analysis['regime']
                    summary["regime_distribution"][regime] = (
                        summary["regime_distribution"].get(regime, 0) + 1
                    )

                return summary

        except Exception as e:
            logger.bind(evt="MARKET").error("market_summary_generation_failed", error=str(e))
            return {"error": str(e)}

    def train_models(self, training_data: list[dict[str, Any]]) -> bool:
        """Train market analysis models with new data"""
        try:
            if not self.enable_ml or not training_data:
                return False

            logger.bind(evt="MARKET").info(
                "starting_model_training", data_points=len(training_data)
            )

            # Prepare training data for regime classifier
            X_regime = []
            y_regime = []

            for data_point in training_data:
                features = self._extract_regime_features(
                    data_point['context'], data_point.get('historical_data', [])
                )
                if features is not None:
                    X_regime.append(features)
                    y_regime.append(data_point.get('regime_label', 0))

            if len(X_regime) < 10:
                logger.bind(evt="MARKET").warning(
                    "insufficient_regime_training_data", samples=len(X_regime)
                )
                return False

            X_regime = np.array(X_regime)
            y_regime = np.array(y_regime)

            # Train regime classifier
            if self.regime_classifier:
                self.regime_classifier.fit(X_regime, y_regime)

                # Evaluate model performance
                y_pred = self.regime_classifier.predict(X_regime)
                accuracy = accuracy_score(y_regime, y_pred)

                self.model_performance['regime_classifier'] = {
                    'accuracy': accuracy,
                    'last_trained': datetime.now(UTC).isoformat(),
                }

                # Update regime accuracy
                self.regime_accuracy.append(accuracy)
                if len(self.regime_accuracy) > 100:
                    self.regime_accuracy = self.regime_accuracy[-100:]

            # Save models
            self._save_models()

            logger.bind(evt="MARKET").info(
                "model_training_completed",
                samples=len(X_regime),
                accuracy=accuracy if 'accuracy' in locals() else None,
            )

            return True

        except Exception as e:
            logger.bind(evt="MARKET").error("model_training_failed", error=str(e))
            return False

    def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            if self.regime_classifier:
                joblib.dump(self.regime_classifier, self.model_dir / "regime_classifier.pkl")

            if self.volatility_detector:
                joblib.dump(self.volatility_detector, self.model_dir / "volatility_detector.pkl")

            if self.pattern_classifier:
                joblib.dump(self.pattern_classifier, self.model_dir / "pattern_classifier.pkl")

            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "market_scaler.pkl")

            logger.bind(evt="MARKET").info("market_models_saved_to_disk")

        except Exception as e:
            logger.bind(evt="MARKET").error("model_saving_failed", error=str(e))

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._stop_analysis.set()
            if self._analysis_thread and self._analysis_thread.is_alive():
                self._analysis_thread.join(timeout=5.0)

            # Save models before cleanup
            self._save_models()

            logger.bind(evt="MARKET").info("advanced_market_analyzer_cleanup_complete")

        except Exception as e:
            logger.bind(evt="MARKET").error("cleanup_failed", error=str(e))
