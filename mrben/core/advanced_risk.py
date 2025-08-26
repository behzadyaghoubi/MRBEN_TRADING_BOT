#!/usr/bin/env python3
"""
MR BEN - Advanced Risk Analytics System
Enhanced risk modeling, predictive assessment, and dynamic threshold adjustment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple, Callable
from enum import Enum
import threading
from pathlib import Path
import json
import pickle
from collections import deque

from loguru import logger
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from .typesx import DecisionCard, MarketContext
from .metricsx import observe_risk_metric, observe_risk_prediction


class RiskModelType(str, Enum):
    """Types of risk models"""
    ISOLATION_FOREST = "isolation_forest"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class RiskMetricType(str, Enum):
    """Types of risk metrics"""
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    REGIME = "regime"


@dataclass
class RiskProfile:
    """Comprehensive risk profile for a trading decision"""
    decision_id: str
    timestamp: datetime
    overall_risk_score: float
    volatility_risk: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    regime_risk: float
    confidence_interval: Tuple[float, float]
    risk_factors: Dict[str, float]
    recommendations: List[str]
    model_confidence: float


@dataclass
class RiskCorrelation:
    """Risk correlation analysis result"""
    factor1: str
    factor2: str
    correlation_coefficient: float
    significance: float
    sample_size: int
    last_updated: datetime


@dataclass
class DynamicThreshold:
    """Dynamic threshold configuration"""
    metric_name: str
    base_threshold: float
    current_threshold: float
    adjustment_factor: float
    min_threshold: float
    max_threshold: float
    last_adjusted: datetime
    adjustment_reason: str


class AdvancedRiskAnalytics:
    """
    Advanced Risk Analytics System
    
    Provides enhanced risk modeling, predictive assessment, and dynamic
    threshold adjustment using machine learning and statistical analysis.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_dir: str = "risk_models",
        enable_ml: bool = True,
        enable_dynamic_thresholds: bool = True,
        enable_correlation_analysis: bool = True
    ):
        self.config_path = config_path or "risk_analytics_config.json"
        self.model_dir = Path(model_dir)
        self.enable_ml = enable_ml
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        self.enable_correlation_analysis = enable_correlation_analysis
        
        # Create model directory
        self.model_dir.mkdir(exist_ok=True)
        
        # Risk models
        self.isolation_forest: Optional[IsolationForest] = None
        self.random_forest: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Data storage
        self.risk_history: deque = deque(maxlen=10000)
        self.correlation_matrix: Dict[str, Dict[str, RiskCorrelation]] = {}
        self.dynamic_thresholds: Dict[str, DynamicThreshold] = {}
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.prediction_accuracy: List[float] = []
        
        # Threading
        self._lock = threading.RLock()
        self._training_thread: Optional[threading.Thread] = None
        self._stop_training = threading.Event()
        
        # Load configuration and models
        self._load_config()
        self._load_models()
        self._initialize_dynamic_thresholds()
        
        logger.bind(evt="RISK").info("advanced_risk_analytics_initialized",
                                    ml_enabled=enable_ml,
                                    dynamic_thresholds=enable_dynamic_thresholds)
    
    def _load_config(self) -> None:
        """Load risk analytics configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Update configuration
                if 'enable_ml' in config:
                    self.enable_ml = config['enable_ml']
                if 'enable_dynamic_thresholds' in config:
                    self.enable_dynamic_thresholds = config['enable_dynamic_thresholds']
                if 'enable_correlation_analysis' in config:
                    self.enable_correlation_analysis = config['enable_correlation_analysis']
                
                logger.bind(evt="RISK").info("risk_analytics_config_loaded", config_path=self.config_path)
            else:
                logger.bind(evt="RISK").info("risk_analytics_config_not_found", config_path=self.config_path)
        except Exception as e:
            logger.bind(evt="RISK").warning("risk_analytics_config_load_failed", error=str(e))
    
    def _load_models(self) -> None:
        """Load pre-trained risk models"""
        try:
            if not self.enable_ml:
                return
            
            # Load isolation forest for anomaly detection
            isolation_model_path = self.model_dir / "isolation_forest.pkl"
            if isolation_model_path.exists():
                self.isolation_forest = joblib.load(isolation_model_path)
                logger.bind(evt="RISK").info("isolation_forest_model_loaded")
            
            # Load random forest for risk prediction
            rf_model_path = self.model_dir / "random_forest.pkl"
            if rf_model_path.exists():
                self.random_forest = joblib.load(rf_model_path)
                logger.bind(evt="RISK").info("random_forest_model_loaded")
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.bind(evt="RISK").info("scaler_model_loaded")
            
            # If no models exist, create default ones
            if not self.isolation_forest:
                self._create_default_models()
                
        except Exception as e:
            logger.bind(evt="RISK").error("model_loading_failed", error=str(e))
            self._create_default_models()
    
    def _create_default_models(self) -> None:
        """Create default risk models"""
        try:
            # Create isolation forest for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Create random forest for risk prediction
            self.random_forest = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Create scaler
            self.scaler = StandardScaler()
            
            logger.bind(evt="RISK").info("default_risk_models_created")
            
        except Exception as e:
            logger.bind(evt="RISK").error("default_model_creation_failed", error=str(e))
    
    def _initialize_dynamic_thresholds(self) -> None:
        """Initialize dynamic thresholds for various risk metrics"""
        try:
            base_thresholds = {
                "volatility": 0.3,
                "correlation": 0.7,
                "concentration": 0.4,
                "liquidity": 0.5,
                "regime": 0.6
            }
            
            for metric, base_threshold in base_thresholds.items():
                self.dynamic_thresholds[metric] = DynamicThreshold(
                    metric_name=metric,
                    base_threshold=base_threshold,
                    current_threshold=base_threshold,
                    adjustment_factor=1.0,
                    min_threshold=base_threshold * 0.5,
                    max_threshold=base_threshold * 1.5,
                    last_adjusted=datetime.now(timezone.utc),
                    adjustment_reason="initialization"
                )
            
            logger.bind(evt="RISK").info("dynamic_thresholds_initialized", count=len(base_thresholds))
            
        except Exception as e:
            logger.bind(evt="RISK").error("dynamic_thresholds_initialization_failed", error=str(e))
    
    def analyze_risk(
        self, 
        decision: DecisionCard, 
        context: MarketContext,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> RiskProfile:
        """
        Perform comprehensive risk analysis
        
        Args:
            decision: Trading decision to analyze
            context: Market context
            historical_data: Optional historical data for analysis
            
        Returns:
            Comprehensive risk profile
        """
        try:
            with self._lock:
                # Generate unique decision ID
                decision_id = f"{decision.action}_{decision.dir}_{int(datetime.now().timestamp())}"
                
                # Calculate individual risk components
                volatility_risk = self._calculate_volatility_risk(context, historical_data)
                correlation_risk = self._calculate_correlation_risk(context, historical_data)
                concentration_risk = self._calculate_concentration_risk(decision, context)
                liquidity_risk = self._calculate_liquidity_risk(context)
                regime_risk = self._calculate_regime_risk(context)
                
                # Calculate overall risk score
                overall_risk_score = self._calculate_overall_risk_score(
                    volatility_risk, correlation_risk, concentration_risk,
                    liquidity_risk, regime_risk, decision.dyn_conf
                )
                
                # Generate confidence interval
                confidence_interval = self._calculate_confidence_interval(overall_risk_score)
                
                # Identify risk factors
                risk_factors = self._identify_risk_factors(
                    volatility_risk, correlation_risk, concentration_risk,
                    liquidity_risk, regime_risk
                )
                
                # Generate recommendations
                recommendations = self._generate_risk_recommendations(
                    overall_risk_score, risk_factors, context
                )
                
                # Calculate model confidence
                model_confidence = self._calculate_model_confidence()
                
                # Create risk profile
                risk_profile = RiskProfile(
                    decision_id=decision_id,
                    timestamp=datetime.now(timezone.utc),
                    overall_risk_score=overall_risk_score,
                    volatility_risk=volatility_risk,
                    correlation_risk=correlation_risk,
                    concentration_risk=concentration_risk,
                    liquidity_risk=liquidity_risk,
                    regime_risk=regime_risk,
                    confidence_interval=confidence_interval,
                    risk_factors=risk_factors,
                    recommendations=recommendations,
                    model_confidence=model_confidence
                )
                
                # Store in history
                self.risk_history.append(asdict(risk_profile))
                
                # Update dynamic thresholds
                if self.enable_dynamic_thresholds:
                    self._update_dynamic_thresholds(risk_profile)
                
                # Update correlation matrix
                if self.enable_correlation_analysis:
                    self._update_correlation_analysis(risk_profile)
                
                # Record metrics
                observe_risk_metric("overall_risk", overall_risk_score)
                observe_risk_metric("volatility_risk", volatility_risk)
                observe_risk_metric("correlation_risk", correlation_risk)
                
                logger.bind(evt="RISK").info("risk_analysis_completed",
                                           decision_id=decision_id,
                                           overall_risk=overall_risk_score,
                                           model_confidence=model_confidence)
                
                return risk_profile
                
        except Exception as e:
            logger.bind(evt="RISK").error("risk_analysis_failed", error=str(e))
            # Return default risk profile
            return self._create_default_risk_profile(decision, context)
    
    def _calculate_volatility_risk(self, context: MarketContext, historical_data: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate volatility-based risk"""
        try:
            # Use ATR for volatility assessment
            atr_risk = min(context.atr_pts / 100.0, 1.0)  # Normalize ATR
            
            # If historical data available, calculate rolling volatility
            if historical_data and len(historical_data) > 20:
                prices = [d.get('close', 0) for d in historical_data[-20:]]
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    vol_risk = min(volatility, 1.0)
                    return (atr_risk + vol_risk) / 2.0
            
            return atr_risk
            
        except Exception as e:
            logger.bind(evt="RISK").warning("volatility_risk_calculation_failed", error=str(e))
            return 0.5  # Default moderate risk
    
    def _calculate_correlation_risk(self, context: MarketContext, historical_data: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate correlation-based risk"""
        try:
            # Check for high correlation with existing positions
            if context.open_positions > 0:
                # Higher correlation risk with more positions
                correlation_risk = min(context.open_positions * 0.2, 1.0)
            else:
                correlation_risk = 0.1
            
            # Check market regime correlation
            if context.regime == "HIGH":
                correlation_risk += 0.2
            elif context.regime == "LOW":
                correlation_risk += 0.1
            
            return min(correlation_risk, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("correlation_risk_calculation_failed", error=str(e))
            return 0.3  # Default moderate risk
    
    def _calculate_concentration_risk(self, decision: DecisionCard, context: MarketContext) -> float:
        """Calculate concentration risk"""
        try:
            # Position size concentration
            position_concentration = min(decision.lot / 2.0, 1.0)
            
            # Account size concentration
            if context.equity > 0:
                account_concentration = min(decision.lot * 100000 / context.equity, 1.0)
            else:
                account_concentration = 0.5
            
            # Session concentration
            session_concentration = 0.0
            if context.session == "overlap":
                session_concentration = 0.2
            elif context.session in ["london", "newyork"]:
                session_concentration = 0.1
            
            return min((position_concentration + account_concentration + session_concentration) / 3.0, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("concentration_risk_calculation_failed", error=str(e))
            return 0.4  # Default moderate risk
    
    def _calculate_liquidity_risk(self, context: MarketContext) -> float:
        """Calculate liquidity risk"""
        try:
            # Spread-based liquidity assessment
            spread_risk = min(context.spread_pts / 50.0, 1.0)
            
            # Session-based liquidity
            session_liquidity = 0.0
            if context.session == "overlap":
                session_liquidity = 0.1  # High liquidity during overlap
            elif context.session in ["asia", "weekend"]:
                session_liquidity = 0.4  # Lower liquidity
            
            # Market regime liquidity
            regime_liquidity = 0.0
            if context.regime == "HIGH":
                regime_liquidity = 0.3  # Lower liquidity in high volatility
            elif context.regime == "LOW":
                regime_liquidity = 0.1  # Higher liquidity in low volatility
            
            return min((spread_risk + session_liquidity + regime_liquidity) / 3.0, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("liquidity_risk_calculation_failed", error=str(e))
            return 0.3  # Default moderate risk
    
    def _calculate_regime_risk(self, context: MarketContext) -> float:
        """Calculate regime-based risk"""
        try:
            regime_risk = 0.0
            
            if context.regime == "HIGH":
                regime_risk = 0.8
            elif context.regime == "MEDIUM":
                regime_risk = 0.5
            elif context.regime == "LOW":
                regime_risk = 0.2
            else:
                regime_risk = 0.5
            
            # Adjust for session
            if context.session == "overlap":
                regime_risk += 0.1
            elif context.session == "weekend":
                regime_risk += 0.2
            
            return min(regime_risk, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("regime_risk_calculation_failed", error=str(e))
            return 0.5  # Default moderate risk
    
    def _calculate_overall_risk_score(
        self,
        volatility_risk: float,
        correlation_risk: float,
        concentration_risk: float,
        liquidity_risk: float,
        regime_risk: float,
        decision_confidence: float
    ) -> float:
        """Calculate overall risk score using weighted combination"""
        try:
            # Weighted risk components
            weights = {
                'volatility': 0.25,
                'correlation': 0.20,
                'concentration': 0.20,
                'liquidity': 0.15,
                'regime': 0.20
            }
            
            weighted_risk = (
                volatility_risk * weights['volatility'] +
                correlation_risk * weights['correlation'] +
                concentration_risk * weights['concentration'] +
                liquidity_risk * weights['liquidity'] +
                regime_risk * weights['regime']
            )
            
            # Adjust for decision confidence
            confidence_adjustment = (1.0 - decision_confidence) * 0.3
            
            overall_risk = weighted_risk + confidence_adjustment
            
            return min(overall_risk, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("overall_risk_calculation_failed", error=str(e))
            return 0.5  # Default moderate risk
    
    def _calculate_confidence_interval(self, risk_score: float) -> Tuple[float, float]:
        """Calculate confidence interval for risk score"""
        try:
            # Simple confidence interval based on model confidence
            confidence = 0.1  # 10% confidence interval
            
            lower_bound = max(0.0, risk_score - confidence)
            upper_bound = min(1.0, risk_score + confidence)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("confidence_interval_calculation_failed", error=str(e))
            return (risk_score, risk_score)
    
    def _identify_risk_factors(
        self,
        volatility_risk: float,
        correlation_risk: float,
        concentration_risk: float,
        liquidity_risk: float,
        regime_risk: float
    ) -> Dict[str, float]:
        """Identify key risk factors"""
        try:
            risk_factors = {}
            
            if volatility_risk > 0.7:
                risk_factors['high_volatility'] = volatility_risk
            
            if correlation_risk > 0.6:
                risk_factors['high_correlation'] = correlation_risk
            
            if concentration_risk > 0.5:
                risk_factors['position_concentration'] = concentration_risk
            
            if liquidity_risk > 0.6:
                risk_factors['low_liquidity'] = liquidity_risk
            
            if regime_risk > 0.7:
                risk_factors['high_regime_risk'] = regime_risk
            
            return risk_factors
            
        except Exception as e:
            logger.bind(evt="RISK").warning("risk_factors_identification_failed", error=str(e))
            return {}
    
    def _generate_risk_recommendations(
        self,
        overall_risk: float,
        risk_factors: Dict[str, float],
        context: MarketContext
    ) -> List[str]:
        """Generate risk-based recommendations"""
        try:
            recommendations = []
            
            if overall_risk > 0.8:
                recommendations.append("Consider reducing position size significantly")
                recommendations.append("Review risk management parameters")
            
            elif overall_risk > 0.6:
                recommendations.append("Consider reducing position size")
                recommendations.append("Monitor market conditions closely")
            
            if 'high_volatility' in risk_factors:
                recommendations.append("Use wider stop losses due to high volatility")
            
            if 'high_correlation' in risk_factors:
                recommendations.append("Diversify positions to reduce correlation risk")
            
            if 'position_concentration' in risk_factors:
                recommendations.append("Consider splitting position into smaller sizes")
            
            if 'low_liquidity' in risk_factors:
                recommendations.append("Allow for wider spreads in order execution")
            
            if context.session == "weekend":
                recommendations.append("Be cautious of weekend gap risk")
            
            if not recommendations:
                recommendations.append("Risk levels are within acceptable range")
            
            return recommendations
            
        except Exception as e:
            logger.bind(evt="RISK").warning("recommendations_generation_failed", error=str(e))
            return ["Risk analysis incomplete - proceed with caution"]
    
    def _calculate_model_confidence(self) -> float:
        """Calculate confidence in risk models"""
        try:
            if not self.prediction_accuracy:
                return 0.5  # Default confidence
            
            # Use recent prediction accuracy
            recent_accuracy = np.mean(self.prediction_accuracy[-10:])
            return min(recent_accuracy, 1.0)
            
        except Exception as e:
            logger.bind(evt="RISK").warning("model_confidence_calculation_failed", error=str(e))
            return 0.5  # Default confidence
    
    def _update_dynamic_thresholds(self, risk_profile: RiskProfile) -> None:
        """Update dynamic thresholds based on recent risk analysis"""
        try:
            for metric_name, threshold in self.dynamic_thresholds.items():
                if metric_name == "volatility":
                    current_risk = risk_profile.volatility_risk
                elif metric_name == "correlation":
                    current_risk = risk_profile.correlation_risk
                elif metric_name == "concentration":
                    current_risk = risk_profile.concentration_risk
                elif metric_name == "liquidity":
                    current_risk = risk_profile.liquidity_risk
                elif metric_name == "regime":
                    current_risk = risk_profile.regime_risk
                else:
                    continue
                
                # Calculate adjustment factor
                if current_risk > threshold.current_threshold:
                    # Increase threshold if risk is high
                    adjustment = 1.1
                    reason = "risk_above_threshold"
                else:
                    # Decrease threshold if risk is low
                    adjustment = 0.95
                    reason = "risk_below_threshold"
                
                # Apply adjustment
                new_threshold = threshold.current_threshold * adjustment
                new_threshold = max(threshold.min_threshold, min(threshold.max_threshold, new_threshold))
                
                if abs(new_threshold - threshold.current_threshold) > 0.01:  # Only update if significant change
                    threshold.current_threshold = new_threshold
                    threshold.adjustment_factor = adjustment
                    threshold.last_adjusted = datetime.now(timezone.utc)
                    threshold.adjustment_reason = reason
                    
                    logger.bind(evt="RISK").info("dynamic_threshold_adjusted",
                                               metric=metric_name,
                                               old_threshold=threshold.current_threshold / adjustment,
                                               new_threshold=new_threshold,
                                               reason=reason)
            
        except Exception as e:
            logger.bind(evt="RISK").error("dynamic_threshold_update_failed", error=str(e))
    
    def _update_correlation_analysis(self, risk_profile: RiskProfile) -> None:
        """Update correlation analysis with new risk data"""
        try:
            # This is a simplified correlation analysis
            # In production, you would use more sophisticated statistical methods
            
            risk_components = {
                'volatility': risk_profile.volatility_risk,
                'correlation': risk_profile.correlation_risk,
                'concentration': risk_profile.concentration_risk,
                'liquidity': risk_profile.liquidity_risk,
                'regime': risk_profile.regime_risk
            }
            
            # Update correlation matrix (simplified)
            for factor1 in risk_components:
                if factor1 not in self.correlation_matrix:
                    self.correlation_matrix[factor1] = {}
                
                for factor2 in risk_components:
                    if factor1 != factor2:
                        if factor2 not in self.correlation_matrix[factor1]:
                            # Initialize correlation
                            self.correlation_matrix[factor1][factor2] = RiskCorrelation(
                                factor1=factor1,
                                factor2=factor2,
                                correlation_coefficient=0.0,
                                significance=0.0,
                                sample_size=1,
                                last_updated=datetime.now(timezone.utc)
                            )
                        
                        # Simple correlation calculation (in production, use proper statistical methods)
                        correlation = self._calculate_simple_correlation(
                            risk_components[factor1], risk_components[factor2]
                        )
                        
                        self.correlation_matrix[factor1][factor2].correlation_coefficient = correlation
                        self.correlation_matrix[factor1][factor2].last_updated = datetime.now(timezone.utc)
                        self.correlation_matrix[factor1][factor2].sample_size += 1
            
        except Exception as e:
            logger.bind(evt="RISK").error("correlation_analysis_update_failed", error=str(e))
    
    def _calculate_simple_correlation(self, value1: float, value2: float) -> float:
        """Calculate simple correlation between two values"""
        try:
            # This is a simplified correlation calculation
            # In production, use proper statistical correlation methods
            
            # Normalize values to [-1, 1] range
            norm1 = (value1 - 0.5) * 2  # Convert [0,1] to [-1,1]
            norm2 = (value2 - 0.5) * 2
            
            # Simple correlation approximation
            correlation = norm1 * norm2
            
            return max(-1.0, min(1.0, correlation))
            
        except Exception as e:
            logger.bind(evt="RISK").warning("simple_correlation_calculation_failed", error=str(e))
            return 0.0
    
    def _create_default_risk_profile(self, decision: DecisionCard, context: MarketContext) -> RiskProfile:
        """Create default risk profile when analysis fails"""
        return RiskProfile(
            decision_id=f"default_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(timezone.utc),
            overall_risk_score=0.5,
            volatility_risk=0.5,
            correlation_risk=0.3,
            concentration_risk=0.4,
            liquidity_risk=0.3,
            regime_risk=0.5,
            confidence_interval=(0.4, 0.6),
            risk_factors={'default_analysis': 0.5},
            recommendations=['Risk analysis incomplete - proceed with caution'],
            model_confidence=0.3
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            with self._lock:
                if not self.risk_history:
                    return {"message": "No risk data available"}
                
                recent_risks = list(self.risk_history)[-100:]  # Last 100 analyses
                
                summary = {
                    "total_analyses": len(self.risk_history),
                    "recent_analyses": len(recent_risks),
                    "average_risk_score": np.mean([r['overall_risk_score'] for r in recent_risks]),
                    "risk_distribution": {
                        "low": len([r for r in recent_risks if r['overall_risk_score'] < 0.4]),
                        "medium": len([r for r in recent_risks if 0.4 <= r['overall_risk_score'] < 0.7]),
                        "high": len([r for r in recent_risks if r['overall_risk_score'] >= 0.7])
                    },
                    "dynamic_thresholds": {
                        name: {
                            "current": th.current_threshold,
                            "base": th.base_threshold,
                            "last_adjusted": th.last_adjusted.isoformat()
                        }
                        for name, th in self.dynamic_thresholds.items()
                    },
                    "correlation_insights": len(self.correlation_matrix),
                    "model_performance": self.model_performance,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
                return summary
                
        except Exception as e:
            logger.bind(evt="RISK").error("risk_summary_generation_failed", error=str(e))
            return {"error": str(e)}
    
    def train_models(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train risk models with new data"""
        try:
            if not self.enable_ml or not training_data:
                return False
            
            logger.bind(evt="RISK").info("starting_model_training", data_points=len(training_data))
            
            # Prepare training data
            X = []
            y = []
            
            for data_point in training_data:
                features = self._extract_features(data_point)
                if features is not None:
                    X.append(features)
                    y.append(data_point.get('risk_score', 0.5))
            
            if len(X) < 10:
                logger.bind(evt="RISK").warning("insufficient_training_data", samples=len(X))
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Train isolation forest
            if self.isolation_forest:
                self.isolation_forest.fit(X)
            
            # Train random forest
            if self.random_forest:
                self.random_forest.fit(X, y)
                
                # Evaluate model performance
                y_pred = self.random_forest.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                self.model_performance['random_forest'] = {
                    'mse': mse,
                    'r2': r2,
                    'last_trained': datetime.now(timezone.utc).isoformat()
                }
                
                # Update prediction accuracy
                accuracy = 1.0 - mse  # Simple accuracy metric
                self.prediction_accuracy.append(accuracy)
                
                # Keep only recent accuracy scores
                if len(self.prediction_accuracy) > 100:
                    self.prediction_accuracy = self.prediction_accuracy[-100:]
            
            # Save models
            self._save_models()
            
            logger.bind(evt="RISK").info("model_training_completed",
                                        samples=len(X),
                                        mse=mse if 'mse' in locals() else None,
                                        r2=r2 if 'r2' in locals() else None)
            
            return True
            
        except Exception as e:
            logger.bind(evt="RISK").error("model_training_failed", error=str(e))
            return False
    
    def _extract_features(self, data_point: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features from data point for model training"""
        try:
            features = []
            
            # Extract numerical features
            for key in ['atr_pts', 'spread_pts', 'equity', 'balance', 'open_positions']:
                value = data_point.get(key, 0)
                if isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)
            
            # Extract categorical features (encoded)
            session_encoding = {
                'london': 1.0, 'newyork': 2.0, 'asia': 3.0,
                'overlap': 4.0, 'weekend': 5.0
            }
            session = data_point.get('session', 'london')
            features.append(session_encoding.get(session, 1.0))
            
            regime_encoding = {'LOW': 1.0, 'NORMAL': 2.0, 'HIGH': 3.0}
            regime = data_point.get('regime', 'NORMAL')
            features.append(regime_encoding.get(regime, 2.0))
            
            # Extract decision features
            decision = data_point.get('decision', {})
            features.extend([
                float(decision.get('score', 0.5)),
                float(decision.get('dyn_conf', 0.5)),
                float(decision.get('lot', 0.0))
            ])
            
            return features
            
        except Exception as e:
            logger.bind(evt="RISK").warning("feature_extraction_failed", error=str(e))
            return None
    
    def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, self.model_dir / "isolation_forest.pkl")
            
            if self.random_forest:
                joblib.dump(self.random_forest, self.model_dir / "random_forest.pkl")
            
            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            
            logger.bind(evt="RISK").info("models_saved_to_disk")
            
        except Exception as e:
            logger.bind(evt="RISK").error("model_saving_failed", error=str(e))
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._stop_training.set()
            if self._training_thread and self._training_thread.is_alive():
                self._training_thread.join(timeout=5.0)
            
            # Save models before cleanup
            self._save_models()
            
            logger.bind(evt="RISK").info("advanced_risk_analytics_cleanup_complete")
            
        except Exception as e:
            logger.bind(evt="RISK").error("cleanup_failed", error=str(e))
