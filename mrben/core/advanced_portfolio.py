#!/usr/bin/env python3
"""
MR BEN - Advanced Portfolio Management System
Portfolio-level risk management, correlation analysis, dynamic allocation, and optimization
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .metricsx import observe_portfolio_allocation, observe_portfolio_metric


class PortfolioStrategy(str, Enum):
    """Portfolio management strategies"""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    BLACK_LITTERMAN = "black_litterman"
    DYNAMIC_ALLOCATION = "dynamic_allocation"
    CUSTOM = "custom"


class RiskMetric(str, Enum):
    """Portfolio risk metrics"""

    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    CORRELATION = "correlation"
    BETA = "beta"


class AllocationMethod(str, Enum):
    """Portfolio allocation methods"""

    STATIC = "static"
    DYNAMIC = "dynamic"
    REBALANCING = "rebalancing"
    MOMENTUM_BASED = "momentum_based"
    REGIME_BASED = "regime_based"
    ML_OPTIMIZED = "ml_optimized"


@dataclass
class PortfolioAsset:
    """Individual asset in portfolio"""

    symbol: str
    weight: float  # Current weight (0.0 to 1.0)
    target_weight: float  # Target weight
    position_size: float  # Current position size
    unrealized_pnl: float  # Unrealized profit/loss
    risk_score: float  # Individual asset risk (0.0 to 1.0)
    correlation_group: str  # Correlation grouping
    last_updated: datetime


@dataclass
class PortfolioRisk:
    """Portfolio risk assessment"""

    total_volatility: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_score: float
    diversification_score: float
    risk_timestamp: datetime


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""

    strategy: PortfolioStrategy
    method: AllocationMethod
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    rebalancing_needed: bool
    rebalancing_trades: list[dict[str, Any]]
    expected_return: float
    expected_risk: float
    confidence_score: float
    allocation_timestamp: datetime


class AdvancedPortfolioManager:
    """
    Advanced Portfolio Management System

    Provides portfolio-level risk management, correlation analysis,
    dynamic allocation strategies, and optimization algorithms.
    """

    def __init__(
        self,
        config_path: str | None = None,
        model_dir: str = "portfolio_models",
        enable_ml: bool = True,
        enable_correlation: bool = True,
        enable_optimization: bool = True,
    ):
        self.config_path = config_path or "advanced_portfolio_config.json"
        self.model_dir = Path(model_dir)
        self.enable_ml = enable_ml
        self.enable_correlation = enable_correlation
        self.enable_optimization = enable_optimization

        # Create model directory
        self.model_dir.mkdir(exist_ok=True)

        # ML models
        self.risk_predictor: RandomForestRegressor | None = None
        self.correlation_predictor: RandomForestRegressor | None = None
        self.allocation_optimizer: RandomForestRegressor | None = None
        self.scaler: StandardScaler | None = None

        # Portfolio data
        self.portfolio_assets: dict[str, PortfolioAsset] = {}
        self.portfolio_history: deque = deque(maxlen=10000)
        self.risk_history: deque = deque(maxlen=5000)
        self.allocation_history: deque = deque(maxlen=5000)

        # Performance tracking
        self.model_performance: dict[str, dict[str, float]] = {}
        self.portfolio_metrics: dict[str, list[float]] = {}

        # Threading
        self._lock = threading.RLock()
        self._portfolio_thread: threading.Thread | None = None
        self._stop_portfolio = threading.Event()

        # Load configuration and models
        self._load_config()
        self._load_models()

        logger.bind(evt="PORTFOLIO").info(
            "advanced_portfolio_manager_initialized",
            ml_enabled=enable_ml,
            correlation_enabled=enable_correlation,
        )

    def _load_config(self) -> None:
        """Load advanced portfolio management configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    config = json.load(f)

                # Update configuration
                if 'enable_ml' in config:
                    self.enable_ml = config['enable_ml']
                if 'enable_correlation' in config:
                    self.enable_correlation = config['enable_correlation']
                if 'enable_optimization' in config:
                    self.enable_optimization = config['enable_optimization']

                logger.bind(evt="PORTFOLIO").info(
                    "advanced_portfolio_config_loaded", config_path=self.config_path
                )
            else:
                logger.bind(evt="PORTFOLIO").info(
                    "advanced_portfolio_config_not_found", config_path=self.config_path
                )
        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning(
                "advanced_portfolio_config_load_failed", error=str(e)
            )

    def _load_models(self) -> None:
        """Load pre-trained portfolio management models"""
        try:
            if not self.enable_ml:
                return

            # Load risk predictor
            risk_path = self.model_dir / "risk_predictor.pkl"
            if risk_path.exists():
                self.risk_predictor = joblib.load(risk_path)
                logger.bind(evt="PORTFOLIO").info("risk_predictor_loaded")

            # Load correlation predictor
            correlation_path = self.model_dir / "correlation_predictor.pkl"
            if correlation_path.exists():
                self.correlation_predictor = joblib.load(correlation_path)
                logger.bind(evt="PORTFOLIO").info("correlation_predictor_loaded")

            # Load allocation optimizer
            allocation_path = self.model_dir / "allocation_optimizer.pkl"
            if allocation_path.exists():
                self.allocation_optimizer = joblib.load(allocation_path)
                logger.bind(evt="PORTFOLIO").info("allocation_optimizer_loaded")

            # Load scaler
            scaler_path = self.model_dir / "portfolio_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.bind(evt="PORTFOLIO").info("portfolio_scaler_loaded")

            # If no models exist, create default ones
            if not self.risk_predictor:
                self._create_default_models()

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("model_loading_failed", error=str(e))
            self._create_default_models()

    def _create_default_models(self) -> None:
        """Create default portfolio management models"""
        try:
            # Create risk predictor
            self.risk_predictor = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )

            # Create correlation predictor
            self.correlation_predictor = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )

            # Create allocation optimizer
            self.allocation_optimizer = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42
            )

            # Create scaler
            self.scaler = StandardScaler()

            logger.bind(evt="PORTFOLIO").info("default_portfolio_models_created")

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("default_model_creation_failed", error=str(e))

    def add_asset(
        self, symbol: str, weight: float, position_size: float = 0.0, unrealized_pnl: float = 0.0
    ) -> None:
        """Add or update asset in portfolio"""
        try:
            with self._lock:
                asset = PortfolioAsset(
                    symbol=symbol,
                    weight=weight,
                    target_weight=weight,
                    position_size=position_size,
                    unrealized_pnl=unrealized_pnl,
                    risk_score=0.5,  # Default risk score
                    correlation_group="default",
                    last_updated=datetime.now(UTC),
                )

                self.portfolio_assets[symbol] = asset

                # Record metrics
                observe_portfolio_metric("asset_count", len(self.portfolio_assets))
                observe_portfolio_metric(
                    "total_weight", sum(a.weight for a in self.portfolio_assets.values())
                )

                logger.bind(evt="PORTFOLIO").info(
                    "asset_added_to_portfolio", symbol=symbol, weight=weight
                )

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("asset_addition_failed", symbol=symbol, error=str(e))

    def remove_asset(self, symbol: str) -> bool:
        """Remove asset from portfolio"""
        try:
            with self._lock:
                if symbol in self.portfolio_assets:
                    del self.portfolio_assets[symbol]

                    # Record metrics
                    observe_portfolio_metric("asset_count", len(self.portfolio_assets))
                    observe_portfolio_metric(
                        "total_weight", sum(a.weight for a in self.portfolio_assets.values())
                    )

                    logger.bind(evt="PORTFOLIO").info("asset_removed_from_portfolio", symbol=symbol)
                    return True
                else:
                    logger.bind(evt="PORTFOLIO").warning(
                        "asset_not_found_in_portfolio", symbol=symbol
                    )
                    return False

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("asset_removal_failed", symbol=symbol, error=str(e))
            return False

    def update_asset_position(
        self, symbol: str, position_size: float, unrealized_pnl: float
    ) -> bool:
        """Update asset position and PnL"""
        try:
            with self._lock:
                if symbol in self.portfolio_assets:
                    asset = self.portfolio_assets[symbol]
                    asset.position_size = position_size
                    asset.unrealized_pnl = unrealized_pnl
                    asset.last_updated = datetime.now(UTC)

                    # Record metrics
                    observe_portfolio_metric(
                        "total_unrealized_pnl",
                        sum(a.unrealized_pnl for a in self.portfolio_assets.values()),
                    )

                    logger.bind(evt="PORTFOLIO").info(
                        "asset_position_updated",
                        symbol=symbol,
                        position=position_size,
                        pnl=unrealized_pnl,
                    )
                    return True
                else:
                    logger.bind(evt="PORTFOLIO").warning(
                        "asset_not_found_for_update", symbol=symbol
                    )
                    return False

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error(
                "asset_position_update_failed", symbol=symbol, error=str(e)
            )
            return False

    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            with self._lock:
                if not self.portfolio_assets:
                    return self._create_default_portfolio_risk()

                # Calculate basic metrics
                total_value = sum(abs(a.position_size) for a in self.portfolio_assets.values())
                total_pnl = sum(a.unrealized_pnl for a in self.portfolio_assets.values())

                if total_value == 0:
                    return self._create_default_portfolio_risk()

                # Calculate weights
                weights = np.array(
                    [abs(a.position_size) / total_value for a in self.portfolio_assets.values()]
                )

                # Mock correlation matrix (in real implementation, this would use historical data)
                n_assets = len(self.portfolio_assets)
                correlation_matrix = (
                    np.eye(n_assets) * 0.3 + np.random.random((n_assets, n_assets)) * 0.4
                )
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                np.fill_diagonal(correlation_matrix, 1.0)

                # Mock volatility (in real implementation, this would use historical data)
                volatilities = np.array([0.15 + np.random.random() * 0.1 for _ in range(n_assets)])

                # Calculate portfolio volatility
                portfolio_variance = np.dot(
                    weights.T,
                    np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights),
                )
                portfolio_volatility = np.sqrt(portfolio_variance)

                # Calculate VaR and CVaR (simplified)
                var_95 = portfolio_volatility * 1.645
                cvar_95 = portfolio_volatility * 2.06

                # Calculate Sharpe ratio (simplified)
                risk_free_rate = 0.02  # 2% annual
                portfolio_return = total_pnl / total_value if total_value > 0 else 0
                sharpe_ratio = (
                    (portfolio_return - risk_free_rate) / portfolio_volatility
                    if portfolio_volatility > 0
                    else 0
                )

                # Calculate other ratios
                sortino_ratio = sharpe_ratio  # Simplified
                calmar_ratio = sharpe_ratio  # Simplified

                # Calculate correlation and diversification scores
                correlation_score = np.mean(correlation_matrix[np.triu_indices(n_assets, k=1)])
                diversification_score = 1.0 - correlation_score

                # Mock max drawdown
                max_drawdown = portfolio_volatility * 0.5

                risk_metrics = PortfolioRisk(
                    total_volatility=portfolio_volatility,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    calmar_ratio=calmar_ratio,
                    correlation_score=correlation_score,
                    diversification_score=diversification_score,
                    risk_timestamp=datetime.now(UTC),
                )

                # Store in history
                self.risk_history.append(asdict(risk_metrics))

                # Record metrics
                observe_portfolio_metric("portfolio_volatility", portfolio_volatility)
                observe_portfolio_metric("portfolio_var", var_95)
                observe_portfolio_metric("portfolio_sharpe", sharpe_ratio)
                observe_portfolio_metric("diversification_score", diversification_score)

                return risk_metrics

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("portfolio_risk_calculation_failed", error=str(e))
            return self._create_default_portfolio_risk()

    def optimize_portfolio_allocation(
        self,
        strategy: PortfolioStrategy = PortfolioStrategy.RISK_PARITY,
        method: AllocationMethod = AllocationMethod.ML_OPTIMIZED,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation based on strategy and method"""
        try:
            with self._lock:
                if not self.portfolio_assets:
                    return self._create_default_portfolio_allocation()

                # Get current weights
                current_weights = {
                    symbol: asset.weight for symbol, asset in self.portfolio_assets.items()
                }

                # Calculate target weights based on strategy
                if strategy == PortfolioStrategy.EQUAL_WEIGHT:
                    target_weights = self._equal_weight_allocation()
                elif strategy == PortfolioStrategy.RISK_PARITY:
                    target_weights = self._risk_parity_allocation()
                elif strategy == PortfolioStrategy.MAX_SHARPE:
                    target_weights = self._max_sharpe_allocation()
                elif strategy == PortfolioStrategy.MIN_VARIANCE:
                    target_weights = self._min_variance_allocation()
                else:
                    target_weights = self._equal_weight_allocation()

                # Apply ML optimization if enabled
                if self.enable_ml and method == AllocationMethod.ML_OPTIMIZED:
                    target_weights = self._ml_optimize_allocation(target_weights, current_weights)

                # Calculate rebalancing needs
                rebalancing_needed = self._check_rebalancing_needed(current_weights, target_weights)
                rebalancing_trades = (
                    self._calculate_rebalancing_trades(current_weights, target_weights)
                    if rebalancing_needed
                    else []
                )

                # Calculate expected metrics
                expected_return = self._calculate_expected_return(target_weights)
                expected_risk = self._calculate_expected_risk(target_weights)
                confidence_score = self._calculate_allocation_confidence(
                    target_weights, current_weights
                )

                allocation = PortfolioAllocation(
                    strategy=strategy,
                    method=method,
                    target_weights=target_weights,
                    current_weights=current_weights,
                    rebalancing_needed=rebalancing_needed,
                    rebalancing_trades=rebalancing_trades,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    confidence_score=confidence_score,
                    allocation_timestamp=datetime.now(UTC),
                )

                # Store in history
                self.allocation_history.append(asdict(allocation))

                # Record metrics
                observe_portfolio_allocation(strategy.value, method.value, confidence_score)

                return allocation

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error(
                "portfolio_allocation_optimization_failed", error=str(e)
            )
            return self._create_default_portfolio_allocation()

    def _equal_weight_allocation(self) -> dict[str, float]:
        """Equal weight allocation strategy"""
        n_assets = len(self.portfolio_assets)
        if n_assets == 0:
            return {}

        equal_weight = 1.0 / n_assets
        return {symbol: equal_weight for symbol in self.portfolio_assets.keys()}

    def _risk_parity_allocation(self) -> dict[str, float]:
        """Risk parity allocation strategy"""
        try:
            if not self.portfolio_assets:
                return {}

            # Mock risk scores (in real implementation, this would use historical data)
            risk_scores = {
                symbol: 0.1 + np.random.random() * 0.2 for symbol in self.portfolio_assets.keys()
            }

            # Calculate inverse risk weights
            inverse_risks = {symbol: 1.0 / risk for symbol, risk in risk_scores.items()}
            total_inverse_risk = sum(inverse_risks.values())

            # Normalize weights
            target_weights = {
                symbol: inv_risk / total_inverse_risk for symbol, inv_risk in inverse_risks.items()
            }

            return target_weights

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("risk_parity_allocation_failed", error=str(e))
            return self._equal_weight_allocation()

    def _max_sharpe_allocation(self) -> dict[str, float]:
        """Maximum Sharpe ratio allocation strategy"""
        try:
            if not self.portfolio_assets:
                return {}

            # Mock expected returns and risks (in real implementation, this would use historical data)
            expected_returns = {
                symbol: 0.05 + np.random.random() * 0.1 for symbol in self.portfolio_assets.keys()
            }
            risks = {
                symbol: 0.1 + np.random.random() * 0.2 for symbol in self.portfolio_assets.keys()
            }

            # Calculate Sharpe ratios
            risk_free_rate = 0.02
            sharpe_ratios = {
                symbol: (ret - risk_free_rate) / risk
                for symbol, (ret, risk) in zip(
                    expected_returns.keys(),
                    zip(expected_returns.values(), risks.values(), strict=False),
                    strict=False,
                )
            }

            # Weight by Sharpe ratio
            total_sharpe = sum(max(0, sharpe) for sharpe in sharpe_ratios.values())
            if total_sharpe == 0:
                return self._equal_weight_allocation()

            target_weights = {
                symbol: max(0, sharpe) / total_sharpe for symbol, sharpe in sharpe_ratios.items()
            }

            return target_weights

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("max_sharpe_allocation_failed", error=str(e))
            return self._equal_weight_allocation()

    def _min_variance_allocation(self) -> dict[str, float]:
        """Minimum variance allocation strategy"""
        try:
            if not self.portfolio_assets:
                return {}

            # Mock risks (in real implementation, this would use historical data)
            risks = {
                symbol: 0.1 + np.random.random() * 0.2 for symbol in self.portfolio_assets.keys()
            }

            # Weight by inverse variance
            inverse_variances = {symbol: 1.0 / (risk**2) for symbol, risk in risks.items()}
            total_inverse_variance = sum(inverse_variances.values())

            target_weights = {
                symbol: inv_var / total_inverse_variance
                for symbol, inv_var in inverse_variances.items()
            }

            return target_weights

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("min_variance_allocation_failed", error=str(e))
            return self._equal_weight_allocation()

    def _ml_optimize_allocation(
        self, base_weights: dict[str, float], current_weights: dict[str, float]
    ) -> dict[str, float]:
        """ML-based allocation optimization"""
        try:
            if not self.allocation_optimizer or not self.scaler:
                return base_weights

            # Extract features for optimization
            features = self._extract_allocation_features(base_weights, current_weights)
            if features is None:
                return base_weights

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict optimal weights
            prediction = self.allocation_optimizer.predict(features_scaled)[0]

            # Apply ML adjustment to base weights
            symbols = list(base_weights.keys())
            if len(symbols) > 0:
                # Simple adjustment based on ML prediction
                adjustment_factor = min(max(prediction, 0.5), 1.5)
                adjusted_weights = {
                    symbol: weight * adjustment_factor for symbol, weight in base_weights.items()
                }

                # Normalize weights
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    adjusted_weights = {
                        symbol: weight / total_weight for symbol, weight in adjusted_weights.items()
                    }
                    return adjusted_weights

            return base_weights

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("ml_allocation_optimization_failed", error=str(e))
            return base_weights

    def _extract_allocation_features(
        self, base_weights: dict[str, float], current_weights: dict[str, float]
    ) -> list[float] | None:
        """Extract features for ML-based allocation optimization"""
        try:
            features = []

            # Weight features
            if base_weights:
                features.extend(
                    [
                        np.mean(list(base_weights.values())),
                        np.std(list(base_weights.values())),
                        np.max(list(base_weights.values())),
                        np.min(list(base_weights.values())),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Current weight features
            if current_weights:
                features.extend(
                    [
                        np.mean(list(current_weights.values())),
                        np.std(list(current_weights.values())),
                        np.max(list(current_weights.values())),
                        np.min(list(current_weights.values())),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Portfolio features
            features.extend(
                [
                    len(self.portfolio_assets),
                    sum(a.unrealized_pnl for a in self.portfolio_assets.values()),
                    sum(abs(a.position_size) for a in self.portfolio_assets.values()),
                ]
            )

            return features

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning(
                "allocation_features_extraction_failed", error=str(e)
            )
            return None

    def _check_rebalancing_needed(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        threshold: float = 0.05,
    ) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            for symbol in current_weights:
                if symbol in target_weights:
                    current = current_weights[symbol]
                    target = target_weights[symbol]
                    if abs(current - target) > threshold:
                        return True
            return False

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("rebalancing_check_failed", error=str(e))
            return False

    def _calculate_rebalancing_trades(
        self, current_weights: dict[str, float], target_weights: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Calculate required rebalancing trades"""
        try:
            trades = []

            for symbol in target_weights:
                if symbol in current_weights:
                    current = current_weights[symbol]
                    target = target_weights[symbol]
                    weight_change = target - current

                    if abs(weight_change) > 0.01:  # 1% threshold
                        trade = {
                            "symbol": symbol,
                            "action": "buy" if weight_change > 0 else "sell",
                            "weight_change": abs(weight_change),
                            "current_weight": current,
                            "target_weight": target,
                        }
                        trades.append(trade)

            return trades

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning(
                "rebalancing_trades_calculation_failed", error=str(e)
            )
            return []

    def _calculate_expected_return(self, weights: dict[str, float]) -> float:
        """Calculate expected portfolio return"""
        try:
            # Mock expected returns (in real implementation, this would use historical data)
            expected_returns = {
                symbol: 0.05 + np.random.random() * 0.1 for symbol in weights.keys()
            }

            expected_return = sum(weights[symbol] * expected_returns[symbol] for symbol in weights)
            return expected_return

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("expected_return_calculation_failed", error=str(e))
            return 0.05

    def _calculate_expected_risk(self, weights: dict[str, float]) -> float:
        """Calculate expected portfolio risk"""
        try:
            # Mock risks (in real implementation, this would use historical data)
            risks = {symbol: 0.1 + np.random.random() * 0.2 for symbol in weights.keys()}

            # Simplified risk calculation
            expected_risk = sum(weights[symbol] * risks[symbol] for symbol in weights)
            return expected_risk

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("expected_risk_calculation_failed", error=str(e))
            return 0.15

    def _calculate_allocation_confidence(
        self, target_weights: dict[str, float], current_weights: dict[str, float]
    ) -> float:
        """Calculate confidence in allocation recommendation"""
        try:
            # Base confidence
            confidence = 0.7

            # Adjust based on weight distribution
            if target_weights:
                weight_std = np.std(list(target_weights.values()))
                if weight_std < 0.1:  # Even distribution
                    confidence += 0.1
                elif weight_std > 0.3:  # Concentrated distribution
                    confidence -= 0.1

            # Adjust based on rebalancing needs
            rebalancing_needed = self._check_rebalancing_needed(current_weights, target_weights)
            if not rebalancing_needed:
                confidence += 0.1

            # Adjust based on portfolio size
            if len(self.portfolio_assets) >= 5:
                confidence += 0.1

            return max(0.1, min(confidence, 1.0))

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning(
                "allocation_confidence_calculation_failed", error=str(e)
            )
            return 0.5

    def _create_default_portfolio_risk(self) -> PortfolioRisk:
        """Create default portfolio risk metrics"""
        return PortfolioRisk(
            total_volatility=0.0,
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            correlation_score=0.0,
            diversification_score=1.0,
            risk_timestamp=datetime.now(UTC),
        )

    def _create_default_portfolio_allocation(self) -> PortfolioAllocation:
        """Create default portfolio allocation"""
        return PortfolioAllocation(
            strategy=PortfolioStrategy.EQUAL_WEIGHT,
            method=AllocationMethod.STATIC,
            target_weights={},
            current_weights={},
            rebalancing_needed=False,
            rebalancing_trades=[],
            expected_return=0.0,
            expected_risk=0.0,
            confidence_score=0.0,
            allocation_timestamp=datetime.now(UTC),
        )

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            with self._lock:
                summary = {
                    "total_assets": len(self.portfolio_assets),
                    "total_weight": sum(a.weight for a in self.portfolio_assets.values()),
                    "total_position_size": sum(
                        abs(a.position_size) for a in self.portfolio_assets.values()
                    ),
                    "total_unrealized_pnl": sum(
                        a.unrealized_pnl for a in self.portfolio_assets.values()
                    ),
                    "assets": {},
                    "risk_metrics": {},
                    "allocation_history": len(self.allocation_history),
                    "last_updated": datetime.now(UTC).isoformat(),
                }

                # Asset details
                for symbol, asset in self.portfolio_assets.items():
                    summary["assets"][symbol] = {
                        "weight": asset.weight,
                        "target_weight": asset.target_weight,
                        "position_size": asset.position_size,
                        "unrealized_pnl": asset.unrealized_pnl,
                        "risk_score": asset.risk_score,
                        "correlation_group": asset.correlation_group,
                        "last_updated": asset.last_updated.isoformat(),
                    }

                # Risk metrics
                if self.risk_history:
                    latest_risk = self.risk_history[-1]
                    summary["risk_metrics"] = latest_risk

                return summary

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("portfolio_summary_generation_failed", error=str(e))
            return {"error": str(e)}

    def train_models(self, training_data: list[dict[str, Any]]) -> bool:
        """Train portfolio management models with new data"""
        try:
            if not self.enable_ml or not training_data:
                return False

            logger.bind(evt="PORTFOLIO").info(
                "starting_portfolio_model_training", data_points=len(training_data)
            )

            # Prepare training data for risk predictor
            X_risk = []
            y_risk = []

            for data_point in training_data:
                features = self._extract_risk_features(data_point)
                if features is not None:
                    X_risk.append(features)
                    y_risk.append(data_point.get('risk_label', 0.0))

            if len(X_risk) < 20:
                logger.bind(evt="PORTFOLIO").warning(
                    "insufficient_risk_training_data", samples=len(X_risk)
                )
                return False

            X_risk = np.array(X_risk)
            y_risk = np.array(y_risk)

            # Train risk predictor
            if self.risk_predictor:
                self.risk_predictor.fit(X_risk, y_risk)

                # Evaluate model performance
                y_pred = self.risk_predictor.predict(X_risk)
                mse = mean_squared_error(y_risk, y_pred)
                r2 = r2_score(y_risk, y_pred)

                self.model_performance['risk_predictor'] = {
                    'mse': mse,
                    'r2': r2,
                    'last_trained': datetime.now(UTC).isoformat(),
                }

            # Save models
            self._save_models()

            logger.bind(evt="PORTFOLIO").info(
                "portfolio_model_training_completed",
                samples=len(X_risk),
                mse=mse if 'mse' in locals() else None,
                r2=r2 if 'r2' in locals() else None,
            )

            return True

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("portfolio_model_training_failed", error=str(e))
            return False

    def _extract_risk_features(self, data_point: dict[str, Any]) -> list[float] | None:
        """Extract features for risk model training"""
        try:
            features = []

            # Portfolio features
            features.extend(
                [
                    data_point.get('asset_count', 0),
                    data_point.get('total_weight', 0.0),
                    data_point.get('total_position_size', 0.0),
                    data_point.get('total_unrealized_pnl', 0.0),
                ]
            )

            # Risk features
            features.extend(
                [
                    data_point.get('volatility', 0.0),
                    data_point.get('var', 0.0),
                    data_point.get('correlation_score', 0.0),
                    data_point.get('diversification_score', 0.0),
                ]
            )

            return features if len(features) > 0 else None

        except Exception as e:
            logger.bind(evt="PORTFOLIO").warning("risk_features_extraction_failed", error=str(e))
            return None

    def _save_models(self) -> None:
        """Save trained portfolio models to disk"""
        try:
            if self.risk_predictor:
                joblib.dump(self.risk_predictor, self.model_dir / "risk_predictor.pkl")

            if self.correlation_predictor:
                joblib.dump(
                    self.correlation_predictor, self.model_dir / "correlation_predictor.pkl"
                )

            if self.allocation_optimizer:
                joblib.dump(self.allocation_optimizer, self.model_dir / "allocation_optimizer.pkl")

            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "portfolio_scaler.pkl")

            logger.bind(evt="PORTFOLIO").info("portfolio_models_saved_to_disk")

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("portfolio_model_saving_failed", error=str(e))

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._stop_portfolio.set()
            if self._portfolio_thread and self._portfolio_thread.is_alive():
                self._portfolio_thread.join(timeout=5.0)

            # Save models before cleanup
            self._save_models()

            logger.bind(evt="PORTFOLIO").info("advanced_portfolio_manager_cleanup_complete")

        except Exception as e:
            logger.bind(evt="PORTFOLIO").error("cleanup_failed", error=str(e))
