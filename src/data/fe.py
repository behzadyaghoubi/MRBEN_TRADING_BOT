"""
Feature Engineering for MR BEN Pro Strategy
"""

import warnings

import numpy as np
import pandas as pd

from src.strategy.indicators import TechnicalIndicators
from src.strategy.pa import PriceActionValidator
from src.strategy.structure import MarketStructureAnalyzer

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Comprehensive feature engineering for professional trading"""

    def __init__(self, config: dict):
        self.config = config
        self.tech_indicators = TechnicalIndicators()
        self.structure_analyzer = MarketStructureAnalyzer()
        self.pa_validator = PriceActionValidator(config.get('pa_config', {}))

        # Feature configuration
        self.feature_config = config.get('feature_config', {})
        self.lookback_periods = self.feature_config.get('lookback_periods', [5, 10, 20, 50])
        self.volume_thresholds = self.feature_config.get('volume_thresholds', [1.0, 1.5, 2.0])

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            # Create copy to avoid modifying original
            features_df = df.copy()

            # Add technical features
            features_df = self._add_technical_features(features_df)

            # Add structural features
            features_df = self._add_structural_features(features_df)

            # Add Price Action features
            features_df = self._add_pa_features(features_df)

            # Add volume features
            features_df = self._add_volume_features(features_df)

            # Add derived features
            features_df = self._add_derived_features(features_df)

            # Clean and validate features
            features_df = self._clean_features(features_df)

            return features_df

        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # Moving Averages
            for period in self.lookback_periods:
                df[f'sma_{period}'] = self.tech_indicators.sma(df['close'], period)
                df[f'ema_{period}'] = self.tech_indicators.ema(df['close'], period)
                df[f'wma_{period}'] = self.tech_indicators.wma(df['close'], period)

            # Momentum Indicators
            df['rsi'] = self.tech_indicators.rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = self.tech_indicators.macd(df['close'])
            df['stoch_k'], df['stoch_d'] = self.tech_indicators.stochastic(
                df['high'], df['low'], df['close']
            )
            df['williams_r'] = self.tech_indicators.williams_r(df['high'], df['low'], df['close'])

            # Volatility Indicators
            df['atr'] = self.tech_indicators.atr(df['high'], df['low'], df['close'], 14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.tech_indicators.bollinger_bands(
                df['close']
            )
            df['kc_upper'], df['kc_middle'], df['kc_lower'] = self.tech_indicators.keltner_channels(
                df['high'], df['low'], df['close']
            )

            # Trend Indicators
            df['adx'] = self.tech_indicators.adx(df['high'], df['low'], df['close'])
            df['cci'] = self.tech_indicators.cci(df['high'], df['low'], df['close'])
            df['mfi'] = self.tech_indicators.money_flow_index(
                df['high'], df['low'], df['close'], df['volume']
            )

            # Support/Resistance
            df['pivot'] = self.tech_indicators.pivot_points(df['high'], df['low'], df['close'])
            df['fib_23'] = self.tech_indicators.fibonacci_retracement(df['high'], df['low'], 0.236)
            df['fib_38'] = self.tech_indicators.fibonacci_retracement(df['high'], df['low'], 0.382)
            df['fib_50'] = self.tech_indicators.fibonacci_retracement(df['high'], df['low'], 0.5)
            df['fib_61'] = self.tech_indicators.fibonacci_retracement(df['high'], df['low'], 0.618)

            return df

        except Exception as e:
            print(f"Error adding technical features: {e}")
            return df

    def _add_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        try:
            # Analyze market structure
            structure = self.structure_analyzer.analyze_market_structure(df)

            # Swing point features
            df['swing_high_count'] = len([s for s in structure.swings if s.type == 'HIGH'])
            df['swing_low_count'] = len([s for s in structure.swings if s.type == 'LOW'])
            df['trend_strength'] = structure.trend_strength
            df['structure_quality'] = structure.structure_quality

            # Trend direction encoding
            df['trend_up'] = (structure.last_trend == 'UP').astype(int)
            df['trend_down'] = (structure.last_trend == 'DOWN').astype(int)
            df['trend_range'] = (structure.last_trend == 'RANGE').astype(int)

            # BOS/CHOCH features
            if structure.last_bos:
                df['bos_distance'] = structure.last_bos.get('distance', 0)
                df['bos_strength'] = structure.last_bos.get('strength', 0)
            else:
                df['bos_distance'] = 0
                df['bos_strength'] = 0

            if structure.last_choch:
                df['choch_distance'] = structure.last_choch.get('distance', 0)
                df['choch_strength'] = structure.last_choch.get('strength', 0)
            else:
                df['choch_distance'] = 0
                df['choch_strength'] = 0

            return df

        except Exception as e:
            print(f"Error adding structural features: {e}")
            return df

    def _add_pa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Price Action features"""
        try:
            # Get PA patterns for each bar
            pa_results = []
            for i in range(len(df)):
                if i >= 5:  # Need at least 5 bars for pattern detection
                    bar_df = df.iloc[max(0, i - 10) : i + 1]  # Look at last 10 bars
                    patterns = self.pa_validator.validate_price_action(bar_df)
                    pa_results.append(patterns)
                else:
                    pa_results.append([])

            # Extract PA features
            df['pa_pattern_count'] = [len(patterns) for patterns in pa_results]
            df['pa_bullish_count'] = [
                len([p for p in patterns if p.direction == 'BULLISH']) for patterns in pa_results
            ]
            df['pa_bearish_count'] = [
                len([p for p in patterns if p.direction == 'BEARISH']) for patterns in pa_results
            ]
            df['pa_neutral_count'] = [
                len([p for p in patterns if p.direction == 'NEUTRAL']) for patterns in pa_results
            ]

            # Pattern strength and confidence
            df['pa_avg_strength'] = [
                np.mean([p.strength for p in patterns]) if patterns else 0
                for patterns in pa_results
            ]
            df['pa_avg_confidence'] = [
                np.mean([p.confidence for p in patterns]) if patterns else 0
                for patterns in pa_results
            ]
            df['pa_max_strength'] = [
                max([p.strength for p in patterns]) if patterns else 0 for patterns in pa_results
            ]
            df['pa_max_confidence'] = [
                max([p.confidence for p in patterns]) if patterns else 0 for patterns in pa_results
            ]

            # Volume confirmation
            df['pa_volume_confirmed'] = [
                any(p.volume_confirmed for p in patterns) if patterns else False
                for patterns in pa_results
            ]
            df['pa_volume_confirmed'] = df['pa_volume_confirmed'].astype(int)

            # Specific pattern types
            df['pa_engulfing_count'] = [
                len([p for p in patterns if 'ENGULFING' in p.pattern_type])
                for patterns in pa_results
            ]
            df['pa_fvg_count'] = [
                len([p for p in patterns if 'FVG' in p.pattern_type]) for patterns in pa_results
            ]
            df['pa_sweep_count'] = [
                len([p for p in patterns if 'LIQUIDITY_SWEEP' in p.pattern_type])
                for patterns in pa_results
            ]

            return df

        except Exception as e:
            print(f"Error adding PA features: {e}")
            return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume and liquidity features"""
        try:
            # Volume moving averages
            for period in [5, 10, 20]:
                df[f'volume_sma_{period}'] = self.tech_indicators.sma(df['volume'], period)
                df[f'volume_ema_{period}'] = self.tech_indicators.ema(df['volume'], period)

            # Volume ratios
            df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
            df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
            df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']

            # Volume thresholds
            for threshold in self.volume_thresholds:
                df[f'volume_above_{threshold}x'] = (
                    df['volume'] > threshold * df['volume_sma_20']
                ).astype(int)

            # Price-volume relationship
            df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])

            # Volume concentration
            df['volume_concentration'] = df['volume'] / df['volume'].rolling(20).sum()

            return df

        except Exception as e:
            print(f"Error adding volume features: {e}")
            return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features"""
        try:
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)

            # Volatility measures
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_10'] = df['price_change'].rolling(10).std()
            df['volatility_20'] = df['price_change'].rolling(20).std()

            # Trend strength indicators
            df['trend_strength_5'] = abs(df['ema_20'] - df['ema_50']) / df['atr']
            df['trend_strength_10'] = abs(df['ema_20'] - df['ema_50']) / df['atr']

            # Momentum indicators
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

            # Support/Resistance proximity
            df['distance_to_pivot'] = (df['close'] - df['pivot']) / df['atr']
            df['distance_to_fib_50'] = (df['close'] - df['fib_50']) / df['atr']

            # Pattern clustering
            df['pa_pattern_density'] = df['pa_pattern_count'] / 10  # Normalize by lookback

            return df

        except Exception as e:
            print(f"Error adding derived features: {e}")
            return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)

            # Forward fill missing values
            df = df.fillna(method='ffill')

            # Remove rows with remaining NaN values
            df = df.dropna()

            # Ensure all features are numeric
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]

            return df

        except Exception as e:
            print(f"Error cleaning features: {e}")
            return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names"""
        return [
            col
            for col in self.get_sample_features().columns
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ]

    def get_sample_features(self) -> pd.DataFrame:
        """Get sample features for analysis"""
        # This would be implemented with actual data
        return pd.DataFrame()


# Convenience function
def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Quick feature engineering"""
    engineer = FeatureEngineer(config)
    return engineer.engineer_features(df)
