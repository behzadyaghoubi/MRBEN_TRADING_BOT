"""
Label Engineering for MR BEN Pro Strategy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from src.strategy.rules import evaluate_rules
from src.strategy.structure import analyze_market_structure

warnings.filterwarnings('ignore')

class LabelEngineer:
    """Professional label engineering for trading strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.label_config = config.get('label_config', {})
        
        # Labeling parameters
        self.return_periods = self.label_config.get('return_periods', [1, 4, 24])  # 1h, 4h, 1d
        self.confidence_thresholds = self.label_config.get('confidence_thresholds', [0.6, 0.8])
        self.risk_adjusted = self.label_config.get('risk_adjusted', True)
        
    def create_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive labels for training"""
        try:
            # Create copy to avoid modifying original
            labels_df = features_df.copy()
            
            # Add rule-based signals
            labels_df = self._add_rule_signals(labels_df, df)
            
            # Add return-based labels
            labels_df = self._add_return_labels(labels_df, df)
            
            # Add confidence labels
            labels_df = self._add_confidence_labels(labels_df)
            
            # Add risk-adjusted labels
            if self.risk_adjusted:
                labels_df = self._add_risk_adjusted_labels(labels_df, df)
            
            # Clean labels
            labels_df = self._clean_labels(labels_df)
            
            return labels_df
            
        except Exception as e:
            print(f"Error creating labels: {e}")
            return features_df
    
    def _add_rule_signals(self, labels_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add rule-based trading signals"""
        try:
            # Analyze market structure
            structure = analyze_market_structure(df)
            
            # Get rule decisions for each bar
            signals = []
            confidences = []
            rule_types = []
            
            for i in range(len(df)):
                if i >= 50:  # Need enough bars for analysis
                    bar_df = df.iloc[:i+1]
                    try:
                        decision = evaluate_rules(bar_df, structure, self.config)
                        signals.append(decision.side)
                        confidences.append(decision.score)
                        rule_types.append(decision.rule_type)
                    except:
                        signals.append(0)
                        confidences.append(0.5)
                        rule_types.append('UNKNOWN')
                else:
                    signals.append(0)
                    confidences.append(0.5)
                    rule_types.append('UNKNOWN')
            
            labels_df['rule_signal'] = signals
            labels_df['rule_confidence'] = confidences
            labels_df['rule_type'] = rule_types
            
            # Binary classification
            labels_df['signal_buy'] = (labels_df['rule_signal'] == 1).astype(int)
            labels_df['signal_sell'] = (labels_df['rule_signal'] == -1).astype(int)
            labels_df['signal_hold'] = (labels_df['rule_signal'] == 0).astype(int)
            
            # Multi-class classification
            labels_df['signal_class'] = labels_df['rule_signal'].map({
                1: 'BUY',
                -1: 'SELL',
                0: 'HOLD'
            })
            
            return labels_df
            
        except Exception as e:
            print(f"Error adding rule signals: {e}")
            return labels_df
    
    def _add_return_labels(self, labels_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based labels"""
        try:
            for period in self.return_periods:
                # Calculate forward returns
                future_return = df['close'].shift(-period) / df['close'] - 1
                
                # Create binary labels
                labels_df[f'return_{period}h_buy'] = (future_return > 0.001).astype(int)  # 0.1% threshold
                labels_df[f'return_{period}h_sell'] = (future_return < -0.001).astype(int)
                
                # Create multi-class labels
                labels_df[f'return_{period}h_class'] = pd.cut(
                    future_return,
                    bins=[-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf],
                    labels=['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
                )
                
                # Create regression labels
                labels_df[f'return_{period}h_value'] = future_return
                
                # Risk-adjusted returns
                if 'atr' in labels_df.columns:
                    labels_df[f'return_{period}h_risk_adj'] = future_return / labels_df['atr']
            
            return labels_df
            
        except Exception as e:
            print(f"Error adding return labels: {e}")
            return labels_df
    
    def _add_confidence_labels(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Add confidence-based labels"""
        try:
            # Confidence thresholds
            low_conf = self.confidence_thresholds[0]
            high_conf = self.confidence_thresholds[1]
            
            # Rule confidence labels
            labels_df['confidence_low'] = (labels_df['rule_confidence'] < low_conf).astype(int)
            labels_df['confidence_medium'] = ((labels_df['rule_confidence'] >= low_conf) & 
                                            (labels_df['rule_confidence'] < high_conf)).astype(int)
            labels_df['confidence_high'] = (labels_df['rule_confidence'] >= high_conf).astype(int)
            
            # PA confidence labels
            if 'pa_avg_confidence' in labels_df.columns:
                labels_df['pa_confidence_low'] = (labels_df['pa_avg_confidence'] < low_conf).astype(int)
                labels_df['pa_confidence_medium'] = ((labels_df['pa_avg_confidence'] >= low_conf) & 
                                                   (labels_df['pa_avg_confidence'] < high_conf)).astype(int)
                labels_df['pa_confidence_high'] = (labels_df['pa_avg_confidence'] >= high_conf).astype(int)
            
            return labels_df
            
        except Exception as e:
            print(f"Error adding confidence labels: {e}")
            return labels_df
    
    def _add_risk_adjusted_labels(self, labels_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted labels"""
        try:
            # Calculate rolling Sharpe ratio
            if 'return_24h_value' in labels_df.columns and 'atr' in labels_df.columns:
                returns = labels_df['return_24h_value']
                volatility = labels_df['atr'] / df['close']  # Normalized volatility
                
                # Rolling Sharpe (simplified)
                labels_df['sharpe_ratio'] = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)
                
                # Risk-adjusted signal strength
                labels_df['risk_adj_signal'] = labels_df['rule_confidence'] * (1 / (1 + volatility))
            
            # Drawdown-based labels
            if 'return_24h_value' in labels_df.columns:
                cumulative_returns = (1 + labels_df['return_24h_value']).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                
                labels_df['max_drawdown'] = drawdown.rolling(20).min()
                labels_df['drawdown_severe'] = (labels_df['max_drawdown'] < -0.05).astype(int)  # 5% threshold
            
            return labels_df
            
        except Exception as e:
            print(f"Error adding risk-adjusted labels: {e}")
            return labels_df
    
    def _clean_labels(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate labels"""
        try:
            # Remove infinite values
            labels_df = labels_df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values
            labels_df = labels_df.fillna(method='ffill')
            
            # Remove rows with remaining NaN values
            labels_df = labels_df.dropna()
            
            # Ensure label columns are properly typed
            label_columns = [col for col in labels_df.columns if 'signal_' in col or 'return_' in col or 'confidence_' in col]
            for col in label_columns:
                if labels_df[col].dtype == 'object':
                    labels_df[col] = pd.Categorical(labels_df[col])
            
            return labels_df
            
        except Exception as e:
            print(f"Error cleaning labels: {e}")
            return labels_df
    
    def get_label_names(self) -> List[str]:
        """Get list of all label names"""
        return [col for col in self.get_sample_labels().columns if 'signal_' in col or 'return_' in col or 'confidence_' in col]
    
    def get_sample_labels(self) -> pd.DataFrame:
        """Get sample labels for analysis"""
        # This would be implemented with actual data
        return pd.DataFrame()

# Convenience function
def create_labels(df: pd.DataFrame, features_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Quick label creation"""
    engineer = LabelEngineer(config)
    return engineer.create_labels(df, features_df)
