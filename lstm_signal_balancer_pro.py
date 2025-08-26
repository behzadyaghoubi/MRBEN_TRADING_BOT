#!/usr/bin/env python3
"""
MRBEN LSTM Signal Balancer Pro - Professional Signal Generation System
=====================================================================

Advanced LSTM signal balancing system that generates real BUY and SELL signals
from imbalanced LSTM probability distributions.

Problem: LSTM produces mostly HOLD (0) signals, very few BUY (1) and SELL (-1)
Solution: Intelligent threshold management with dynamic signal generation

Features:
- Configurable thresholds for each signal class
- Dynamic signal generation based on probability differences
- Signal confidence scoring
- Distribution analysis and visualization
- Export capabilities with detailed signal information

Author: MRBEN Trading System (Professional Version)
Version: 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_signal_balancer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Professional signal configuration with advanced parameters"""
    buy_threshold: float = 0.15      # Lower threshold for BUY signals
    sell_threshold: float = 0.15     # Lower threshold for SELL signals
    hold_threshold: float = 0.70     # Higher threshold for HOLD signals
    confidence_boost: float = 0.08   # Confidence boost for edge cases
    min_signal_gap: float = 0.05     # Minimum gap between competing signals
    aggressive_mode: bool = True      # More aggressive signal generation
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not (0 <= self.buy_threshold <= 1):
            logger.error(f"Invalid buy_threshold: {self.buy_threshold}")
            return False
        if not (0 <= self.sell_threshold <= 1):
            logger.error(f"Invalid sell_threshold: {self.sell_threshold}")
            return False
        if not (0 <= self.hold_threshold <= 1):
            logger.error(f"Invalid hold_threshold: {self.hold_threshold}")
            return False
        return True

class LSTMSignalBalancerPro:
    """
    Professional LSTM signal balancer with advanced signal generation algorithms
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialize professional signal balancer
        
        Args:
            config: SignalConfig object with custom parameters
        """
        self.config = config or SignalConfig()
        if not self.config.validate():
            raise ValueError("Invalid signal configuration")
        
        # Signal mapping
        self.signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        self.reverse_map = {v: k for k, v in self.signal_map.items()}
        
        logger.info("=== LSTM Signal Balancer Pro Initialized ===")
        logger.info(f"BUY threshold: {self.config.buy_threshold}")
        logger.info(f"SELL threshold: {self.config.sell_threshold}")
        logger.info(f"HOLD threshold: {self.config.hold_threshold}")
        logger.info(f"Confidence boost: {self.config.confidence_boost}")
        logger.info(f"Aggressive mode: {self.config.aggressive_mode}")

    def generate_balanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate balanced signals from LSTM probabilities using advanced algorithms
        
        Args:
            df: DataFrame with LSTM probability columns
            
        Returns:
            DataFrame with balanced signals and detailed analysis
        """
        logger.info("Starting professional signal generation...")
        
        # Validate required columns
        required_cols = ['lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['balanced_signal'] = 0  # Default to HOLD
        result_df['signal_confidence'] = 0.0
        result_df['signal_reason'] = 'HOLD'
        result_df['signal_strength'] = 0.0
        result_df['signal_priority'] = 0

        # Generate signals for each row using advanced algorithms
        for idx, row in result_df.iterrows():
            buy_prob = row['lstm_buy_proba']
            hold_prob = row['lstm_hold_proba']
            sell_prob = row['lstm_sell_proba']

            # Generate signal using professional algorithm
            signal, confidence, reason, strength, priority = self._generate_single_signal(
                buy_prob, hold_prob, sell_prob
            )

            result_df.at[idx, 'balanced_signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason
            result_df.at[idx, 'signal_strength'] = strength
            result_df.at[idx, 'signal_priority'] = priority

        # Add signal labels and categories
        result_df['signal_label'] = result_df['balanced_signal'].map(self.signal_map)
        result_df['signal_category'] = result_df['signal_confidence'].apply(self._categorize_signal)

        logger.info("Professional signal generation completed")
        return result_df

    def _generate_single_signal(self, buy_prob: float, hold_prob: float, sell_prob: float) -> Tuple[int, float, str, float, int]:
        """
        Generate a single signal using advanced professional algorithms
        
        Returns:
            Tuple of (signal_value, confidence, reason, strength, priority)
        """
        # Validate and normalize probabilities
        if not (0 <= buy_prob <= 1 and 0 <= hold_prob <= 1 and 0 <= sell_prob <= 1):
            return 0, 0.0, "INVALID_PROBABILITIES", 0.0, 0
        
        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob
        
        # Strategy 1: Clear winner with high confidence
        max_prob = max(buy_prob, hold_prob, sell_prob)
        
        # BUY signal generation
        if buy_prob >= self.config.buy_threshold and buy_prob == max_prob:
            confidence = min(buy_prob + self.config.confidence_boost, 1.0)
            strength = buy_prob - max(hold_prob, sell_prob)
            priority = 1 if buy_prob > (hold_prob + self.config.min_signal_gap) else 2
            return 1, confidence, f"BUY_STRONG_{buy_prob:.3f}", strength, priority
        
        # SELL signal generation
        if sell_prob >= self.config.sell_threshold and sell_prob == max_prob:
            confidence = min(sell_prob + self.config.confidence_boost, 1.0)
            strength = sell_prob - max(hold_prob, buy_prob)
            priority = 1 if sell_prob > (hold_prob + self.config.min_signal_gap) else 2
            return -1, confidence, f"SELL_STRONG_{sell_prob:.3f}", strength, priority
        
        # HOLD signal generation
        if hold_prob >= self.config.hold_threshold and hold_prob == max_prob:
            confidence = hold_prob
            strength = hold_prob - max(buy_prob, sell_prob)
            priority = 1
            return 0, confidence, f"HOLD_STRONG_{hold_prob:.3f}", strength, priority
        
        # Strategy 2: Competitive analysis (BUY vs SELL)
        if buy_prob >= self.config.buy_threshold and sell_prob >= self.config.sell_threshold:
            if buy_prob > sell_prob:
                confidence = min(buy_prob - sell_prob + self.config.confidence_boost, 1.0)
                strength = buy_prob - sell_prob
                priority = 2
                return 1, confidence, f"BUY_VS_SELL_{buy_prob:.3f}", strength, priority
            else:
                confidence = min(sell_prob - buy_prob + self.config.confidence_boost, 1.0)
                strength = sell_prob - buy_prob
                priority = 2
                return -1, confidence, f"SELL_VS_BUY_{sell_prob:.3f}", strength, priority
        
        # Strategy 3: Aggressive mode - lower thresholds for edge cases
        if self.config.aggressive_mode:
            # Lower thresholds for aggressive signal generation
            aggressive_buy_threshold = self.config.buy_threshold - 0.03
            aggressive_sell_threshold = self.config.sell_threshold - 0.03
            
            if buy_prob >= aggressive_buy_threshold and buy_prob > hold_prob and buy_prob > sell_prob:
                confidence = buy_prob
                strength = buy_prob - max(hold_prob, sell_prob)
                priority = 3
                return 1, confidence, f"BUY_AGGRESSIVE_{buy_prob:.3f}", strength, priority
            
            if sell_prob >= aggressive_sell_threshold and sell_prob > hold_prob and sell_prob > buy_prob:
                confidence = sell_prob
                strength = sell_prob - max(hold_prob, buy_prob)
                priority = 3
                return -1, confidence, f"SELL_AGGRESSIVE_{sell_prob:.3f}", strength, priority
        
        # Strategy 4: Relative strength analysis
        if buy_prob > hold_prob and buy_prob > sell_prob and buy_prob >= (self.config.buy_threshold - 0.05):
            confidence = buy_prob
            strength = buy_prob - max(hold_prob, sell_prob)
            priority = 4
            return 1, confidence, f"BUY_RELATIVE_{buy_prob:.3f}", strength, priority
        
        if sell_prob > hold_prob and sell_prob > buy_prob and sell_prob >= (self.config.sell_threshold - 0.05):
            confidence = sell_prob
            strength = sell_prob - max(hold_prob, buy_prob)
            priority = 4
            return -1, confidence, f"SELL_RELATIVE_{sell_prob:.3f}", strength, priority
        
        # Default: HOLD with calculated confidence
        confidence = hold_prob
        strength = hold_prob - max(buy_prob, sell_prob)
        priority = 5
        return 0, confidence, f"HOLD_DEFAULT_{hold_prob:.3f}", strength, priority
    
    def _categorize_signal(self, confidence: float) -> str:
        """Categorize signal based on confidence level"""
        if confidence >= 0.8:
            return "VERY_HIGH"
        elif confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MEDIUM"
        elif confidence >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def analyze_distribution(self, df: pd.DataFrame, signal_col: str = 'balanced_signal') -> Dict[str, Any]:
        """
        Comprehensive signal distribution analysis
        
        Returns:
            Dictionary with detailed distribution statistics
        """
        signal_counts = df[signal_col].value_counts()
        total_signals = len(df)
        
        # Calculate percentages
        signal_percentages = (signal_counts / total_signals * 100).round(2)

        # Calculate balance score (target: 30% BUY, 40% HOLD, 30% SELL)
        ideal_distribution = {1: 30, 0: 40, -1: 30}
        total_deviation = sum(abs(signal_percentages.get(signal, 0) - ideal_pct) 
                            for signal, ideal_pct in ideal_distribution.items())
        balance_score = max(0, 100 - (total_deviation / 200) * 100)

        # Calculate confidence statistics
        confidence_stats = df['signal_confidence'].describe()
        
        # Calculate strength statistics
        strength_stats = df['signal_strength'].describe()
        
        # Priority distribution
        priority_counts = df['signal_priority'].value_counts().sort_index()
        
        analysis = {
            'total_signals': total_signals,
            'signal_counts': signal_counts.to_dict(),
            'signal_percentages': signal_percentages.to_dict(),
            'balance_score': balance_score,
            'confidence_stats': confidence_stats.to_dict(),
            'strength_stats': strength_stats.to_dict(),
            'priority_distribution': priority_counts.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log comprehensive results
        logger.info("=== Professional Signal Distribution Analysis ===")
        logger.info(f"Total signals: {total_signals}")
        for signal, count in signal_counts.items():
            percentage = signal_percentages[signal]
            label = self.signal_map.get(signal, str(signal))
            logger.info(f"{label}: {count} ({percentage}%)")
        logger.info(f"Balance score: {balance_score:.2f}")
        logger.info(f"Average confidence: {confidence_stats['mean']:.3f}")
        logger.info(f"Average strength: {strength_stats['mean']:.3f}")
        
        return analysis

    def plot_distribution(self, df: pd.DataFrame, signal_col: str = 'balanced_signal', save_path: str = None):
        """
        Create professional visualization of signal distribution
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM Professional Signal Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Signal count bar plot
        signal_counts = df[signal_col].value_counts().sort_index()
        signal_labels = [self.signal_map.get(s, str(s)) for s in signal_counts.index]
        colors = ['green', 'gray', 'red']

        bars = ax1.bar(signal_labels, signal_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Signal Counts', fontweight='bold')
        ax1.set_ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, signal_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(signal_counts.values),
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        # 2. Signal percentage pie chart
        percentages = (signal_counts / len(df) * 100).round(1)
        ax2.pie(percentages.values, labels=[f"{label}\n{pct}%" for label, pct in zip(signal_labels, percentages.values)],
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Signal Distribution (%)', fontweight='bold')
        
        # 3. Confidence distribution by signal type
        if 'signal_confidence' in df.columns:
            confidence_data = []
            confidence_labels = []
            for signal in signal_counts.index:
                signal_data = df[df[signal_col] == signal]['signal_confidence']
                confidence_data.append(signal_data.values)
                confidence_labels.append(self.signal_map.get(signal, str(signal)))
            
            ax3.boxplot(confidence_data, labels=confidence_labels, patch_artist=True)
            ax3.set_title('Signal Confidence Distribution', fontweight='bold')
            ax3.set_ylabel('Confidence Score')
            ax3.grid(True, alpha=0.3)
        
        # 4. Signal strength distribution
        if 'signal_strength' in df.columns:
            strength_data = []
            strength_labels = []
            for signal in signal_counts.index:
                signal_data = df[df[signal_col] == signal]['signal_strength']
                strength_data.append(signal_data.values)
                strength_labels.append(self.signal_map.get(signal, str(signal)))
            
            ax4.boxplot(strength_data, labels=strength_labels, patch_artist=True)
            ax4.set_title('Signal Strength Distribution', fontweight='bold')
            ax4.set_ylabel('Strength Score')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Professional distribution plot saved to: {save_path}")
        
        plt.show()

    def export_signals(self, df: pd.DataFrame, output_path: str, signal_col: str = 'balanced_signal'):
        """
        Export signals to CSV with comprehensive information
        """
        export_columns = [
            'time', 'close', 'lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba',
            signal_col, 'signal_confidence', 'signal_reason', 'signal_strength', 
            'signal_priority', 'signal_label', 'signal_category'
        ]
        
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()
        
        export_df.to_csv(output_path, index=False)
        logger.info(f"Professional signals exported to: {output_path}")
        logger.info(f"Exported {len(export_df)} signals with {len(export_df.columns)} columns")

def test_threshold_combinations_pro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test different threshold combinations to find optimal balance
    """
    test_configs = [
        SignalConfig(buy_threshold=0.12, sell_threshold=0.12, hold_threshold=0.75, aggressive_mode=True),
        SignalConfig(buy_threshold=0.15, sell_threshold=0.15, hold_threshold=0.70, aggressive_mode=True),
        SignalConfig(buy_threshold=0.18, sell_threshold=0.18, hold_threshold=0.65, aggressive_mode=True),
        SignalConfig(buy_threshold=0.20, sell_threshold=0.20, hold_threshold=0.60, aggressive_mode=True),
        SignalConfig(buy_threshold=0.15, sell_threshold=0.18, hold_threshold=0.68, aggressive_mode=True),
        SignalConfig(buy_threshold=0.18, sell_threshold=0.15, hold_threshold=0.68, aggressive_mode=True),
        SignalConfig(buy_threshold=0.10, sell_threshold=0.10, hold_threshold=0.80, aggressive_mode=False),
        SignalConfig(buy_threshold=0.13, sell_threshold=0.13, hold_threshold=0.75, aggressive_mode=False),
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        logger.info(f"Testing configuration {i+1}/{len(test_configs)}")
        
        try:
            balancer = LSTMSignalBalancerPro(config)
            result_df = balancer.generate_balanced_signals(df)
            analysis = balancer.analyze_distribution(result_df)
            
            result = {
                'test_id': i + 1,
                'buy_threshold': config.buy_threshold,
                'sell_threshold': config.sell_threshold,
                'hold_threshold': config.hold_threshold,
                'aggressive_mode': config.aggressive_mode,
                'total_signals': analysis['total_signals'],
                'buy_count': analysis['signal_counts'].get(1, 0),
                'hold_count': analysis['signal_counts'].get(0, 0),
                'sell_count': analysis['signal_counts'].get(-1, 0),
                'buy_percentage': analysis['signal_percentages'].get(1, 0),
                'hold_percentage': analysis['signal_percentages'].get(0, 0),
                'sell_percentage': analysis['signal_percentages'].get(-1, 0),
                'balance_score': analysis['balance_score'],
                'avg_confidence': analysis['confidence_stats']['mean'],
                'avg_strength': analysis['strength_stats']['mean']
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error testing configuration {i+1}: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    """
    Main execution function with professional testing and analysis
    """
    # Load LSTM signals data
    try:
        df = pd.read_csv("lstm_signals_pro.csv")
        logger.info(f"Loaded professional data: {len(df)} rows")
    except FileNotFoundError:
        logger.error("File 'lstm_signals_pro.csv' not found!")
        return

    # Test different threshold combinations
    logger.info("Testing professional threshold combinations...")
    test_results = test_threshold_combinations_pro(df)
    
    # Display comprehensive results
    print("\n" + "="*100)
    print("PROFESSIONAL THRESHOLD TESTING RESULTS")
    print("="*100)
    print(test_results.to_string(index=False))
    
    # Find best configuration
    best_result = test_results.loc[test_results['balance_score'].idxmax()]
    print(f"\n🏆 BEST PROFESSIONAL CONFIGURATION:")
    print(f"   Balance Score: {best_result['balance_score']:.2f}")
    print(f"   Buy Threshold: {best_result['buy_threshold']}")
    print(f"   Sell Threshold: {best_result['sell_threshold']}")
    print(f"   Hold Threshold: {best_result['hold_threshold']}")
    print(f"   Aggressive Mode: {best_result['aggressive_mode']}")
    print(f"   Buy: {best_result['buy_percentage']:.1f}% | Hold: {best_result['hold_percentage']:.1f}% | Sell: {best_result['sell_percentage']:.1f}%")
    print(f"   Avg Confidence: {best_result['avg_confidence']:.3f}")
    print(f"   Avg Strength: {best_result['avg_strength']:.3f}")
    
    # Generate final signals with best configuration
    best_config = SignalConfig(
        buy_threshold=best_result['buy_threshold'],
        sell_threshold=best_result['sell_threshold'],
        hold_threshold=best_result['hold_threshold'],
        aggressive_mode=best_result['aggressive_mode']
    )
    
    balancer = LSTMSignalBalancerPro(best_config)
    final_df = balancer.generate_balanced_signals(df)
    
    # Comprehensive analysis and visualization
    analysis = balancer.analyze_distribution(final_df)
    balancer.plot_distribution(final_df, save_path="lstm_professional_signals_distribution.png")
    balancer.export_signals(final_df, "lstm_professional_signals_final.csv")
    
    # Save test results
    test_results.to_csv("professional_threshold_testing_results.csv", index=False)
    
    print(f"\n✅ Professional balanced signals generated successfully!")
    print(f"   Output file: lstm_professional_signals_final.csv")
    print(f"   Distribution plot: lstm_professional_signals_distribution.png")
    print(f"   Test results: professional_threshold_testing_results.csv")
    print(f"   Log file: lstm_signal_balancer.log")

if __name__ == "__main__":
    main()