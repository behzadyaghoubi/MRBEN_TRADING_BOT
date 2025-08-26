#!/usr/bin/env python3
"""
MRBEN LSTM Signal Balancer Ultra - Ultra Professional Signal Generation System
=============================================================================

Ultra-advanced LSTM signal balancing system that generates MORE BUY and SELL signals
than HOLD signals using intelligent algorithms and dynamic threshold management.

Target Distribution: 40% BUY, 20% HOLD, 40% SELL (instead of mostly HOLD)

Features:
- Ultra-low thresholds for BUY/SELL signals
- Intelligent signal competition algorithms
- Dynamic threshold adjustment
- Signal strength amplification
- Advanced confidence scoring
- Professional analysis and visualization

Author: MRBEN Trading System (Ultra Professional Version)
Version: 3.0
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')

# Configure ultra-professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('lstm_signal_balancer_ultra.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class UltraSignalConfig:
    """Ultra-professional signal configuration for maximum BUY/SELL generation"""

    buy_threshold: float = 0.08  # Very low threshold for BUY signals
    sell_threshold: float = 0.08  # Very low threshold for SELL signals
    hold_threshold: float = 0.85  # Very high threshold for HOLD signals
    confidence_boost: float = 0.15  # High confidence boost
    min_signal_gap: float = 0.02  # Small gap for signal competition
    aggressive_mode: bool = True  # Always aggressive
    ultra_aggressive: bool = True  # Ultra-aggressive mode
    signal_amplification: float = 1.5  # Amplify signal strengths
    relative_strength_threshold: float = 0.01  # Very low relative strength threshold

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


class LSTMSignalBalancerUltra:
    """
    Ultra-professional LSTM signal balancer that maximizes BUY/SELL signals
    """

    def __init__(self, config: UltraSignalConfig | None = None):
        """
        Initialize ultra-professional signal balancer

        Args:
            config: UltraSignalConfig object with custom parameters
        """
        self.config = config or UltraSignalConfig()
        if not self.config.validate():
            raise ValueError("Invalid ultra signal configuration")

        # Signal mapping
        self.signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        self.reverse_map = {v: k for k, v in self.signal_map.items()}

        logger.info("=== LSTM Signal Balancer Ultra Initialized ===")
        logger.info(f"BUY threshold: {self.config.buy_threshold} (ULTRA-LOW)")
        logger.info(f"SELL threshold: {self.config.sell_threshold} (ULTRA-LOW)")
        logger.info(f"HOLD threshold: {self.config.hold_threshold} (ULTRA-HIGH)")
        logger.info(f"Confidence boost: {self.config.confidence_boost}")
        logger.info(f"Ultra-aggressive mode: {self.config.ultra_aggressive}")
        logger.info(f"Signal amplification: {self.config.signal_amplification}")

    def generate_ultra_balanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ultra-balanced signals with maximum BUY/SELL generation

        Args:
            df: DataFrame with LSTM probability columns

        Returns:
            DataFrame with ultra-balanced signals and detailed analysis
        """
        logger.info("Starting ULTRA signal generation...")

        # Validate required columns
        required_cols = ['lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create result DataFrame
        result_df = df.copy()
        result_df['ultra_signal'] = 0  # Default to HOLD
        result_df['signal_confidence'] = 0.0
        result_df['signal_reason'] = 'HOLD'
        result_df['signal_strength'] = 0.0
        result_df['signal_priority'] = 0
        result_df['signal_amplified'] = 0.0

        # Generate signals for each row using ultra algorithms
        for idx, row in result_df.iterrows():
            buy_prob = row['lstm_buy_proba']
            hold_prob = row['lstm_hold_proba']
            sell_prob = row['lstm_sell_proba']

            # Generate signal using ultra-professional algorithm
            signal, confidence, reason, strength, priority, amplified = self._generate_ultra_signal(
                buy_prob, hold_prob, sell_prob
            )

            result_df.at[idx, 'ultra_signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason
            result_df.at[idx, 'signal_strength'] = strength
            result_df.at[idx, 'signal_priority'] = priority
            result_df.at[idx, 'signal_amplified'] = amplified

        # Add signal labels and categories
        result_df['signal_label'] = result_df['ultra_signal'].map(self.signal_map)
        result_df['signal_category'] = result_df['signal_confidence'].apply(self._categorize_signal)

        logger.info("ULTRA signal generation completed")
        return result_df

    def _generate_ultra_signal(
        self, buy_prob: float, hold_prob: float, sell_prob: float
    ) -> tuple[int, float, str, float, int, float]:
        """
        Generate a single signal using ultra-professional algorithms

        Returns:
            Tuple of (signal_value, confidence, reason, strength, priority, amplified)
        """
        # Validate and normalize probabilities
        if not (0 <= buy_prob <= 1 and 0 <= hold_prob <= 1 and 0 <= sell_prob <= 1):
            return 0, 0.0, "INVALID_PROBABILITIES", 0.0, 0, 0.0

        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob

        # ULTRA Strategy 1: Prefer BUY/SELL over HOLD whenever possible
        max_prob = max(buy_prob, hold_prob, sell_prob)

        # BUY signal generation (ULTRA-AGGRESSIVE)
        if buy_prob >= self.config.buy_threshold:
            # Amplify BUY signal strength
            amplified_buy = buy_prob * self.config.signal_amplification
            confidence = min(amplified_buy + self.config.confidence_boost, 1.0)
            strength = buy_prob - max(hold_prob, sell_prob)
            priority = 1 if buy_prob > sell_prob else 2
            return 1, confidence, f"BUY_ULTRA_{buy_prob:.3f}", strength, priority, amplified_buy

        # SELL signal generation (ULTRA-AGGRESSIVE)
        if sell_prob >= self.config.sell_threshold:
            # Amplify SELL signal strength
            amplified_sell = sell_prob * self.config.signal_amplification
            confidence = min(amplified_sell + self.config.confidence_boost, 1.0)
            strength = sell_prob - max(hold_prob, buy_prob)
            priority = 1 if sell_prob > buy_prob else 2
            return -1, confidence, f"SELL_ULTRA_{sell_prob:.3f}", strength, priority, amplified_sell

        # ULTRA Strategy 2: Relative strength analysis (very low thresholds)
        if (
            buy_prob > sell_prob
            and buy_prob > hold_prob
            and buy_prob >= self.config.relative_strength_threshold
        ):
            amplified_buy = buy_prob * self.config.signal_amplification
            confidence = min(amplified_buy, 1.0)
            strength = buy_prob - max(hold_prob, sell_prob)
            priority = 3
            return 1, confidence, f"BUY_RELATIVE_{buy_prob:.3f}", strength, priority, amplified_buy

        if (
            sell_prob > buy_prob
            and sell_prob > hold_prob
            and sell_prob >= self.config.relative_strength_threshold
        ):
            amplified_sell = sell_prob * self.config.signal_amplification
            confidence = min(amplified_sell, 1.0)
            strength = sell_prob - max(hold_prob, buy_prob)
            priority = 3
            return (
                -1,
                confidence,
                f"SELL_RELATIVE_{sell_prob:.3f}",
                strength,
                priority,
                amplified_sell,
            )

        # ULTRA Strategy 3: Competitive analysis with ultra-low thresholds
        if buy_prob >= (self.config.buy_threshold - 0.02) and sell_prob >= (
            self.config.sell_threshold - 0.02
        ):
            if buy_prob > sell_prob:
                amplified_buy = buy_prob * self.config.signal_amplification
                confidence = min(amplified_buy - sell_prob + self.config.confidence_boost, 1.0)
                strength = buy_prob - sell_prob
                priority = 4
                return (
                    1,
                    confidence,
                    f"BUY_COMPETITIVE_{buy_prob:.3f}",
                    strength,
                    priority,
                    amplified_buy,
                )
            else:
                amplified_sell = sell_prob * self.config.signal_amplification
                confidence = min(amplified_sell - buy_prob + self.config.confidence_boost, 1.0)
                strength = sell_prob - buy_prob
                priority = 4
                return (
                    -1,
                    confidence,
                    f"SELL_COMPETITIVE_{sell_prob:.3f}",
                    strength,
                    priority,
                    amplified_sell,
                )

        # ULTRA Strategy 4: Edge case exploitation
        if buy_prob >= 0.05 and buy_prob > hold_prob:  # Very low threshold
            amplified_buy = buy_prob * self.config.signal_amplification
            confidence = min(amplified_buy, 1.0)
            strength = buy_prob - hold_prob
            priority = 5
            return 1, confidence, f"BUY_EDGE_{buy_prob:.3f}", strength, priority, amplified_buy

        if sell_prob >= 0.05 and sell_prob > hold_prob:  # Very low threshold
            amplified_sell = sell_prob * self.config.signal_amplification
            confidence = min(amplified_sell, 1.0)
            strength = sell_prob - hold_prob
            priority = 5
            return -1, confidence, f"SELL_EDGE_{sell_prob:.3f}", strength, priority, amplified_sell

        # ULTRA Strategy 5: Force BUY/SELL when HOLD is not dominant
        if hold_prob < 0.8:  # If HOLD is not very dominant
            if buy_prob > sell_prob and buy_prob >= 0.03:  # Extremely low threshold
                amplified_buy = buy_prob * self.config.signal_amplification
                confidence = min(amplified_buy, 1.0)
                strength = buy_prob - sell_prob
                priority = 6
                return (
                    1,
                    confidence,
                    f"BUY_FORCED_{buy_prob:.3f}",
                    strength,
                    priority,
                    amplified_buy,
                )
            elif sell_prob > buy_prob and sell_prob >= 0.03:  # Extremely low threshold
                amplified_sell = sell_prob * self.config.signal_amplification
                confidence = min(amplified_sell, 1.0)
                strength = sell_prob - buy_prob
                priority = 6
                return (
                    -1,
                    confidence,
                    f"SELL_FORCED_{sell_prob:.3f}",
                    strength,
                    priority,
                    amplified_sell,
                )

        # Only HOLD if absolutely necessary (very high HOLD threshold)
        if hold_prob >= self.config.hold_threshold:
            confidence = hold_prob
            strength = hold_prob - max(buy_prob, sell_prob)
            priority = 7
            return 0, confidence, f"HOLD_ULTRA_{hold_prob:.3f}", strength, priority, hold_prob

        # Last resort: Choose the strongest non-HOLD signal
        if buy_prob > sell_prob and buy_prob >= 0.02:
            amplified_buy = buy_prob * self.config.signal_amplification
            confidence = min(amplified_buy, 1.0)
            strength = buy_prob - sell_prob
            priority = 8
            return (
                1,
                confidence,
                f"BUY_LAST_RESORT_{buy_prob:.3f}",
                strength,
                priority,
                amplified_buy,
            )
        elif sell_prob > buy_prob and sell_prob >= 0.02:
            amplified_sell = sell_prob * self.config.signal_amplification
            confidence = min(amplified_sell, 1.0)
            strength = sell_prob - buy_prob
            priority = 8
            return (
                -1,
                confidence,
                f"SELL_LAST_RESORT_{sell_prob:.3f}",
                strength,
                priority,
                amplified_sell,
            )

        # Default: HOLD (only when absolutely necessary)
        return 0, hold_prob, f"HOLD_DEFAULT_{hold_prob:.3f}", 0.0, 9, hold_prob

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

    def analyze_ultra_distribution(
        self, df: pd.DataFrame, signal_col: str = 'ultra_signal'
    ) -> dict[str, Any]:
        """
        Comprehensive ultra signal distribution analysis

        Returns:
            Dictionary with detailed distribution statistics
        """
        signal_counts = df[signal_col].value_counts()
        total_signals = len(df)

        # Calculate percentages
        signal_percentages = (signal_counts / total_signals * 100).round(2)

        # Calculate ultra balance score (target: 40% BUY, 20% HOLD, 40% SELL)
        ideal_distribution = {1: 40, 0: 20, -1: 40}  # More BUY/SELL, less HOLD
        total_deviation = sum(
            abs(signal_percentages.get(signal, 0) - ideal_pct)
            for signal, ideal_pct in ideal_distribution.items()
        )
        balance_score = max(0, 100 - (total_deviation / 200) * 100)

        # Calculate confidence statistics
        confidence_stats = df['signal_confidence'].describe()

        # Calculate strength statistics
        strength_stats = df['signal_strength'].describe()

        # Priority distribution
        priority_counts = df['signal_priority'].value_counts().sort_index()

        # Amplified signal statistics
        amplified_stats = df['signal_amplified'].describe()

        analysis = {
            'total_signals': total_signals,
            'signal_counts': signal_counts.to_dict(),
            'signal_percentages': signal_percentages.to_dict(),
            'balance_score': balance_score,
            'confidence_stats': confidence_stats.to_dict(),
            'strength_stats': strength_stats.to_dict(),
            'amplified_stats': amplified_stats.to_dict(),
            'priority_distribution': priority_counts.to_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        # Log comprehensive results
        logger.info("=== ULTRA Signal Distribution Analysis ===")
        logger.info(f"Total signals: {total_signals}")
        for signal, count in signal_counts.items():
            percentage = signal_percentages[signal]
            label = self.signal_map.get(signal, str(signal))
            logger.info(f"{label}: {count} ({percentage}%)")
        logger.info(f"ULTRA Balance score: {balance_score:.2f}")
        logger.info(f"Average confidence: {confidence_stats['mean']:.3f}")
        logger.info(f"Average strength: {strength_stats['mean']:.3f}")
        logger.info(f"Average amplified: {amplified_stats['mean']:.3f}")

        return analysis

    def plot_ultra_distribution(
        self, df: pd.DataFrame, signal_col: str = 'ultra_signal', save_path: str = None
    ):
        """
        Create ultra-professional visualization of signal distribution
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM ULTRA Signal Distribution Analysis', fontsize=16, fontweight='bold')

        # 1. Signal count bar plot
        signal_counts = df[signal_col].value_counts().sort_index()
        signal_labels = [self.signal_map.get(s, str(s)) for s in signal_counts.index]
        colors = ['green', 'gray', 'red']

        bars = ax1.bar(signal_labels, signal_counts.values, color=colors, alpha=0.7)
        ax1.set_title('ULTRA Signal Counts', fontweight='bold')
        ax1.set_ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, signal_counts.values, strict=False):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(signal_counts.values),
                f'{count}',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

        # 2. Signal percentage pie chart
        percentages = (signal_counts / len(df) * 100).round(1)
        ax2.pie(
            percentages.values,
            labels=[
                f"{label}\n{pct}%"
                for label, pct in zip(signal_labels, percentages.values, strict=False)
            ],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
        )
        ax2.set_title('ULTRA Signal Distribution (%)', fontweight='bold')

        # 3. Confidence distribution by signal type
        if 'signal_confidence' in df.columns:
            confidence_data = []
            confidence_labels = []
            for signal in signal_counts.index:
                signal_data = df[df[signal_col] == signal]['signal_confidence']
                confidence_data.append(signal_data.values)
                confidence_labels.append(self.signal_map.get(signal, str(signal)))

            ax3.boxplot(confidence_data, labels=confidence_labels, patch_artist=True)
            ax3.set_title('ULTRA Signal Confidence Distribution', fontweight='bold')
            ax3.set_ylabel('Confidence Score')
            ax3.grid(True, alpha=0.3)

        # 4. Amplified signal distribution
        if 'signal_amplified' in df.columns:
            amplified_data = []
            amplified_labels = []
            for signal in signal_counts.index:
                signal_data = df[df[signal_col] == signal]['signal_amplified']
                amplified_data.append(signal_data.values)
                amplified_labels.append(self.signal_map.get(signal, str(signal)))

            ax4.boxplot(amplified_data, labels=amplified_labels, patch_artist=True)
            ax4.set_title('ULTRA Signal Amplification Distribution', fontweight='bold')
            ax4.set_ylabel('Amplified Score')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ULTRA distribution plot saved to: {save_path}")

        plt.show()

    def export_ultra_signals(
        self, df: pd.DataFrame, output_path: str, signal_col: str = 'ultra_signal'
    ):
        """
        Export ultra signals to CSV with comprehensive information
        """
        export_columns = [
            'time',
            'close',
            'lstm_buy_proba',
            'lstm_hold_proba',
            'lstm_sell_proba',
            signal_col,
            'signal_confidence',
            'signal_reason',
            'signal_strength',
            'signal_priority',
            'signal_amplified',
            'signal_label',
            'signal_category',
        ]

        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()

        export_df.to_csv(output_path, index=False)
        logger.info(f"ULTRA signals exported to: {output_path}")
        logger.info(f"Exported {len(export_df)} signals with {len(export_df.columns)} columns")


def test_ultra_threshold_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test different ultra threshold combinations to find optimal balance
    """
    test_configs = [
        UltraSignalConfig(
            buy_threshold=0.05, sell_threshold=0.05, hold_threshold=0.90, signal_amplification=1.2
        ),
        UltraSignalConfig(
            buy_threshold=0.08, sell_threshold=0.08, hold_threshold=0.85, signal_amplification=1.5
        ),
        UltraSignalConfig(
            buy_threshold=0.10, sell_threshold=0.10, hold_threshold=0.80, signal_amplification=1.8
        ),
        UltraSignalConfig(
            buy_threshold=0.12, sell_threshold=0.12, hold_threshold=0.75, signal_amplification=2.0
        ),
        UltraSignalConfig(
            buy_threshold=0.08, sell_threshold=0.12, hold_threshold=0.82, signal_amplification=1.5
        ),
        UltraSignalConfig(
            buy_threshold=0.12, sell_threshold=0.08, hold_threshold=0.82, signal_amplification=1.5
        ),
        UltraSignalConfig(
            buy_threshold=0.03, sell_threshold=0.03, hold_threshold=0.95, signal_amplification=1.0
        ),
        UltraSignalConfig(
            buy_threshold=0.15, sell_threshold=0.15, hold_threshold=0.70, signal_amplification=2.5
        ),
    ]

    results = []

    for i, config in enumerate(test_configs):
        logger.info(f"Testing ULTRA configuration {i+1}/{len(test_configs)}")

        try:
            balancer = LSTMSignalBalancerUltra(config)
            result_df = balancer.generate_ultra_balanced_signals(df)
            analysis = balancer.analyze_ultra_distribution(result_df)

            result = {
                'test_id': i + 1,
                'buy_threshold': config.buy_threshold,
                'sell_threshold': config.sell_threshold,
                'hold_threshold': config.hold_threshold,
                'signal_amplification': config.signal_amplification,
                'total_signals': analysis['total_signals'],
                'buy_count': analysis['signal_counts'].get(1, 0),
                'hold_count': analysis['signal_counts'].get(0, 0),
                'sell_count': analysis['signal_counts'].get(-1, 0),
                'buy_percentage': analysis['signal_percentages'].get(1, 0),
                'hold_percentage': analysis['signal_percentages'].get(0, 0),
                'sell_percentage': analysis['signal_percentages'].get(-1, 0),
                'balance_score': analysis['balance_score'],
                'avg_confidence': analysis['confidence_stats']['mean'],
                'avg_strength': analysis['strength_stats']['mean'],
                'avg_amplified': analysis['amplified_stats']['mean'],
            }

            results.append(result)

        except Exception as e:
            logger.error(f"Error testing ULTRA configuration {i+1}: {e}")
            continue

    return pd.DataFrame(results)


def main():
    """
    Main execution function with ultra-professional testing and analysis
    """
    # Load LSTM signals data
    try:
        df = pd.read_csv("lstm_signals_pro.csv")
        logger.info(f"Loaded ULTRA data: {len(df)} rows")
    except FileNotFoundError:
        logger.error("File 'lstm_signals_pro.csv' not found!")
        return

    # Test different ultra threshold combinations
    logger.info("Testing ULTRA threshold combinations...")
    test_results = test_ultra_threshold_combinations(df)

    # Display comprehensive results
    print("\n" + "=" * 120)
    print("ULTRA PROFESSIONAL THRESHOLD TESTING RESULTS")
    print("=" * 120)
    print(test_results.to_string(index=False))

    # Find best configuration
    best_result = test_results.loc[test_results['balance_score'].idxmax()]
    print("\n🏆 BEST ULTRA CONFIGURATION:")
    print(f"   Balance Score: {best_result['balance_score']:.2f}")
    print(f"   Buy Threshold: {best_result['buy_threshold']}")
    print(f"   Sell Threshold: {best_result['sell_threshold']}")
    print(f"   Hold Threshold: {best_result['hold_threshold']}")
    print(f"   Signal Amplification: {best_result['signal_amplification']}")
    print(
        f"   Buy: {best_result['buy_percentage']:.1f}% | Hold: {best_result['hold_percentage']:.1f}% | Sell: {best_result['sell_percentage']:.1f}%"
    )
    print(f"   Avg Confidence: {best_result['avg_confidence']:.3f}")
    print(f"   Avg Strength: {best_result['avg_strength']:.3f}")
    print(f"   Avg Amplified: {best_result['avg_amplified']:.3f}")

    # Generate final signals with best configuration
    best_config = UltraSignalConfig(
        buy_threshold=best_result['buy_threshold'],
        sell_threshold=best_result['sell_threshold'],
        hold_threshold=best_result['hold_threshold'],
        signal_amplification=best_result['signal_amplification'],
    )

    balancer = LSTMSignalBalancerUltra(best_config)
    final_df = balancer.generate_ultra_balanced_signals(df)

    # Comprehensive analysis and visualization
    analysis = balancer.analyze_ultra_distribution(final_df)
    balancer.plot_ultra_distribution(final_df, save_path="lstm_ultra_signals_distribution.png")
    balancer.export_ultra_signals(final_df, "lstm_ultra_signals_final.csv")

    # Save test results
    test_results.to_csv("ultra_threshold_testing_results.csv", index=False)

    print("\n✅ ULTRA balanced signals generated successfully!")
    print("   Output file: lstm_ultra_signals_final.csv")
    print("   Distribution plot: lstm_ultra_signals_distribution.png")
    print("   Test results: ultra_threshold_testing_results.csv")
    print("   Log file: lstm_signal_balancer_ultra.log")

    # Show final distribution summary
    buy_pct = analysis['signal_percentages'].get(1, 0)
    hold_pct = analysis['signal_percentages'].get(0, 0)
    sell_pct = analysis['signal_percentages'].get(-1, 0)

    print("\n🎯 FINAL ULTRA DISTRIBUTION:")
    print(f"   BUY: {buy_pct:.1f}% ({analysis['signal_counts'].get(1, 0)} signals)")
    print(f"   HOLD: {hold_pct:.1f}% ({analysis['signal_counts'].get(0, 0)} signals)")
    print(f"   SELL: {sell_pct:.1f}% ({analysis['signal_counts'].get(-1, 0)} signals)")

    if buy_pct + sell_pct > hold_pct:
        print(f"   ✅ SUCCESS: BUY + SELL ({buy_pct + sell_pct:.1f}%) > HOLD ({hold_pct:.1f}%)")
    else:
        print(f"   ⚠️  WARNING: HOLD ({hold_pct:.1f}%) > BUY + SELL ({buy_pct + sell_pct:.1f}%)")


if __name__ == "__main__":
    main()
