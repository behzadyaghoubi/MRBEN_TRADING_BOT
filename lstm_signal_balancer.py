#!/usr/bin/env python3
"""
LSTM Signal Balancer - Professional Signal Generation System
===========================================================

Solves the problem of imbalanced LSTM signal distribution by implementing
intelligent threshold-based signal generation with configurable parameters.

Problem: LSTM produces mostly HOLD (0) signals, very few BUY (1) and SELL (-1)
Solution: Configurable thresholds for each class to achieve balanced distribution

Author: MRBEN Trading System
Version: 1.0
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMSignalBalancer:
    """
    Professional LSTM signal balancer with configurable thresholds
    """

    def __init__(
        self,
        buy_threshold: float = 0.25,
        sell_threshold: float = 0.25,
        hold_threshold: float = 0.50,
        confidence_boost: float = 0.05,
    ):
        """
        Initialize signal balancer with configurable thresholds

        Args:
            buy_threshold: Minimum probability for BUY signal (default: 0.25)
            sell_threshold: Minimum probability for SELL signal (default: 0.25)
            hold_threshold: Minimum probability for HOLD signal (default: 0.50)
            confidence_boost: Additional confidence for edge cases (default: 0.05)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.hold_threshold = hold_threshold
        self.confidence_boost = confidence_boost

        # Signal mapping
        self.signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}

        logger.info("LSTM Signal Balancer initialized:")
        logger.info(f"  BUY threshold: {buy_threshold}")
        logger.info(f"  SELL threshold: {sell_threshold}")
        logger.info(f"  HOLD threshold: {hold_threshold}")
        logger.info(f"  Confidence boost: {confidence_boost}")

    def generate_balanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate balanced signals from LSTM probabilities

        Args:
            df: DataFrame with columns: lstm_buy_proba, lstm_hold_proba, lstm_sell_proba

        Returns:
            DataFrame with balanced signals and confidence scores
        """
        logger.info("Generating balanced signals...")

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

        # Generate signals for each row
        for idx, row in result_df.iterrows():
            buy_prob = row['lstm_buy_proba']
            hold_prob = row['lstm_hold_proba']
            sell_prob = row['lstm_sell_proba']

            # Generate signal using intelligent threshold system
            signal, confidence, reason = self._generate_single_signal(
                buy_prob, hold_prob, sell_prob
            )

            result_df.at[idx, 'balanced_signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason

        # Add signal labels
        result_df['signal_label'] = result_df['balanced_signal'].map(self.signal_map)

        logger.info("Balanced signal generation completed")
        return result_df

    def _generate_single_signal(
        self, buy_prob: float, hold_prob: float, sell_prob: float
    ) -> tuple[int, float, str]:
        """
        Generate a single signal using intelligent threshold system

        Returns:
            Tuple of (signal_value, confidence, reason)
        """
        # Validate probabilities
        if not (0 <= buy_prob <= 1 and 0 <= hold_prob <= 1 and 0 <= sell_prob <= 1):
            return 0, 0.0, "INVALID_PROBABILITIES"

        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob

        # Strategy 1: Clear winner with high confidence
        max_prob = max(buy_prob, hold_prob, sell_prob)

        if max_prob == buy_prob and buy_prob >= self.buy_threshold:
            confidence = min(buy_prob + self.confidence_boost, 1.0)
            return 1, confidence, f"BUY_HIGH_{buy_prob:.3f}"

        elif max_prob == sell_prob and sell_prob >= self.sell_threshold:
            confidence = min(sell_prob + self.confidence_boost, 1.0)
            return -1, confidence, f"SELL_HIGH_{sell_prob:.3f}"

        elif max_prob == hold_prob and hold_prob >= self.hold_threshold:
            return 0, hold_prob, f"HOLD_HIGH_{hold_prob:.3f}"

        # Strategy 2: Competitive analysis
        if buy_prob >= self.buy_threshold and sell_prob >= self.sell_threshold:
            if buy_prob > sell_prob:
                confidence = min(buy_prob - sell_prob + self.confidence_boost, 1.0)
                return 1, confidence, f"BUY_VS_SELL_{buy_prob:.3f}"
            else:
                confidence = min(sell_prob - buy_prob + self.confidence_boost, 1.0)
                return -1, confidence, f"SELL_VS_BUY_{sell_prob:.3f}"

        # Strategy 3: Edge cases
        if buy_prob >= (self.buy_threshold - 0.05) and buy_prob > hold_prob:
            return 1, buy_prob, f"BUY_EDGE_{buy_prob:.3f}"

        if sell_prob >= (self.sell_threshold - 0.05) and sell_prob > hold_prob:
            return -1, sell_prob, f"SELL_EDGE_{sell_prob:.3f}"

        # Default: HOLD
        return 0, hold_prob, f"HOLD_DEFAULT_{hold_prob:.3f}"

    def analyze_distribution(self, df: pd.DataFrame, signal_col: str = 'balanced_signal') -> dict:
        """
        Analyze signal distribution

        Returns:
            Dictionary with distribution statistics
        """
        signal_counts = df[signal_col].value_counts()
        total_signals = len(df)

        # Calculate percentages
        signal_percentages = (signal_counts / total_signals * 100).round(2)

        # Calculate balance score
        ideal_distribution = {1: 30, 0: 40, -1: 30}  # BUY, HOLD, SELL
        total_deviation = sum(
            abs(signal_percentages.get(signal, 0) - ideal_pct)
            for signal, ideal_pct in ideal_distribution.items()
        )
        balance_score = max(0, 100 - (total_deviation / 200) * 100)

        analysis = {
            'total_signals': total_signals,
            'signal_counts': signal_counts.to_dict(),
            'signal_percentages': signal_percentages.to_dict(),
            'balance_score': balance_score,
            'timestamp': datetime.now().isoformat(),
        }

        # Log results
        logger.info("=== Signal Distribution Analysis ===")
        logger.info(f"Total signals: {total_signals}")
        for signal, count in signal_counts.items():
            percentage = signal_percentages[signal]
            label = self.signal_map.get(signal, str(signal))
            logger.info(f"{label}: {count} ({percentage}%)")
        logger.info(f"Balance score: {balance_score:.2f}")

        return analysis

    def plot_distribution(
        self, df: pd.DataFrame, signal_col: str = 'balanced_signal', save_path: str = None
    ):
        """
        Create visualization of signal distribution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('LSTM Balanced Signal Distribution', fontsize=16, fontweight='bold')

        # Bar plot
        signal_counts = df[signal_col].value_counts()
        signal_labels = [self.signal_map.get(s, str(s)) for s in signal_counts.index]
        colors = ['green', 'gray', 'red']

        bars = ax1.bar(signal_labels, signal_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Signal Counts', fontweight='bold')
        ax1.set_ylabel('Count')

        # Add value labels
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

        # Pie chart
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
        ax2.set_title('Signal Distribution (%)', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to: {save_path}")

        plt.show()

    def export_signals(
        self, df: pd.DataFrame, output_path: str, signal_col: str = 'balanced_signal'
    ):
        """
        Export signals to CSV file
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
            'signal_label',
        ]

        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()

        export_df.to_csv(output_path, index=False)
        logger.info(f"Signals exported to: {output_path}")


def test_threshold_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test different threshold combinations to find optimal balance
    """
    test_configs = [
        {'buy_threshold': 0.20, 'sell_threshold': 0.20, 'hold_threshold': 0.60},
        {'buy_threshold': 0.25, 'sell_threshold': 0.25, 'hold_threshold': 0.50},
        {'buy_threshold': 0.30, 'sell_threshold': 0.30, 'hold_threshold': 0.40},
        {'buy_threshold': 0.35, 'sell_threshold': 0.35, 'hold_threshold': 0.30},
        {'buy_threshold': 0.25, 'sell_threshold': 0.30, 'hold_threshold': 0.45},
        {'buy_threshold': 0.30, 'sell_threshold': 0.25, 'hold_threshold': 0.45},
    ]

    results = []

    for i, config in enumerate(test_configs):
        logger.info(f"Testing configuration {i+1}/{len(test_configs)}")

        try:
            balancer = LSTMSignalBalancer(**config)
            result_df = balancer.generate_balanced_signals(df)
            analysis = balancer.analyze_distribution(result_df)

            result = {
                'test_id': i + 1,
                'buy_threshold': config['buy_threshold'],
                'sell_threshold': config['sell_threshold'],
                'hold_threshold': config['hold_threshold'],
                'total_signals': analysis['total_signals'],
                'buy_count': analysis['signal_counts'].get(1, 0),
                'hold_count': analysis['signal_counts'].get(0, 0),
                'sell_count': analysis['signal_counts'].get(-1, 0),
                'buy_percentage': analysis['signal_percentages'].get(1, 0),
                'hold_percentage': analysis['signal_percentages'].get(0, 0),
                'sell_percentage': analysis['signal_percentages'].get(-1, 0),
                'balance_score': analysis['balance_score'],
            }

            results.append(result)

        except Exception as e:
            logger.error(f"Error testing configuration {i+1}: {e}")
            continue

    return pd.DataFrame(results)


def main():
    """
    Main execution function
    """
    # Load LSTM signals data
    try:
        df = pd.read_csv("lstm_signals_pro.csv")
        logger.info(f"Loaded data: {len(df)} rows")
    except FileNotFoundError:
        logger.error("File 'lstm_signals_pro.csv' not found!")
        return

    # Test different threshold combinations
    logger.info("Testing threshold combinations...")
    test_results = test_threshold_combinations(df)

    # Display results
    print("\n" + "=" * 80)
    print("THRESHOLD TESTING RESULTS")
    print("=" * 80)
    print(test_results.to_string(index=False))

    # Find best configuration
    best_result = test_results.loc[test_results['balance_score'].idxmax()]
    print("\n🏆 BEST CONFIGURATION:")
    print(f"   Balance Score: {best_result['balance_score']:.2f}")
    print(f"   Buy Threshold: {best_result['buy_threshold']}")
    print(f"   Sell Threshold: {best_result['sell_threshold']}")
    print(f"   Hold Threshold: {best_result['hold_threshold']}")
    print(
        f"   Buy: {best_result['buy_percentage']:.1f}% | Hold: {best_result['hold_percentage']:.1f}% | Sell: {best_result['sell_percentage']:.1f}%"
    )

    # Generate final signals with best configuration
    balancer = LSTMSignalBalancer(
        buy_threshold=best_result['buy_threshold'],
        sell_threshold=best_result['sell_threshold'],
        hold_threshold=best_result['hold_threshold'],
    )

    final_df = balancer.generate_balanced_signals(df)
    analysis = balancer.analyze_distribution(final_df)

    # Visualize and export results
    balancer.plot_distribution(final_df, save_path="lstm_balanced_signals_distribution.png")
    balancer.export_signals(final_df, "lstm_balanced_signals_final.csv")

    # Save test results
    test_results.to_csv("threshold_testing_results.csv", index=False)

    print("\n✅ Final balanced signals generated!")
    print("   Output: lstm_balanced_signals_final.csv")
    print("   Plot: lstm_balanced_signals_distribution.png")
    print("   Test results: threshold_testing_results.csv")


if __name__ == "__main__":
    main()
