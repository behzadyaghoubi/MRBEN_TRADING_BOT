#!/usr/bin/env python3
"""
MRBEN LSTM Balanced Signal Generator
===================================

Professional signal generation system for LSTM models with configurable thresholds
to achieve balanced signal distribution across BUY, HOLD, and SELL classes.

Author: MRBEN Trading System
Version: 1.0
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SignalThresholds:
    """Configurable thresholds for signal generation"""

    buy_threshold: float = 0.25  # Minimum probability for BUY signal
    sell_threshold: float = 0.25  # Minimum probability for SELL signal
    hold_threshold: float = 0.50  # Minimum probability for HOLD signal
    confidence_boost: float = 0.05  # Additional confidence boost for edge cases

    def validate(self) -> bool:
        """Validate threshold values"""
        if not (0 <= self.buy_threshold <= 1):
            logger.error(f"Invalid buy_threshold: {self.buy_threshold}")
            return False
        if not (0 <= self.sell_threshold <= 1):
            logger.error(f"Invalid sell_threshold: {self.sell_threshold}")
            return False
        if not (0 <= self.hold_threshold <= 1):
            logger.error(f"Invalid hold_threshold: {self.hold_threshold}")
            return False
        if not (0 <= self.confidence_boost <= 0.2):
            logger.error(f"Invalid confidence_boost: {self.confidence_boost}")
            return False
        return True


class LSTMBalancedSignalGenerator:
    """
    Professional LSTM signal generator with balanced distribution

    Features:
    - Configurable thresholds for each signal class
    - Intelligent signal selection based on probability differences
    - Signal confidence scoring
    - Distribution analysis and visualization
    - Export capabilities
    """

    def __init__(self, thresholds: SignalThresholds | None = None):
        """
        Initialize the signal generator

        Args:
            thresholds: SignalThresholds object with custom thresholds
        """
        self.thresholds = thresholds or SignalThresholds()
        if not self.thresholds.validate():
            raise ValueError("Invalid threshold configuration")

        self.signal_mapping = {1: "BUY", 0: "HOLD", -1: "SELL"}

        self.reverse_mapping = {v: k for k, v in self.signal_mapping.items()}

        logger.info(
            f"LSTM Balanced Signal Generator initialized with thresholds: "
            f"BUY={self.thresholds.buy_threshold}, "
            f"SELL={self.thresholds.sell_threshold}, "
            f"HOLD={self.thresholds.hold_threshold}"
        )

    def generate_balanced_signals(
        self,
        df: pd.DataFrame,
        buy_proba_col: str = 'lstm_buy_proba',
        hold_proba_col: str = 'lstm_hold_proba',
        sell_proba_col: str = 'lstm_sell_proba',
        close_col: str = 'close',
        time_col: str = 'time',
    ) -> pd.DataFrame:
        """
        Generate balanced trading signals from LSTM probabilities

        Args:
            df: DataFrame with LSTM probability columns
            buy_proba_col: Column name for BUY probabilities
            hold_proba_col: Column name for HOLD probabilities
            sell_proba_col: Column name for SELL probabilities
            close_col: Column name for close prices
            time_col: Column name for timestamps

        Returns:
            DataFrame with balanced signals and confidence scores
        """
        logger.info("Starting balanced signal generation...")

        # Validate input data
        required_cols = [buy_proba_col, hold_proba_col, sell_proba_col, close_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Initialize signal columns
        result_df['balanced_signal'] = 0  # Default to HOLD
        result_df['signal_confidence'] = 0.0
        result_df['signal_reason'] = 'HOLD'

        # Generate signals for each row
        for idx, row in result_df.iterrows():
            buy_prob = row[buy_proba_col]
            hold_prob = row[hold_proba_col]
            sell_prob = row[sell_proba_col]

            # Generate signal using intelligent threshold system
            signal, confidence, reason = self._generate_single_signal(
                buy_prob, hold_prob, sell_prob
            )

            result_df.at[idx, 'balanced_signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason

        # Add signal labels for easier interpretation
        result_df['signal_label'] = result_df['balanced_signal'].map(self.signal_mapping)

        logger.info("Balanced signal generation completed")
        return result_df

    def _generate_single_signal(
        self, buy_prob: float, hold_prob: float, sell_prob: float
    ) -> tuple[int, float, str]:
        """
        Generate a single signal using intelligent threshold system

        Args:
            buy_prob: Probability for BUY signal
            hold_prob: Probability for HOLD signal
            sell_prob: Probability for SELL signal

        Returns:
            Tuple of (signal_value, confidence, reason)
        """
        # Validate probabilities
        if not (0 <= buy_prob <= 1 and 0 <= hold_prob <= 1 and 0 <= sell_prob <= 1):
            logger.warning(
                f"Invalid probabilities: BUY={buy_prob}, HOLD={hold_prob}, SELL={sell_prob}"
            )
            return 0, 0.0, "INVALID_PROBABILITIES"

        # Normalize probabilities (in case they don't sum to 1)
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob

        # Strategy 1: Clear winner with high confidence
        max_prob = max(buy_prob, hold_prob, sell_prob)

        if max_prob == buy_prob and buy_prob >= self.thresholds.buy_threshold:
            confidence = buy_prob + self.thresholds.confidence_boost
            return 1, min(confidence, 1.0), f"BUY_HIGH_CONF_{buy_prob:.3f}"

        elif max_prob == sell_prob and sell_prob >= self.thresholds.sell_threshold:
            confidence = sell_prob + self.thresholds.confidence_boost
            return -1, min(confidence, 1.0), f"SELL_HIGH_CONF_{sell_prob:.3f}"

        elif max_prob == hold_prob and hold_prob >= self.thresholds.hold_threshold:
            confidence = hold_prob
            return 0, confidence, f"HOLD_HIGH_CONF_{hold_prob:.3f}"

        # Strategy 2: Competitive analysis (when no clear winner)
        # Check if BUY and SELL are competing
        if (
            buy_prob >= self.thresholds.buy_threshold
            and sell_prob >= self.thresholds.sell_threshold
        ):
            # Choose the stronger signal
            if buy_prob > sell_prob:
                confidence = buy_prob - sell_prob + self.thresholds.confidence_boost
                return 1, min(confidence, 1.0), f"BUY_VS_SELL_{buy_prob:.3f}_vs_{sell_prob:.3f}"
            else:
                confidence = sell_prob - buy_prob + self.thresholds.confidence_boost
                return -1, min(confidence, 1.0), f"SELL_VS_BUY_{sell_prob:.3f}_vs_{buy_prob:.3f}"

        # Strategy 3: Edge case handling
        # If BUY is close to threshold, give it a chance
        if buy_prob >= (self.thresholds.buy_threshold - 0.05) and buy_prob > hold_prob:
            confidence = buy_prob
            return 1, confidence, f"BUY_EDGE_CASE_{buy_prob:.3f}"

        # If SELL is close to threshold, give it a chance
        if sell_prob >= (self.thresholds.sell_threshold - 0.05) and sell_prob > hold_prob:
            confidence = sell_prob
            return -1, confidence, f"SELL_EDGE_CASE_{sell_prob:.3f}"

        # Default: HOLD
        return 0, hold_prob, f"HOLD_DEFAULT_{hold_prob:.3f}"

    def analyze_signal_distribution(
        self, df: pd.DataFrame, signal_col: str = 'balanced_signal'
    ) -> dict[str, Any]:
        """
        Analyze the distribution of generated signals

        Args:
            df: DataFrame with signals
            signal_col: Column name containing signals

        Returns:
            Dictionary with distribution statistics
        """
        if signal_col not in df.columns:
            raise ValueError(f"Signal column '{signal_col}' not found in DataFrame")

        # Count signals
        signal_counts = df[signal_col].value_counts()
        total_signals = len(df)

        # Calculate percentages
        signal_percentages = (signal_counts / total_signals * 100).round(2)

        # Create analysis dictionary
        analysis = {
            'total_signals': total_signals,
            'signal_counts': signal_counts.to_dict(),
            'signal_percentages': signal_percentages.to_dict(),
            'signal_labels': {k: self.signal_mapping.get(k, str(k)) for k in signal_counts.index},
            'balance_score': self._calculate_balance_score(signal_percentages),
            'timestamp': datetime.now().isoformat(),
        }

        # Log analysis results
        logger.info("=== Signal Distribution Analysis ===")
        logger.info(f"Total signals: {total_signals}")
        for signal, count in signal_counts.items():
            percentage = signal_percentages[signal]
            label = analysis['signal_labels'][signal]
            logger.info(f"{label}: {count} ({percentage}%)")
        logger.info(f"Balance score: {analysis['balance_score']:.2f}")

        return analysis

    def _calculate_balance_score(self, percentages: pd.Series) -> float:
        """
        Calculate a balance score (0-100) for signal distribution

        Args:
            percentages: Series with signal percentages

        Returns:
            Balance score (higher is better)
        """
        # Ideal distribution: 30% BUY, 40% HOLD, 30% SELL
        ideal_distribution = {1: 30, 0: 40, -1: 30}  # BUY  # HOLD  # SELL

        # Calculate deviation from ideal
        total_deviation = 0
        for signal, ideal_pct in ideal_distribution.items():
            actual_pct = percentages.get(signal, 0)
            deviation = abs(actual_pct - ideal_pct)
            total_deviation += deviation

        # Convert to score (0-100, higher is better)
        max_possible_deviation = 200  # Worst case: 100% deviation for all signals
        balance_score = max(0, 100 - (total_deviation / max_possible_deviation) * 100)

        return balance_score

    def plot_signal_distribution(
        self, df: pd.DataFrame, signal_col: str = 'balanced_signal', save_path: str | None = None
    ) -> None:
        """
        Create visualization of signal distribution

        Args:
            df: DataFrame with signals
            signal_col: Column name containing signals
            save_path: Optional path to save the plot
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LSTM Balanced Signal Distribution Analysis', fontsize=16, fontweight='bold')

        # 1. Signal count bar plot
        signal_counts = df[signal_col].value_counts()
        signal_labels = [self.signal_mapping.get(s, str(s)) for s in signal_counts.index]
        colors = ['green', 'gray', 'red']

        bars = ax1.bar(signal_labels, signal_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Signal Counts', fontweight='bold')
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
        ax2.set_title('Signal Distribution (%)', fontweight='bold')

        # 3. Signal confidence distribution
        if 'signal_confidence' in df.columns:
            confidence_data = []
            confidence_labels = []
            for signal in signal_counts.index:
                signal_data = df[df[signal_col] == signal]['signal_confidence']
                confidence_data.append(signal_data.values)
                confidence_labels.append(self.signal_mapping.get(signal, str(signal)))

            ax3.boxplot(confidence_data, labels=confidence_labels, patch_artist=True)
            ax3.set_title('Signal Confidence Distribution', fontweight='bold')
            ax3.set_ylabel('Confidence Score')
            ax3.grid(True, alpha=0.3)

        # 4. Signal over time (last 100 signals)
        if 'time' in df.columns:
            recent_df = df.tail(100).copy()
            recent_df['time'] = pd.to_datetime(recent_df['time'])
            recent_df = recent_df.sort_values('time')

            # Create color mapping for signals
            color_map = {1: 'green', 0: 'gray', -1: 'red'}
            colors = [color_map.get(s, 'black') for s in recent_df[signal_col]]

            ax4.scatter(range(len(recent_df)), recent_df[signal_col], c=colors, alpha=0.7, s=30)
            ax4.set_title('Recent Signals (Last 100)', fontweight='bold')
            ax4.set_ylabel('Signal')
            ax4.set_xlabel('Time Index')
            ax4.set_yticks([-1, 0, 1])
            ax4.set_yticklabels(['SELL', 'HOLD', 'BUY'])
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to: {save_path}")

        plt.show()

    def export_signals(
        self, df: pd.DataFrame, output_path: str, signal_col: str = 'balanced_signal'
    ) -> None:
        """
        Export signals to CSV file

        Args:
            df: DataFrame with signals
            output_path: Path to save the CSV file
            signal_col: Column name containing signals
        """
        # Select relevant columns for export
        export_columns = [
            'time',
            'close',
            'lstm_buy_proba',
            'lstm_hold_proba',
            'lstm_sell_proba',
            signal_col,
            'signal_confidence',
            'signal_reason',
        ]

        # Filter columns that exist in the DataFrame
        available_columns = [col for col in export_columns if col in df.columns]

        export_df = df[available_columns].copy()

        # Add signal label if not present
        if 'signal_label' not in export_df.columns and signal_col in export_df.columns:
            export_df['signal_label'] = export_df[signal_col].map(self.signal_mapping)

        # Export to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Signals exported to: {output_path}")
        logger.info(f"Exported {len(export_df)} signals with {len(export_df.columns)} columns")


def test_threshold_combinations(
    df: pd.DataFrame, test_thresholds: list[SignalThresholds]
) -> pd.DataFrame:
    """
    Test different threshold combinations to find optimal balance

    Args:
        df: DataFrame with LSTM probabilities
        test_thresholds: List of threshold configurations to test

    Returns:
        DataFrame with test results
    """
    results = []

    for i, thresholds in enumerate(test_thresholds):
        logger.info(f"Testing threshold combination {i+1}/{len(test_thresholds)}")

        try:
            # Create generator with current thresholds
            generator = LSTMBalancedSignalGenerator(thresholds)

            # Generate signals
            result_df = generator.generate_balanced_signals(df)

            # Analyze distribution
            analysis = generator.analyze_signal_distribution(result_df)

            # Store results
            result = {
                'test_id': i + 1,
                'buy_threshold': thresholds.buy_threshold,
                'sell_threshold': thresholds.sell_threshold,
                'hold_threshold': thresholds.hold_threshold,
                'confidence_boost': thresholds.confidence_boost,
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
            logger.error(f"Error testing threshold combination {i+1}: {e}")
            continue

    return pd.DataFrame(results)


# ============================================================================
# MAIN EXECUTION AND TESTING
# ============================================================================


def main():
    """Main execution function with example usage"""

    # Load your LSTM signals data
    try:
        df = pd.read_csv("lstm_signals_pro.csv")
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        logger.error("File 'lstm_signals_pro.csv' not found. Please ensure the file exists.")
        return

    # Define test threshold combinations
    test_thresholds = [
        SignalThresholds(buy_threshold=0.20, sell_threshold=0.20, hold_threshold=0.60),
        SignalThresholds(buy_threshold=0.25, sell_threshold=0.25, hold_threshold=0.50),
        SignalThresholds(buy_threshold=0.30, sell_threshold=0.30, hold_threshold=0.40),
        SignalThresholds(buy_threshold=0.35, sell_threshold=0.35, hold_threshold=0.30),
        SignalThresholds(buy_threshold=0.25, sell_threshold=0.30, hold_threshold=0.45),
        SignalThresholds(buy_threshold=0.30, sell_threshold=0.25, hold_threshold=0.45),
    ]

    # Test different threshold combinations
    logger.info("Testing threshold combinations...")
    test_results = test_threshold_combinations(df, test_thresholds)

    # Display test results
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
    best_thresholds = SignalThresholds(
        buy_threshold=best_result['buy_threshold'],
        sell_threshold=best_result['sell_threshold'],
        hold_threshold=best_result['hold_threshold'],
    )

    generator = LSTMBalancedSignalGenerator(best_thresholds)
    final_df = generator.generate_balanced_signals(df)

    # Analyze and visualize results
    analysis = generator.analyze_signal_distribution(final_df)
    generator.plot_signal_distribution(final_df, save_path="lstm_balanced_signals_distribution.png")

    # Export final results
    generator.export_signals(final_df, "lstm_balanced_signals_final.csv")

    # Save test results
    test_results.to_csv("threshold_testing_results.csv", index=False)
    logger.info("Threshold testing results saved to: threshold_testing_results.csv")

    print("\n✅ Final balanced signals generated and saved!")
    print("   Output file: lstm_balanced_signals_final.csv")
    print("   Distribution plot: lstm_balanced_signals_distribution.png")


if __name__ == "__main__":
    main()
