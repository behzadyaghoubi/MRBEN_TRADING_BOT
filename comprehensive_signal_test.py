#!/usr/bin/env python3
"""
Comprehensive Signal Distribution Test
Tests the current system's signal generation and analyzes BUY/SELL/HOLD distribution
"""

import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Import our trading system components
sys.path.append('.')
from live_trader_clean import MT5Config, MT5DataManager, MT5SignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SignalDistributionTester:
    """Comprehensive signal distribution tester"""

    def __init__(self):
        self.signals = []
        self.lstm_signals = []
        self.ta_signals = []
        self.ml_signals = []
        self.final_signals = []
        self.test_results = {}

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False

            # Load settings
            with open('config/settings.json') as f:
                settings = json.load(f)

            # Login to MT5
            if not mt5.login(
                login=settings['mt5_login'],
                password=settings['mt5_password'],
                server=settings['mt5_server'],
            ):
                logger.error("MT5 login failed")
                return False

            logger.info("✅ MT5 connected successfully")
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def test_signal_generation(self, num_tests=100):
        """Test signal generation multiple times"""
        logger.info(f"🧪 Testing signal generation ({num_tests} times)...")

        # Initialize components
        config = MT5Config()
        data_manager = MT5DataManager()
        signal_generator = MT5SignalGenerator(config)

        for i in range(num_tests):
            try:
                # Get market data
                df = data_manager.get_latest_data(bars=100)
                if df is not None and not df.empty:
                    # Generate signal
                    signal_result = signal_generator.generate_enhanced_signal(df)
                else:
                    signal_result = None

                if signal_result:
                    # Extract individual components
                    lstm_signal = signal_result.get('lstm_signal', 0)
                    ta_signal = signal_result.get('ta_signal', 0)
                    ml_signal = signal_result.get('ml_signal', 0)
                    final_signal = signal_result.get('final_signal', 0)

                    # Store signals
                    self.lstm_signals.append(lstm_signal)
                    self.ta_signals.append(ta_signal)
                    self.ml_signals.append(ml_signal)
                    self.final_signals.append(final_signal)

                    # Log every 10th test
                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"   Test {i+1}/{num_tests}: LSTM={lstm_signal}, TA={ta_signal}, ML={ml_signal}, Final={final_signal}"
                        )

            except Exception as e:
                logger.error(f"Signal generation error in test {i+1}: {e}")
                continue

        logger.info(
            f"✅ Signal generation test completed: {len(self.final_signals)} successful tests"
        )

    def analyze_signal_distribution(self):
        """Analyze signal distribution"""
        logger.info("📊 Analyzing signal distribution...")

        if not self.final_signals:
            logger.error("❌ No signals to analyze")
            return

        # Convert signals to labels
        signal_labels = []
        for signal in self.final_signals:
            if signal == 1:
                signal_labels.append('BUY')
            elif signal == -1:
                signal_labels.append('SELL')
            else:
                signal_labels.append('HOLD')

        # Count signals
        signal_counts = Counter(signal_labels)
        total_signals = len(signal_labels)

        # Calculate percentages
        signal_percentages = {}
        for signal, count in signal_counts.items():
            percentage = (count / total_signals) * 100
            signal_percentages[signal] = percentage

        # Store results
        self.test_results = {
            'total_tests': total_signals,
            'signal_counts': dict(signal_counts),
            'signal_percentages': signal_percentages,
            'lstm_signals': self.lstm_signals,
            'ta_signals': self.ta_signals,
            'ml_signals': self.ml_signals,
            'final_signals': self.final_signals,
        }

        # Log results
        logger.info("📈 Signal Distribution Results:")
        logger.info(f"   Total Tests: {total_signals}")
        for signal, count in signal_counts.items():
            percentage = signal_percentages[signal]
            logger.info(f"   {signal}: {count} ({percentage:.1f}%)")

    def analyze_component_distribution(self):
        """Analyze distribution of individual components"""
        logger.info("🔍 Analyzing component distribution...")

        components = {
            'LSTM': self.lstm_signals,
            'Technical Analysis': self.ta_signals,
            'ML Filter': self.ml_signals,
        }

        for component_name, signals in components.items():
            if not signals:
                continue

            signal_counts = Counter(signals)
            total = len(signals)

            logger.info(f"📊 {component_name} Distribution:")
            for signal, count in signal_counts.items():
                percentage = (count / total) * 100
                signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[signal]
                logger.info(f"   {signal_name}: {count} ({percentage:.1f}%)")

    def check_bias(self):
        """Check for bias in signal distribution"""
        logger.info("🎯 Checking for bias...")

        if not self.test_results:
            logger.error("❌ No test results to analyze")
            return

        percentages = self.test_results['signal_percentages']

        # Check for extreme bias (>80% in one direction)
        bias_threshold = 80
        bias_detected = False

        for signal, percentage in percentages.items():
            if percentage > bias_threshold:
                logger.warning(
                    f"⚠️ BIAS DETECTED: {signal} signals are {percentage:.1f}% (>{bias_threshold}%)"
                )
                bias_detected = True

        # Check for balanced distribution (ideal: 30-40% each)
        balanced_threshold = 30
        balanced_count = 0

        for signal, percentage in percentages.items():
            if percentage >= balanced_threshold:
                balanced_count += 1

        if balanced_count >= 2:
            logger.info("✅ Distribution appears balanced")
        else:
            logger.warning("⚠️ Distribution may be imbalanced")

        if not bias_detected:
            logger.info("✅ No extreme bias detected")

        return not bias_detected

    def create_visualization(self):
        """Create visualization of signal distribution"""
        try:
            logger.info("📊 Creating visualization...")

            # Prepare data
            signal_labels = []
            for signal in self.final_signals:
                if signal == 1:
                    signal_labels.append('BUY')
                elif signal == -1:
                    signal_labels.append('SELL')
                else:
                    signal_labels.append('HOLD')

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Signal Distribution Analysis', fontsize=16, fontweight='bold')

            # Final signal distribution
            signal_counts = Counter(signal_labels)
            ax1.pie(
                signal_counts.values(),
                labels=signal_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
            )
            ax1.set_title('Final Signal Distribution')

            # Component comparison
            components = ['LSTM', 'Technical Analysis', 'ML Filter']
            buy_percentages = []
            sell_percentages = []
            hold_percentages = []

            for signals in [self.lstm_signals, self.ta_signals, self.ml_signals]:
                if signals:
                    counts = Counter(signals)
                    total = len(signals)
                    buy_percentages.append((counts.get(1, 0) / total) * 100)
                    sell_percentages.append((counts.get(-1, 0) / total) * 100)
                    hold_percentages.append((counts.get(0, 0) / total) * 100)

            x = np.arange(len(components))
            width = 0.25

            ax2.bar(x - width, buy_percentages, width, label='BUY', color='green', alpha=0.7)
            ax2.bar(x, sell_percentages, width, label='SELL', color='red', alpha=0.7)
            ax2.bar(x + width, hold_percentages, width, label='HOLD', color='gray', alpha=0.7)

            ax2.set_xlabel('Components')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Component Signal Distribution')
            ax2.set_xticks(x)
            ax2.set_xticklabels(components)
            ax2.legend()

            # Signal timeline
            ax3.plot(range(len(self.final_signals)), self.final_signals, 'o-', alpha=0.7)
            ax3.set_xlabel('Test Number')
            ax3.set_ylabel('Signal Value')
            ax3.set_title('Signal Timeline')
            ax3.grid(True, alpha=0.3)

            # Signal frequency
            signal_series = pd.Series(signal_labels)
            signal_series.value_counts().plot(kind='bar', ax=ax4, color=['green', 'red', 'gray'])
            ax4.set_title('Signal Frequency')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save plot
            plot_path = 'logs/signal_distribution_analysis.png'
            os.makedirs('logs', exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Visualization saved: {plot_path}")

            # Show plot
            plt.show()

        except Exception as e:
            logger.error(f"❌ Visualization error: {e}")

    def save_detailed_report(self):
        """Save detailed test report"""
        try:
            logger.info("📝 Saving detailed report...")

            os.makedirs('logs', exist_ok=True)
            report_path = 'logs/signal_distribution_report.txt'

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("SIGNAL DISTRIBUTION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Tests: {self.test_results.get('total_tests', 0)}\n\n")

                f.write("FINAL SIGNAL DISTRIBUTION:\n")
                f.write("-" * 30 + "\n")
                for signal, count in self.test_results.get('signal_counts', {}).items():
                    percentage = self.test_results.get('signal_percentages', {}).get(signal, 0)
                    f.write(f"{signal}: {count} ({percentage:.1f}%)\n")

                f.write("\nCOMPONENT ANALYSIS:\n")
                f.write("-" * 20 + "\n")

                components = {
                    'LSTM': self.lstm_signals,
                    'Technical Analysis': self.ta_signals,
                    'ML Filter': self.ml_signals,
                }

                for component_name, signals in components.items():
                    if signals:
                        f.write(f"\n{component_name}:\n")
                        counts = Counter(signals)
                        total = len(signals)
                        for signal, count in counts.items():
                            percentage = (count / total) * 100
                            signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[signal]
                            f.write(f"  {signal_name}: {count} ({percentage:.1f}%)\n")

                f.write("\nBIAS ANALYSIS:\n")
                f.write("-" * 15 + "\n")
                percentages = self.test_results.get('signal_percentages', {})

                # Check for extreme bias
                bias_threshold = 80
                for signal, percentage in percentages.items():
                    if percentage > bias_threshold:
                        f.write(
                            f"⚠️ BIAS DETECTED: {signal} signals are {percentage:.1f}% (>{bias_threshold}%)\n"
                        )
                    else:
                        f.write(f"✅ {signal}: {percentage:.1f}% (within normal range)\n")

                f.write("\nRECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")

                if all(p <= 80 for p in percentages.values()):
                    f.write("✅ System appears to have balanced signal distribution\n")
                    f.write("✅ Ready for live trading\n")
                else:
                    f.write("⚠️ Bias detected - consider model retraining\n")
                    f.write("⚠️ Review training data balance\n")

            logger.info(f"✅ Detailed report saved: {report_path}")

        except Exception as e:
            logger.error(f"❌ Report saving error: {e}")

    def run_comprehensive_test(self, num_tests=100):
        """Run comprehensive signal distribution test"""
        logger.info("🚀 STARTING COMPREHENSIVE SIGNAL DISTRIBUTION TEST")
        logger.info("=" * 60)

        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("❌ MT5 initialization failed - cannot proceed")
            return False

        # Test signal generation
        self.test_signal_generation(num_tests)

        # Analyze distribution
        self.analyze_signal_distribution()
        self.analyze_component_distribution()

        # Check for bias
        bias_ok = self.check_bias()

        # Create visualization
        self.create_visualization()

        # Save report
        self.save_detailed_report()

        # Final summary
        logger.info("=" * 60)
        logger.info("📋 TEST SUMMARY:")
        logger.info(f"   Total Tests: {self.test_results.get('total_tests', 0)}")

        percentages = self.test_results.get('signal_percentages', {})
        for signal, percentage in percentages.items():
            logger.info(f"   {signal}: {percentage:.1f}%")

        if bias_ok:
            logger.info("✅ SYSTEM READY FOR LIVE TRADING")
        else:
            logger.warning("⚠️ BIAS DETECTED - REVIEW REQUIRED")

        return bias_ok


def main():
    """Main function"""
    tester = SignalDistributionTester()
    success = tester.run_comprehensive_test(num_tests=100)

    if success:
        logger.info("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
        return True
    else:
        logger.error("❌ COMPREHENSIVE TEST FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
