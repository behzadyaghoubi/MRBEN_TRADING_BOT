#!/usr/bin/env python3
"""
MRBEN LSTM Trading System - Simple Runner
=========================================

Simple script to run the complete LSTM trading system with default settings.

Usage:
    python run_trading_system.py

This will:
1. Load the LSTM signals data
2. Train the LSTM model
3. Generate balanced signals
4. Run backtesting
5. Generate comprehensive reports
6. Save all results to 'outputs/' directory

Author: MRBEN Trading System
"""

import os

from lstm_trading_system_pro import LSTMTradingSystem, TradingConfig


def main():
    """Run the complete trading system"""
    print("🚀 Starting MRBEN LSTM Trading System...")

    # Check if data file exists
    data_file = "lstm_signals_fixed.csv"  # Use fixed data file
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file '{data_file}' not found!")
        print("Please run 'python fix_data_columns.py' first to prepare the data.")
        return

    try:
        # Create trading system with default configuration
        config = TradingConfig()
        trading_system = LSTMTradingSystem(config)

        # Run complete system
        results = trading_system.run_complete_system(data_file)

        # Print results
        performance = results['performance']
        print("\n✅ Trading System Completed Successfully!")
        print(f"📊 Total Trades: {performance.get('total_trades', 0):,}")
        print(f"🎯 Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
        print(f"💰 Total Return: {performance.get('total_return', 0)*100:.1f}%")
        print(f"📉 Max Drawdown: {performance.get('max_drawdown', 0)*100:.1f}%")
        print(f"💵 Final Balance: ${performance.get('final_balance', 0):,.2f}")

        print("\n📁 All outputs saved to 'outputs/' directory:")
        print("   - trading_report.png (Comprehensive analysis)")
        print("   - signals_with_predictions.csv (All signals)")
        print("   - lstm_trading_model.h5 (Trained model)")
        print("   - trading_config.json (Configuration)")
        print("   - performance_summary.json (Performance metrics)")

    except Exception as e:
        print(f"❌ Error running trading system: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
