#!/usr/bin/env python3
"""
Fix Signal Values - Update all signal values to be compatible with LSTM
======================================================================

This script updates all signal values from [-1, 0, 1] to [0, 1, 2] format
to be compatible with LSTM sparse_categorical_crossentropy loss.

Author: MRBEN Trading System
"""

import re


def fix_signal_values():
    """Fix signal values in the trading system files"""

    # Files to update
    files_to_update = [
        'lstm_trading_system_pro.py',
        'test_trade_count.py',
        'high_frequency_trader.py',
    ]

    # Signal value mappings
    signal_mappings = [
        # Signal map updates
        (
            r'self\.signal_map = \{1: "BUY", 0: "HOLD", -1: "SELL"\}',
            'self.signal_map = {2: "BUY", 1: "HOLD", 0: "SELL"}',
        ),
        # Return statement updates
        (r'return 1, confidence, f"BUY_', 'return 2, confidence, f"BUY_'),
        (r'return -1, confidence, f"SELL_', 'return 0, confidence, f"SELL_'),
        (r'return 0, hold_prob, f"HOLD_', 'return 1, hold_prob, f"HOLD_'),
        # Signal value updates in return statements
        (r'return 1, buy_prob, f"BUY_', 'return 2, buy_prob, f"BUY_'),
        (r'return -1, sell_prob, f"SELL_', 'return 0, sell_prob, f"SELL_'),
        # Signal comparisons in backtesting
        (r'trade\[\'signal\'\] == 1', "trade['signal'] == 2"),
        (r'trade\[\'signal\'\] == -1', "trade['signal'] == 0"),
        (r'trade\[\'signal\'\] == 0', "trade['signal'] == 1"),
        # Signal checks in backtesting
        (r'if signal == 1:', 'if signal == 2:'),
        (r'elif signal == -1:', 'elif signal == 0:'),
        (r'elif signal == 0:', 'elif signal == 1:'),
        # PnL calculations
        (r'if trade\[\'signal\'\] == 1:', 'if trade[\'signal\'] == 2:'),
        (r'else:  # SELL', 'else:  # SELL'),
        # Signal filtering
        (r'if signal != 0:', 'if signal != 1:'),
        (r'if signal == 0:', 'if signal == 1:'),
        # Signal distribution analysis
        (
            r'signal_counts = df\[\'signal\'\]\.value_counts\(\)',
            'signal_counts = df[\'signal\'].value_counts()',
        ),
        (r'df\[\'signal\'\] == 1', "df['signal'] == 2"),
        (r'df\[\'signal\'\] == -1', "df['signal'] == 0"),
        (r'df\[\'signal\'\] == 0', "df['signal'] == 1"),
    ]

    for filename in files_to_update:
        try:
            print(f"Updating {filename}...")

            # Read file
            with open(filename, encoding='utf-8') as f:
                content = f.read()

            # Apply all mappings
            original_content = content
            for pattern, replacement in signal_mappings:
                content = re.sub(pattern, replacement, content)

            # Write updated content
            if content != original_content:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Updated {filename}")
            else:
                print(f"⚠️  No changes needed in {filename}")

        except FileNotFoundError:
            print(f"❌ File {filename} not found")
        except Exception as e:
            print(f"❌ Error updating {filename}: {e}")


def main():
    """Main function"""
    print("=== Fixing Signal Values ===")
    fix_signal_values()
    print("Signal values fixed successfully!")


if __name__ == "__main__":
    main()
