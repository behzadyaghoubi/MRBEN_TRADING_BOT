#!/usr/bin/env python3
"""
Signal Confidence Threshold Adjustment Script
Adjusts confidence thresholds based on newly trained models
"""

import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def adjust_signal_thresholds():
    """Adjust signal confidence thresholds in live_trader_clean.py"""
    logger.info("Adjusting signal confidence thresholds...")

    # Read the current file
    with open("live_trader_clean.py", encoding="utf-8") as f:
        content = f.read()

    # Define new threshold values based on model performance
    new_thresholds = {
        'base_confidence_threshold': 0.3,  # Lower base threshold for more trades
        'ml_filter_confidence_threshold': 0.4,  # Lower ML filter threshold
        'ta_combined_threshold': 0.4,  # Lower TA combined threshold
        'rsi_strong_thresholds': (30, 70),  # More balanced RSI
        'rsi_weak_thresholds': (40, 60),  # More balanced RSI weak signals
        'macd_strength_threshold': 0.08,  # Lower MACD strength threshold
        'macd_moderate_threshold': 0.04,  # Lower MACD moderate threshold
    }

    # Apply changes to the file
    modified_content = content

    # Update base confidence threshold in EnhancedRiskManager
    modified_content = modified_content.replace(
        'base_confidence_threshold=0.5',
        f'base_confidence_threshold={new_thresholds["base_confidence_threshold"]}',
    )

    # Update ML filter confidence threshold
    modified_content = modified_content.replace(
        'if ml_confidence < 0.5:  # Lower threshold for more HOLD signals',
        f'if ml_confidence < {new_thresholds["ml_filter_confidence_threshold"]}:  # Adjusted threshold for better signal balance',
    )

    # Update TA combined threshold
    modified_content = modified_content.replace(
        'if total_signal >= 0.5:  # Even lower threshold for BUY (was 0.6)',
        f'if total_signal >= {new_thresholds["ta_combined_threshold"]}:  # Adjusted threshold for better signal balance',
    )
    modified_content = modified_content.replace(
        'elif total_signal <= -0.5:  # Even lower threshold for SELL (was -0.6)',
        f'elif total_signal <= -{new_thresholds["ta_combined_threshold"]}:  # Adjusted threshold for better signal balance',
    )

    # Update RSI thresholds
    modified_content = modified_content.replace(
        'if rsi < 25:  # Less extreme oversold (was 20)',
        f'if rsi < {new_thresholds["rsi_strong_thresholds"][0]}:  # Adjusted oversold threshold',
    )
    modified_content = modified_content.replace(
        'elif rsi > 75:  # Less extreme overbought (was 80)',
        f'elif rsi > {new_thresholds["rsi_strong_thresholds"][1]}:  # Adjusted overbought threshold',
    )
    modified_content = modified_content.replace(
        'elif rsi < 35:  # Oversold (was 30)',
        f'elif rsi < {new_thresholds["rsi_weak_thresholds"][0]}:  # Adjusted weak oversold threshold',
    )
    modified_content = modified_content.replace(
        'elif rsi > 65:  # Overbought (was 70)',
        f'elif rsi > {new_thresholds["rsi_weak_thresholds"][1]}:  # Adjusted weak overbought threshold',
    )

    # Update MACD thresholds
    modified_content = modified_content.replace(
        'if macd_strength > 0.12:  # Lower threshold for strong signals (was 0.15)',
        f'if macd_strength > {new_thresholds["macd_strength_threshold"]}:  # Adjusted strong signal threshold',
    )
    modified_content = modified_content.replace(
        'elif macd_strength > 0.06:  # Lower threshold for moderate divergence (was 0.08)',
        f'elif macd_strength > {new_thresholds["macd_moderate_threshold"]}:  # Adjusted moderate signal threshold',
    )

    # Write the modified content back
    with open("live_trader_clean.py", "w", encoding="utf-8") as f:
        f.write(modified_content)

    logger.info("✅ Signal confidence thresholds adjusted successfully!")

    # Create a report
    create_threshold_report(new_thresholds)

    return new_thresholds


def create_threshold_report(thresholds):
    """Create a report of the adjusted thresholds"""
    logger.info("Creating threshold adjustment report...")

    if not os.path.exists("models"):
        os.makedirs("models")

    with open("models/threshold_adjustment_report.txt", "w") as f:
        f.write("MR BEN Signal Confidence Threshold Adjustment Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Adjustment Date: {datetime.now()}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("ADJUSTED THRESHOLDS:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Base Confidence Threshold: {thresholds['base_confidence_threshold']}\n")
        f.write(f"ML Filter Confidence Threshold: {thresholds['ml_filter_confidence_threshold']}\n")
        f.write(f"TA Combined Threshold: {thresholds['ta_combined_threshold']}\n")
        f.write(f"RSI Strong Thresholds: {thresholds['rsi_strong_thresholds']}\n")
        f.write(f"RSI Weak Thresholds: {thresholds['rsi_weak_thresholds']}\n")
        f.write(f"MACD Strength Threshold: {thresholds['macd_strength_threshold']}\n")
        f.write(f"MACD Moderate Threshold: {thresholds['macd_moderate_threshold']}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("EXPECTED IMPACT:\n")
        f.write("=" * 60 + "\n")
        f.write("• More balanced BUY/SELL/HOLD signal distribution\n")
        f.write("• Increased trade frequency due to lower thresholds\n")
        f.write("• Better adaptation to current market conditions\n")
        f.write("• Improved signal diversity and reduced bias\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("=" * 60 + "\n")
        f.write("• Monitor signal distribution for the next 24-48 hours\n")
        f.write("• Adjust thresholds further if needed based on performance\n")
        f.write("• Consider increasing thresholds if too many weak signals\n")
        f.write("• Consider decreasing thresholds if too few signals\n")

    logger.info("📄 Threshold adjustment report saved to models/threshold_adjustment_report.txt")


def main():
    """Main function"""
    logger.info("Starting Signal Confidence Threshold Adjustment...")

    try:
        thresholds = adjust_signal_thresholds()

        logger.info("✅ Signal confidence threshold adjustment completed successfully!")
        logger.info("📁 Updated file: live_trader_clean.py")
        logger.info("📄 Report saved to: models/threshold_adjustment_report.txt")

        # Print summary
        logger.info("\n📊 ADJUSTED THRESHOLDS SUMMARY:")
        logger.info(f"   Base Confidence: {thresholds['base_confidence_threshold']}")
        logger.info(f"   ML Filter: {thresholds['ml_filter_confidence_threshold']}")
        logger.info(f"   TA Combined: {thresholds['ta_combined_threshold']}")
        logger.info(f"   RSI Strong: {thresholds['rsi_strong_thresholds']}")
        logger.info(f"   MACD Strength: {thresholds['macd_strength_threshold']}")

    except Exception as e:
        logger.error(f"❌ Error adjusting thresholds: {e}")


if __name__ == "__main__":
    main()
