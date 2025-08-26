#!/usr/bin/env python3
"""
Test script to verify signal generation with new thresholds
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the live trader components
from live_trader_clean import MRBENAdvancedAISystem


def test_signal_generation():
    """Test signal generation with various market conditions"""

    print("🧪 Testing MR BEN Signal Generation with New Thresholds")
    print("=" * 60)

    # Initialize the AI system
    ai_system = MRBENAdvancedAISystem()

    # Test cases with different market conditions
    test_cases = [
        {
            'name': 'Strong BUY Signal',
            'data': {
                'time': datetime.now().isoformat(),
                'open': 3300.0,
                'high': 3310.0,
                'low': 3295.0,
                'close': 3308.0,  # Strong upward movement
                'tick_volume': 1000,
            },
        },
        {
            'name': 'Strong SELL Signal',
            'data': {
                'time': datetime.now().isoformat(),
                'open': 3300.0,
                'high': 3305.0,
                'low': 3285.0,
                'close': 3288.0,  # Strong downward movement
                'tick_volume': 1000,
            },
        },
        {
            'name': 'Weak BUY Signal',
            'data': {
                'time': datetime.now().isoformat(),
                'open': 3300.0,
                'high': 3302.0,
                'low': 3298.0,
                'close': 3301.0,  # Small upward movement
                'tick_volume': 1000,
            },
        },
        {
            'name': 'Neutral Signal',
            'data': {
                'time': datetime.now().isoformat(),
                'open': 3300.0,
                'high': 3300.5,
                'low': 3299.5,
                'close': 3300.0,  # No movement
                'tick_volume': 1000,
            },
        },
    ]

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['name']}")
        print("-" * 40)

        # Generate signal
        signal_data = ai_system.generate_ensemble_signal(test_case['data'])

        # Display results
        print(
            f"Signal: {signal_data['signal']} ({'BUY' if signal_data['signal'] == 1 else 'SELL' if signal_data['signal'] == -1 else 'HOLD'})"
        )
        print(f"Confidence: {signal_data['confidence']:.3f}")
        print(f"Score: {signal_data['score']:.3f}")
        print(f"Source: {signal_data.get('source', 'Unknown')}")

        # Check if signal meets new thresholds
        if signal_data['confidence'] >= 0.1:
            print("✅ Signal meets confidence threshold (>= 0.1)")
        else:
            print("❌ Signal below confidence threshold (< 0.1)")

        if signal_data['signal'] != 0:
            print("✅ Signal is actionable (BUY/SELL)")
        else:
            print("ℹ️ Signal is HOLD")

    print("\n" + "=" * 60)
    print("🎯 Signal Generation Test Complete!")
    print("📈 New thresholds should generate more signals:")
    print("   - Confidence threshold: 0.1 (reduced from 0.3)")
    print("   - Ensemble score threshold: 0.05 (reduced from 0.1)")
    print("   - RSI thresholds: 40/60 (reduced from 35/65)")
    print("   - Price change threshold: 0.02% (reduced from 0.05%)")


if __name__ == "__main__":
    test_signal_generation()
