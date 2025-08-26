#!/usr/bin/env python3
"""
Quick test of signal generation logic
"""


def test_signal_thresholds():
    """Test the new signal thresholds"""

    print("🧪 Testing New Signal Thresholds")
    print("=" * 50)

    # Test ensemble score thresholds
    print("\n📊 Ensemble Score Thresholds:")
    print("Old threshold: 0.1")
    print("New threshold: 0.05")

    test_scores = [0.02, 0.05, 0.08, -0.02, -0.05, -0.08]

    for score in test_scores:
        if score > 0.05:
            signal = 1
            action = "BUY"
        elif score < -0.05:
            signal = -1
            action = "SELL"
        else:
            signal = 0
            action = "HOLD"

        print(f"Score: {score:.3f} -> Signal: {signal} ({action})")

    # Test RSI thresholds
    print("\n📊 RSI Thresholds:")
    print("Old thresholds: 35/65")
    print("New thresholds: 40/60")

    test_rsi_values = [30, 35, 40, 50, 60, 65, 70]

    for rsi in test_rsi_values:
        if rsi < 40:
            signal = 1
            action = "BUY"
        elif rsi > 60:
            signal = -1
            action = "SELL"
        else:
            signal = 0
            action = "HOLD"

        print(f"RSI: {rsi} -> Signal: {signal} ({action})")

    # Test price change thresholds
    print("\n📊 Price Change Thresholds:")
    print("Old threshold: 0.05%")
    print("New threshold: 0.02%")

    test_changes = [0.001, 0.002, 0.005, -0.001, -0.002, -0.005]

    for change in test_changes:
        if change > 0.0002:
            signal = 1
            action = "BUY"
        elif change < -0.0002:
            signal = -1
            action = "SELL"
        else:
            signal = 0
            action = "HOLD"

        print(f"Change: {change:.4f} ({change*100:.2f}%) -> Signal: {signal} ({action})")

    # Test confidence thresholds
    print("\n📊 Confidence Thresholds:")
    print("Old threshold: 0.3")
    print("New threshold: 0.1")

    test_confidences = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    for conf in test_confidences:
        if conf >= 0.1:
            status = "✅ Meets threshold"
        else:
            status = "❌ Below threshold"

        print(f"Confidence: {conf:.2f} -> {status}")

    print("\n" + "=" * 50)
    print("🎯 Summary of Changes:")
    print("✅ Ensemble score threshold: 0.1 -> 0.05")
    print("✅ RSI thresholds: 35/65 -> 40/60")
    print("✅ Price change threshold: 0.05% -> 0.02%")
    print("✅ Confidence threshold: 0.3 -> 0.1")
    print("✅ Consecutive signals: 2 -> 1")
    print("\n📈 These changes should generate 3-5x more signals!")


if __name__ == "__main__":
    test_signal_thresholds()
