#!/usr/bin/env python3


def run_analysis():
    """Run the advanced LSTM analysis directly"""

    print("🚀 Running Advanced LSTM Analysis Directly...")
    print("=" * 60)

    try:
        # Import and run the analysis
        import advanced_debug_lstm

        # Run the main analysis
        print("📊 Starting model behavior analysis...")
        success = advanced_debug_lstm.analyze_model_behavior()

        if success:
            print("🧪 Testing with extreme inputs...")
            test_success = advanced_debug_lstm.test_model_with_extreme_inputs()

            print("💡 Suggesting fixes...")
            advanced_debug_lstm.suggest_fixes()

            print("\n✅ Advanced analysis completed successfully!")
        else:
            print("❌ Analysis failed!")

    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_analysis()
