# combined_signal_generator.py - FINAL PROFESSIONAL VERSION

import pandas as pd
from book_strategy import generate_book_signals
from rl_predictor import load_trained_agent, predict_next_signal

def generate_filtered_signals(df, threshold=0.7, agent_path="trained_q_table.npy", epsilon=0.01):
    """
    Generate combined trading signals using both book (classic) strategy and RL (Reinforcement Learning) agent.
    :param df: DataFrame with price & indicator columns
    :param threshold: [NOT USED] – can be used for future RL signal confidence
    :param agent_path: Path to trained RL agent model (Q-table .npy, etc)
    :param epsilon: RL agent exploration rate (should be very low for live trading)
    :return: DataFrame with new column 'filtered_signal'
    """
    # Step 1: Generate classic strategy signals
    df = generate_book_signals(df)

    # Step 2: Load RL agent
    try:
        agent = load_trained_agent(agent_path, epsilon=epsilon)
    except Exception as e:
        print(f"⛔️ RL Agent loading failed: {e}")
        agent = None

    filtered_signals = []
    for i in range(len(df)):
        signal = df.iloc[i]["signal"]
        # Only evaluate RL filter for actionable signals (BUY/SELL)
        if signal in ["BUY", "SELL"] and agent is not None:
            try:
                temp_df = df.iloc[:i+1].copy()
                score_signal = predict_next_signal(agent, temp_df)
                # Accept only if RL agent agrees, else HOLD
                if score_signal == signal:
                    filtered_signals.append(signal)
                else:
                    filtered_signals.append("HOLD")
            except Exception as e:
                print(f"⚠️ RL filter error at row {i}: {e}")
                filtered_signals.append("HOLD")
        else:
            filtered_signals.append("HOLD")

    df["filtered_signal"] = filtered_signals
    return df

if __name__ == "__main__":
    # Test: Run filtered signal generator on sample data
    data_path = "xauusd_m15.csv"
    output_path = "final_signal.csv"
    try:
        df = pd.read_csv(data_path)
        df = generate_filtered_signals(df)
        df.to_csv(output_path, index=False)
        print(f"✅ Filtered combined signals saved to {output_path}")
    except Exception as e:
        print(f"⛔️ Error: {e}")