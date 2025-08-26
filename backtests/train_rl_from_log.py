# train_rl_from_log.py
import pandas as pd
import numpy as np
from rl_agent import DQNAgent

LOG_FILE = "live_trades_log.csv"
MODEL_FILE = "dqn_rl_weights.h5"
BATCH_SIZE = 32

df = pd.read_csv(LOG_FILE)
# فرض: state = [SMA_FAST, RSI, ...] | action = 0/1/2 | reward = profit | done = آخرین معامله یا بسته‌شدن حساب

# فرض می‌کنیم ستون‌ها: SMA_FAST, RSI, action, profit (یا باید محاسبه کنی)
states = df[["SMA_FAST", "RSI", "pinbar", "engulfing", "close"]].values
actions = df["action"].map({"BUY": 0, "SELL": 1, "HOLD": 2}).values
rewards = df.get("profit", df.get("result", pd.Series(np.zeros(len(df))))).values
dones = [False] * (len(df)-1) + [True]

agent = DQNAgent(state_size=states.shape[1], action_size=3, model_path=MODEL_FILE)

for i in range(len(df)-1):
    agent.remember(states[i], actions[i], rewards[i], states[i+1], dones[i])
agent.replay(batch_size=BATCH_SIZE)

agent.save(MODEL_FILE)
print("[OK] RL agent trained and saved.")