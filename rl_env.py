import pandas as pd
import numpy as np

class TradingEnv:
    def __init__(self, csv_path="live_trades.csv", window_size=3):
        self.data = pd.read_csv(csv_path)
        self.window_size = window_size
        self.current_step = window_size
        self.done = False

        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        return self.get_state()

    def get_state(self):
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        state = window[["price"]].values.flatten()
        return state

    def step(self, action):
        reward = 0
        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
            result = row.get("result", 0)
            reward = result if action == row["signal"] else -abs(result)

        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        next_state = self.get_state()
        return next_state, reward, self.done

    def get_total_steps(self):
        return len(self.data) - self.window_size
