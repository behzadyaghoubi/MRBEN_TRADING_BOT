import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnvDQN(gym.Env):
    """
    محیط تریدینگ برای آموزش DQN
    - فضای state قابل توسعه (OHLC + اندیکاتور)
    - Reward بر اساس سود واقعی پوزیشن
    - Action: 0=BUY, 1=SELL, 2=HOLD
    """

    def __init__(self, df, window_size=10):
        super(TradingEnvDQN, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = window_size
        self.done = False
        self.position = 0  # +1: Buy, -1: Sell, 0: Flat
        self.entry_price = 0
        self.total_profit = 0

        # Define action & observation space
        # Action space: 0=BUY, 1=SELL, 2=HOLD
        self.action_space = spaces.Discrete(3)
        # Observation: OHLC + (در آینده: اندیکاتورها) در پنجره window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 4), dtype=np.float32
        )

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step][
            ['open', 'high', 'low', 'close']
        ].values
        return obs

    def step(self, action):
        reward = 0
        price = self.df['close'].iloc[self.current_step]

        # مدیریت پوزیشن (پوزیشن تک‌تایی)
        if self.position == 0:
            if action == 0:  # BUY
                self.position = 1
                self.entry_price = price
            elif action == 1:  # SELL
                self.position = -1
                self.entry_price = price
        else:
            # خروج از پوزیشن یا ادامه
            if (self.position == 1 and action == 1) or (self.position == -1 and action == 0):
                # خروج معکوس: محاسبه سود/زیان و بسته شدن پوزیشن
                reward = (price - self.entry_price) * self.position
                self.total_profit += reward
                self.position = 0
                self.entry_price = 0

        # پاداش برای پوزیشن باز (مارکت بازی)
        if self.position != 0:
            reward += (
                (price - self.entry_price) * self.position * 0.01
            )  # سود/زیان لحظه‌ای (قابل تنظیم)

        # پایان اپیزود
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            # بستن پوزیشن باز آخر (اگر باشد)
            if self.position != 0:
                reward += (price - self.entry_price) * self.position
                self.total_profit += reward
                self.position = 0
                self.entry_price = 0

        next_obs = self._next_observation()
        return next_obs, reward, self.done, {'total_profit': self.total_profit}


# نمونه تست اولیه
if __name__ == "__main__":
    df = pd.read_csv('ohlc_data.csv')
    env = TradingEnvDQN(df)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # تصادفی برای تست
        next_state, reward, done, info = env.step(action)
        total_reward += reward
    print(
        f"Total reward from random policy: {total_reward:.2f} | Total profit: {info['total_profit']:.2f}"
    )
