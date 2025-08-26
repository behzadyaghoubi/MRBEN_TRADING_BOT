import gym
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# ====================
# 1. Load market data
# ====================
df = pd.read_csv("data/XAUUSD_PRO_M15_history.csv")  # مسیر داده آموزشی

# ویژگی‌های ورودی برای RL
features = ['close', 'open', 'high', 'low', 'tick_volume']
df = df[features].dropna()

# =====================
# 2. Scale the data
# =====================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
joblib.dump(scaler, "models/mrben_rl_scaler.save")


# =====================
# 3. ساخت محیط سفارشی
# =====================
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.index = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32
        )
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # 0: flat, 1: long, -1: short

    def reset(self):
        self.index = 0
        self.balance = self.initial_balance
        self.position = 0
        return self.data[self.index]

    def step(self, action):
        reward = 0
        done = False
        price = self.data[self.index][0]

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
            elif self.position == -1:
                reward = -1
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
            elif self.position == 1:
                reward = -1

        self.index += 1
        if self.index >= len(self.data) - 1:
            done = True

        return self.data[self.index], reward, done, {}


# =====================
# 4. آموزش مدل RL
# =====================
env = TradingEnv(scaled_data)
env = DummyVecEnv([lambda: env])

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("models/mrben_rl_model.pth")
print("✅ RL Model saved to models/mrben_rl_model.pth")
