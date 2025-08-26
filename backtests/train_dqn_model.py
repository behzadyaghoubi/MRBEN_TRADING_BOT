import numpy as np
import pandas as pd
from dqn_env import TradingEnvDQN
from dqn_agent import DQNAgent

# Load historical data
df = pd.read_csv('ohlc_data.csv')

# Initialize the trading environment
env = TradingEnvDQN(df)
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(state_shape, action_size)

# Training settings
episodes = 100
batch_size = 32

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay(batch_size)

    print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {round(agent.epsilon, 3)}")

# Save the trained model
agent.model.save('trained_dqn_model.h5')
print("✅ Trained DQN model saved as trained_dqn_model.h5")