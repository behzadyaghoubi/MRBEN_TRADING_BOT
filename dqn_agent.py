# dqn_agent.py - FINAL PROFESSIONAL VERSION

import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Trading Automation.
    - Modular and robust.
    - Target network for stable learning.
    - Supports save/load, exploration decay, custom memory, and more.
    """

    def __init__(self, state_shape, action_size, model_path=None, memory_size=5000, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Build the main and target models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Optional: Load weights
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.target_model.load_weights(model_path)
            print(f"✅ DQN weights loaded from {model_path}")

    def _build_model(self):
        """Create the neural network architecture."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=False):
        """
        Select action:
        - random (explore) if not greedy and epsilon condition
        - best Q-value (exploit) otherwise
        """
        if not greedy and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self, batch_size=32, update_target_freq=10):
        """
        Train model on random minibatch and periodically update target.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])

        targets = self.model.predict(states, verbose=0)
        t_next = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(t_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodic target network update
        if random.randint(1, update_target_freq) == 1:
            self.update_target_model()

    def save(self, path="dqn_weights.h5"):
        """Save model weights."""
        self.model.save_weights(path)
        print(f"✅ DQN weights saved at {path}")

    def load(self, path="dqn_weights.h5"):
        """Load model weights."""
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print(f"✅ DQN weights loaded from {path}")

# --- Simple test (dev only) ---
if __name__ == "__main__":
    agent = DQNAgent(state_shape=(10,), action_size=3)
    state = np.random.rand(10)
    action = agent.act(state)
    print("Test Action:", action)