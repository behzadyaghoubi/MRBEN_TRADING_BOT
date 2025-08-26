# rl_agent.py
import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size, model_path="dqn_rl_weights.h5", memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.target_model.load_weights(model_path)
            print(f"[OK] DQN weights loaded from {model_path}")

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32, update_target_freq=10):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(np.array([next_state]), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            self.model.fit(np.array([state]), target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if random.randint(1, update_target_freq) == 1:
            self.update_target_model()

    def save(self, path="dqn_rl_weights.h5"):
        self.model.save_weights(path)
        print(f"[OK] DQN weights saved to {path}")