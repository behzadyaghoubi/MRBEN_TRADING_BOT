import os
from collections import deque

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


# --- Feature Extraction ---
def extract_features(df):
    # Add technical indicators if not present
    if 'rsi' not in df.columns:
        df['rsi'] = (
            df['close']
            .rolling(14)
            .apply(
                lambda x: 100
                - 100
                / (
                    1
                    + (
                        np.mean(np.maximum(x.diff(), 0))
                        / (np.mean(np.abs(np.minimum(x.diff(), 0))) + 1e-6)
                    )
                )
            )
        )
    if 'macd' not in df.columns:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum.reduce([high_low, high_close, low_close])
        df['atr'] = pd.Series(tr).rolling(14).mean()
    df = df.fillna(0)
    return df


# --- RL Training ---
def train_dqn_from_log(
    log_path,
    model_path,
    scaler_path=None,
    history_path=None,
    log_txt_path=None,
    epochs=20,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
):
    df = pd.read_csv(log_path, header=None)
    # Infer columns
    base = ['time', 'signal', 'price', 'sl', 'tp', 'result', 'buy_proba', 'sell_proba']
    extra = [f'feature_{i}' for i in range(df.shape[1] - len(base))]
    columns = base + extra
    df.columns = columns[: df.shape[1]]
    # Add technicals
    df = extract_features(df)
    # State features
    state_cols = ['price', 'sl', 'tp', 'buy_proba', 'sell_proba', 'atr', 'rsi', 'macd']
    state_cols = [c for c in state_cols if c in df.columns]
    # Label encoding for actions
    action_map = {0: 0, 1: 1, -1: 2}
    reward_map = {'WIN': 1, 'LOSS': -1, 'PENDING': 0, 1: 1, 0: 0, -1: -1}
    # Build transitions
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in range(len(df) - 1):
        s = df.iloc[i][state_cols].values.astype(np.float32)
        a = action_map.get(int(df.iloc[i]['signal']), 0)
        r = reward_map.get(df.iloc[i]['result'], 0)
        s2 = df.iloc[i + 1][state_cols].values.astype(np.float32)
        done = 0 if i < len(df) - 2 else 1
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s2)
        dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    # Normalize states
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    states = scaler.fit_transform(states)
    next_states = scaler.transform(next_states)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    # DQN setup
    state_dim = states.shape[1]
    action_dim = 3
    dqn = DQN(state_dim, action_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(max_size=10000)
    # Fill buffer
    for i in range(len(states)):
        buffer.push((states[i], actions[i], rewards[i], next_states[i], dones[i]))
    # Training loop
    gamma = 0.99
    epsilon = epsilon_start
    episode_rewards = []
    log_lines = []
    for epoch in range(epochs):
        losses = []
        total_reward = 0
        for _ in range(len(buffer) // batch_size):
            batch = buffer.sample(batch_size)
            s, a, r, s2, d = zip(*batch, strict=False)
            s = torch.tensor(np.array(s), dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.long)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(np.array(s2), dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                a = torch.randint(0, action_dim, a.shape)
            q = dqn(s)
            q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = dqn(s2).max(1)[0]
                q_target = r + gamma * q_next * (1 - d)
            loss = loss_fn(q_a, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_reward += r.sum().item()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        log_line = f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.4f} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
        print(log_line)
        log_lines.append(log_line)
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            'model_state_dict': dqn.state_dict(),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'state_cols': state_cols,
        },
        model_path,
    )
    print(f"✅ RL model saved to {model_path}")
    if scaler_path:
        print(f"✅ RL scaler saved to {scaler_path}")
    # Save training history plot
    if history_path:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards, label='Episode Reward')
        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')
        plt.title('RL Training Reward History')
        plt.legend()
        plt.tight_layout()
        plt.savefig(history_path)
        plt.close()
        print(f"✅ RL training history plot saved to {history_path}")
    # Save log
    if log_txt_path:
        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        with open(log_txt_path, 'w', encoding='utf-8') as f:
            for line in log_lines:
                f.write(line + '\n')
        print(f"✅ RL training log saved to {log_txt_path}")
    return dqn, scaler, state_cols


# --- RL Inference ---
def load_rl_model(model_path, scaler_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    dqn = DQN(checkpoint['state_dim'], checkpoint['action_dim'])
    dqn.load_state_dict(checkpoint['model_state_dict'])
    dqn.eval()
    scaler = joblib.load(scaler_path)
    state_cols = checkpoint.get(
        'state_cols', ['price', 'sl', 'tp', 'buy_proba', 'sell_proba', 'atr', 'rsi', 'macd']
    )
    return dqn, scaler, state_cols


def predict_signal_rl(state, dqn, scaler, state_cols):
    # state: dict of features
    x = np.array([[state.get(col, 0) for col in state_cols]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    with torch.no_grad():
        q = dqn(torch.tensor(x_scaled, dtype=torch.float32))
        action = int(torch.argmax(q, dim=1)[0])
    # Map back: 0=Hold, 1=Buy, 2=Sell
    if action == 1:
        return 1
    elif action == 2:
        return -1
    else:
        return 0


if __name__ == "__main__":
    log_path = "data/trade_log_clean.csv"
    model_path = "models/mrben_rl_model.pth"
    scaler_path = "models/mrben_rl_scaler.save"
    history_path = "models/rl_training_history.png"
    log_txt_path = "logs/rl_training_log.txt"
    train_dqn_from_log(
        log_path, model_path, scaler_path, history_path, log_txt_path, epochs=30, batch_size=64
    )
