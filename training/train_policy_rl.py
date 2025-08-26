#!/usr/bin/env python3
"""
Reinforcement Learning Policy Training for MR BEN AI Architecture
Trains an RL agent to optimize TP/SL/Trailing/Position sizing decisions
"""
import json
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TradingEnvironment:
    """
    Gym-like environment for training RL policy
    Simulates trading with historical data and realistic execution
    """

    def __init__(
        self,
        data_path: str = "data/labeled_events.csv",
        price_data_path: str = "data/XAUUSD_PRO_M15_history.csv",
    ):
        self.logger = logging.getLogger("TradingEnv")

        # Load training data
        self.labeled_data = pd.read_csv(data_path) if os.path.exists(data_path) else None
        self.price_data = pd.read_csv(price_data_path) if os.path.exists(price_data_path) else None

        if self.price_data is not None:
            self.price_data['time'] = pd.to_datetime(self.price_data['time'])
            self.price_data = self.price_data.sort_values('time').reset_index(drop=True)

        # Environment state
        self.current_step = 0
        self.max_steps = 1000
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.position = None
        self.trade_history = []

        # Simulation parameters
        self.spread = 30  # points
        self.commission = 0.0  # simplified

        # State and action dimensions
        self.state_dim = 15  # features + context
        self.action_dim = 6  # [accept/reject, tp1_mult, tp2_mult, sl_mult, tp_share, size_mult]

        # Reward shaping parameters
        self.reward_params = {
            'profit_scale': 1.0,
            'drawdown_penalty': 2.0,
            'variance_penalty': 0.5,
            'tp1_bonus': 0.1,
            'consistency_bonus': 0.2,
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.trade_history = []

        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Args:
            action: [accept_prob, tp1_mult, tp2_mult, sl_mult, tp_share, size_mult]

        Returns:
            (next_state, reward, done, info)
        """
        # Parse action
        accept_prob = torch.sigmoid(torch.tensor(action[0])).item()
        tp1_mult = 0.5 + 1.5 * torch.sigmoid(torch.tensor(action[1])).item()  # 0.5 to 2.0
        tp2_mult = 1.0 + 2.0 * torch.sigmoid(torch.tensor(action[2])).item()  # 1.0 to 3.0
        sl_mult = 0.8 + 0.8 * torch.sigmoid(torch.tensor(action[3])).item()  # 0.8 to 1.6
        tp_share = torch.sigmoid(torch.tensor(action[4])).item()  # 0.0 to 1.0
        size_mult = 0.5 + 1.5 * torch.sigmoid(torch.tensor(action[5])).item()  # 0.5 to 2.0

        reward = 0.0
        info = {}

        # Generate synthetic signal for this step
        signal_data = self._generate_signal()

        # Decision: accept or reject the signal
        if random.random() < accept_prob and signal_data['signal'] != 0:
            # Execute trade
            trade_result = self._execute_trade(
                signal_data, tp1_mult, tp2_mult, sl_mult, tp_share, size_mult
            )
            reward = self._calculate_reward(trade_result)
            info = trade_result
        else:
            # No trade executed
            reward = 0.0
            info = {'action': 'no_trade', 'reason': 'rejected_or_no_signal'}

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._get_state()

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        # Create a state vector with market features + context

        # Market features (normalized)
        market_features = np.array(
            [
                3400.0,  # close price (normalized)
                0.001,  # return
                3398.0,  # sma_20
                3395.0,  # sma_50
                15.0,  # atr
                55.0,  # rsi
                2.1,  # macd
                1.8,  # macd_signal
                10.0,  # hour
                1.0,  # day of week
            ]
        )

        # Context features
        context_features = np.array(
            [
                self.balance / self.initial_balance - 1.0,  # equity change
                len(self.trade_history),  # trades executed
                self._get_recent_pnl(),  # recent performance
                self._get_drawdown(),  # current drawdown
                float(self.current_step) / self.max_steps,  # progress
            ]
        )

        return np.concatenate([market_features, context_features]).astype(np.float32)

    def _generate_signal(self) -> dict:
        """Generate synthetic trading signal"""
        # Simple synthetic signal generation
        signal = random.choice([-1, 0, 1])
        confidence = random.uniform(0.5, 0.9) if signal != 0 else 0.0

        return {'signal': signal, 'confidence': confidence, 'source': 'synthetic'}

    def _execute_trade(
        self,
        signal_data: dict,
        tp1_mult: float,
        tp2_mult: float,
        sl_mult: float,
        tp_share: float,
        size_mult: float,
    ) -> dict:
        """Simulate trade execution with realistic fills"""

        # Calculate position size
        risk_per_trade = 0.01 * size_mult  # 1% base risk
        position_size = (self.balance * risk_per_trade) / 100  # Simplified

        # Entry price with spread
        entry_price = 3400.0  # Simplified
        if signal_data['signal'] == 1:  # BUY
            entry_price += self.spread / 2
        else:  # SELL
            entry_price -= self.spread / 2

        # Calculate TP/SL levels
        atr = 15.0  # Simplified ATR
        risk_amount = atr * sl_mult

        if signal_data['signal'] == 1:  # BUY
            sl_price = entry_price - risk_amount
            tp1_price = entry_price + (risk_amount * tp1_mult)
            tp2_price = entry_price + (risk_amount * tp2_mult)
        else:  # SELL
            sl_price = entry_price + risk_amount
            tp1_price = entry_price - (risk_amount * tp1_mult)
            tp2_price = entry_price - (risk_amount * tp2_mult)

        # Simulate market movement and fills
        outcome = self._simulate_trade_outcome(
            signal_data, entry_price, sl_price, tp1_price, tp2_price, tp_share
        )

        # Calculate PnL
        pnl = outcome['pnl'] * position_size
        self.balance += pnl

        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'signal': signal_data['signal'],
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'tp_share': tp_share,
            'position_size': position_size,
            'pnl': pnl,
            'outcome': outcome['result'],
            'r_multiple': outcome['r_multiple'],
        }

        self.trade_history.append(trade_record)

        return trade_record

    def _simulate_trade_outcome(
        self, signal_data: dict, entry: float, sl: float, tp1: float, tp2: float, tp_share: float
    ) -> dict:
        """Simulate realistic trade outcome"""

        # Simple simulation based on signal confidence
        confidence = signal_data['confidence']

        # Probability of hitting TP1, TP2, or SL
        tp1_prob = confidence * 0.6
        tp2_prob = confidence * 0.3
        sl_prob = 1.0 - confidence

        outcome = random.choices(['tp1', 'tp2', 'sl'], weights=[tp1_prob, tp2_prob, sl_prob])[0]

        # Calculate R-multiple and PnL
        risk = abs(entry - sl)

        if outcome == 'tp1':
            profit1 = abs(tp1 - entry)
            r_mult = (tp_share * profit1) / risk
            pnl = tp_share * profit1 - (1 - tp_share) * risk  # Assume TP2 hit SL
            result = 'partial_win'
        elif outcome == 'tp2':
            profit1 = abs(tp1 - entry)
            profit2 = abs(tp2 - entry)
            r_mult = (tp_share * profit1 + (1 - tp_share) * profit2) / risk
            pnl = tp_share * profit1 + (1 - tp_share) * profit2
            result = 'full_win'
        else:  # sl
            r_mult = -1.0
            pnl = -risk
            result = 'loss'

        return {'result': result, 'r_multiple': r_mult, 'pnl': pnl / entry}  # Normalized PnL

    def _calculate_reward(self, trade_result: dict) -> float:
        """Calculate reward for the RL agent"""

        # Base reward from PnL
        pnl = trade_result.get('pnl', 0.0)
        reward = pnl * self.reward_params['profit_scale']

        # Bonus for hitting TP1
        if trade_result.get('outcome') in ['partial_win', 'full_win']:
            reward += self.reward_params['tp1_bonus']

        # Penalty for drawdown
        current_dd = self._get_drawdown()
        if current_dd > 0.02:  # 2% drawdown penalty
            reward -= current_dd * self.reward_params['drawdown_penalty']

        # Consistency bonus (Sharpe-like)
        if len(self.trade_history) > 10:
            recent_returns = [t['pnl'] for t in self.trade_history[-10:]]
            if np.std(recent_returns) > 0:
                sharpe_like = np.mean(recent_returns) / np.std(recent_returns)
                reward += sharpe_like * self.reward_params['consistency_bonus']

        return reward

    def _get_recent_pnl(self, n: int = 5) -> float:
        """Get recent PnL performance"""
        if len(self.trade_history) < n:
            return 0.0
        return sum(t['pnl'] for t in self.trade_history[-n:])

    def _get_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.trade_history:
            return 0.0

        peak_balance = self.initial_balance
        for trade in self.trade_history:
            peak_balance = max(peak_balance, self.balance)

        return max(0.0, (peak_balance - self.balance) / peak_balance)


class PolicyNetwork(nn.Module):
    """Neural network for policy learning"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class PPOTrainer:
    """Proximal Policy Optimization trainer"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4

        self.memory = deque(maxlen=2000)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(state_tensor)

        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """Update policy using collected experiences"""
        if len(self.memory) < 100:
            return

        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in self.memory]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in self.memory]).to(self.device)

        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Forward pass
            action_pred = self.policy(states)

            # Calculate loss (simplified)
            loss = nn.MSELoss()(action_pred, actions) - 0.01 * returns.mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.memory.clear()


def train_policy_rl(episodes: int = 1000, save_interval: int = 100):
    """Main training function"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PolicyTrainer")

    # Initialize environment and trainer
    env = TradingEnvironment()
    trainer = PPOTrainer(env.state_dim, env.action_dim)

    logger.info(f"Starting RL policy training for {episodes} episodes")

    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            # Select action
            action = trainer.select_action(state)

            # Execute step
            next_state, reward, done, info = env.step(action)

            # Store transition
            trainer.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update policy
        trainer.update()

        episode_rewards.append(episode_reward)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(
                f"Episode {episode}, Avg Reward: {avg_reward:.3f}, " f"Balance: {env.balance:.2f}"
            )

        # Save model
        if episode % save_interval == 0 and episode > 0:
            model_path = f"models/policy_rl_episode_{episode}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(trainer.policy.state_dict(), model_path)
            logger.info(f"Model saved: {model_path}")

    # Save final model
    final_model_path = "models/policy_rl.pt"
    torch.save(trainer.policy.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # Save training results
    results = {
        "episode_rewards": episode_rewards,
        "final_performance": {
            "avg_reward": np.mean(episode_rewards[-100:]),
            "total_episodes": episodes,
            "final_balance": env.balance,
        },
    }

    with open("models/rl_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("RL policy training completed!")


if __name__ == "__main__":
    train_policy_rl(episodes=500, save_interval=50)
