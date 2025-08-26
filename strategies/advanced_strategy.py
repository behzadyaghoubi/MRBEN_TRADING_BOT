"""
Advanced Trading Strategy
Combines multiple technical indicators with sophisticated signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedStrategy:
    """
    Advanced trading strategy combining multiple technical indicators
    with sophisticated signal generation and risk management
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize advanced strategy with configuration
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        
        # Strategy parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_multiplier = self.config.get('atr_multiplier', 2)
        
        self.ema_short = self.config.get('ema_short', 9)
        self.ema_long = self.config.get('ema_long', 21)
        
        self.stoch_k = self.config.get('stoch_k', 14)
        self.stoch_d = self.config.get('stoch_d', 3)
        
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
        
        # Signal thresholds
        self.signal_threshold = self.config.get('signal_threshold', 0.7)
        self.confirmation_threshold = self.config.get('confirmation_threshold', 0.5)
        
        # Risk management
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5%
        self.max_open_trades = self.config.get('max_open_trades', 3)
        
        logger.info(f"Advanced Strategy initialized with config: {self.config}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(
                df['close'], self.macd_fast, self.macd_slow, self.macd_signal
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(
                df['close'], self.bb_period, self.bb_std
            )
            
            # ATR
            df['atr'] = self._calculate_atr(df, self.atr_period)
            
            # EMAs
            df['ema_short'] = self._calculate_ema(df['close'], self.ema_short)
            df['ema_long'] = self._calculate_ema(df['close'], self.ema_long)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
                df, self.stoch_k, self.stoch_d
            )
            
            # Volume indicators
            df['volume_ma'] = df['tick_volume'].rolling(self.volume_ma_period).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            # Price action
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            
            # Trend indicators
            df['trend_strength'] = abs(df['ema_short'] - df['ema_long']) / df['ema_long']
            df['price_vs_ema'] = (df['close'] - df['ema_short']) / df['ema_short']
            
            logger.debug(f"Calculated indicators for {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with signals added
        """
        try:
            # Initialize signal columns
            df['signal'] = 0
            df['signal_strength'] = 0.0
            df['signal_reason'] = ''
            
            for i in range(len(df)):
                if i < 50:  # Skip first 50 candles for indicator stability
                    continue
                
                signal, strength, reason = self._analyze_candle(df, i)
                df.loc[df.index[i], 'signal'] = signal
                df.loc[df.index[i], 'signal_strength'] = strength
                df.loc[df.index[i], 'signal_reason'] = reason
            
            logger.info(f"Generated signals for {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return df
    
    def _analyze_candle(self, df: pd.DataFrame, index: int) -> Tuple[int, float, str]:
        """
        Analyze a single candle for trading signals
        
        Args:
            df: DataFrame with indicators
            index: Current candle index
            
        Returns:
            Tuple of (signal, strength, reason)
        """
        try:
            current = df.iloc[index]
            prev = df.iloc[index - 1]
            
            # Initialize scores
            bullish_score = 0.0
            bearish_score = 0.0
            reasons = []
            
            # RSI Analysis
            if current['rsi'] < self.rsi_oversold:
                bullish_score += 0.2
                reasons.append("RSI oversold")
            elif current['rsi'] > self.rsi_overbought:
                bearish_score += 0.2
                reasons.append("RSI overbought")
            
            # MACD Analysis
            if (current['macd'] > current['macd_signal'] and 
                prev['macd'] <= prev['macd_signal']):
                bullish_score += 0.15
                reasons.append("MACD bullish crossover")
            elif (current['macd'] < current['macd_signal'] and 
                  prev['macd'] >= prev['macd_signal']):
                bearish_score += 0.15
                reasons.append("MACD bearish crossover")
            
            # Bollinger Bands Analysis
            if current['close'] < current['bb_lower']:
                bullish_score += 0.15
                reasons.append("Price below BB lower")
            elif current['close'] > current['bb_upper']:
                bearish_score += 0.15
                reasons.append("Price above BB upper")
            
            # EMA Analysis
            if (current['ema_short'] > current['ema_long'] and 
                current['close'] > current['ema_short']):
                bullish_score += 0.1
                reasons.append("Price above EMAs")
            elif (current['ema_short'] < current['ema_long'] and 
                  current['close'] < current['ema_short']):
                bearish_score += 0.1
                reasons.append("Price below EMAs")
            
            # Stochastic Analysis
            if current['stoch_k'] < 20 and current['stoch_d'] < 20:
                bullish_score += 0.1
                reasons.append("Stochastic oversold")
            elif current['stoch_k'] > 80 and current['stoch_d'] > 80:
                bearish_score += 0.1
                reasons.append("Stochastic overbought")
            
            # Volume Analysis
            if current['volume_ratio'] > 1.5:
                if bullish_score > bearish_score:
                    bullish_score += 0.1
                    reasons.append("High volume bullish")
                elif bearish_score > bullish_score:
                    bearish_score += 0.1
                    reasons.append("High volume bearish")
            
            # Price Action Analysis
            if current['body_ratio'] > 0.6:  # Strong body
                if current['close'] > current['open']:  # Bullish candle
                    bullish_score += 0.1
                    reasons.append("Strong bullish candle")
                else:  # Bearish candle
                    bearish_score += 0.1
                    reasons.append("Strong bearish candle")
            
            # Trend Strength Analysis
            if current['trend_strength'] > 0.01:  # Strong trend
                if current['price_vs_ema'] > 0:
                    bullish_score += 0.1
                    reasons.append("Strong uptrend")
                else:
                    bearish_score += 0.1
                    reasons.append("Strong downtrend")
            
            # Determine final signal
            if bullish_score > bearish_score and bullish_score >= self.signal_threshold:
                return 1, bullish_score, " | ".join(reasons)
            elif bearish_score > bullish_score and bearish_score >= self.signal_threshold:
                return -1, bearish_score, " | ".join(reasons)
            else:
                return 0, max(bullish_score, bearish_score), "No clear signal"
                
        except Exception as e:
            logger.error(f"Error analyzing candle {index}: {e}")
            return 0, 0.0, f"Error: {e}"
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, 
                              stop_loss_pips: float, pip_value: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade
            stop_loss_pips: Stop loss in pips
            pip_value: Value of one pip
            
        Returns:
            Position size in lots
        """
        try:
            risk_amount = account_balance * risk_per_trade
            stop_loss_amount = stop_loss_pips * pip_value
            
            if stop_loss_amount > 0:
                position_size = risk_amount / stop_loss_amount
                return round(position_size, 2)
            else:
                return 0.01  # Minimum position size
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal: int, atr_multiplier: float = None) -> float:
        """
        Calculate dynamic stop loss based on ATR
        
        Args:
            df: DataFrame with indicators
            signal: Signal direction (1 for buy, -1 for sell)
            atr_multiplier: ATR multiplier for stop loss
            
        Returns:
            Stop loss price
        """
        try:
            if atr_multiplier is None:
                atr_multiplier = self.atr_multiplier
            
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            if signal == 1:  # Buy signal
                stop_loss = current_price - (current_atr * atr_multiplier)
            elif signal == -1:  # Sell signal
                stop_loss = current_price + (current_atr * atr_multiplier)
            else:
                return 0.0
            
            return round(stop_loss, 5)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return 0.0
    
    def calculate_take_profit(self, df: pd.DataFrame, signal: int, 
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            df: DataFrame with indicators
            signal: Signal direction
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            Take profit price
        """
        try:
            current_price = df['close'].iloc[-1]
            stop_loss = self.calculate_stop_loss(df, signal)
            
            if stop_loss == 0.0:
                return 0.0
            
            if signal == 1:  # Buy signal
                risk = current_price - stop_loss
                take_profit = current_price + (risk * risk_reward_ratio)
            elif signal == -1:  # Sell signal
                risk = stop_loss - current_price
                take_profit = current_price - (risk * risk_reward_ratio)
            else:
                return 0.0
            
            return round(take_profit, 5)
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                             d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and parameters"""
        return {
            'name': 'Advanced Strategy',
            'description': 'Multi-indicator strategy with sophisticated signal generation',
            'parameters': {
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'macd_fast': self.macd_fast,
                'macd_slow': self.macd_slow,
                'macd_signal': self.macd_signal,
                'bb_period': self.bb_period,
                'bb_std': self.bb_std,
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier,
                'ema_short': self.ema_short,
                'ema_long': self.ema_long,
                'stoch_k': self.stoch_k,
                'stoch_d': self.stoch_d,
                'volume_ma_period': self.volume_ma_period,
                'signal_threshold': self.signal_threshold,
                'confirmation_threshold': self.confirmation_threshold
            },
            'risk_management': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_loss': self.max_daily_loss,
                'max_open_trades': self.max_open_trades
            }
        } 