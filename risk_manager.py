import logging
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
from scipy.stats import pearsonr

class RiskManager:
    def __init__(
        self,
        base_risk=0.01,
        min_lot=0.01,
        max_lot=2.0,
        max_open_trades=3,
        max_drawdown=0.10,
        dynamic=True,
        dynamic_sensitivity=0.5,
        correlation_threshold=0.8,
        atr_period=14,
        sl_multiplier=1.5
    ):
        self.base_risk = base_risk
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.max_open_trades = max_open_trades
        self.max_drawdown = max_drawdown
        self.dynamic = dynamic
        self.dynamic_sensitivity = dynamic_sensitivity
        self.correlation_threshold = correlation_threshold
        self.atr_period = atr_period
        self.sl_multiplier = sl_multiplier
        self.logger = logging.getLogger("RiskManager")

    def calc_lot_size(self, symbol, balance, stop_loss_pips, pip_value, open_trades, start_balance, atr=None):
        if self.dynamic and atr:
            risk_fraction = min(1, self.dynamic_sensitivity * atr / stop_loss_pips)
        else:
            risk_fraction = 1.0

        available_balance = balance * (1 - self.max_drawdown)
        risk_amt = available_balance * self.base_risk * risk_fraction
        lot = risk_amt / (stop_loss_pips * pip_value)
        lot = np.clip(lot, self.min_lot, self.max_lot)

        if open_trades >= self.max_open_trades:
            self.logger.warning("Max open trades reached, lot size set to zero.")
            return 0
        return round(lot, 2)

    def check_drawdown(self, balance, start_balance):
        dd = 1 - (balance / start_balance)
        if dd > self.max_drawdown:
            self.logger.error(f"Max drawdown exceeded: {dd:.2%}")
            return False
        return True

    def can_trade(self, balance, open_trades, start_balance):
        if not self.check_drawdown(balance, start_balance):
            return False
        if open_trades >= self.max_open_trades:
            self.logger.warning("Max open trades exceeded.")
            return False
        return True

    def get_atr(self, symbol, timeframe=mt5.TIMEFRAME_H1, bars=100):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < self.atr_period:
            self.logger.error("ATR data not available.")
            return None
        df = pd.DataFrame(rates)
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift())
        df['L-PC'] = abs(df['low'] - df['close'].shift())
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=self.atr_period).mean()
        return df['ATR'].iloc[-1]

    def is_correlated_pair_open(self, symbol, open_symbols, timeframe=mt5.TIMEFRAME_H1, bars=100):
        base_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if base_data is None:
            return False
        base_close = pd.DataFrame(base_data)['close']

        for other_symbol in open_symbols:
            if other_symbol == symbol:
                continue
            other_data = mt5.copy_rates_from_pos(other_symbol, timeframe, 0, bars)
            if other_data is None:
                continue
            other_close = pd.DataFrame(other_data)['close']
            if len(base_close) != len(other_close):
                continue
            corr, _ = pearsonr(base_close, other_close)
            if abs(corr) >= self.correlation_threshold:
                self.logger.warning(f"High correlation with {other_symbol}: {corr:.2f}")
                return True
        return False

    def calculate_chandelier_exit(self, symbol, timeframe=mt5.TIMEFRAME_H1, bars=100):
        data = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if data is None or len(data) < self.atr_period:
            self.logger.warning("Chandelier data not available.")
            return None
        df = pd.DataFrame(data)
        atr = self.get_atr(symbol, timeframe, bars)
        if atr is None:
            return None
        highest_high = df['high'].rolling(window=self.atr_period).max().iloc[-1]
        exit_price = highest_high - self.sl_multiplier * atr
        return round(exit_price, 5)