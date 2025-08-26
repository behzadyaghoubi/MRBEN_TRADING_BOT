#!/usr/bin/env python3
"""
Professional Hybrid Signal Engine
================================

یک ماژول سیگنال‌دهی کاملاً حرفه‌ای برای سیستم تریدینگ:
- ترکیب خروجی LSTM با اندیکاتورهای تکنیکال (RSI, MACD)
- فیلتر هوشمند بر اساس confidence و شرایط بازار
- کاملاً ماژولار و قابل تنظیم

Author: MRBEN Trading System
"""

import numpy as np
import pandas as pd

class ProfessionalSignalEngine:
    def __init__(self, lstm_threshold=0.1, rsi_bounds=(35, 65), macd_confirm=True, min_confidence=0.1):
        self.lstm_threshold = lstm_threshold
        self.rsi_bounds = rsi_bounds
        self.macd_confirm = macd_confirm
        self.min_confidence = min_confidence
        self.signal_map = {2: "BUY", 1: "HOLD", 0: "SELL"}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate professional hybrid signals for the entire DataFrame"""
        df = df.copy()
        signals = []
        confidences = []
        reasons = []
        for idx, row in df.iterrows():
            signal, conf, reason = self.professional_signal(row)
            signals.append(signal)
            confidences.append(conf)
            reasons.append(reason)
        df['signal'] = signals
        df['signal_confidence'] = confidences
        df['signal_reason'] = reasons
        df['signal_label'] = df['signal'].map(self.signal_map)
        return df

    def professional_signal(self, row) -> tuple:
        lstm_probs = [row.get('lstm_sell_proba', 0), row.get('lstm_hold_proba', 0), row.get('lstm_buy_proba', 0)]
        lstm_signal = int(np.argmax(lstm_probs))
        lstm_conf = float(np.max(lstm_probs))
        rsi = row.get('RSI', 50)
        if rsi < self.rsi_bounds[0]:
            rsi_signal = 0  # SELL
        elif rsi > self.rsi_bounds[1]:
            rsi_signal = 2  # BUY
        else:
            rsi_signal = 1  # HOLD
        macd = row.get('MACD', 0)
        macd_signal = 2 if macd > 0 else (0 if macd < 0 else 1)
        # شرط confidence منعطف‌تر
        if lstm_conf < self.lstm_threshold or lstm_conf < self.min_confidence:
            return 1, lstm_conf, 'HOLD_LOW_CONF'
        # اگر حداقل دو تا از سه شرط BUY باشند
        if [lstm_signal, rsi_signal, macd_signal].count(2) >= 2:
            return 2, lstm_conf, 'BUY_COMBO'
        # اگر حداقل دو تا از سه شرط SELL باشند
        if [lstm_signal, rsi_signal, macd_signal].count(0) >= 2:
            return 0, lstm_conf, 'SELL_COMBO'
        # اگر فقط LSTM سیگنال BUY و یکی از اندیکاتورها هم BUY
        if lstm_signal == 2 and (rsi_signal == 2 or macd_signal == 2):
            return 2, lstm_conf, 'BUY_LSTM_PLUS'
        if lstm_signal == 0 and (rsi_signal == 0 or macd_signal == 0):
            return 0, lstm_conf, 'SELL_LSTM_PLUS'
        # در غیر این صورت HOLD
        return 1, lstm_conf, 'HOLD_FILTERED'

if __name__ == "__main__":
    df = pd.read_csv('lstm_signals_fixed.csv')
    engine = ProfessionalSignalEngine(lstm_threshold=0.1, rsi_bounds=(35, 65), macd_confirm=True, min_confidence=0.1)
    signals_df = engine.generate_signals(df)
    signals_df.to_csv('outputs/professional_signals.csv', index=False)
    print('Professional signals generated and saved to outputs/professional_signals.csv')
    print(signals_df['signal_label'].value_counts()) 