# logger/trade_logger.py

import csv
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, log_dir='logs/trades'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{date_str}_trades.csv")

    def log_trade(self, symbol, signal, predicted_confidence, actual_result,
                  entry_price, exit_price, profit, feature_dict):
        log_path = self._get_log_path()

        # ساخت فایل و نوشتن هدر اگر وجود نداشت
        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "signal", "predicted_confidence", "actual_result",
                    "entry_price", "exit_price", "profit", "features"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                signal,
                round(predicted_confidence, 4),
                actual_result,
                entry_price,
                exit_price,
                round(profit, 2),
                ";".join(f"{k}={v}" for k, v in feature_dict.items())
            ])