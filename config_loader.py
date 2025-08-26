import json
import os
import logging
import MetaTrader5 as mt5

# Mapping string timeframe from JSON to MetaTrader5 constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1
}

class TradingConfig:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        data = self.config
        trading = data.get("trading", {})
        # Essential Trading Parameters
        self.SYMBOL = trading.get("symbol", "XAUUSD.PRO")
        self.TIMEFRAME = TIMEFRAME_MAP.get(trading.get("timeframe", "M15"), mt5.TIMEFRAME_M15)
        self.BARS = 500  # Default number of candles to retrieve
        self.VOLUME = trading.get("min_lot", 0.1)
        self.MAGIC = trading.get("magic_number", 123456)
        self.DEVIATION = trading.get("deviation", 10)
        self._set_attributes()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}. Using defaults.")
            return {}

    def _set_attributes(self):
        # Trading
        trading = self.config.get('trading', {})
        self.symbol = trading.get('symbol', 'XAUUSD.PRO')
        self.timeframe = trading.get('timeframe', 'M15')
        self.base_risk = trading.get('base_risk', 0.01)
        self.min_lot = trading.get('min_lot', 0.01)
        self.max_lot = trading.get('max_lot', 2.0)
        self.max_open_trades = trading.get('max_open_trades', 3)
        self.dynamic_sensitivity = trading.get('dynamic_sensitivity', 0.5)
        self.start_balance = trading.get('start_balance', 100000)
        self.stop_loss_pips = trading.get('stop_loss_pips', 30)
        self.take_profit_pips = trading.get('take_profit_pips', 60)
        self.DEVIATION = trading.get("deviation", 10)
        self.magic_number = trading.get('magic_number', 20250721)
        self.VOLUME = trading.get("min_lot", 0.1)
        # Models
        models = self.config.get('models', {})
        self.use_rl = models.get('use_rl', True)
        self.use_lstm = models.get('use_lstm', True)
        self.use_technical = models.get('use_technical', True)
        self.ml_filter_threshold = models.get('ml_filter_threshold', 0.5)
        # MT5
        mt5 = self.config.get('mt5', {})
        self.demo_mode = mt5.get('demo_mode', True)
        self.login = mt5.get('login', 0)
        self.password = mt5.get('password', '')
        self.server = mt5.get('server', '')
        self.SIMULATION_MODE = mt5.get('simulation_mode', False)
        self.MT5_LOGIN = self.login
        self.MT5_PASSWORD = self.password
        self.MT5_SERVER = self.server
        self.MT5_TIMEOUT = mt5.get('timeout', 60000)
        # Logging
        logging_cfg = self.config.get('logging', {})
        self.logging_enabled = logging_cfg.get('enabled', True)
        self.logging_level = logging_cfg.get('level', 'INFO')
        self.log_file = logging_cfg.get('log_file', 'logs/trading_bot.log')
        self.LOGS_DIR = logging_cfg.get('logs_dir', 'logs')
        self.LOG_LEVEL = self.logging_level
        # Notifications
        notifications = self.config.get('notifications', {})
        self.telegram_enabled = notifications.get('telegram_enabled', False)
        self.email_enabled = notifications.get('email_enabled', False)
        self.daily_summary = notifications.get('daily_summary', True)
        # --- سازگاری با کد اصلی ---
        self.SYMBOL = self.symbol
        self.ML_FILTER_MODEL_PATH = self.config.get('ai', {}).get('model_path', "models/mrben_ai_signal_filter_xgb.joblib")
        self.RETRY_DELAY = 10
        self.SLEEP_SECONDS = 60
        self.DATA_DIR = "data"

    def reload(self):
        self.config = self._load_config()
        self._set_attributes() 