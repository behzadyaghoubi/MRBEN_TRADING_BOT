import os

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# === General Settings ===
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M15")
START_BALANCE = float(os.getenv("START_BALANCE", "100000"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "3000"))
POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "2.0"))

# === MetaTrader 5 ===
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "12345678"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")

# === AI Settings ===
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "models/mrben_ai_model.joblib")
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.6"))

# === Database ===
DB_PATH = os.getenv("DB_PATH", "data/trading.db")

# === Logging ===
LOG_DIR = os.getenv("LOG_DIR", "logs/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === Notification (Optional) ===
EMAIL_ALERTS = os.getenv("EMAIL_ALERTS", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
