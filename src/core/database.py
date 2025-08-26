"""
Database management for MR BEN Trading Bot.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings

from .logger import get_logger

logger = get_logger("database")


class DatabaseManager:
    """Manages SQLite database operations for the trading bot."""

    def __init__(self):
        self.trades_db_path = Path(settings.database.trades_db)
        self.signals_db_path = Path(settings.database.signals_db)

        # Create data directory
        self.trades_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.signals_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_databases()

    def _init_databases(self):
        """Initialize database tables."""
        self._init_trades_db()
        self._init_signals_db()

    def _init_trades_db(self):
        """Initialize trades database."""
        with sqlite3.connect(self.trades_db_path) as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    sl REAL,
                    tp REAL,
                    lot_size REAL NOT NULL,
                    profit REAL,
                    balance REAL,
                    ai_decision INTEGER,
                    ai_confidence REAL,
                    result_code INTEGER,
                    comment TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            '''
            )

            # Performance metrics table
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_profit REAL,
                    win_rate REAL,
                    avg_profit REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            '''
            )

            conn.commit()
            logger.info("Trades database initialized")

    def _init_signals_db(self):
        """Initialize signals database."""
        with sqlite3.connect(self.signals_db_path) as conn:
            cursor = conn.cursor()

            # Signals table
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL,
                    ai_filtered BOOLEAN,
                    executed BOOLEAN DEFAULT FALSE,
                    features TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            '''
            )

            # AI model performance table
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS ai_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_signals INTEGER,
                    approved_signals INTEGER,
                    rejected_signals INTEGER,
                    accuracy REAL,
                    avg_confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            '''
            )

            conn.commit()
            logger.info("Signals database initialized")

    def save_trade(self, trade_data: dict[str, Any]) -> int:
        """Save a trade to the database."""
        try:
            with sqlite3.connect(self.trades_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    '''
                    INSERT INTO trades (
                        timestamp, symbol, action, entry_price, exit_price,
                        sl, tp, lot_size, profit, balance, ai_decision,
                        ai_confidence, result_code, comment, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        trade_data.get('timestamp'),
                        trade_data.get('symbol'),
                        trade_data.get('action'),
                        trade_data.get('entry_price'),
                        trade_data.get('exit_price'),
                        trade_data.get('sl'),
                        trade_data.get('tp'),
                        trade_data.get('lot_size'),
                        trade_data.get('profit'),
                        trade_data.get('balance'),
                        trade_data.get('ai_decision'),
                        trade_data.get('ai_confidence'),
                        trade_data.get('result_code'),
                        trade_data.get('comment'),
                        trade_data.get('status', 'open'),
                    ),
                )

                trade_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Trade saved with ID: {trade_id}")
                return trade_id

        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            raise

    def update_trade(self, trade_id: int, update_data: dict[str, Any]) -> bool:
        """Update a trade in the database."""
        try:
            with sqlite3.connect(self.trades_db_path) as conn:
                cursor = conn.cursor()

                # Build update query dynamically
                set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
                query = f"UPDATE trades SET {set_clause} WHERE id = ?"

                values = list(update_data.values()) + [trade_id]
                cursor.execute(query, values)

                conn.commit()
                logger.info(f"Trade {trade_id} updated successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False

    def get_trades(self, limit: int | None = None, status: str | None = None) -> pd.DataFrame:
        """Get trades from database."""
        try:
            with sqlite3.connect(self.trades_db_path) as conn:
                query = "SELECT * FROM trades"
                params = []

                if status:
                    query += " WHERE status = ?"
                    params.append(status)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += f" LIMIT {limit}"

                df = pd.read_sql_query(query, conn, params=params)
                return df

        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return pd.DataFrame()

    def save_signal(self, signal_data: dict[str, Any]) -> int:
        """Save a signal to the database."""
        try:
            with sqlite3.connect(self.signals_db_path) as conn:
                cursor = conn.cursor()

                features_json = json.dumps(signal_data.get('features', {}))

                cursor.execute(
                    '''
                    INSERT INTO signals (
                        timestamp, symbol, timeframe, signal_type, price,
                        confidence, ai_filtered, executed, features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        signal_data.get('timestamp'),
                        signal_data.get('symbol'),
                        signal_data.get('timeframe'),
                        signal_data.get('signal_type'),
                        signal_data.get('price'),
                        signal_data.get('confidence'),
                        signal_data.get('ai_filtered'),
                        signal_data.get('executed', False),
                        features_json,
                    ),
                )

                signal_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Signal saved with ID: {signal_id}")
                return signal_id

        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            raise

    def get_signals(self, limit: int | None = None, executed: bool | None = None) -> pd.DataFrame:
        """Get signals from database."""
        try:
            with sqlite3.connect(self.signals_db_path) as conn:
                query = "SELECT * FROM signals"
                params = []

                if executed is not None:
                    query += " WHERE executed = ?"
                    params.append(executed)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += f" LIMIT {limit}"

                df = pd.read_sql_query(query, conn, params=params)

                # Parse features JSON
                if 'features' in df.columns:
                    df['features'] = df['features'].apply(lambda x: json.loads(x) if x else {})

                return df

        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return pd.DataFrame()

    def calculate_performance_metrics(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Calculate performance metrics for a date range."""
        try:
            with sqlite3.connect(self.trades_db_path) as conn:
                query = "SELECT * FROM trades WHERE status = 'closed'"
                params = []

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                df = pd.read_sql_query(query, conn, params=params)

                if df.empty:
                    return {}

                total_trades = len(df)
                winning_trades = len(df[df['profit'] > 0])
                losing_trades = len(df[df['profit'] < 0])
                total_profit = df['profit'].sum()
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                avg_profit = total_profit / total_trades if total_trades > 0 else 0

                # Calculate max drawdown
                cumulative_profit = df['profit'].cumsum()
                running_max = cumulative_profit.expanding().max()
                drawdown = cumulative_profit - running_max
                max_drawdown = drawdown.min()

                # Calculate Sharpe ratio (simplified)
                returns = df['profit'] / df['entry_price']
                sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

                metrics = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'total_profit': total_profit,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                }

                return metrics

        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}

    def backup_database(self) -> bool:
        """Create a backup of the database."""
        if not settings.database.backup_enabled:
            return True

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Backup trades database
            backup_trades_path = self.trades_db_path.parent / f"trades_backup_{timestamp}.db"
            with sqlite3.connect(self.trades_db_path) as source:
                with sqlite3.connect(backup_trades_path) as backup:
                    source.backup(backup)

            # Backup signals database
            backup_signals_path = self.signals_db_path.parent / f"signals_backup_{timestamp}.db"
            with sqlite3.connect(self.signals_db_path) as source:
                with sqlite3.connect(backup_signals_path) as backup:
                    source.backup(backup)

            logger.info(f"Database backup created: {backup_trades_path}, {backup_signals_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()
