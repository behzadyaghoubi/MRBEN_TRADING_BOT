#!/usr/bin/env python3
"""
Dynamic Position Manager
Advanced position management with ATR-based trailing stop loss and dynamic take profit
"""

import json
import time
from datetime import datetime

import pandas as pd

# MT5 Integration
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not available")
    MT5_AVAILABLE = False


class DynamicPositionManager:
    """Advanced position management with ATR-based trailing stop loss and dynamic take profit."""

    def __init__(self, config_path: str = 'config/settings.json'):
        """Initialize the position manager."""
        self.config = self._load_config(config_path)
        self.mt5_connected = False
        self.positions = {}  # Track active positions
        self.atr_period = 14
        self.atr_multiplier = 2.0  # ATR multiplier for trailing stop
        self.tp_trail_percent = 0.5  # TP trailing percentage
        self.min_atr_distance = 10  # Minimum ATR distance in points
        self.max_positions = self.config.get('max_open_trades', 2)

        # Initialize MT5 connection
        self._initialize_mt5()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {
                'trading': {
                    'symbol': 'XAUUSD.PRO',
                    'volume': 0.01,
                    'magic_number': 654321,
                    'max_open_trades': 2,
                },
                'mt5_login': 1104123,
                'mt5_password': '-4YcBgRd',
                'mt5_server': 'OxSecurities-Demo',
            }

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 not available")
            return False

        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
                return False

            # Login to MT5
            if not mt5.login(
                login=self.config['mt5_login'],
                password=self.config['mt5_password'],
                server=self.config['mt5_server'],
            ):
                print(f"❌ Failed to login to MT5: {mt5.last_error()}")
                return False

            # Select symbol
            symbol = self.config['trading']['symbol']
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not found")
                return False

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    print(f"❌ Failed to select symbol {symbol}")
                    return False

            self.mt5_connected = True
            print("✅ MT5 connected successfully")
            print(f"📊 Symbol: {symbol} | Point: {symbol_info.point}")
            return True

        except Exception as e:
            print(f"❌ MT5 initialization error: {e}")
            return False

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = true_range.rolling(window=period).mean()
            return atr.iloc[-1]

        except Exception as e:
            print(f"❌ ATR calculation error: {e}")
            return 20.0  # Default ATR value

    def get_current_positions(self) -> list:
        """Get all current open positions."""
        try:
            if not self.mt5_connected:
                return []

            positions = mt5.positions_get(symbol=self.config['trading']['symbol'])
            if positions is None:
                return []

            return list(positions)

        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []

    def update_position_tracking(self):
        """Update internal position tracking."""
        try:
            positions = self.get_current_positions()
            current_tickets = set()

            for pos in positions:
                ticket = pos.ticket
                current_tickets.add(ticket)

                # Add new position to tracking
                if ticket not in self.positions:
                    self.positions[ticket] = {
                        'ticket': ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit,
                        'time': pos.time,
                        'high_since_entry': pos.price_open,  # Track highest price since entry
                        'low_since_entry': pos.price_open,  # Track lowest price since entry
                        'atr_at_entry': 0,
                        'last_update': datetime.now(),
                    }

                    # Calculate initial ATR
                    df = self._get_market_data(100)
                    if df is not None:
                        self.positions[ticket]['atr_at_entry'] = self.calculate_atr(df)
                        print(
                            f"📊 Position {ticket} opened | ATR: {self.positions[ticket]['atr_at_entry']:.2f}"
                        )

                # Update existing position
                else:
                    self.positions[ticket].update(
                        {
                            'price_current': pos.price_current,
                            'sl': pos.sl,
                            'tp': pos.tp,
                            'profit': pos.profit,
                            'last_update': datetime.now(),
                        }
                    )

                    # Update high/low tracking
                    if pos.price_current > self.positions[ticket]['high_since_entry']:
                        self.positions[ticket]['high_since_entry'] = pos.price_current
                    if pos.price_current < self.positions[ticket]['low_since_entry']:
                        self.positions[ticket]['low_since_entry'] = pos.price_current

            # Remove closed positions
            closed_tickets = set(self.positions.keys()) - current_tickets
            for ticket in closed_tickets:
                if ticket in self.positions:
                    pos_info = self.positions[ticket]
                    print(f"📊 Position {ticket} closed | Final Profit: {pos_info['profit']:.2f}")
                    del self.positions[ticket]

        except Exception as e:
            print(f"❌ Error updating position tracking: {e}")

    def _get_market_data(self, bars: int = 100) -> pd.DataFrame | None:
        """Get market data from MT5."""
        try:
            if not self.mt5_connected:
                return None

            symbol = self.config['trading']['symbol']
            timeframe = mt5.TIMEFRAME_M5

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            return None

    def calculate_trailing_stop(self, position_info: dict, current_atr: float) -> float | None:
        """Calculate trailing stop loss based on ATR and price movement."""
        try:
            if position_info['type'] == 'BUY':
                # For long positions: SL = highest price - (ATR * multiplier)
                highest_price = position_info['high_since_entry']
                atr_distance = current_atr * self.atr_multiplier
                new_sl = highest_price - atr_distance

                # Ensure minimum distance
                min_distance = self.min_atr_distance * 0.1  # Convert points to price
                if highest_price - new_sl < min_distance:
                    new_sl = highest_price - min_distance

                # Only move SL up (never down)
                current_sl = position_info['sl']
                if current_sl == 0 or new_sl > current_sl:
                    return new_sl

            else:  # SELL
                # For short positions: SL = lowest price + (ATR * multiplier)
                lowest_price = position_info['low_since_entry']
                atr_distance = current_atr * self.atr_multiplier
                new_sl = lowest_price + atr_distance

                # Ensure minimum distance
                min_distance = self.min_atr_distance * 0.1
                if new_sl - lowest_price < min_distance:
                    new_sl = lowest_price + min_distance

                # Only move SL down (never up)
                current_sl = position_info['sl']
                if current_sl == 0 or new_sl < current_sl:
                    return new_sl

            return None

        except Exception as e:
            print(f"❌ Error calculating trailing stop: {e}")
            return None

    def calculate_trailing_take_profit(self, position_info: dict) -> float | None:
        """Calculate trailing take profit."""
        try:
            if position_info['type'] == 'BUY':
                # For long positions: move TP up as price increases
                current_price = position_info['price_current']
                current_tp = position_info['tp']

                if current_tp == 0:
                    # Set initial TP
                    entry_price = position_info['price_open']
                    initial_tp_distance = entry_price * 0.01  # 1% initial TP
                    return entry_price + initial_tp_distance

                # Trail TP up
                trail_distance = current_price * (self.tp_trail_percent / 100)
                new_tp = current_price + trail_distance

                # Only move TP up
                if new_tp > current_tp:
                    return new_tp

            else:  # SELL
                # For short positions: move TP down as price decreases
                current_price = position_info['price_current']
                current_tp = position_info['tp']

                if current_tp == 0:
                    # Set initial TP
                    entry_price = position_info['price_open']
                    initial_tp_distance = entry_price * 0.01  # 1% initial TP
                    return entry_price - initial_tp_distance

                # Trail TP down
                trail_distance = current_price * (self.tp_trail_percent / 100)
                new_tp = current_price - trail_distance

                # Only move TP down
                if new_tp < current_tp or current_tp == 0:
                    return new_tp

            return None

        except Exception as e:
            print(f"❌ Error calculating trailing TP: {e}")
            return None

    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify position's stop loss and take profit."""
        try:
            if not self.mt5_connected:
                return False

            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False

            position = positions[0]

            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "sl": sl,
                "tp": tp,
                "position": ticket,
            }

            # Send modification request
            result = mt5.order_send(request)
            if result is None:
                print(f"❌ Failed to modify position {ticket}: Result is None")
                return False

            if result.retcode == 10009:  # TRADE_RETCODE_DONE
                print(f"✅ Position {ticket} modified | SL: {sl:.2f} | TP: {tp:.2f}")
                return True
            else:
                print(f"❌ Failed to modify position {ticket}: {result.retcode} - {result.comment}")
                return False

        except Exception as e:
            print(f"❌ Error modifying position: {e}")
            return False

    def can_open_new_position(self) -> bool:
        """Check if a new position can be opened."""
        try:
            current_positions = len(self.positions)
            max_positions = self.config['trading']['max_open_trades']

            if current_positions >= max_positions:
                print(f"📊 Max positions reached ({current_positions}/{max_positions})")
                return False

            return True

        except Exception as e:
            print(f"❌ Error checking position limit: {e}")
            return False

    def manage_positions(self):
        """Main position management loop."""
        try:
            # Update position tracking
            self.update_position_tracking()

            if not self.positions:
                return

            # Get current market data
            df = self._get_market_data(100)
            if df is None:
                return

            # Calculate current ATR
            current_atr = self.calculate_atr(df, self.atr_period)

            print(f"📊 Managing {len(self.positions)} positions | ATR: {current_atr:.2f}")

            # Manage each position
            for ticket, position_info in self.positions.items():
                try:
                    # Calculate new trailing stop
                    new_sl = self.calculate_trailing_stop(position_info, current_atr)

                    # Calculate new trailing take profit
                    new_tp = self.calculate_trailing_take_profit(position_info)

                    # Apply modifications if needed
                    current_sl = position_info['sl']
                    current_tp = position_info['tp']

                    if new_sl is not None and abs(new_sl - current_sl) > 0.1:
                        print(
                            f"📊 Position {ticket} | Updating SL: {current_sl:.2f} → {new_sl:.2f}"
                        )
                        self.modify_position(ticket, new_sl, current_tp)

                    if new_tp is not None and abs(new_tp - current_tp) > 0.1:
                        print(
                            f"📊 Position {ticket} | Updating TP: {current_tp:.2f} → {new_tp:.2f}"
                        )
                        self.modify_position(ticket, current_sl, new_tp)

                except Exception as e:
                    print(f"❌ Error managing position {ticket}: {e}")

        except Exception as e:
            print(f"❌ Error in position management: {e}")

    def get_position_summary(self) -> dict:
        """Get summary of all positions."""
        try:
            summary = {'total_positions': len(self.positions), 'total_profit': 0.0, 'positions': []}

            for ticket, pos_info in self.positions.items():
                summary['total_profit'] += pos_info['profit']
                summary['positions'].append(
                    {
                        'ticket': ticket,
                        'type': pos_info['type'],
                        'volume': pos_info['volume'],
                        'entry_price': pos_info['price_open'],
                        'current_price': pos_info['price_current'],
                        'sl': pos_info['sl'],
                        'tp': pos_info['tp'],
                        'profit': pos_info['profit'],
                        'atr_at_entry': pos_info['atr_at_entry'],
                    }
                )

            return summary

        except Exception as e:
            print(f"❌ Error getting position summary: {e}")
            return {'total_positions': 0, 'total_profit': 0.0, 'positions': []}

    def run_management_loop(self, interval_seconds: int = 30):
        """Run continuous position management loop."""
        print("🚀 Starting Dynamic Position Manager")
        print(f"📊 ATR Period: {self.atr_period} | ATR Multiplier: {self.atr_multiplier}")
        print(f"📊 TP Trail Percent: {self.tp_trail_percent}%")
        print(f"📊 Max Positions: {self.max_positions}")
        print(f"📊 Management Interval: {interval_seconds} seconds")
        print("=" * 60)

        try:
            while True:
                try:
                    # Manage positions
                    self.manage_positions()

                    # Print summary
                    summary = self.get_position_summary()
                    if summary['total_positions'] > 0:
                        print(
                            f"📊 Summary: {summary['total_positions']} positions | Total Profit: {summary['total_profit']:.2f}"
                        )

                    # Wait for next iteration
                    time.sleep(interval_seconds)

                except KeyboardInterrupt:
                    print("\n🛑 Position management stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Error in management loop: {e}")
                    time.sleep(interval_seconds)

        except Exception as e:
            print(f"❌ Fatal error in management loop: {e}")
        finally:
            if self.mt5_connected:
                mt5.shutdown()


def main():
    """Main function for testing."""
    try:
        # Create position manager
        manager = DynamicPositionManager()

        if not manager.mt5_connected:
            print("❌ Failed to connect to MT5")
            return

        # Run management loop
        manager.run_management_loop(interval_seconds=30)

    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()
