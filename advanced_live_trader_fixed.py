#!/usr/bin/env python3
"""
Advanced MR BEN AI Live Trading System (Fixed)
Integrates the advanced AI system with live trading capabilities
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import the fixed advanced system
from advanced_mrben_system_fixed import AdvancedMRBENSystemFixed

class AdvancedLiveTraderFixed:
    """
    Advanced Live Trading System with MR BEN AI (Fixed)
    """
    
    def __init__(self, config_path: str = "advanced_live_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.ai_system = AdvancedMRBENSystemFixed()
        self.trade_history = []
        self.current_position = None
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load live trading configuration"""
        default_config = {
            'symbol': 'XAUUSD',
            'lot_size': 0.01,
            'max_positions': 1,
            'stop_loss_pips': 50,
            'take_profit_pips': 100,
            'max_daily_trades': 10,
            'min_confidence': 0.7,
            'risk_per_trade': 0.02,
            'trading_hours': {
                'start': '00:00',
                'end': '23:59'
            },
            'session_filters': {
                'Asia': True,
                'London': True,
                'NY': True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        else:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_logger(self):
        """Setup live trading logger"""
        os.makedirs('logs', exist_ok=True)
        
        logger = logging.getLogger('AdvancedLiveTraderFixed')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = f'logs/advanced_live_trader_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data (simulated for demo)"""
        # In a real implementation, this would connect to MT5 or other data source
        current_time = datetime.now()
        
        # Simulate market data
        base_price = 3300 + np.random.uniform(-50, 50)
        
        market_data = {
            'time': current_time.isoformat(),
            'open': base_price + np.random.uniform(-10, 10),
            'high': base_price + np.random.uniform(0, 20),
            'low': base_price - np.random.uniform(0, 20),
            'close': base_price + np.random.uniform(-15, 15),
            'tick_volume': np.random.randint(100, 1000),
            'spread': np.random.uniform(0.1, 0.5),
            'real_volume': np.random.randint(50, 500)
        }
        
        return market_data
    
    def calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators for market data"""
        # In a real implementation, this would use historical data
        # For demo, we'll simulate the indicators
        
        close = market_data['close']
        
        indicators = {
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.uniform(-0.5, 0.5),
            'macd_signal': np.random.uniform(-0.5, 0.5),
            'atr': np.random.uniform(5, 25),
            'sma_20': close + np.random.uniform(-30, 30),
            'sma_50': close + np.random.uniform(-50, 50)
        }
        
        return {**market_data, **indicators}
    
    def should_trade(self, signal: Dict[str, Any]) -> bool:
        """Check if we should execute a trade based on signal and conditions"""
        try:
            # Check confidence threshold
            if signal['confidence'] < self.config['min_confidence']:
                return False
            
            # Check if we already have a position
            if self.current_position is not None:
                return False
            
            # Check daily trade limit
            today_trades = sum(1 for trade in self.trade_history 
                             if trade['timestamp'].date() == datetime.now().date())
            if today_trades >= self.config['max_daily_trades']:
                return False
            
            # Check trading hours
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute
            current_time = f"{current_hour:02d}:{current_minute:02d}"
            
            if not (self.config['trading_hours']['start'] <= current_time <= self.config['trading_hours']['end']):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trade conditions: {e}")
            return False
    
    def execute_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Execute a trade based on the signal"""
        try:
            if signal['signal'] == 0:  # HOLD
                return False
            
            # Calculate trade parameters
            entry_price = market_data['close']
            stop_loss_pips = self.config['stop_loss_pips']
            take_profit_pips = self.config['take_profit_pips']
            
            if signal['signal'] == 1:  # BUY
                stop_loss = entry_price - (stop_loss_pips * 0.1)
                take_profit = entry_price + (take_profit_pips * 0.1)
                trade_type = 'BUY'
            else:  # SELL
                stop_loss = entry_price + (stop_loss_pips * 0.1)
                take_profit = entry_price - (take_profit_pips * 0.1)
                trade_type = 'SELL'
            
            # Create trade record
            trade = {
                'id': len(self.trade_history) + 1,
                'timestamp': datetime.now(),
                'type': trade_type,
                'symbol': self.config['symbol'],
                'lot_size': self.config['lot_size'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_confidence': signal['confidence'],
                'signal_score': signal['score'],
                'market_context': market_data,
                'status': 'OPEN'
            }
            
            # In a real implementation, this would send order to MT5
            self.logger.info(f"Executing {trade_type} trade: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
            
            # Record trade
            self.trade_history.append(trade)
            self.current_position = trade
            self.performance_metrics['total_trades'] += 1
            
            # Log trade execution
            self._log_trade_execution(trade, signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def check_position_status(self, current_price: float) -> bool:
        """Check if current position should be closed"""
        if self.current_position is None:
            return False
        
        try:
            entry_price = self.current_position['entry_price']
            stop_loss = self.current_position['stop_loss']
            take_profit = self.current_position['take_profit']
            trade_type = self.current_position['type']
            
            # Check stop loss
            if trade_type == 'BUY' and current_price <= stop_loss:
                self._close_position('STOP_LOSS', current_price)
                return True
            elif trade_type == 'SELL' and current_price >= stop_loss:
                self._close_position('STOP_LOSS', current_price)
                return True
            
            # Check take profit
            if trade_type == 'BUY' and current_price >= take_profit:
                self._close_position('TAKE_PROFIT', current_price)
                return True
            elif trade_type == 'SELL' and current_price <= take_profit:
                self._close_position('TAKE_PROFIT', current_price)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position status: {e}")
            return False
    
    def _close_position(self, reason: str, exit_price: float):
        """Close current position"""
        try:
            if self.current_position is None:
                return
            
            # Calculate profit/loss
            entry_price = self.current_position['entry_price']
            lot_size = self.current_position['lot_size']
            
            if self.current_position['type'] == 'BUY':
                profit = (exit_price - entry_price) * lot_size * 100000  # Convert to account currency
            else:
                profit = (entry_price - exit_price) * lot_size * 100000
            
            # Update trade record
            self.current_position['exit_price'] = exit_price
            self.current_position['exit_time'] = datetime.now()
            self.current_position['close_reason'] = reason
            self.current_position['profit'] = profit
            self.current_position['status'] = 'CLOSED'
            
            # Update performance metrics
            self.performance_metrics['total_profit'] += profit
            if profit > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            # Calculate max drawdown
            if profit < 0:
                self.performance_metrics['max_drawdown'] = min(
                    self.performance_metrics['max_drawdown'], 
                    self.performance_metrics['total_profit']
                )
            
            self.logger.info(f"Position closed: {reason}, Profit: {profit:.2f}")
            
            # Log position closure
            self._log_position_closure(self.current_position)
            
            # Clear current position
            self.current_position = None
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _log_trade_execution(self, trade: Dict[str, Any], signal: Dict[str, Any]):
        """Log trade execution details"""
        log_entry = {
            'timestamp': trade['timestamp'].isoformat(),
            'event': 'TRADE_EXECUTION',
            'trade_id': trade['id'],
            'trade_type': trade['type'],
            'entry_price': trade['entry_price'],
            'signal_confidence': trade['signal_confidence'],
            'signal_score': trade['signal_score'],
            'market_context': trade['market_context']
        }
        
        # Save to file
        log_file = f'logs/trade_executions_{datetime.now().strftime("%Y%m%d")}.json'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_position_closure(self, trade: Dict[str, Any]):
        """Log position closure details"""
        log_entry = {
            'timestamp': trade['exit_time'].isoformat(),
            'event': 'POSITION_CLOSURE',
            'trade_id': trade['id'],
            'exit_price': trade['exit_price'],
            'close_reason': trade['close_reason'],
            'profit': trade['profit'],
            'duration_minutes': (trade['exit_time'] - trade['timestamp']).total_seconds() / 60
        }
        
        # Save to file
        log_file = f'logs/position_closures_{datetime.now().strftime("%Y%m%d")}.json'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            if self.performance_metrics['total_trades'] == 0:
                self.logger.info("No trades executed yet")
                return
            
            win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            avg_profit = self.performance_metrics['total_profit'] / self.performance_metrics['total_trades']
            
            report = f"""
============================================================
ADVANCED LIVE TRADING (FIXED) - PERFORMANCE REPORT
============================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {self.config['symbol']}

TRADING METRICS:
===============
- Total Trades: {self.performance_metrics['total_trades']}
- Winning Trades: {self.performance_metrics['winning_trades']}
- Losing Trades: {self.performance_metrics['losing_trades']}
- Win Rate: {win_rate:.2%}
- Total Profit: {self.performance_metrics['total_profit']:.2f}
- Average Profit per Trade: {avg_profit:.2f}
- Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}

CURRENT STATUS:
==============
- Current Position: {'OPEN' if self.current_position else 'NONE'}
- Daily Trades: {sum(1 for trade in self.trade_history if trade['timestamp'].date() == datetime.now().date())}
- AI System Status: {'ACTIVE' if self.ai_system.models else 'INACTIVE'}

CONFIGURATION:
=============
- Lot Size: {self.config['lot_size']}
- Stop Loss: {self.config['stop_loss_pips']} pips
- Take Profit: {self.config['take_profit_pips']} pips
- Min Confidence: {self.config['min_confidence']}
- Max Daily Trades: {self.config['max_daily_trades']}

RECOMMENDATIONS:
===============
1. Monitor win rate and adjust strategy if needed
2. Review stop loss and take profit levels
3. Check AI system performance regularly
4. Monitor risk management parameters

============================================================
Report generated by Advanced Live Trading System (Fixed)
============================================================
"""
            
            # Save report
            report_path = f'logs/advanced_live_performance_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Performance report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
    
    def run_live_trading(self, duration_minutes: int = 60):
        """Run live trading for specified duration"""
        try:
            self.logger.info(f"Starting Advanced Live Trading (Fixed) for {duration_minutes} minutes")
            
            # Load AI system
            self.ai_system.load_models()
            
            # Train models if needed
            data_paths = [
                'data/XAUUSD_PRO_M5_live.csv',
                'data/XAUUSD_PRO_M5_enhanced.csv'
            ]
            
            for data_path in data_paths:
                if os.path.exists(data_path):
                    if 'lstm' not in self.ai_system.models:
                        self.ai_system.train_advanced_lstm(data_path)
                    if 'ml_filter' not in self.ai_system.models:
                        self.ai_system.train_advanced_ml_filter(data_path)
                    break
            
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                try:
                    # Get market data
                    market_data = self.get_market_data()
                    market_data = self.calculate_technical_indicators(market_data)
                    
                    # Generate AI signal
                    signal = self.ai_system.generate_ensemble_signal(market_data)
                    
                    # Check if we should trade
                    if self.should_trade(signal):
                        self.execute_trade(signal, market_data)
                    
                    # Check position status
                    if self.current_position:
                        self.check_position_status(market_data['close'])
                    
                    # Check for performance drift
                    self.ai_system.check_performance_drift()
                    
                    # Log current status
                    self.logger.info(f"Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}, Position: {'OPEN' if self.current_position else 'NONE'}")
                    
                    # Wait before next iteration
                    time.sleep(5)  # 5-second intervals
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    time.sleep(10)
            
            # Generate final report
            self.generate_performance_report()
            
            self.logger.info("Advanced Live Trading (Fixed) completed")
            
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}")

def main():
    """Main function to run advanced live trading"""
    print("🎯 Advanced MR BEN AI Live Trading System (Fixed)")
    print("=" * 70)
    
    # Create and run live trader
    trader = AdvancedLiveTraderFixed()
    
    # Run for 30 minutes (demo)
    trader.run_live_trading(duration_minutes=30)
    
    print("✅ Advanced Live Trading (Fixed) completed!")
    print("📋 Check logs/ directory for detailed reports")

if __name__ == "__main__":
    main() 