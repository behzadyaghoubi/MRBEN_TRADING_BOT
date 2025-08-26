#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Pro Strategy - Baseline Backtest Implementation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import os

logger = logging.getLogger(__name__)

class BaselineBacktest:
    """Baseline backtest using SMA20/50 strategy with fixed confidence 0.7"""
    
    def __init__(self, config):
        self.config = config
        # Handle case where config might be None or missing SYMBOL
        if config and hasattr(config, 'SYMBOL'):
            self.symbol = config.SYMBOL
        else:
            self.symbol = "XAUUSD.PRO"  # Default symbol
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        
    def run(self, symbol: str, dt_from: str, dt_to: str, cfg: Any) -> Dict[str, Any]:
        """Run baseline backtest"""
        try:
            logger.info(f"🚀 Starting baseline backtest for {symbol}")
            logger.info(f"📅 Period: {dt_from} to {dt_to}")
            
            # Simulate backtest with mock data
            # In production, this would use real MT5 data
            result = self._simulate_backtest(dt_from, dt_to)
            
            # Generate reports
            self._generate_reports()
            
            logger.info("✅ Baseline backtest completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _simulate_backtest(self, dt_from: str, dt_to: str) -> Dict[str, Any]:
        """Simulate backtest with mock data"""
        try:
            # Parse dates
            start_date = datetime.strptime(dt_from, "%Y-%m-%d")
            end_date = datetime.strptime(dt_to, "%Y-%m-%d")
            
            # Generate mock price data
            dates = pd.date_range(start=start_date, end=end_date, freq='15T')
            n_bars = len(dates)
            
            # Simulate price movement (random walk with trend)
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.0001, 0.002, n_bars)  # Small positive drift
            prices = 3300 * np.exp(np.cumsum(returns))  # Start at 3300
            
            # Create mock DataFrame
            df = pd.DataFrame({
                'time': dates,
                'open': prices * (1 + np.random.normal(0, 0.0005, n_bars)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
                'close': prices,
                'tick_volume': np.random.randint(100, 1000, n_bars)
            })
            
            # Calculate indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Generate signals
            signals = self._generate_signals(df)
            
            # Execute trades
            self._execute_trades(df, signals)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            return {
                "status": "completed",
                "trades": len(self.trades),
                "pnl": metrics['net_return'],
                "metrics": metrics,
                "equity_curve": self.equity_curve
            }
            
        except Exception as e:
            logger.error(f"Error in backtest simulation: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using SMA20/50 crossover"""
        signals = pd.Series(0, index=df.index)
        
        # SMA crossover strategy
        for i in range(50, len(df)):
            if pd.isna(df['sma_20'].iloc[i]) or pd.isna(df['sma_50'].iloc[i]):
                continue
                
            if df['sma_20'].iloc[i] > df['sma_50'].iloc[i]:
                signals.iloc[i] = 1  # Buy signal
            elif df['sma_20'].iloc[i] < df['sma_50'].iloc[i]:
                signals.iloc[i] = -1  # Sell signal
        
        return signals
    
    def _execute_trades(self, df: pd.DataFrame, signals: pd.Series):
        """Execute trades based on signals"""
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(df)):
            signal = signals.iloc[i]
            current_price = df['close'].iloc[i]
            current_time = df['time'].iloc[i]
            
            # Close existing position if signal changes
            if position != 0 and signal != position:
                # Calculate P&L
                if position == 1:  # Long position
                    pnl = (current_price - entry_price) / entry_price
                else:  # Short position
                    pnl = (entry_price - current_price) / entry_price
                
                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'side': 'BUY' if position == 1 else 'SELL',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl * 100
                }
                self.trades.append(trade)
                
                # Update balance
                self.current_balance *= (1 + pnl)
                self.equity_curve.append({
                    'time': current_time,
                    'balance': self.current_balance
                })
                
                # Reset position
                position = 0
                entry_price = 0
                entry_time = None
            
            # Open new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_time = current_time
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate backtest metrics"""
        if not self.trades:
            return {
                'net_return': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Basic metrics
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate returns
        returns = [t['pnl'] for t in self.trades]
        avg_return = np.mean(returns) if returns else 0
        
        # Sharpe ratio (simplified)
        sharpe_ratio = avg_return / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'net_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(self.trades) - len(winning_trades)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0
        
        balances = [e['balance'] for e in self.equity_curve]
        peak = balances[0]
        max_dd = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _generate_reports(self):
        """Generate backtest reports and artifacts"""
        try:
            # Create baseline directory
            os.makedirs('docs/pro/01_baseline', exist_ok=True)
            
            # Save trade list
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv('docs/pro/01_baseline/trade-list.csv', index=False)
            
            # Save metrics
            metrics = self._calculate_metrics()
            with open('docs/pro/01_baseline/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.to_csv('docs/pro/01_baseline/equity-curve.csv', index=False)
            
            # Generate backtest report
            self._generate_backtest_report(metrics)
            
            logger.info("✅ Backtest reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def _generate_backtest_report(self, metrics: Dict[str, Any]):
        """Generate comprehensive backtest report"""
        report = f"""# Baseline Backtest Report - MR BEN Pro Strategy

## Test Configuration
- **Symbol**: {self.symbol}
- **Strategy**: SMA20/50 Crossover with fixed confidence 0.7
- **Timeframe**: 15 minutes
- **Initial Balance**: ${self.initial_balance:,.2f}

## Results Summary
- **Net Return**: {metrics['net_return']:.2%}
- **Total Trades**: {metrics['total_trades']}
- **Win Rate**: {metrics['win_rate']:.1%}
- **Average Return per Trade**: {metrics['avg_return']:.2%}
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
- **Maximum Drawdown**: {metrics['max_drawdown']:.2%}

## Trade Breakdown
- **Winning Trades**: {metrics['winning_trades']}
- **Losing Trades**: {metrics['losing_trades']}

## Equity Curve
- **Final Balance**: ${self.current_balance:,.2f}
- **Total P&L**: ${self.current_balance - self.initial_balance:,.2f}

## Analysis
This baseline test establishes the performance benchmark using the current SMA20/50 strategy.
The results will be compared against enhanced strategies in subsequent phases.

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('docs/pro/01_baseline/backtest_report.md', 'w') as f:
            f.write(report)

def run(symbol: str, dt_from: str, dt_to: str, cfg: Any) -> Dict[str, Any]:
    """Main entry point for backtest"""
    backtest = BaselineBacktest(cfg)
    return backtest.run(symbol, dt_from, dt_to, cfg)
