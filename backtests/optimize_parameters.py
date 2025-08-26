#!/usr/bin/env python3
"""
MRBEN LSTM Trading System - Parameter Optimizer
===============================================

Parameter optimization script for the LSTM trading system.
Tests different parameter combinations to find optimal settings.

Usage:
    python optimize_parameters.py

This will:
1. Test different threshold combinations
2. Test different trading parameters
3. Find the best performing configuration
4. Save optimal parameters
5. Generate optimization report

Author: MRBEN Trading System
"""

import pandas as pd
import numpy as np
from itertools import product
from lstm_trading_system_pro import LSTMTradingSystem, TradingConfig
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """Parameter optimization for LSTM trading system"""
    
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')
        
    def optimize_thresholds(self, data_file: str = "lstm_signals_pro.csv") -> Dict:
        """Optimize signal generation thresholds"""
        logger.info("Starting threshold optimization...")
        
        # Define parameter ranges
        buy_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
        sell_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
        hold_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
        amplifications = [1.2, 1.5, 1.8, 2.0, 2.5]
        
        # Test all combinations
        total_combinations = len(buy_thresholds) * len(sell_thresholds) * len(hold_thresholds) * len(amplifications)
        logger.info(f"Testing {total_combinations} threshold combinations...")
        
        for i, (buy_thr, sell_thr, hold_thr, amp) in enumerate(
            product(buy_thresholds, sell_thresholds, hold_thresholds, amplifications)
        ):
            try:
                logger.info(f"Testing combination {i+1}/{total_combinations}")
                
                # Create configuration
                config = TradingConfig(
                    buy_threshold=buy_thr,
                    sell_threshold=sell_thr,
                    hold_threshold=hold_thr,
                    signal_amplification=amp
                )
                
                # Run trading system
                trading_system = LSTMTradingSystem(config)
                results = trading_system.run_complete_system(data_file)
                
                # Calculate score (customize based on your preferences)
                performance = results['performance']
                score = self._calculate_score(performance)
                
                # Store results
                result = {
                    'test_id': i + 1,
                    'buy_threshold': buy_thr,
                    'sell_threshold': sell_thr,
                    'hold_threshold': hold_thr,
                    'signal_amplification': amp,
                    'total_trades': performance.get('total_trades', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'total_return': performance.get('total_return', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'profit_factor': performance.get('profit_factor', 0),
                    'final_balance': performance.get('final_balance', 0),
                    'score': score
                }
                
                self.results.append(result)
                
                # Update best configuration
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                    logger.info(f"New best configuration found! Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing combination {i+1}: {e}")
                continue
        
        return self._get_optimization_results()
    
    def optimize_trading_parameters(self, data_file: str = "lstm_signals_pro.csv") -> Dict:
        """Optimize trading parameters (stop loss, take profit, risk)"""
        logger.info("Starting trading parameter optimization...")
        
        # Use best threshold configuration
        if self.best_config is None:
            self.best_config = TradingConfig()
        
        # Define parameter ranges
        stop_losses = [20, 25, 30, 35, 40]
        take_profits = [40, 50, 60, 70, 80]
        risk_per_trades = [0.01, 0.015, 0.02, 0.025, 0.03]
        
        # Test all combinations
        total_combinations = len(stop_losses) * len(take_profits) * len(risk_per_trades)
        logger.info(f"Testing {total_combinations} trading parameter combinations...")
        
        for i, (sl, tp, risk) in enumerate(product(stop_losses, take_profits, risk_per_trades)):
            try:
                logger.info(f"Testing trading combination {i+1}/{total_combinations}")
                
                # Create configuration with best thresholds and new trading params
                config = TradingConfig(
                    buy_threshold=self.best_config.buy_threshold,
                    sell_threshold=self.best_config.sell_threshold,
                    hold_threshold=self.best_config.hold_threshold,
                    signal_amplification=self.best_config.signal_amplification,
                    stop_loss_pips=sl,
                    take_profit_pips=tp,
                    risk_per_trade=risk
                )
                
                # Run trading system
                trading_system = LSTMTradingSystem(config)
                results = trading_system.run_complete_system(data_file)
                
                # Calculate score
                performance = results['performance']
                score = self._calculate_score(performance)
                
                # Store results
                result = {
                    'test_id': i + 1,
                    'stop_loss_pips': sl,
                    'take_profit_pips': tp,
                    'risk_per_trade': risk,
                    'total_trades': performance.get('total_trades', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'total_return': performance.get('total_return', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'profit_factor': performance.get('profit_factor', 0),
                    'final_balance': performance.get('final_balance', 0),
                    'score': score
                }
                
                self.results.append(result)
                
                # Update best configuration
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                    logger.info(f"New best trading configuration found! Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing trading combination {i+1}: {e}")
                continue
        
        return self._get_optimization_results()
    
    def _calculate_score(self, performance: Dict) -> float:
        """Calculate optimization score (customize based on preferences)"""
        total_return = performance.get('total_return', 0)
        win_rate = performance.get('win_rate', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        profit_factor = performance.get('profit_factor', 0)
        total_trades = performance.get('total_trades', 0)
        
        # Penalize if too few trades
        if total_trades < 10:
            return -1000
        
        # Calculate score (prioritize return, win rate, and low drawdown)
        score = (
            total_return * 100 +  # Return component
            win_rate * 50 +       # Win rate component
            (1 - max_drawdown) * 30 +  # Drawdown component (lower is better)
            min(profit_factor, 5) * 10  # Profit factor component (capped at 5)
        )
        
        return score
    
    def _get_optimization_results(self) -> Dict:
        """Get optimization results summary"""
        if not self.results:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Find top 10 configurations
        top_configs = df.nlargest(10, 'score')
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_tests': len(self.results),
            'top_configurations': top_configs.to_dict('records'),
            'summary_stats': {
                'avg_score': df['score'].mean(),
                'std_score': df['score'].std(),
                'avg_return': df['total_return'].mean(),
                'avg_win_rate': df['win_rate'].mean(),
                'avg_drawdown': df['max_drawdown'].mean()
            }
        }
    
    def save_optimization_results(self, results: Dict, filename: str = "optimization_results.json"):
        """Save optimization results"""
        import json
        
        # Convert config to dict for JSON serialization
        if results.get('best_config'):
            config_dict = {
                'buy_threshold': results['best_config'].buy_threshold,
                'sell_threshold': results['best_config'].sell_threshold,
                'hold_threshold': results['best_config'].hold_threshold,
                'signal_amplification': results['best_config'].signal_amplification,
                'stop_loss_pips': results['best_config'].stop_loss_pips,
                'take_profit_pips': results['best_config'].take_profit_pips,
                'risk_per_trade': results['best_config'].risk_per_trade,
                'learning_rate': results['best_config'].learning_rate,
                'batch_size': results['best_config'].batch_size,
                'epochs': results['best_config'].epochs
            }
            results['best_config'] = config_dict
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filename}")
    
    def generate_optimization_report(self, results: Dict, filename: str = "optimization_report.txt"):
        """Generate optimization report"""
        with open(filename, 'w') as f:
            f.write("MRBEN LSTM Trading System - Parameter Optimization Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {results.get('total_tests', 0):,}\n")
            f.write(f"Best Score: {results.get('best_score', 0):.4f}\n\n")
            
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            if results.get('best_config'):
                config = results['best_config']
                f.write(f"Buy Threshold: {config.get('buy_threshold', 0)}\n")
                f.write(f"Sell Threshold: {config.get('sell_threshold', 0)}\n")
                f.write(f"Hold Threshold: {config.get('hold_threshold', 0)}\n")
                f.write(f"Signal Amplification: {config.get('signal_amplification', 0)}\n")
                f.write(f"Stop Loss (pips): {config.get('stop_loss_pips', 0)}\n")
                f.write(f"Take Profit (pips): {config.get('take_profit_pips', 0)}\n")
                f.write(f"Risk per Trade: {config.get('risk_per_trade', 0)}\n\n")
            
            f.write("TOP 10 CONFIGURATIONS:\n")
            f.write("-" * 25 + "\n")
            for i, config in enumerate(results.get('top_configurations', [])[:10]):
                f.write(f"{i+1}. Score: {config['score']:.4f} | ")
                f.write(f"Return: {config['total_return']*100:.1f}% | ")
                f.write(f"Win Rate: {config['win_rate']*100:.1f}% | ")
                f.write(f"Trades: {config['total_trades']}\n")
            
            f.write(f"\nSUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            stats = results.get('summary_stats', {})
            f.write(f"Average Score: {stats.get('avg_score', 0):.4f}\n")
            f.write(f"Score Std Dev: {stats.get('std_score', 0):.4f}\n")
            f.write(f"Average Return: {stats.get('avg_return', 0)*100:.1f}%\n")
            f.write(f"Average Win Rate: {stats.get('avg_win_rate', 0)*100:.1f}%\n")
            f.write(f"Average Max Drawdown: {stats.get('avg_drawdown', 0)*100:.1f}%\n")
        
        logger.info(f"Optimization report saved to {filename}")

def main():
    """Main optimization function"""
    print("🔧 Starting MRBEN LSTM Trading System Parameter Optimization...")
    
    # Check if data file exists
    data_file = "lstm_signals_pro.csv"
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file '{data_file}' not found!")
        return
    
    try:
        # Create optimizer
        optimizer = ParameterOptimizer()
        
        # Optimize thresholds
        print("\n📊 Optimizing signal generation thresholds...")
        threshold_results = optimizer.optimize_thresholds(data_file)
        
        # Optimize trading parameters
        print("\n📈 Optimizing trading parameters...")
        trading_results = optimizer.optimize_trading_parameters(data_file)
        
        # Combine results
        final_results = {
            'threshold_optimization': threshold_results,
            'trading_optimization': trading_results,
            'best_config': optimizer.best_config,
            'best_score': optimizer.best_score
        }
        
        # Save results
        optimizer.save_optimization_results(final_results, "outputs/optimization_results.json")
        optimizer.generate_optimization_report(final_results, "outputs/optimization_report.txt")
        
        # Print summary
        print("\n✅ Parameter Optimization Completed!")
        print(f"🎯 Best Score: {optimizer.best_score:.4f}")
        print(f"📁 Results saved to 'outputs/' directory")
        
        if optimizer.best_config:
            print("\n🏆 BEST CONFIGURATION:")
            print(f"   Buy Threshold: {optimizer.best_config.buy_threshold}")
            print(f"   Sell Threshold: {optimizer.best_config.sell_threshold}")
            print(f"   Hold Threshold: {optimizer.best_config.hold_threshold}")
            print(f"   Signal Amplification: {optimizer.best_config.signal_amplification}")
            print(f"   Stop Loss: {optimizer.best_config.stop_loss_pips} pips")
            print(f"   Take Profit: {optimizer.best_config.take_profit_pips} pips")
            print(f"   Risk per Trade: {optimizer.best_config.risk_per_trade*100}%")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 