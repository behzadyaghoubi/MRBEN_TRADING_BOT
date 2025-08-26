import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import json

class AdvancedMonitoring:
    def __init__(self):
        self.signal_log = []
        self.trade_log = []
        self.performance_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'accuracy': 0.0
        }
        
    def log_signal(self, signal_type, confidence, timestamp=None):
        """Log a new signal with detailed information"""
        if timestamp is None:
            timestamp = datetime.now()
            
        signal_data = {
            'timestamp': timestamp,
            'signal_type': signal_type,
            'confidence': confidence,
            'signal_id': len(self.signal_log) + 1
        }
        
        self.signal_log.append(signal_data)
        self.performance_stats['total_signals'] += 1
        
        if signal_type == 'BUY':
            self.performance_stats['buy_signals'] += 1
        elif signal_type == 'SELL':
            self.performance_stats['sell_signals'] += 1
        else:
            self.performance_stats['hold_signals'] += 1
            
        # Real-time analysis
        self.analyze_distribution()
        
    def log_trade(self, trade_type, entry_price, exit_price, profit_loss, timestamp=None):
        """Log executed trades"""
        if timestamp is None:
            timestamp = datetime.now()
            
        trade_data = {
            'timestamp': timestamp,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'trade_id': len(self.trade_log) + 1
        }
        
        self.trade_log.append(trade_data)
        
    def analyze_distribution(self):
        """Analyze signal distribution in real-time"""
        total = self.performance_stats['total_signals']
        if total == 0:
            return
            
        buy_pct = (self.performance_stats['buy_signals'] / total) * 100
        sell_pct = (self.performance_stats['sell_signals'] / total) * 100
        hold_pct = (self.performance_stats['hold_signals'] / total) * 100
        
        print(f"\n📊 Real-time Signal Distribution:")
        print(f"BUY: {buy_pct:.1f}% ({self.performance_stats['buy_signals']})")
        print(f"SELL: {sell_pct:.1f}% ({self.performance_stats['sell_signals']})")
        print(f"HOLD: {hold_pct:.1f}% ({self.performance_stats['hold_signals']})")
        print(f"Total: {total} signals")
        
        # Check for bias
        if abs(buy_pct - sell_pct) > 20:
            print("⚠️ WARNING: Potential bias detected!")
        else:
            print("✅ Balanced distribution maintained")
            
    def generate_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.performance_stats,
            'recent_signals': self.signal_log[-10:] if self.signal_log else [],
            'recent_trades': self.trade_log[-10:] if self.trade_log else []
        }
        
        # Save report
        with open('monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def monitor_lstm_training(self, model, X_test, y_test):
        """Monitor LSTM training progress and predictions"""
        print("\n🔍 LSTM Model Analysis:")
        
        # Get predictions
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Analyze distribution
        unique, counts = np.unique(predicted_classes, return_counts=True)
        total = len(predicted_classes)
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        print("Predicted Signal Distribution:")
        for class_id, count in zip(unique, counts):
            percentage = (count / total) * 100
            signal_name = signal_map[class_id]
            print(f"{signal_name}: {percentage:.1f}% ({count})")
            
        # Check for bias
        if len(unique) >= 2:
            buy_count = counts[unique == 2][0] if 2 in unique else 0
            sell_count = counts[unique == 0][0] if 0 in unique else 0
            
            if buy_count > 0 and sell_count > 0:
                ratio = buy_count / sell_count
                print(f"BUY/SELL ratio: {ratio:.2f}")
                
                if 0.7 <= ratio <= 1.3:
                    print("✅ Balanced predictions")
                else:
                    print("⚠️ Potential bias in predictions")

# Global monitoring instance
monitor = AdvancedMonitoring()

def quick_analysis():
    """Quick analysis of current system status"""
    print("\n🚀 MR BEN System Status Check:")
    print("=" * 50)
    
    # Check if LSTM model exists
    if os.path.exists('models/lstm_balanced_model.h5'):
        print("✅ LSTM model found")
    else:
        print("❌ LSTM model not found")
        
    # Check if balanced dataset exists
    if os.path.exists('data/mrben_ai_signal_dataset_synthetic_balanced.csv'):
        print("✅ Balanced dataset found")
    else:
        print("❌ Balanced dataset not found")
        
    # Check if live trader exists
    if os.path.exists('live_trader_clean.py'):
        print("✅ Live trader ready")
    else:
        print("❌ Live trader not found")
        
    print("=" * 50)

if __name__ == "__main__":
    quick_analysis() 