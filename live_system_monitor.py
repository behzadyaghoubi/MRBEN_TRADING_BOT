import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque

class LiveSystemMonitor:
    """Advanced monitoring system for MR BEN trading bot"""
    
    def __init__(self, log_file='logs/live_trades.csv', signal_log='logs/signals.csv'):
        self.log_file = log_file
        self.signal_log = signal_log
        self.trade_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=1000)
        self.monitoring_active = False
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize log files
        self._initialize_logs()
    
    def _initialize_logs(self):
        """Initialize log files with headers"""
        
        # Trade log
        if not os.path.exists(self.log_file):
            trade_headers = ['timestamp', 'signal', 'action', 'price', 'volume', 'profit_loss', 'balance']
            pd.DataFrame(columns=trade_headers).to_csv(self.log_file, index=False)
        
        # Signal log
        if not os.path.exists(self.signal_log):
            signal_headers = ['timestamp', 'lstm_signal', 'technical_signal', 'ml_filter_signal', 'final_signal', 'confidence']
            pd.DataFrame(columns=signal_headers).to_csv(self.signal_log, index=False)
    
    def start_monitoring(self):
        """Start the monitoring system"""
        print("🚀 Starting Live System Monitor...")
        print("=" * 50)
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                self._update_data()
                self._display_status()
                self._check_alerts()
                time.sleep(30)  # Update every 30 seconds
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user.")
                break
            except Exception as e:
                print(f"❌ Error in monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_data(self):
        """Update trade and signal data"""
        
        # Read trade log
        if os.path.exists(self.log_file):
            try:
                trades_df = pd.read_csv(self.log_file)
                if not trades_df.empty:
                    self.trade_history = deque(trades_df.to_dict('records'), maxlen=1000)
            except Exception as e:
                print(f"Warning: Could not read trade log: {e}")
        
        # Read signal log
        if os.path.exists(self.signal_log):
            try:
                signals_df = pd.read_csv(self.signal_log)
                if not signals_df.empty:
                    self.signal_history = deque(signals_df.to_dict('records'), maxlen=1000)
            except Exception as e:
                print(f"Warning: Could not read signal log: {e}")
    
    def _display_status(self):
        """Display current system status"""
        
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🎯 MR BEN Live Trading System Monitor")
        print("=" * 60)
        print(f"📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Trade statistics
        if self.trade_history:
            self._display_trade_stats()
        else:
            print("📊 No trades recorded yet.")
        
        print()
        
        # Signal statistics
        if self.signal_history:
            self._display_signal_stats()
        else:
            print("📈 No signals recorded yet.")
        
        print()
        
        # System health
        self._display_system_health()
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring")
    
    def _display_trade_stats(self):
        """Display trade statistics"""
        
        print("💰 Trade Statistics:")
        print("-" * 30)
        
        trades = list(self.trade_history)
        recent_trades = [t for t in trades if self._is_recent(t['timestamp'], hours=24)]
        
        # Overall stats
        total_trades = len(trades)
        recent_trades_count = len(recent_trades)
        
        print(f"Total Trades: {total_trades}")
        print(f"Recent Trades (24h): {recent_trades_count}")
        
        if trades:
            # Signal distribution
            signal_counts = defaultdict(int)
            for trade in trades:
                signal_counts[trade['signal']] += 1
            
            print("\nSignal Distribution:")
            for signal, count in signal_counts.items():
                percentage = (count / total_trades) * 100
                print(f"  {signal}: {count} ({percentage:.1f}%)")
            
            # Profit/Loss analysis
            if 'profit_loss' in trades[0]:
                profits = [float(t['profit_loss']) for t in trades if t['profit_loss'] != '']
                if profits:
                    total_pnl = sum(profits)
                    avg_pnl = np.mean(profits)
                    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                    
                    print(f"\nProfit/Loss Analysis:")
                    print(f"  Total P&L: ${total_pnl:.2f}")
                    print(f"  Average P&L: ${avg_pnl:.2f}")
                    print(f"  Win Rate: {win_rate:.1f}%")
    
    def _display_signal_stats(self):
        """Display signal statistics"""
        
        print("📈 Signal Statistics:")
        print("-" * 30)
        
        signals = list(self.signal_history)
        recent_signals = [s for s in signals if self._is_recent(s['timestamp'], hours=1)]
        
        total_signals = len(signals)
        recent_signals_count = len(recent_signals)
        
        print(f"Total Signals: {total_signals}")
        print(f"Recent Signals (1h): {recent_signals_count}")
        
        if signals:
            # Signal type distribution
            final_signal_counts = defaultdict(int)
            for signal in signals:
                final_signal_counts[signal['final_signal']] += 1
            
            print("\nFinal Signal Distribution:")
            for signal, count in final_signal_counts.items():
                percentage = (count / total_signals) * 100
                print(f"  {signal}: {count} ({percentage:.1f}%)")
            
            # Confidence analysis
            if 'confidence' in signals[0]:
                confidences = [float(s['confidence']) for s in signals if s['confidence'] != '']
                if confidences:
                    avg_confidence = np.mean(confidences)
                    high_conf_count = len([c for c in confidences if c >= 0.7])
                    high_conf_rate = high_conf_count / len(confidences) * 100
                    
                    print(f"\nConfidence Analysis:")
                    print(f"  Average Confidence: {avg_confidence:.3f}")
                    print(f"  High Confidence (≥0.7): {high_conf_count} ({high_conf_rate:.1f}%)")
    
    def _display_system_health(self):
        """Display system health indicators"""
        
        print("🏥 System Health:")
        print("-" * 30)
        
        # Check if system is active
        current_time = datetime.now()
        
        # Check recent activity
        recent_trades = [t for t in self.trade_history if self._is_recent(t['timestamp'], minutes=30)]
        recent_signals = [s for s in self.signal_history if self._is_recent(s['timestamp'], minutes=5)]
        
        if recent_trades:
            print("✅ Trading System: ACTIVE")
            last_trade_time = self._parse_timestamp(recent_trades[-1]['timestamp'])
            time_since_trade = current_time - last_trade_time
            print(f"   Last Trade: {time_since_trade.total_seconds()/60:.1f} minutes ago")
        else:
            print("⚠️ Trading System: NO RECENT ACTIVITY")
        
        if recent_signals:
            print("✅ Signal Generation: ACTIVE")
            last_signal_time = self._parse_timestamp(recent_signals[-1]['timestamp'])
            time_since_signal = current_time - last_signal_time
            print(f"   Last Signal: {time_since_signal.total_seconds()/60:.1f} minutes ago")
        else:
            print("⚠️ Signal Generation: NO RECENT ACTIVITY")
        
        # Overall health score
        health_score = 0
        if recent_trades:
            health_score += 50
        if recent_signals:
            health_score += 50
        
        if health_score == 100:
            print("🟢 Overall Health: EXCELLENT")
        elif health_score >= 50:
            print("🟡 Overall Health: GOOD")
        else:
            print("🔴 Overall Health: POOR")
    
    def _check_alerts(self):
        """Check for system alerts"""
        
        alerts = []
        
        # Check for no recent activity
        recent_trades = [t for t in self.trade_history if self._is_recent(t['timestamp'], minutes=30)]
        if not recent_trades:
            alerts.append("⚠️ No trades in the last 30 minutes")
        
        recent_signals = [s for s in self.signal_history if self._is_recent(s['timestamp'], minutes=5)]
        if not recent_signals:
            alerts.append("⚠️ No signals in the last 5 minutes")
        
        # Check for signal bias
        if self.signal_history:
            signals = list(self.signal_history)[-100:]  # Last 100 signals
            signal_counts = defaultdict(int)
            for signal in signals:
                signal_counts[signal['final_signal']] += 1
            
            total_signals = len(signals)
            for signal, count in signal_counts.items():
                percentage = (count / total_signals) * 100
                if percentage > 80:  # More than 80% of one signal type
                    alerts.append(f"⚠️ Signal bias detected: {signal} ({percentage:.1f}%)")
        
        # Display alerts
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")
    
    def _is_recent(self, timestamp_str, hours=1, minutes=0):
        """Check if timestamp is recent"""
        try:
            timestamp = self._parse_timestamp(timestamp_str)
            current_time = datetime.now()
            time_diff = current_time - timestamp
            return time_diff.total_seconds() <= (hours * 3600 + minutes * 60)
        except:
            return False
    
    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.now()
        except:
            return datetime.now()
    
    def generate_report(self, hours=24):
        """Generate a detailed report for the specified time period"""
        
        print(f"📊 Generating Report for Last {hours} Hours...")
        print("=" * 50)
        
        # Filter data for the time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_trades = [t for t in self.trade_history 
                        if self._parse_timestamp(t['timestamp']) >= cutoff_time]
        recent_signals = [s for s in self.signal_history 
                         if self._parse_timestamp(s['timestamp']) >= cutoff_time]
        
        # Generate report
        report = {
            'period': f"{hours} hours",
            'start_time': cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades': len(recent_trades),
            'total_signals': len(recent_signals),
            'trade_distribution': defaultdict(int),
            'signal_distribution': defaultdict(int),
            'profit_loss': 0,
            'win_rate': 0
        }
        
        # Analyze trades
        if recent_trades:
            for trade in recent_trades:
                report['trade_distribution'][trade['signal']] += 1
                if 'profit_loss' in trade and trade['profit_loss'] != '':
                    report['profit_loss'] += float(trade['profit_loss'])
            
            # Calculate win rate
            profits = [float(t['profit_loss']) for t in recent_trades 
                      if 'profit_loss' in t and t['profit_loss'] != '']
            if profits:
                wins = len([p for p in profits if p > 0])
                report['win_rate'] = (wins / len(profits)) * 100
        
        # Analyze signals
        if recent_signals:
            for signal in recent_signals:
                report['signal_distribution'][signal['final_signal']] += 1
        
        # Display report
        print(f"📈 Report Period: {report['start_time']} to {report['end_time']}")
        print(f"💰 Total Trades: {report['total_trades']}")
        print(f"📊 Total Signals: {report['total_signals']}")
        
        if report['total_trades'] > 0:
            print(f"\n💵 Profit/Loss: ${report['profit_loss']:.2f}")
            print(f"🎯 Win Rate: {report['win_rate']:.1f}%")
            
            print("\n📊 Trade Distribution:")
            for signal, count in report['trade_distribution'].items():
                percentage = (count / report['total_trades']) * 100
                print(f"  {signal}: {count} ({percentage:.1f}%)")
        
        if report['total_signals'] > 0:
            print("\n📈 Signal Distribution:")
            for signal, count in report['signal_distribution'].items():
                percentage = (count / report['total_signals']) * 100
                print(f"  {signal}: {count} ({percentage:.1f}%)")
        
        return report

def main():
    """Main function to run the monitor"""
    
    print("🎯 MR BEN Live System Monitor")
    print("=" * 50)
    
    monitor = LiveSystemMonitor()
    
    # Start monitoring
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped.")
    
    # Generate final report
    print("\n📊 Generating Final Report...")
    monitor.generate_report(hours=24)

if __name__ == "__main__":
    main() 