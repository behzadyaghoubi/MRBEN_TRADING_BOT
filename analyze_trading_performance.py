#!/usr/bin/env python3
"""
MR BEN Trading Performance Analyzer
Analyzes trade_log_gold.csv and events.jsonl for insights
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class TradingPerformanceAnalyzer:
    def __init__(
        self,
        trade_log_path: str = "data/trade_log_gold.csv",
        events_path: str = "data/events.jsonl",
    ):
        self.trade_log_path = trade_log_path
        self.events_path = events_path
        self.trades_df = None
        self.events_df = None

    def load_data(self) -> bool:
        """Load trade logs and events data."""
        try:
            # Load trade logs
            if os.path.exists(self.trade_log_path):
                self.trades_df = pd.read_csv(self.trade_log_path)
                print(f"✅ Loaded {len(self.trades_df)} trades from {self.trade_log_path}")
            else:
                print(f"⚠️ Trade log not found: {self.trade_log_path}")
                return False

            # Load events
            if os.path.exists(self.events_path):
                events = []
                with open(self.events_path, encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                events.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue

                if events:
                    self.events_df = pd.DataFrame(events)
                    self.events_df['ts'] = pd.to_datetime(self.events_df['ts'])
                    print(f"✅ Loaded {len(self.events_df)} events from {self.events_path}")
                else:
                    print(f"⚠️ No valid events found in {self.events_path}")
            else:
                print(f"⚠️ Events file not found: {self.events_path}")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def analyze_trades(self) -> dict:
        """Analyze trading performance metrics."""
        if self.trades_df is None or len(self.trades_df) == 0:
            return {}

        analysis = {}

        # Basic metrics
        analysis['total_trades'] = len(self.trades_df)
        analysis['winning_trades'] = len(self.trades_df[self.trades_df['pnl'] > 0])
        analysis['losing_trades'] = len(self.trades_df[self.trades_df['pnl'] < 0])
        analysis['win_rate'] = (
            analysis['winning_trades'] / analysis['total_trades']
            if analysis['total_trades'] > 0
            else 0
        )

        # PnL analysis
        analysis['total_pnl'] = self.trades_df['pnl'].sum()
        analysis['avg_win'] = (
            self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean()
            if analysis['winning_trades'] > 0
            else 0
        )
        analysis['avg_loss'] = (
            self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean()
            if analysis['losing_trades'] > 0
            else 0
        )
        analysis['profit_factor'] = (
            abs(analysis['avg_win'] / analysis['avg_loss']) if analysis['avg_loss'] != 0 else 0
        )

        # R-ratio analysis
        if 'R_post' in self.trades_df.columns:
            analysis['avg_R'] = self.trades_df['R_post'].mean()
            analysis['R_distribution'] = {
                'R<1.0': len(self.trades_df[self.trades_df['R_post'] < 1.0]),
                '1.0-1.2': len(
                    self.trades_df[
                        (self.trades_df['R_post'] >= 1.0) & (self.trades_df['R_post'] < 1.2)
                    ]
                ),
                '1.2-1.5': len(
                    self.trades_df[
                        (self.trades_df['R_post'] >= 1.2) & (self.trades_df['R_post'] < 1.5)
                    ]
                ),
                '1.5-2.0': len(
                    self.trades_df[
                        (self.trades_df['R_post'] >= 1.5) & (self.trades_df['R_post'] < 2.0)
                    ]
                ),
                'R>2.0': len(self.trades_df[self.trades_df['R_post'] >= 2.0]),
            }

        # MFE analysis
        if 'mfe' in self.trades_df.columns:
            analysis['avg_mfe'] = self.trades_df['mfe'].mean()
            analysis['mfe_distribution'] = {
                'MFE<0.5': len(self.trades_df[self.trades_df['mfe'] < 0.5]),
                '0.5-0.8': len(
                    self.trades_df[(self.trades_df['mfe'] >= 0.5) & (self.trades_df['mfe'] < 0.8)]
                ),
                '0.8-1.0': len(
                    self.trades_df[(self.trades_df['mfe'] >= 0.8) & (self.trades_df['mfe'] < 1.0)]
                ),
                'MFE>1.0': len(self.trades_df[self.trades_df['mfe'] >= 1.0]),
            }

        # Split TP analysis
        if 'use_split' in self.trades_df.columns:
            split_trades = self.trades_df[self.trades_df['use_split'] == True]
            single_trades = self.trades_df[self.trades_df['use_split'] == False]

            if len(split_trades) > 0:
                analysis['split_tp_performance'] = {
                    'count': len(split_trades),
                    'avg_pnl': split_trades['pnl'].mean(),
                    'win_rate': len(split_trades[split_trades['pnl'] > 0]) / len(split_trades),
                }

            if len(single_trades) > 0:
                analysis['single_tp_performance'] = {
                    'count': len(single_trades),
                    'avg_pnl': single_trades['pnl'].mean(),
                    'win_rate': len(single_trades[single_trades['pnl'] > 0]) / len(single_trades),
                }

        return analysis

    def analyze_events(self) -> dict:
        """Analyze system events and blocking reasons."""
        if self.events_df is None or len(self.events_df) == 0:
            return {}

        analysis = {}

        # Event counts by type
        event_counts = self.events_df['event'].value_counts()
        analysis['event_counts'] = event_counts.to_dict()

        # Blocking analysis
        if 'blocked_by_spread' in self.events_df['event'].values:
            spread_blocks = self.events_df[self.events_df['event'] == 'blocked_by_spread']
            analysis['spread_blocks'] = {
                'count': len(spread_blocks),
                'avg_spread': (
                    spread_blocks['spread'].mean() if 'spread' in spread_blocks.columns else 0
                ),
            }

        if 'order_attempt' in self.events_df['event'].values:
            order_attempts = self.events_df[self.events_df['event'] == 'order_attempt']
            analysis['order_attempts'] = {
                'count': len(order_attempts),
                'avg_confidence': (
                    order_attempts['confidence'].mean()
                    if 'confidence' in order_attempts.columns
                    else 0
                ),
            }

        # Conformal gate analysis
        if 'conformal_rejected' in self.events_df['event'].values:
            conformal_rejects = self.events_df[self.events_df['event'] == 'conformal_rejected']
            analysis['conformal_rejects'] = {
                'count': len(conformal_rejects),
                'avg_p_value': (
                    conformal_rejects['p_value'].mean()
                    if 'p_value' in conformal_rejects.columns
                    else 0
                ),
            }

        return analysis

    def plot_performance(self, save_path: str = "trading_analysis.png"):
        """Create performance visualization plots."""
        if self.trades_df is None or len(self.trades_df) == 0:
            print("❌ No trade data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MR BEN Trading Performance Analysis', fontsize=16)

        # 1. PnL Distribution
        if 'pnl' in self.trades_df.columns:
            axes[0, 0].hist(
                self.trades_df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black'
            )
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
            axes[0, 0].set_title('PnL Distribution')
            axes[0, 0].set_xlabel('PnL')
            axes[0, 0].set_ylabel('Frequency')

        # 2. R-ratio Distribution
        if 'R_post' in self.trades_df.columns:
            axes[0, 1].hist(
                self.trades_df['R_post'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black'
            )
            axes[0, 1].axvline(1.2, color='orange', linestyle='--', alpha=0.8, label='Target R=1.2')
            axes[0, 1].set_title('R-Ratio Distribution')
            axes[0, 1].set_xlabel('R-Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()

        # 3. MFE Distribution
        if 'mfe' in self.trades_df.columns:
            axes[1, 0].hist(
                self.trades_df['mfe'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black'
            )
            axes[1, 0].axvline(0.6, color='orange', linestyle='--', alpha=0.8, label='MFE 60%')
            axes[1, 0].set_title('MFE Distribution')
            axes[1, 0].set_xlabel('MFE')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()

        # 4. Win Rate by Month
        if 'entry_time' in self.trades_df.columns:
            try:
                self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
                self.trades_df['month'] = self.trades_df['entry_time'].dt.to_period('M')
                monthly_stats = (
                    self.trades_df.groupby('month')
                    .agg({'pnl': ['count', lambda x: (x > 0).sum() / len(x)]})
                    .round(3)
                )
                monthly_stats.columns = ['trades', 'win_rate']

                axes[1, 1].plot(
                    range(len(monthly_stats)),
                    monthly_stats['win_rate'],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                )
                axes[1, 1].set_title('Monthly Win Rate')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Win Rate')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].grid(True, alpha=0.3)

                # Set x-axis labels
                month_labels = [str(m) for m in monthly_stats.index]
                axes[1, 1].set_xticks(range(len(month_labels)))
                axes[1, 1].set_xticklabels(month_labels, rotation=45)

            except Exception as e:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    f'Monthly analysis failed:\n{e}',
                    ha='center',
                    va='center',
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title('Monthly Win Rate')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance plots saved to {save_path}")
        plt.show()

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        trades_analysis = self.analyze_trades()
        events_analysis = self.analyze_events()

        report = []
        report.append("=" * 60)
        report.append("MR BEN TRADING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Trading Performance
        if trades_analysis:
            report.append("📊 TRADING PERFORMANCE")
            report.append("-" * 30)
            report.append(f"Total Trades: {trades_analysis.get('total_trades', 0)}")
            report.append(f"Win Rate: {trades_analysis.get('win_rate', 0):.1%}")
            report.append(f"Total PnL: ${trades_analysis.get('total_pnl', 0):.2f}")
            report.append(f"Profit Factor: {trades_analysis.get('profit_factor', 0):.2f}")
            report.append(f"Average R: {trades_analysis.get('avg_R', 0):.2f}")
            report.append("")

            # R distribution
            if 'R_distribution' in trades_analysis:
                report.append("R-Ratio Distribution:")
                for r_range, count in trades_analysis['R_distribution'].items():
                    report.append(f"  {r_range}: {count} trades")
                report.append("")

            # Split TP analysis
            if 'split_tp_performance' in trades_analysis:
                split = trades_analysis['split_tp_performance']
                report.append("Split TP Performance:")
                report.append(f"  Count: {split['count']}, Win Rate: {split['win_rate']:.1%}")
                report.append(f"  Avg PnL: ${split['avg_pnl']:.2f}")
                report.append("")

        # System Events
        if events_analysis:
            report.append("🔍 SYSTEM EVENTS")
            report.append("-" * 30)

            if 'event_counts' in events_analysis:
                report.append("Event Counts:")
                for event, count in events_analysis['event_counts'].items():
                    report.append(f"  {event}: {count}")
                report.append("")

            if 'spread_blocks' in events_analysis:
                spread = events_analysis['spread_blocks']
                report.append(f"Spread Blocks: {spread['count']} (Avg: {spread['avg_spread']:.2f})")

            if 'conformal_rejects' in events_analysis:
                conformal = events_analysis['conformal_rejects']
                report.append(
                    f"Conformal Rejects: {conformal['count']} (Avg p-value: {conformal['avg_p_value']:.3f})"
                )

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Main analysis function."""
    print("🚀 MR BEN Trading Performance Analyzer")
    print("=" * 50)

    analyzer = TradingPerformanceAnalyzer()

    if not analyzer.load_data():
        print("❌ Failed to load data. Exiting.")
        return

    # Generate report
    report = analyzer.generate_report()
    print(report)

    # Save report to file
    with open("trading_performance_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ Report saved to trading_performance_report.txt")

    # Create plots
    try:
        analyzer.plot_performance()
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")


if __name__ == "__main__":
    main()
