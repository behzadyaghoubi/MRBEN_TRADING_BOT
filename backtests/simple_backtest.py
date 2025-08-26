import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleBacktester")


class SimpleBacktester:
    def __init__(self, initial_capital=10000, position_size=0.1):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.trades = []
        self.equity_curve = []
        self.final_capital = initial_capital

    def backtest(self, df):
        df = df.copy()

        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        equity_curve = []

        for i, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']

            if position != 0:
                pnl = (
                    (current_price - entry_price) / entry_price
                    if position == 1
                    else (entry_price - current_price) / entry_price
                )
                current_equity = capital * (1 + pnl * self.position_size)
            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # Open new position
            if signal == 2 and position == 0:
                position = 1
                entry_price = current_price
                entry_time = row['datetime']
                logger.info(f"BUY @ {current_price:.2f} | {entry_time}")

            elif signal == 0 and position == 0:
                position = -1
                entry_price = current_price
                entry_time = row['datetime']
                logger.info(f"SELL @ {current_price:.2f} | {entry_time}")

            # Close position
            elif signal == 1 and position != 0:
                pnl = (
                    (current_price - entry_price) / entry_price
                    if position == 1
                    else (entry_price - current_price) / entry_price
                )
                capital_change = capital * pnl * self.position_size
                capital += capital_change

                self.trades.append(
                    {
                        'entry_time': entry_time,
                        'exit_time': row['datetime'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'pnl': pnl,
                        'capital_change': capital_change,
                    }
                )

                logger.info(f"CLOSE {'LONG' if position == 1 else 'SHORT'}: PnL={pnl*100:.2f}%")
                position = 0
                entry_price = 0
                entry_time = None

        self.equity_curve = equity_curve
        self.final_capital = capital
        return self.calculate_metrics()

    def calculate_metrics(self):
        if not self.trades:
            return {
                'final_capital': self.final_capital,
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
            }

        df = pd.DataFrame(self.trades)
        total_trades = len(df)
        winning = df[df.pnl > 0]
        losing = df[df.pnl < 0]

        win_rate = len(winning) / total_trades
        profit_factor = (
            winning.pnl.sum() / abs(losing.pnl.sum()) if len(losing) > 0 else float('inf')
        )
        equity_series = pd.Series(self.equity_curve)
        max_dd = ((equity_series / equity_series.cummax()) - 1).min()

        return {
            'final_capital': self.final_capital,
            'total_return': (self.final_capital - self.initial_capital) / self.initial_capital,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': winning.pnl.mean() if not winning.empty else 0,
            'avg_loss': losing.pnl.mean() if not losing.empty else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
        }

    def plot_results(self, df, report_path="backtester/reports/simple_backtest.png"):
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.plot(df['datetime'], df['close'], label='Price', alpha=0.6)
        plt.scatter(
            df[df['signal'] == 2]['datetime'],
            df[df['signal'] == 2]['close'],
            marker='^',
            color='green',
            label='BUY',
        )
        plt.scatter(
            df[df['signal'] == 0]['datetime'],
            df[df['signal'] == 0]['close'],
            marker='v',
            color='red',
            label='SELL',
        )
        plt.title("Price & Signals")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.equity_curve, label="Equity Curve", color='blue')
        plt.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        plt.title("Equity Over Time")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(report_path)
        plt.show()

    def save_trade_log(self, path="backtester/reports/trade_log.csv"):
        if self.trades:
            df = pd.DataFrame(self.trades)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"✅ Trade log saved to {path}")
        else:
            logger.warning("⚠️ No trades to save.")


def main():
    try:
        df = pd.read_csv("simple_price_action_signals.csv")
        logger.info(f"✅ Loaded {len(df)} rows from simple_price_action_signals.csv")
    except FileNotFoundError:
        logger.error("❌ File simple_price_action_signals.csv not found.")
        return

    if 'signal' not in df.columns or 'close' not in df.columns:
        logger.error("❌ Required columns (signal, close) missing.")
        return

    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)

    bt = SimpleBacktester(initial_capital=10000, position_size=0.1)
    metrics = bt.backtest(df)

    print("\n=== SIMPLE PRICE ACTION BACKTEST ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.2f}" if isinstance(v, float) else f"{k:>16}: {v}")

    bt.save_trade_log()  # ✅ ذخیره CSV معاملات
    bt.plot_results(df)  # 📈 رسم نمودار نتایج


if __name__ == "__main__":
    main()
