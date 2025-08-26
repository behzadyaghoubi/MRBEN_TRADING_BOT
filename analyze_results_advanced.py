import matplotlib.pyplot as plt
import pandas as pd


def analyze_backtest_results(file_path="ai_trading_results_pro.csv"):
    df = pd.read_csv(file_path)
    if 'equity' not in df.columns:
        print("Error: 'equity' column not found in the CSV file!")
        return

    equity = df['equity']
    max_equity = equity.max()
    min_equity = equity.min()
    final_equity = equity.iloc[-1]

    # Calculate drawdown
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / running_max.max()) * 100

    # Calculate basic stats
    total_trades = df.shape[0] - 1  # assume one trade per row except header
    winning_trades = (df['equity'].diff() > 0).sum()
    losing_trades = (df['equity'].diff() < 0).sum()
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    avg_trade_return = df['equity'].diff().mean()

    # Print summary
    print("=== Backtest Performance Summary ===")
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average trade return: {avg_trade_return:.4f}")
    print(f"Max Equity: {max_equity:.2f}")
    print(f"Min Equity: {min_equity:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
    print(f"Final Equity: {final_equity:.2f}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid()
    plt.savefig("equity_curve.png", dpi=300)
    plt.show()

    # Plot drawdown curve
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown, color='red', label="Drawdown")
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid()
    plt.savefig("drawdown_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    analyze_backtest_results()
