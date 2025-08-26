import numpy as np
import pandas as pd


def backtest_with_params(df, stop_loss_pips, take_profit_pips, risk_per_trade, pip_value=1):
    balance = 10000
    position = None
    entry_price = 0
    sl = 0
    tp = 0
    lots = 0
    win_trades = 0
    loss_trades = 0
    equity_curve = []

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        signal = df['balanced_signal'].iloc[i - 1]

        if position is None:
            if signal == 1:  # long
                entry_price = df['close'].iloc[i - 1]
                sl = entry_price - stop_loss_pips * pip_value
                tp = entry_price + take_profit_pips * pip_value
                lots = (balance * risk_per_trade) / (stop_loss_pips * pip_value)
                position = 'long'
            elif signal == -1:  # short
                entry_price = df['close'].iloc[i - 1]
                sl = entry_price + stop_loss_pips * pip_value
                tp = entry_price - take_profit_pips * pip_value
                lots = (balance * risk_per_trade) / (stop_loss_pips * pip_value)
                position = 'short'
            equity_curve.append(balance)
            continue

        if position == 'long':
            if price <= sl:
                balance -= lots * stop_loss_pips * pip_value
                position = None
                loss_trades += 1
            elif price >= tp:
                balance += lots * take_profit_pips * pip_value
                position = None
                win_trades += 1
            equity_curve.append(balance)
        elif position == 'short':
            if price >= sl:
                balance -= lots * stop_loss_pips * pip_value
                position = None
                loss_trades += 1
            elif price <= tp:
                balance += lots * take_profit_pips * pip_value
                position = None
                win_trades += 1
            equity_curve.append(balance)
        else:
            equity_curve.append(balance)

    if len(equity_curve) < len(df):
        equity_curve += [balance] * (len(df) - len(equity_curve))

    return balance, win_trades, loss_trades, equity_curve


def optimize_parameters(df):
    best_result = None
    best_balance = -np.inf

    stop_loss_range = range(5, 51, 5)  # 5 to 50 pips step 5
    take_profit_range = range(10, 101, 10)  # 10 to 100 pips step 10
    risk_range = [0.005, 0.01, 0.02, 0.03]  # 0.5%, 1%, 2%, 3% risk per trade

    for sl in stop_loss_range:
        for tp in take_profit_range:
            if tp <= sl:
                continue  # TP should be greater than SL for good RR ratio
            for risk in risk_range:
                balance, wins, losses, _ = backtest_with_params(df, sl, tp, risk)
                if balance > best_balance:
                    best_balance = balance
                    best_result = {
                        'sl': sl,
                        'tp': tp,
                        'risk': risk,
                        'final_balance': balance,
                        'wins': wins,
                        'losses': losses,
                    }

    return best_result


def main():
    df = pd.read_csv('lstm_balanced_signals_final.csv')
    result = optimize_parameters(df)

    print("🏆 Best Parameters Found:")
    print(f"SL = {result['sl']} pips")
    print(f"TP = {result['tp']} pips")
    print(f"Risk per Trade = {result['risk']*100:.2f}%")
    print(f"Final Balance = {result['final_balance']:.2f}")
    print(f"Wins = {result['wins']} | Losses = {result['losses']}")


if __name__ == "__main__":
    main()
