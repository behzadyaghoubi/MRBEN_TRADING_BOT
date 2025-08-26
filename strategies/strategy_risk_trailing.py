def calc_position_size(capital, risk_per_trade, atr, stop_multiplier=2):
    if atr == 0:
        return 0.01
    risk_amount = capital * risk_per_trade
    sl_pips = atr * stop_multiplier
    lot_size = risk_amount / sl_pips
    return max(round(lot_size, 2), 0.01)

def trailing_stop(entry_price, highest_price, atr, trail_multiplier=1.5, side='LONG'):
    """
    برای long: حد ضرر رو در فاصله trailing از سقف جدید قرار می‌دهد.
    برای short: حد ضرر رو در فاصله trailing از کف جدید قرار می‌دهد.
    """
    if side == 'LONG':
        return highest_price - trail_multiplier * atr
    else:
        return highest_price + trail_multiplier * atr