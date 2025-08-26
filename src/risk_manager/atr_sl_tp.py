from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SLTPResult:
    sl: float
    tp: float
    used_fallback: bool


Side = Literal["buy", "sell"]


def calc_sltp_from_atr(
    side: Side,
    entry_price: float,
    atr_value: float | None,
    rr: float,
    sl_k: float,
    tp_k: float,
    fallback_sl_pct: float,
    fallback_tp_pct: float,
) -> SLTPResult:
    """
    Calculate SL/TP from ATR. If atr_value is None or non-positive, use fallback fixed %.
    For BUY: SL = entry - sl_k*ATR, TP = entry + tp_k*ATR (or fallback %).
    For SELL: SL = entry + sl_k*ATR, TP = entry - tp_k*ATR (or fallback %).
    """
    used_fallback = False
    if atr_value is None or atr_value <= 0:
        used_fallback = True
        if side == "buy":
            sl = entry_price * (1.0 - fallback_sl_pct)
            tp = entry_price * (1.0 + fallback_tp_pct * rr / 1.5)  # scale RR roughly
        else:
            sl = entry_price * (1.0 + fallback_sl_pct)
            tp = entry_price * (1.0 - fallback_tp_pct * rr / 1.5)
        return SLTPResult(sl=round(sl, 5), tp=round(tp, 5), used_fallback=used_fallback)

    if side == "buy":
        sl = entry_price - sl_k * atr_value
        tp = entry_price + tp_k * atr_value * rr / 1.5  # allow RR scaling vs tp_k
    else:
        sl = entry_price + sl_k * atr_value
        tp = entry_price - tp_k * atr_value * rr / 1.5

    return SLTPResult(sl=round(sl, 5), tp=round(tp, 5), used_fallback=used_fallback)
