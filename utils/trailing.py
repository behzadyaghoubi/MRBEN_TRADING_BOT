# utils/trailing.py
from dataclasses import dataclass


@dataclass
class TrailParams:
    k_atr: float = 1.5
    min_step: float = 0.2  # حداقل حرکت برای به‌روزرسانی


class ChandelierTrailing:
    def __init__(self, params: TrailParams):
        self.params = params

    def calc_new_sl(
        self, is_buy: bool, entry: float, highest: float, lowest: float, atr: float
    ) -> float | None:
        if atr is None or atr <= 0:
            return None
        k = self.params.k_atr
        if is_buy:
            candidate = highest - k * atr
            return candidate
        else:
            candidate = lowest + k * atr
            return candidate
