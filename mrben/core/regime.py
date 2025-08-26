# Simplified regime detector for Phase 1 testing (no numpy dependency)


class RegimeDetector:
    """
    Market regime detector based on ATR volatility.

    Detects LOW, NORMAL, HIGH volatility regimes using percentile analysis.
    Simplified version for Phase 1 testing.
    """

    def __init__(self, win: int = 200):
        """
        Initialize regime detector.

        Args:
            win: Window size for regime calculation (default: 200)
        """
        self.win = win
        self.buf = []
        self.last_regime = "UNKNOWN"
        self.regime_changes = 0

    def _percentile(self, data: list, p: float) -> float:
        """Calculate percentile without numpy."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        return sorted_data[f]

    def _mean(self, data: list) -> float:
        """Calculate mean without numpy."""
        return sum(data) / len(data) if data else 0.0

    def _std(self, data: list) -> float:
        """Calculate standard deviation without numpy."""
        if len(data) < 2:
            return 0.0
        mean_val = self._mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
        return variance**0.5

    def update(self, atr_val: float) -> str:
        """
        Update regime detection with new ATR value.

        Args:
            atr_val: Current ATR value

        Returns:
            Regime identifier: "LOW", "NORMAL", "HIGH", or "UNKNOWN"
        """
        self.buf.append(float(atr_val))

        # Maintain window size
        if len(self.buf) > self.win:
            self.buf.pop(0)

        # Need minimum data for regime detection
        if len(self.buf) < 50:
            return "UNKNOWN"

        # Calculate percentiles
        ql = self._percentile(self.buf, 33)
        qh = self._percentile(self.buf, 90)
        cur = self.buf[-1]

        # Determine regime
        if cur < ql:
            regime = "LOW"
        elif cur > qh:
            regime = "HIGH"
        else:
            regime = "NORMAL"

        # Track regime changes
        if regime != self.last_regime:
            self.regime_changes += 1
            self.last_regime = regime

        return regime

    def get_regime_info(self, atr_val: float) -> dict:
        """
        Get detailed regime information including statistics.

        Args:
            atr_val: Current ATR value

        Returns:
            Dictionary with regime details
        """
        regime = self.update(atr_val)

        if len(self.buf) < 10:
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "atr_current": atr_val,
                "atr_avg": None,
                "regime_changes": self.regime_changes,
                "data_points": len(self.buf),
            }

        # Calculate confidence based on data stability
        atr_avg = self._mean(self.buf)
        atr_std = self._std(self.buf)
        confidence = max(0.0, min(1.0, 1.0 - (atr_std / max(1e-9, atr_avg))))

        return {
            "regime": regime,
            "confidence": confidence,
            "atr_current": atr_val,
            "atr_avg": atr_avg,
            "atr_std": atr_std,
            "regime_changes": self.regime_changes,
            "data_points": len(self.buf),
            "percentiles": {
                "low": float(self._percentile(self.buf, 33)),
                "mid": float(self._percentile(self.buf, 66)),
                "high": float(self._percentile(self.buf, 90)),
            },
        }

    def reset(self):
        """Reset regime detector state."""
        self.buf.clear()
        self.last_regime = "UNKNOWN"
        self.regime_changes = 0
