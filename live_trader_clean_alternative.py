    def _volume_for_trade(self, entry: float, sl: float) -> float:
        """Return dynamic volume calculation capped at 0.1 for risk control."""
        if not self.config.USE_RISK_BASED_VOLUME:
            return float(self.config.FIXED_VOLUME)

        # Calculate dynamic volume based on risk
        acc = self.trade_executor.get_account_info()
        bal = float(acc.get('balance', 10000.0))
        sl_dist = abs(entry - sl)
        dynamic_volume = self.risk_manager.calculate_lot_size(bal, self.config.BASE_RISK, sl_dist, self.config.SYMBOL)

        # Cap the volume at 0.1 for risk control
        return min(dynamic_volume, 0.1)
