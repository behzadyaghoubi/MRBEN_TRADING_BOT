# src/utils/conformal.py
# Simple stub module for conformal functionality


class ConformalGate:
    """Conformal prediction gate for risk management"""

    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.enabled = False  # Disabled for now

    def evaluate(self, features) -> bool:
        """Evaluate conformal prediction"""
        return True  # Placeholder implementation
