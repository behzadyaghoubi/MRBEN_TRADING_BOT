import types
from src.risk_manager.atr_sl_tp import SLTPResult, calc_sltp_from_atr
# Import a function/class from your live/sim execution layer that prepares order params:
# Example (adjust import to your project):
# from core.trader import prepare_order_params


def test_smoke_prepare_order_params_has_sltp(monkeypatch):
    # If real prepare function requires heavy deps, monkeypatch data/ATR.
    # This is a placeholder; adapt to your actual API.
    assert True  # Replace with your real integration once available
