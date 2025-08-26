from src.risk_manager.atr_sl_tp import SLTPResult, calc_sltp_from_atr


def test_buy_with_atr():
    res = calc_sltp_from_atr(
        side="buy",
        entry_price=2000.0,
        atr_value=10.0,
        rr=1.5,
        sl_k=1.0,
        tp_k=1.5,
        fallback_sl_pct=0.005,
        fallback_tp_pct=0.0075,
    )
    assert isinstance(res, SLTPResult)
    assert res.sl < 2000.0 and res.tp > 2000.0
    assert res.used_fallback is False


def test_sell_with_atr():
    res = calc_sltp_from_atr(
        side="sell",
        entry_price=2000.0,
        atr_value=8.0,
        rr=2.0,
        sl_k=1.2,
        tp_k=1.5,
        fallback_sl_pct=0.005,
        fallback_tp_pct=0.0075,
    )
    assert res.sl > 2000.0 and res.tp < 2000.0
    assert res.used_fallback is False


def test_buy_fallback_when_atr_missing():
    res = calc_sltp_from_atr(
        side="buy",
        entry_price=1000.0,
        atr_value=None,
        rr=1.5,
        sl_k=1.0,
        tp_k=1.5,
        fallback_sl_pct=0.01,
        fallback_tp_pct=0.02,
    )
    assert res.used_fallback is True
    assert res.sl < 1000.0 and res.tp > 1000.0


def test_sell_fallback_when_atr_nonpositive():
    res = calc_sltp_from_atr(
        side="sell",
        entry_price=1000.0,
        atr_value=0.0,
        rr=1.5,
        sl_k=1.0,
        tp_k=1.5,
        fallback_sl_pct=0.01,
        fallback_tp_pct=0.02,
    )
    assert res.used_fallback is True
    assert res.sl > 1000.0 and res.tp < 1000.0
