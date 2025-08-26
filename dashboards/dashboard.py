# mrben_streamlit_dashboard.py - FINAL PROFESSIONAL VERSION

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="MR BEN Trading Dashboard")

st.title("📊 MR BEN Trading Dashboard (Live)")
st.markdown("---")

TRADES_FILE = "live_trades_log.csv"
SIGNALS_FILE = "live_signals_log.csv"

# --- Load trades log ---
try:
    trades = pd.read_csv(TRADES_FILE)
    st.success(f"تعداد کل معاملات ثبت شده: {len(trades)}")
except Exception as e:
    st.error(f"❌ خطا در خواندن فایل معاملات: {e}")
    st.stop()

# --- Summary stats ---
st.subheader("📃 خلاصه معاملات")
summary_cols = []
if "pnl" in trades.columns:
    total_pnl = trades["pnl"].sum()
    st.metric("جمع سود/زیان (pnl)", f"{total_pnl:,.2f}")
    summary_cols.append("pnl")
if "balance" in trades.columns:
    last_balance = trades["balance"].iloc[-1]
    st.metric("آخرین موجودی", f"{last_balance:,.2f}")
    summary_cols.append("balance")

# --- Open trades section ---
if "close_time" in trades.columns:
    open_trades = trades[trades["close_time"].isnull() | (trades["close_time"] == "")]
    st.subheader("📂 معاملات باز (Open Trades)")
    if len(open_trades) > 0:
        st.dataframe(open_trades, use_container_width=True)
    else:
        st.info("هیچ معامله بازی وجود ندارد.")
else:
    st.info("ستون close_time در فایل معاملات وجود ندارد.")

# --- Latest closed trades ---
st.subheader("🕒 آخرین معاملات بسته شده")
if len(trades) > 0:
    st.dataframe(trades.tail(10), use_container_width=True)
else:
    st.info("معامله‌ای ثبت نشده.")

# --- Live signals section ---
try:
    signals = pd.read_csv(SIGNALS_FILE)
    st.subheader("🟢 سیگنال‌های زنده")
    st.dataframe(signals.tail(10), use_container_width=True)
except Exception:
    st.info("فایل سیگنال زنده وجود ندارد یا سیگنالی ثبت نشده.")

# --- Equity curve chart ---
if "pnl" in trades.columns and "timestamp" in trades.columns:
    st.subheader("📈 رشد سرمایه (Equity Curve)")
    try:
        equity = 10000 + trades["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=trades["timestamp"], y=equity, mode='lines+markers', name='Equity')
        )
        fig.update_layout(
            title="Equity Curve", xaxis_title="زمان", yaxis_title="سرمایه", template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"نمودار رشد سرمایه قابل رسم نیست: {e}")
else:
    st.warning("برای نمایش نمودار، ستون‌های 'pnl' و 'timestamp' لازم است.")

# --- Buy/Sell count chart ---
if "action" in trades.columns:
    st.subheader("🔄 آمار BUY/SELL")
    try:
        action_counts = trades["action"].value_counts()
        fig2 = go.Figure(
            [go.Bar(x=action_counts.index, y=action_counts.values, marker_color=['green', 'red'])]
        )
        fig2.update_layout(
            title="Buy/Sell Count",
            xaxis_title="Action",
            yaxis_title="Count",
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"خطا در رسم آمار BUY/SELL: {e}")

st.markdown("---")
st.caption("MR BEN Automated Trading Dashboard | Powered by Streamlit & Plotly")
