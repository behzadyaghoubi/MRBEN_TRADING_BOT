import streamlit as st
import pandas as pd

st.set_page_config(page_title="MR BEN Dashboard", layout="wide")
st.title("📊 داشبورد حرفه‌ای ربات معاملاتی MR BEN")

csv_file = st.text_input("نام فایل گزارش معاملات:", "mrben_multi_strategy_module_trades_EURUSD.csv")
df = None
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"خطا در خواندن فایل: {e}")

if df is not None and not df.empty:
    start_capital = df['capital'].iloc[0]
    end_capital = df['capital'].iloc[-1]
    profit = end_capital - start_capital
    num_trades = df[df['action'].str.contains('CLOSE')].shape[0]
    win_trades = df[(df['action'].str.contains('CLOSE')) & (df['capital'].diff() > 0)].shape[0]
    loss_trades = num_trades - win_trades
    win_rate = win_trades / num_trades * 100 if num_trades > 0 else 0
    max_drawdown = (df['capital'].cummax() - df['capital']).max()
    max_drawdown_pct = (max_drawdown / df['capital'].cummax().max()) * 100 if df['capital'].cummax().max() > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("سرمایه اولیه", f"${start_capital:,.2f}")
    col2.metric("سرمایه نهایی", f"${end_capital:,.2f}", delta=f"${profit:,.2f}")
    col3.metric("سود خالص", f"${profit:,.2f}")

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("تعداد معاملات", f"{num_trades}")
    col5.metric("نرخ برد", f"{win_rate:.1f}%")
    col6.metric("بیشترین افت سرمایه (Drawdown)", f"${max_drawdown:,.2f} ({max_drawdown_pct:.1f}%)")

    st.markdown("### 📈 نمودار سرمایه (Equity Curve)")
    st.line_chart(df['capital'])

    st.markdown("### 📋 آخرین 10 معامله")
    st.dataframe(df[['date', 'action', 'price', 'capital', 'lot', 'trailing_stop']].tail(10), use_container_width=True)

    with st.expander("مشاهده جدول کامل معاملات"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("فایل معاملات معتبر انتخاب نشده یا فایل خالی است.")