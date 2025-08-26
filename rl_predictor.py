import numpy as np

def load_trained_agent(agent_path, epsilon=0.01):
    q_table = np.load(agent_path, allow_pickle=True).item()
    print("✅ Q-Table بارگذاری شد.")
    return {'q_table': q_table, 'epsilon': epsilon}

def predict_next_signal(agent, df):
    # فرض: فقط از قیمت بسته شدن برای state استفاده می‌شود
    # مطمئن شو که df ستون 'close' را دارد نه 'price'
    if 'close' not in df.columns:
        raise Exception("ستون 'close' در داده یافت نشد!")

    state = tuple(df['close'].tail(10).round(2))  # یا هر نوع encoding مناسب
    q_table = agent['q_table']
    epsilon = agent['epsilon']

    # جستجو در Q-table (اگر state موجود بود)
    if state in q_table:
        q_values = q_table[state]
        action = int(np.argmax(q_values))
    else:
        action = np.random.choice([0, 1, 2])  # [BUY, SELL, HOLD]

    # نگاشت اکشن به سیگنال
    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
    return action_map[action]