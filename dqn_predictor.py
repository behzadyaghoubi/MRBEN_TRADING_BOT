import numpy as np
import pandas as pd
import tensorflow as tf

def load_dqn_model(model_path='trained_dqn_model.h5'):
    return tf.keras.models.load_model(model_path)

def predict_action(model, df, window_size=10):
    # آماده‌سازی state
    state = df[['open', 'high', 'low', 'close']].values[-window_size:]
    state = np.expand_dims(state, axis=0)
    q_values = model.predict(state, verbose=0)
    action = np.argmax(q_values[0])  # 0: BUY, 1: SELL, 2: HOLD
    return action

if __name__ == "__main__":
    df = pd.read_csv('ohlc_data.csv')
    model = load_dqn_model()
    action = predict_action(model, df)
    action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    print(f"DQN Action: {action_map[action]}")