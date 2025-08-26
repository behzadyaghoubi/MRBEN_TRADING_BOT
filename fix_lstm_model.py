import numpy as np
from tensorflow.keras.models import load_model

# مسیر مدل
model = load_model('models/lstm_trading_model.h5')

# ورودی تست با شکل (1, 60, 23) - دقیقا ورودی مورد انتظار
dummy_input = np.random.rand(1, 60, 23)

# یک بار اجرا کنیم تا مدل output رو بسازه
_ = model.predict(dummy_input)

# دوباره ذخیره کنیم به همان مسیر
model.save('models/lstm_trading_model.h5')

print("✅ LSTM model structure fixed and re-saved.")
