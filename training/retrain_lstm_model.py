import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm_model import LSTMModel

# --- Load live trading log ---
df = pd.read_csv("logs/live_trades.csv")

# --- Required features ---
features = ['price', 'rsi', 'macd', 'atr', 'buy_proba', 'sell_proba']
df = df[features + ['signal']].dropna()

# --- Load existing scaler ---
scaler = joblib.load("models/lstm_scaler.pkl")
X = scaler.transform(df[features])
y = df['signal'].apply(lambda x: 1 if x == 1 else 0).values  # Buy = 1, else = 0

# --- Create sequences ---
seq_len = 10
X_seq, y_seq = [], []
for i in range(len(X) - seq_len):
    X_seq.append(X[i : i + seq_len])
    y_seq.append(y[i + seq_len])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

# --- Load existing LSTM model ---
checkpoint_path = "models/mrben_lstm_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
input_dim = X_tensor.shape[2]

model = LSTMModel(input_dim=input_dim)
model.load_state_dict(checkpoint)
model.train()

# --- Fine-tune the model ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

# --- Save updated model ---
torch.save(model.state_dict(), checkpoint_path)
print("✅ LSTM model fine-tuned and saved.")
