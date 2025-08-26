import pandas as pd
import numpy as np
import os

def calculate_technical_indicators(df):
    """Calculate technical indicators for enhanced features."""
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=14).mean()
    
    return rsi, macd, macd_signal, atr

def enhance_gold_data():
    """Enhance XAUUSD.PRO data with additional features."""
    
    print("🔧 Enhancing XAUUSD.PRO Data...")
    
    # Load original data
    input_file = 'data/XAUUSD_PRO_M5_data.csv'
    output_file = 'data/XAUUSD_PRO_M5_enhanced.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} rows from {input_file}")
    
    # Calculate technical indicators
    print("📊 Calculating technical indicators...")
    rsi, macd, macd_signal, atr = calculate_technical_indicators(df)
    
    # Add new features
    df['rsi'] = rsi
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['atr'] = atr
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Select the 8 features that match the LSTM model
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
    
    # Create enhanced dataset
    enhanced_df = df[['time'] + features].copy()
    
    # Save enhanced data
    enhanced_df.to_csv(output_file, index=False)
    print(f"✅ Enhanced data saved to {output_file}")
    print(f"📊 Features: {features}")
    print(f"📊 Shape: {enhanced_df.shape}")
    print(f"📊 Sample data:")
    print(enhanced_df.head(3))
    
    return enhanced_df

if __name__ == "__main__":
    enhance_gold_data() 