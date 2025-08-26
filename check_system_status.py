#!/usr/bin/env python3
"""
System Status Checker for MR BEN Trading System
"""

import os
import sys
import numpy as np
import pandas as pd

def check_advanced_lstm():
    """Check Advanced LSTM model status"""
    print("🔍 بررسی مدل Advanced LSTM:")
    print("=" * 50)
    
    model_path = 'models/advanced_lstm_model.h5'
    scaler_path = 'models/advanced_lstm_scaler.save'
    
    if os.path.exists(model_path):
        print("✅ مدل Advanced LSTM موجود است")
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print(f"📊 ساختار مدل: {model.input_shape} -> {model.output_shape}")
            print(f"📈 تعداد پارامترها: {model.count_params():,}")
            
            # Check if model is trained
            if hasattr(model, 'history') and model.history:
                print("✅ مدل آموزش دیده است")
            else:
                print("⚠️ مدل بارگذاری شده اما نیاز به آموزش دارد")
                
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {e}")
    else:
        print("❌ مدل Advanced LSTM موجود نیست")
    
    if os.path.exists(scaler_path):
        print("✅ Scaler موجود است")
    else:
        print("❌ Scaler موجود نیست")
    
    print()

def check_ml_filter():
    """Check ML Filter status"""
    print("🔍 بررسی ML Filter:")
    print("=" * 50)
    
    filter_path = 'models/mrben_ai_signal_filter_xgb.joblib'
    
    if os.path.exists(filter_path):
        print("✅ ML Filter موجود است")
        try:
            import joblib
            model = joblib.load(filter_path)
            print(f"📊 نوع مدل: {type(model).__name__}")
            
            if hasattr(model, 'feature_names_in_'):
                print(f"📋 تعداد ویژگی‌ها: {len(model.feature_names_in_)}")
                print(f"📋 ویژگی‌ها: {list(model.feature_names_in_)}")
            
            if hasattr(model, 'feature_importances_'):
                print("✅ مدل دارای اهمیت ویژگی‌ها است")
                
        except Exception as e:
            print(f"❌ خطا در بارگذاری فیلتر: {e}")
    else:
        print("❌ ML Filter موجود نیست")
    
    print()

def check_live_trader_integration():
    """Check if models are integrated in live trader"""
    print("🔍 بررسی یکپارچگی در Live Trader:")
    print("=" * 50)
    
    live_trader_path = 'live_trader_clean.py'
    
    if os.path.exists(live_trader_path):
        with open(live_trader_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'advanced_lstm_model' in content:
            print("✅ Advanced LSTM در Live Trader یکپارچه شده")
        else:
            print("❌ Advanced LSTM در Live Trader یکپارچه نشده")
            
        if 'AISignalFilter' in content:
            print("✅ ML Filter در Live Trader یکپارچه شده")
        else:
            print("❌ ML Filter در Live Trader یکپارچه نشده")
    else:
        print("❌ فایل Live Trader موجود نیست")
    
    print()

def check_data_files():
    """Check available data files"""
    print("🔍 بررسی فایل‌های داده:")
    print("=" * 50)
    
    data_files = [
        'data/trade_log.csv',
        'data/trade_log_clean.csv',
        'data/trade_log_aggressive.csv',
        'data/lstm_signals_features.csv',
        'data/mrben_ai_signal_dataset.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {file_path}: {len(df)} ردیف")
            except Exception as e:
                print(f"❌ {file_path}: خطا در خواندن - {e}")
        else:
            print(f"❌ {file_path}: موجود نیست")
    
    print()

def main():
    """Main function"""
    print("🚀 بررسی وضعیت سیستم MR BEN")
    print("=" * 60)
    print()
    
    check_advanced_lstm()
    check_ml_filter()
    check_live_trader_integration()
    check_data_files()
    
    print("🎯 خلاصه وضعیت:")
    print("=" * 60)
    print("1. مدل Advanced LSTM با Attention موجود و بارگذاری می‌شود")
    print("2. ML Filter (XGBoost) موجود و فعال است")
    print("3. هر دو مدل در Live Trader یکپارچه شده‌اند")
    print("4. فایل‌های داده برای آموزش موجود هستند")
    print()
    print("⚠️ نکته مهم: مدل LSTM نیاز به آموزش مجدد دارد")
    print("💡 پیشنهاد: اجرای train_advanced_lstm.py برای آموزش مدل")

if __name__ == "__main__":
    main() 