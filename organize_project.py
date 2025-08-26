import os
import shutil
import hashlib
from datetime import datetime

BASE_DIR = os.getcwd()
LOG_FILE = os.path.join(BASE_DIR, "organize_log.txt")

# دسته‌بندی پوشه‌ها و کلیدواژه‌ها
folder_map = {
    'src': ['main.py', 'main_runner.py', 'app.py', 'settings.json'],
    'data': ['.csv', '.npy', 'XAUUSD_PRO'],
    'models': ['.h5', '.save', '.joblib'],
    'strategies': ['strategy', 'book_', 'signal_generator', 'lstm_trading_model', 'ml_signal_filter'],
    'tools': ['download_', 'calculate_', 'fix_', 'export_', 'prepare_', 'feature_', 'plot_', 'check_', 'generate_'],
    'tests': ['test_', 'run_tests'],
    'dashboards': ['dashboard', 'mrben_dashboard', 'web'],
    'backtests': ['backtest', 'optimize_', 'sl_tp', 'train_'],
}

# مسیر ساخت پوشه بک‌آپ
BACKUP_DIR = os.path.join(BASE_DIR, "backup_duplicates")
os.makedirs(BACKUP_DIR, exist_ok=True)

# ساخت پوشه‌ها
for folder in folder_map:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# تابع هش فایل برای تشخیص تکراری بودن
def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# فایل‌های تکراری بر اساس هش
hashes_seen = {}

# باز کردن لاگ
with open(LOG_FILE, 'w', encoding='utf-8') as log:
    log.write(f"==== MR BEN PROJECT ORGANIZATION LOG ====\n{datetime.now()}\n\n")

    for file in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, file)
        if os.path.isfile(file_path) and file != os.path.basename(__file__) and file != os.path.basename(LOG_FILE):
            moved = False

            # تشخیص هش فایل
            file_hash = hash_file(file_path)
            if file_hash in hashes_seen:
                backup_name = f"{file}_DUPLICATE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = os.path.join(BACKUP_DIR, backup_name)
                shutil.move(file_path, backup_path)
                log.write(f"[DUPLICATE REMOVED] {file} ➜ backup_duplicates/\n")
                continue
            else:
                hashes_seen[file_hash] = file

            for folder, keywords in folder_map.items():
                if any(keyword in file for keyword in keywords):
                    dest_path = os.path.join(BASE_DIR, folder, file)

                    # اگر فایل قبلاً وجود داره، بک‌آپ بگیر
                    if os.path.exists(dest_path):
                        backup_file = f"{file}_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        backup_path = os.path.join(BACKUP_DIR, backup_file)
                        shutil.move(dest_path, backup_path)
                        log.write(f"[BACKUP] Existing {file} ➜ backup_duplicates/\n")

                    shutil.move(file_path, dest_path)
                    log.write(f"[MOVED] {file} ➜ {folder}/\n")
                    moved = True
                    break

            if not moved:
                log.write(f"[SKIPPED] {file} (No match found)\n")

    log.write("\n✅ Organization complete.\n")

print("🎯 Project has been organized successfully! Check 'organize_log.txt' for details.")