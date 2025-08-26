import subprocess

# اجرای داشبورد
dash_proc = subprocess.Popen(["python", "app.py"])
# اجرای ترید خودکار
trade_proc = subprocess.Popen(["python", "live_loop.py"])

try:
    dash_proc.wait()
    trade_proc.wait()
except KeyboardInterrupt:
    print("Shutting down both processes ...")
    dash_proc.terminate()
    trade_proc.terminate()