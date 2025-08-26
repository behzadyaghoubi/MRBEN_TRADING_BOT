# main_runner_loop.py - FINAL PROFESSIONAL VERSION
import time
import subprocess
import signal
import sys

LOOP_INTERVAL = 60  # Interval in seconds (e.g., 60 = every minute)

def signal_handler(sig, frame):
    print("\n[!] Stopping main_runner_loop gracefully ...")
    sys.exit(0)

# Handle Ctrl+C for clean exit
signal.signal(signal.SIGINT, signal_handler)

while True:
    print("=== [MR BEN AUTO RUNNER] Starting main_runner.py ... ===")
    try:
        result = subprocess.run(
            ["python", "main_runner.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        print(result.stdout)
        if result.stderr:
            print("!! ERROR:", result.stderr)
    except Exception as e:
        print(f"!! EXCEPTION: {e}")

    print(f"[MR BEN] Sleeping for {LOOP_INTERVAL} seconds ...\n")
    time.sleep(LOOP_INTERVAL)