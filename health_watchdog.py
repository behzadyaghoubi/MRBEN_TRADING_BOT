#!/usr/bin/env python3
"""
MR BEN - Health Watchdog
Automatically creates halt.flag if dangerous conditions detected
"""

import os
import re
import time
import urllib.request
from datetime import datetime

# Configuration
URL = os.environ.get("MRBEN_METRICS", "http://127.0.0.1:8765/metrics")
DD_LIMIT = float(os.environ.get("MRBEN_DD_LIMIT", "1.5"))  # %
BLOCK_SPIKE = int(os.environ.get("MRBEN_BLOCK_SPIKE", "20"))  # in 10 minutes
CHECK_EVERY = 60  # seconds

# State tracking
last_blocks = None
last_check = None
halt_created = False


def fetch_metrics():
    """Fetch metrics from MRBEN"""
    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            return r.read().decode()
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Failed to fetch metrics: {e}")
        return None


def get_gauge(name, text):
    """Extract gauge metric value"""
    m = re.search(rf"^{name}\s+([0-9eE\.\+-]+)$", text, re.M)
    return float(m.group(1)) if m else None


def sum_counter(regex, text):
    """Sum counter metric values"""
    matches = re.finditer(regex, text, re.M)
    return sum(float(m.group(1)) for m in matches)


def ensure_halt():
    """Create halt.flag emergency stop"""
    global halt_created
    if not halt_created:
        try:
            with open("halt.flag", "w") as f:
                f.write(f"EMERGENCY STOP - Auto-created by watchdog at {datetime.now()}\n")
                f.write("Reason: Health threshold exceeded\n")
            halt_created = True
            print(f"[{datetime.now()}] 🚨 E-STOP: halt.flag created automatically!")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Failed to create halt.flag: {e}")


def check_health():
    """Check system health and trigger halt if needed"""
    global last_blocks, last_check

    try:
        text = fetch_metrics()
        if not text:
            return

        # Get key metrics
        dd = get_gauge("mrben_drawdown_pct", text) or 0.0
        blocks = sum_counter(r"^mrben_blocks_total\{.*\}\s+([0-9eE\.\+-]+)$", text)
        equity = get_gauge("mrben_equity", text) or 0.0
        exposure = get_gauge("mrben_exposure_positions", text) or 0

        now = time.time()
        spike = False

        # Check for block spike in last 10 minutes
        if last_blocks is not None and last_check is not None:
            if (now - last_check) <= 600:  # 10 min window
                if (blocks - last_blocks) >= BLOCK_SPIKE:
                    spike = True

        # Status display
        status = "✅"
        if dd > DD_LIMIT:
            status = "🚨"
        elif spike:
            status = "⚠️"

        print(
            f"[{datetime.now()}] {status} DD: {dd:.2f}% | Blocks: {blocks:.0f} | Equity: {equity:.2f} | Exposure: {exposure}"
        )

        # Check conditions for halt
        halt_reasons = []

        if dd > DD_LIMIT:
            halt_reasons.append(f"Drawdown {dd:.2f}% > {DD_LIMIT}%")

        if spike:
            halt_reasons.append(f"Block spike {(blocks - last_blocks):.0f} > {BLOCK_SPIKE}")

        if exposure > 2:
            halt_reasons.append(f"High exposure {exposure} positions")

        # Trigger halt if any condition met
        if halt_reasons:
            print(f"[{datetime.now()}] 🚨 Health thresholds exceeded:")
            for reason in halt_reasons:
                print(f"   - {reason}")
            ensure_halt()

        # Update state
        last_blocks, last_check = blocks, now

    except Exception as e:
        print(f"[{datetime.now()}] ❌ Watchdog error: {e}")


def main():
    """Main watchdog loop"""
    print("🚨 MR BEN Health Watchdog Starting...")
    print(f"Metrics URL: {URL}")
    print(f"Drawdown Limit: {DD_LIMIT}%")
    print(f"Block Spike Limit: {BLOCK_SPIKE} in 10min")
    print(f"Check Interval: {CHECK_EVERY}s")
    print("Press Ctrl+C to stop`n")

    try:
        while True:
            check_health()
            time.sleep(CHECK_EVERY)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 🛑 Watchdog stopped by user")
        if halt_created:
            print(f"[{datetime.now()}] ℹ️  halt.flag was created during this session")


if __name__ == "__main__":
    main()
