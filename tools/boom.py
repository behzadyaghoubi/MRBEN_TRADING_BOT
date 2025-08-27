# tools/boom.py
import os
import sys
import time

print("Boom test starting...")
time.sleep(1)

# Feature flag: if PRODUCTION_TEST=1, exit cleanly instead of crashing
if os.getenv("PRODUCTION_TEST") == "1":
    print("PRODUCTION_TEST=1: Exiting cleanly (no crash)")
    sys.exit(0)
else:
    print("PRODUCTION_TEST not set: Intentional crash for testing")
    raise RuntimeError("Intentional test crash from boom.py")
