#!/usr/bin/env python3
import os
import sys

print("Simple test without keyboard input")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Test file access
files = ["src/main_runner.py", "config/settings.json"]
for file in files:
    if os.path.exists(file):
        print(f"✅ {file}: Found")
    else:
        print(f"❌ {file}: Not found")

print("Test completed!")
