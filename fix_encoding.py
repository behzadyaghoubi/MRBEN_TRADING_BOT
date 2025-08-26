#!/usr/bin/env python3


# Read the file with UTF-8-SIG to handle BOM
with open('live_trader_clean.py', encoding='utf-8-sig') as f:
    content = f.read()

# Write the file with clean UTF-8 encoding
with open('live_trader_clean_clean.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File encoding fixed successfully!")
