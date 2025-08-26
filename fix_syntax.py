#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read the file
with open('live_trader_clean_final.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the syntax error in the _normalize_volume method
for i, line in enumerate(lines):
    if 'v = max(vmin, min(vmax, math.floor(vol/step)*step))' in line:
        # Remove the extra closing parenthesis
        lines[i] = line.replace('))', ')')
        print(f"Fixed line {i+1}: {line.strip()} -> {lines[i].strip()}")

# Write the fixed file
with open('live_trader_clean_fixed_syntax.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Syntax error fixed!")
