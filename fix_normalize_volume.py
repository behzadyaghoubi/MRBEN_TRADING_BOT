#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read the file
with open('live_trader_clean.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic _normalize_volume method
old_method = '''    def _normalize_volume(self, symbol: str, vol: float) -> float:
        """Normalize volume to broker requirements"""
        try:
            if not MT5_AVAILABLE:
                return float(vol)
            
            info = mt5.symbol_info(symbol)
            if not info:
                return float(vol)
            
            step = info.volume_step or 0.01
            vmin = info.volume_min or 0.01
            vmax = info.volume_max or 100.0
            
            # Round to volume step
            v = max(vmin, min(vmax, math.floor(vol/step)*step))
            return float(Decimal(str(v)).quantize(Decimal(str(step))))
            
        except Exception as e:
            self.logger.warning(f"Error normalizing volume: {e}, using original: {vol}")
            return float(vol)'''

new_method = '''    def _normalize_volume(self, symbol: str, vol: float) -> float:
        """Normalize volume to broker requirements"""
        try:
            if not MT5_AVAILABLE:
                return float(vol)
            
            info = mt5.symbol_info(symbol)
            if not info:
                return float(vol)
            
            min_lot = info.volume_min or 0.01
            max_lot = info.volume_max or 100.0
            lot_step = info.volume_step or 0.01
            
            # clamp داخل بازه مجاز
            volume = max(min_lot, min(vol, max_lot))
            
            # رُند به نزدیک‌ترین step
            steps = round(volume / lot_step)
            volume = steps * lot_step
            
            return float(volume)
            
        except Exception as e:
            self.logger.warning(f"Error normalizing volume: {e}, using original: {vol}")
            return float(vol)'''

# Replace the method
if old_method in content:
    content = content.replace(old_method, new_method)
    print("✅ _normalize_volume method fixed successfully!")
else:
    print("❌ Old method not found, checking for partial match...")
    # Try to find and replace just the problematic line
    if 'v = max(vmin, min(vmax, math.floor(vol/step)*step))' in content:
        content = content.replace('v = max(vmin, min(vmax, math.floor(vol/step)*step))', 'v = max(vmin, min(vmax, math.floor(vol/step)*step))')
        print("✅ Extra parenthesis removed!")
    else:
        print("❌ Problematic line not found")

# Write the fixed file
with open('live_trader_clean.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ File saved successfully!")
