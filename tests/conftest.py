# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ریشه پروژه
SRC = ROOT / "src"
p = str(SRC)
if p not in sys.path:
    sys.path.insert(0, p)
