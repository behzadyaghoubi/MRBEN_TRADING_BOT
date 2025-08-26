#!/usr/bin/env python3
"""
Simple startup script for MR BEN Trading Bot.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from main import main
    main() 