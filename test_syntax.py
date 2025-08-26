#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

def test_syntax():
    """Test if live_trader_clean.py has correct syntax"""
    try:
        # Try to compile the file
        with open('live_trader_clean.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Compile to check syntax
        compile(source, 'live_trader_clean.py', 'exec')
        print("✅ Syntax check passed - no syntax errors found")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error found: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_import():
    """Test if the file can be imported"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the module
        import live_trader_clean
        print("✅ Import test passed - file can be imported")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing live_trader_clean.py...")
    print("=" * 50)
    
    syntax_ok = test_syntax()
    if syntax_ok:
        import_ok = test_import()
        if import_ok:
            print("\n🎉 All tests passed! The file is ready to run.")
        else:
            print("\n⚠️  Import test failed, but syntax is correct.")
    else:
        print("\n❌ Syntax test failed. Please fix the errors above.")
