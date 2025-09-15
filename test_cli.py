#!/usr/bin/env python3
"""Test script for CLI functionality."""

import sys
import os
import subprocess
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cli_commands():
    """Test all CLI commands."""
    print("🧪 Testing CLI commands...")
    print()
    
    # Test scrape command
    print("1. Testing scrape command...")
    try:
        result = subprocess.run([
            sys.executable, "cli.py", "scrape", "iPhone 15", "--limit", "2"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Scrape command works")
        else:
            print(f"❌ Scrape command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Scrape command error: {e}")
        return False
    
    # Test analyze command
    print("2. Testing analyze command...")
    try:
        output_file = "test_cli_output.json"
        
        result = subprocess.run([
            sys.executable, "cli.py", "analyze", "iPhone 15", "--limit", "2", "--out", output_file
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Analyze command works")
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print("✅ Output file created successfully")
                os.unlink(output_file)  # Clean up
            else:
                print("❌ Output file not created or empty")
                return False
        else:
            print(f"❌ Analyze command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Analyze command error: {e}")
        return False
    
    # Test export command (create a test file first)
    print("3. Testing export command...")
    try:
        test_data = {
            "query": "test",
            "summary": {"total": 1, "positive": 1, "negative": 0, "neutral": 0, "average_stars": 4.0},
            "pros": ["Great product"],
            "cons": [],
            "reviews": [{"id": "test", "text": "Great!", "author": "test_user"}]
        }
        
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            test_file = f.name
        
        result = subprocess.run([
            sys.executable, "cli.py", "export", "--in", test_file, "--pretty"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Export command works")
            os.unlink(test_file)  # Clean up
        else:
            print(f"❌ Export command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Export command error: {e}")
        return False
    
    print()
    print("🎉 All CLI commands work correctly!")
    return True

if __name__ == "__main__":
    success = test_cli_commands()
    if not success:
        sys.exit(1)
