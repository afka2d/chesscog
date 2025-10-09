#!/usr/bin/env python3
"""
Simple script to start the YOLO upgrade test API
"""

import subprocess
import sys
import os

def start_test_api():
    """Start the test API on port 8012"""
    print("🚀 Starting YOLO Upgrade Test API...")
    print("📍 This will run on port 8012 - completely separate from production!")
    print("")
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    # Start the API
    os.system("python3 test_yolo_upgrade_api.py")

if __name__ == "__main__":
    start_test_api()

