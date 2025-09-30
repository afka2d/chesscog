#!/usr/bin/env python3
"""
Launcher script for the Marshall Improved API.
This starts the API on port 8003 with improved Marshall models.
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("🚀 Starting Marshall Improved API...")
    print("📍 Port: 8003")
    print("🎯 Models: Marshall occupancy + Original color + Combined piece classification")
    print("=" * 60)
    
    # Check if the API file exists
    api_file = Path("marshall_improved_api.py")
    if not api_file.exists():
        print("❌ Error: marshall_improved_api.py not found")
        return 1
    
    # Check if required models exist
    required_models = [
        "models_marshall_improved/occupancy_marshall.pt",
        "models/color_classifier_simple.pt", 
        "models_marshall_improved/combined_piece_classifier.pt"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        print("❌ Error: Missing required models:")
        for model in missing_models:
            print(f"   - {model}")
        return 1
    
    print("✅ All required models found")
    print("🚀 Starting API server...")
    print("=" * 60)
    
    try:
        # Start the API
        subprocess.run([sys.executable, "marshall_improved_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
