#!/usr/bin/env python3
"""
Script to help switch between local development and production APIs for testing.
"""

import json
import os
from pathlib import Path

def switch_to_local():
    """Switch to local development API"""
    config = {
        "api_url": "http://localhost:8001/recognize_chess_position_with_corners",
        "health_url": "http://localhost:8001/health",
        "environment": "local_development",
        "port": 8001
    }
    
    with open("api_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("🔄 Switched to LOCAL DEVELOPMENT API")
    print("📍 URL: http://localhost:8001")
    print("🔧 Includes debug features")
    print("⚠️  Make sure local API is running: ./start_local_dev.sh")

def switch_to_production():
    """Switch to production API"""
    config = {
        "api_url": "https://api.chesspositionscanner.store/recognize_chess_position_with_corners",
        "health_url": "https://api.chesspositionscanner.store/health",
        "environment": "production",
        "port": 443
    }
    
    with open("api_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("🔄 Switched to PRODUCTION API")
    print("📍 URL: https://api.chesspositionscanner.store")
    print("✅ Stable for App Store")

def show_current():
    """Show current API configuration"""
    if Path("api_config.json").exists():
        with open("api_config.json", "r") as f:
            config = json.load(f)
        
        print("📋 Current API Configuration:")
        print(f"   Environment: {config.get('environment', 'unknown')}")
        print(f"   API URL: {config.get('api_url', 'unknown')}")
        print(f"   Health URL: {config.get('health_url', 'unknown')}")
        print(f"   Port: {config.get('port', 'unknown')}")
    else:
        print("❌ No API configuration found")
        print("💡 Run this script with 'local' or 'production' to set up")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python switch_api.py [local|production|status]")
        print("")
        print("Commands:")
        print("  local      - Switch to local development API (port 8001)")
        print("  production - Switch to production API")
        print("  status     - Show current configuration")
        return
    
    command = sys.argv[1].lower()
    
    if command == "local":
        switch_to_local()
    elif command == "production":
        switch_to_production()
    elif command == "status":
        show_current()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'local', 'production', or 'status'")

if __name__ == "__main__":
    main()
