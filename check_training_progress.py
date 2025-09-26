#!/usr/bin/env python3
"""
Check Marshall Training Progress
Monitor the progress of the background training processes
"""

import time
import psutil
import subprocess
from pathlib import Path

def check_training_progress():
    """Check the progress of Marshall model training"""
    print("🔍 Checking Marshall Training Progress")
    print("=" * 50)
    
    # Check for running Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'marshall' in cmdline.lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        print(f"🔄 Found {len(python_processes)} Marshall training processes running:")
        for proc in python_processes:
            print(f"   PID {proc['pid']}: {proc['cmdline']}")
    else:
        print("❌ No Marshall training processes found running")
    
    # Check for created model files
    output_dir = Path("models_marshall_improved")
    if output_dir.exists():
        model_files = list(output_dir.glob("*.pt"))
        if model_files:
            print(f"\n📁 Created model files ({len(model_files)}):")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"   ✅ {model_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"\n📁 Output directory exists but no model files yet")
    else:
        print(f"\n📁 Output directory not created yet")
    
    # Check for log files
    log_files = list(Path(".").glob("*marshall*.log"))
    if log_files:
        print(f"\n📋 Log files found:")
        for log_file in log_files:
            print(f"   📄 {log_file.name}")
    
    print(f"\n⏰ Checked at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_training_progress()
