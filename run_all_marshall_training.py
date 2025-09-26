#!/usr/bin/env python3
"""
Run all Marshall model training sequentially
This script will train occupancy, color, and piece models one after another
"""

import subprocess
import sys
import time
from pathlib import Path

def run_training_script(script_name, model_name):
    """Run a training script and log results"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {model_name} Training")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=3600)  # 1 hour timeout per model
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {model_name} training completed in {duration/60:.1f} minutes")
        
        if result.returncode == 0:
            print(f"✅ {model_name} training successful!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {model_name} training failed!")
            print("Error:", result.stderr[-500:])  # Last 500 chars
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running {model_name} training: {e}")
        return False

def main():
    """Run all training scripts in sequence"""
    print("🌙 Starting Overnight Marshall Model Training")
    print("This will train: Occupancy → Color → Piece Classification")
    print("Each model has a 1-hour timeout and early stopping")
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Training scripts in order
    training_scripts = [
        ("train_occupancy_marshall.py", "Occupancy Detection"),
        ("train_color_marshall.py", "Color Classification"), 
        ("train_piece_marshall.py", "Piece Classification")
    ]
    
    results = {}
    overall_start = time.time()
    
    for script, model_name in training_scripts:
        if not Path(script).exists():
            print(f"❌ Script {script} not found!")
            results[model_name] = False
            continue
            
        success = run_training_script(script, model_name)
        results[model_name] = success
        
        if not success:
            print(f"⚠️  {model_name} failed, but continuing with next model...")
        
        # Small delay between models
        time.sleep(5)
    
    # Final summary
    overall_end = time.time()
    total_duration = overall_end - overall_start
    
    print(f"\n{'='*60}")
    print("🏁 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print()
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    # Check if all models were saved
    print(f"\n📁 Checking saved models in {output_dir}:")
    for model_file in output_dir.glob("*.pt"):
        print(f"  ✅ {model_file.name}")
    
    print(f"\n🎉 Overnight training complete!")
    print("All improved models saved to models_marshall_improved/")

if __name__ == "__main__":
    main()
