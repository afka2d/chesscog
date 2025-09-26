#!/usr/bin/env python3
"""
Train All Marshall Improved Models
Sequentially trains occupancy, color, and piece classification models
"""

import subprocess
import sys
import time
from pathlib import Path

def run_training_script(script_name, model_name):
    """Run a training script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {model_name} Training")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=3600)  # 1 hour timeout per model
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {model_name} training failed!")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running {model_name} training: {e}")
        return False
    
    return True

def main():
    """Main training function"""
    print("🎯 Marshall Model Training Pipeline")
    print("=" * 60)
    print("This will train 3 improved models using Marshall data:")
    print("1. Occupancy Detection")
    print("2. Color Classification") 
    print("3. Piece Classification")
    print("=" * 60)
    print("Starting training...")
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Training scripts and their names
    training_scripts = [
        ("train_occupancy_marshall.py", "Occupancy Detection"),
        ("train_color_marshall.py", "Color Classification"),
        ("train_piece_marshall.py", "Piece Classification")
    ]
    
    successful_models = []
    failed_models = []
    
    for script, model_name in training_scripts:
        if not Path(script).exists():
            print(f"❌ Training script {script} not found!")
            failed_models.append(model_name)
            continue
        
        success = run_training_script(script, model_name)
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Small delay between trainings
        time.sleep(5)
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"✅ Successful models: {len(successful_models)}")
    for model in successful_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\n📁 Models saved to: {output_dir}")
    print("🔗 You can now use the Marshall Improved API on port 8006")
    
    # Check what models were created
    model_files = list(output_dir.glob("*.pt"))
    if model_files:
        print(f"\n📋 Created model files:")
        for model_file in model_files:
            print(f"   - {model_file.name}")

if __name__ == "__main__":
    main()
