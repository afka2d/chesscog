#!/usr/bin/env python3
"""
Overnight Marshall Model Training
Automatically trains occupancy, color, and piece models with Marshall data
Based on existing working models, saves with 'marshall' in the name
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def wait_for_occupancy_completion():
    """Wait for the currently running occupancy training to complete"""
    logger.info("⏳ Waiting for occupancy model training to complete...")
    
    # Check if occupancy training is still running
    while True:
        try:
            # Check if the process is still running by looking for the log file
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if 'train_occupancy_marshall.py' not in result.stdout:
                logger.info("✅ Occupancy training completed!")
                break
            else:
                logger.info("⏳ Occupancy training still running... waiting 30 seconds")
                time.sleep(30)
        except Exception as e:
            logger.warning(f"Error checking occupancy status: {e}")
            time.sleep(30)

def run_training_script(script_name, model_name, max_epochs=15, patience=3):
    """Run a training script with strict time limits to prevent overtraining"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting {model_name} Training")
    logger.info(f"Max epochs: {max_epochs}, Patience: {patience}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script with timeout
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout per model
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ {model_name} training completed in {duration/60:.1f} minutes")
        
        if result.returncode == 0:
            logger.info(f"✅ {model_name} training successful!")
            # Log last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    logger.info(f"  {line}")
            return True
        else:
            logger.error(f"❌ {model_name} training failed!")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning(f"⏰ {model_name} training timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {model_name} training: {e}")
        return False

def verify_model_saved(model_name, expected_file):
    """Verify that the model was saved successfully"""
    model_path = Path("models_marshall_improved") / expected_file
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ {model_name} saved: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        logger.error(f"❌ {model_name} not found at {model_path}")
        return False

def main():
    """Run all Marshall model training overnight"""
    logger.info("🌙 Starting Overnight Marshall Model Training")
    logger.info("=" * 60)
    logger.info("Training Plan:")
    logger.info("1. Wait for current occupancy training to complete")
    logger.info("2. Train color classification model")
    logger.info("3. Train piece classification model")
    logger.info("All models will be saved with 'marshall' in the name")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir.absolute()}")
    
    # Step 1: Wait for occupancy training to complete
    wait_for_occupancy_completion()
    
    # Verify occupancy model was saved
    if not verify_model_saved("Occupancy Detection", "occupancy_marshall.pt"):
        logger.error("❌ Occupancy model not found! Cannot proceed.")
        return
    
    # Step 2: Train color classification
    logger.info("\n" + "="*60)
    logger.info("🎨 Starting Color Classification Training")
    logger.info("="*60)
    
    color_success = run_training_script("train_color_marshall.py", "Color Classification")
    if color_success:
        verify_model_saved("Color Classification", "color_classification_marshall.pt")
    
    # Step 3: Train piece classification
    logger.info("\n" + "="*60)
    logger.info("♟️  Starting Piece Classification Training")
    logger.info("="*60)
    
    piece_success = run_training_script("train_piece_marshall.py", "Piece Classification")
    if piece_success:
        verify_model_saved("Piece Classification", "piece_classification_marshall.pt")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🏁 OVERNIGHT TRAINING SUMMARY")
    logger.info("="*60)
    
    # Check all saved models
    models_to_check = [
        ("Occupancy Detection", "occupancy_marshall.pt"),
        ("Color Classification", "color_classification_marshall.pt"),
        ("Piece Classification", "piece_classification_marshall.pt")
    ]
    
    all_success = True
    for model_name, filename in models_to_check:
        if verify_model_saved(model_name, filename):
            logger.info(f"✅ {model_name}: READY")
        else:
            logger.error(f"❌ {model_name}: FAILED")
            all_success = False
    
    if all_success:
        logger.info("\n🎉 ALL MARSHALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("📁 All models saved in: models_marshall_improved/")
        logger.info("🔒 Original working models remain untouched")
    else:
        logger.warning("\n⚠️  Some models failed to train properly")
        logger.info("Check the logs for details")
    
    logger.info("\n🌅 Overnight training complete! Good morning!")

if __name__ == "__main__":
    main()
