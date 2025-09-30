#!/usr/bin/env python3
"""
Run all Marshall model training with correct architectures.
This ensures compatibility with the existing API.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_training_script(script_name, timeout_minutes=60):
    """Run a training script with timeout"""
    logger.info(f"🚀 Starting {script_name}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            timeout=timeout_minutes * 60,
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  {script_name} completed in {elapsed_time:.1f} seconds")
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} succeeded")
            return True
        else:
            logger.error(f"❌ {script_name} failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {script_name} timed out after {timeout_minutes} minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run all Marshall model training scripts"""
    logger.info("🎯 Starting Marshall model training with correct architectures")
    logger.info("=" * 80)
    
    # Training scripts in order
    training_scripts = [
        ("train_occupancy_marshall_correct_architecture.py", 60),
        ("train_color_marshall_correct_architecture.py", 60),
        ("train_piece_marshall_correct_architecture.py", 60)
    ]
    
    results = []
    
    for script_name, timeout_minutes in training_scripts:
        logger.info(f"\n{'='*20} {script_name} {'='*20}")
        
        success = run_training_script(script_name, timeout_minutes)
        results.append((script_name, success))
        
        if not success:
            logger.error(f"❌ {script_name} failed. Stopping training pipeline.")
            break
        
        logger.info(f"✅ {script_name} completed successfully")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TRAINING SUMMARY")
    logger.info("=" * 80)
    
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{script_name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info(f"\n🎯 Overall: {successful}/{total} models trained successfully")
    
    if successful == total:
        logger.info("🎉 All Marshall models trained successfully with correct architectures!")
        logger.info("📍 Models saved in: models_marshall_improved/")
        logger.info("   - occupancy_marshall_correct_architecture.pt")
        logger.info("   - color_marshall_correct_architecture.pt")
        logger.info("   - piece_marshall_correct_architecture.pt")
    else:
        logger.error("❌ Some models failed to train. Check logs above for details.")

if __name__ == "__main__":
    main()
