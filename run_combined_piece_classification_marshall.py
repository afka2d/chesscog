#!/usr/bin/env python3
"""
Run Combined Marshall piece classification training without manual intervention
Uses both previous training data AND Marshall data for better performance
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_combined_piece_classification_training():
    """Run the combined piece classification training"""
    logger.info("🚀 Starting Combined Marshall Piece Classification Training")
    logger.info("=" * 70)
    
    # Check if Marshall annotations exist
    annotations_path = Path("marshall_chess_annotations/annotations.json")
    if not annotations_path.exists():
        logger.error(f"❌ Marshall annotations not found at {annotations_path}")
        return False
    
    # Check if Marshall photos directory exists
    photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    if not photos_dir.exists():
        logger.error(f"❌ Marshall photos directory not found at {photos_dir}")
        return False
    
    # Check if previous training data exists
    previous_data_paths = [
        "data/pieces",
        "models/piece_classifier/train",
        "runs/piece_classifier/ResNet/train"
    ]
    
    previous_data_found = False
    for path in previous_data_paths:
        if Path(path).exists():
            previous_data_found = True
            logger.info(f"✅ Found previous training data at: {path}")
            break
    
    if not previous_data_found:
        logger.warning("⚠️ No previous training data found - will use Marshall data only")
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Run training
    try:
        logger.info("📚 Starting combined piece classification training...")
        logger.info("⏰ This may take 45-90 minutes depending on your hardware and data size")
        logger.info("🔄 Training will run automatically without manual intervention")
        logger.info("🎯 This approach combines previous training data with Marshall data")
        
        # Run the training script
        result = subprocess.run([
            sys.executable, "train_piece_classification_combined_marshall.py"
        ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        # Log output
        if result.stdout:
            logger.info("Training output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("Training warnings/errors:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        # Check if training was successful
        if result.returncode == 0:
            logger.info("✅ Combined piece classification training completed successfully!")
            
            # Check if model was saved
            model_path = output_dir / "piece_classification_combined_marshall.pt"
            if model_path.exists():
                logger.info(f"✅ Model saved: {model_path}")
                logger.info(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                logger.warning("⚠️ Model file not found after training")
            
            return True
        else:
            logger.error(f"❌ Training failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"❌ Error running training: {e}")
        return False

def main():
    """Main function"""
    logger.info("🎯 Combined Marshall Piece Classification Training Launcher")
    logger.info("=" * 70)
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    
    # Check Python packages
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        from sklearn.model_selection import train_test_split
        logger.info("✅ All required Python packages are available")
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        return
    
    # Check HEIC support
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        logger.info("✅ HEIC support is available")
    except ImportError:
        logger.warning("⚠️ HEIC support not available - HEIC files may not load properly")
    
    # Run training
    success = run_combined_piece_classification_training()
    
    if success:
        logger.info("\n🎉 SUCCESS!")
        logger.info("✅ Combined Marshall piece classification model trained successfully")
        logger.info("📁 Model saved in: models_marshall_improved/piece_classification_combined_marshall.pt")
        logger.info("🔧 This model combines previous training data with Marshall data for better performance")
        logger.info("🚀 You can now use this improved model in your API")
    else:
        logger.error("\n❌ FAILED!")
        logger.error("❌ Combined piece classification training failed")
        logger.error("🔍 Check the logs above for error details")

if __name__ == "__main__":
    main()
