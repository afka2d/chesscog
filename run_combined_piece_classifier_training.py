#!/usr/bin/env python3
"""
Run Combined Piece Classifier Training
Trains a model using both grey background and Marshall data for maximum generalization
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_combined_piece_classifier_training():
    """Run the combined piece classifier training"""
    logger.info("🚀 Starting Combined Piece Classifier Training")
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
    
    # Check if grey background dataset exists
    grey_dataset_path = Path("grey_background_dataset/pieces")
    if not grey_dataset_path.exists():
        logger.error(f"❌ Grey background dataset not found at {grey_dataset_path}")
        return False
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Run training
    try:
        logger.info("📚 Starting combined piece classifier training...")
        logger.info("⏰ This may take 2-4 hours depending on your hardware and data size")
        logger.info("🔄 Training will run automatically without manual intervention")
        logger.info("🎯 This approach combines grey background and Marshall data for maximum generalization")
        
        # Run the training script
        result = subprocess.run([
            sys.executable, "train_combined_piece_classifier.py"
        ], capture_output=True, text=True, timeout=14400)  # 4 hour timeout
        
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
            logger.info("✅ Combined piece classifier training completed successfully!")
            
            # Check if model was saved
            model_path = output_dir / "combined_piece_classifier.pt"
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
        logger.error("❌ Training timed out after 4 hours")
        return False
    except Exception as e:
        logger.error(f"❌ Error running training: {e}")
        return False

def main():
    """Main function"""
    logger.info("🎯 Combined Piece Classifier Training Launcher")
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
    success = run_combined_piece_classifier_training()
    
    if success:
        logger.info("\n🎉 SUCCESS!")
        logger.info("✅ Combined piece classifier model trained successfully")
        logger.info("📁 Model saved in: models_marshall_improved/combined_piece_classifier.pt")
        logger.info("🔧 This model combines grey background and Marshall data for maximum generalization")
        logger.info("🚀 You can now use this improved model in your API")
    else:
        logger.error("\n❌ FAILED!")
        logger.error("❌ Combined piece classifier training failed")
        logger.error("🔍 Check the logs above for error details")

if __name__ == "__main__":
    main()
