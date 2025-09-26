#!/usr/bin/env python3
"""
Run Marshall piece classification training without manual intervention
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_piece_classification_training():
    """Run the piece classification training"""
    logger.info("🚀 Starting Marshall Piece Classification Training")
    logger.info("=" * 60)
    
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
    
    # Create output directory
    output_dir = Path("models_marshall_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Run training
    try:
        logger.info("📚 Starting piece classification training...")
        logger.info("⏰ This may take 30-60 minutes depending on your hardware")
        logger.info("🔄 Training will run automatically without manual intervention")
        
        # Run the training script
        result = subprocess.run([
            sys.executable, "train_piece_classification_marshall.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
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
            logger.info("✅ Piece classification training completed successfully!")
            
            # Check if model was saved
            model_path = output_dir / "piece_classification_marshall.pt"
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
        logger.error("❌ Training timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"❌ Error running training: {e}")
        return False

def main():
    """Main function"""
    logger.info("🎯 Marshall Piece Classification Training Launcher")
    logger.info("=" * 60)
    
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
    success = run_piece_classification_training()
    
    if success:
        logger.info("\n🎉 SUCCESS!")
        logger.info("✅ Marshall piece classification model trained successfully")
        logger.info("📁 Model saved in: models_marshall_improved/piece_classification_marshall.pt")
        logger.info("🔧 You can now use this model in your API without affecting the current working model")
    else:
        logger.error("\n❌ FAILED!")
        logger.error("❌ Marshall piece classification training failed")
        logger.error("🔍 Check the logs above for error details")

if __name__ == "__main__":
    main()
